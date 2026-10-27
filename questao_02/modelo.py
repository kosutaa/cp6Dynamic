"""
Questão 2 — modelo do problema (12 semanas, 3 dívidas, incerteza).

Índices: j = SKU (J), t = semana (T = 12), k = nível de preço (K níveis + SUSPENDER).

Demanda semanal do SKU j no nível k, semana t, cenário w (Poisson com média aleatória):
    lambda = demanda_base_semanal_j * m_k^(-e_j) * (pc_tj / pc0_j)^(gama*e_j) * eps_tj * uplift_promo
    D ~ Poisson(lambda),  vendas = min(estoque, D)

Fontes de incerteza (as 3 exigidas, parâmetros de dispersão do q2_config):
  1. demanda:      eps_tj = exp(sigma_d * (sqrt(rho)*Z_t + sqrt(1-rho)*Z_tj) - sigma_d^2/2)
                   (choque de mercado Z_t comum + choque próprio do SKU; média 1)
  2. concorrente:  log(pc_tj/pc0_j) = x_tj, AR(1): x_t = phi*x_{t-1} + sigma_c*sqrt(1-phi^2)*Z  (reverte à média)
  3. logística:    L_t = exp(sigma_l*Z_t - sigma_l^2/2), comum a todos os SKUs (combustível/frete); custo = L_t*base_j

Caixa semanal = sum_j vendas*(p*(1-taxa) - custo - L_t*log_base) - custos promocionais.
Dívidas pagas nas semanas 4, 8, 12 a partir do saldo de caixa (caixa inicial + acumulado).
"""

from dataclasses import dataclass, field

import numpy as np

PARAMETROS_PADRAO = {
    "gama_concorrente": 0.5,        # elasticidade cruzada = gama * elasticidade própria
    "rho_mercado": 0.3,             # correlação do choque de demanda entre SKUs
    "phi_concorrente": 0.6,         # persistência semanal do preço do concorrente
    # cenário-surpresa
    "choque_preco_concorrente": 0.0,  # -0.08 = concorrente reduz 8%
    "ajuste_custo_logistico": 1.0,    # 1.15 = +15%
    "ajuste_orcamento": 1.0,          # 0.75 = -25%
    "ajuste_dividas": 1.0,            # 1.10 = dívidas +10%
    "skus_removidos": [],
}


@dataclass
class InstanciaQ2:
    skus: np.ndarray
    categorias: np.ndarray
    preco_atual: np.ndarray
    custo: np.ndarray
    estoque: np.ndarray            # int
    demanda_base: np.ndarray
    elasticidade: np.ndarray
    taxa: np.ndarray
    logistica_base: np.ndarray
    preco_concorrente: np.ndarray
    niveis: np.ndarray
    semanas: int
    caixa_inicial: float
    dividas: dict                  # {semana: valor}
    sigma_demanda: float
    sigma_concorrente: float
    sigma_logistica: float
    orcamento_semanal: float
    limite_risco_falha: float
    params: dict = field(default_factory=dict)

    @property
    def J(self):
        return len(self.skus)

    @property
    def K(self):
        return len(self.niveis)

    @property
    def T(self):
        return self.semanas

    def precos(self):
        """(J, K+1): preço por SKU e nível; última coluna = SUSPENDER (preço 0)."""
        p = self.preco_atual[:, None] * self.niveis[None, :]
        return np.hstack([p, np.zeros((self.J, 1))])

    def margem_unitaria(self, mult_logistica):
        """(..., J, K+1): margem por unidade dado o multiplicador de logística (escalar ou array (...))."""
        mult = np.asarray(mult_logistica, dtype=float)
        base = self.precos() * (1 - self.taxa[:, None]) - self.custo[:, None]
        m = base - mult[..., None, None] * self.logistica_base[:, None]
        m[..., -1] = 0.0
        return m


def construir_instancia(catalogo, config, parametros=None):
    p = {**PARAMETROS_PADRAO, **(parametros or {})}
    cat = catalogo[~catalogo["sku"].isin(p["skus_removidos"])].reset_index(drop=True)
    if len(cat) == 0:
        raise ValueError("Catálogo vazio após remoção de SKUs.")
    if (cat["estoque_inicial"] <= 0).any() or (cat["demanda_base_semanal"] <= 0).any():
        raise ValueError("Estoque e demanda devem ser positivos.")
    var = config["variabilidade"]
    dividas = {int(k.split("_")[1]): float(v) * p["ajuste_dividas"] for k, v in config["dividas"].items()}
    return InstanciaQ2(
        skus=cat["sku"].to_numpy(),
        categorias=cat["categoria"].to_numpy(),
        preco_atual=cat["preco_atual"].to_numpy(dtype=float),
        custo=cat["custo_unitario"].to_numpy(dtype=float),
        estoque=cat["estoque_inicial"].to_numpy(dtype=int),
        demanda_base=cat["demanda_base_semanal"].to_numpy(dtype=float),
        elasticidade=cat["elasticidade"].to_numpy(dtype=float),
        taxa=cat["taxa_marketplace"].to_numpy(dtype=float),
        logistica_base=cat["custo_logistico_base"].to_numpy(dtype=float) * p["ajuste_custo_logistico"],
        preco_concorrente=cat["preco_concorrente_base"].to_numpy(dtype=float) * (1 + p["choque_preco_concorrente"]),
        niveis=np.asarray(config["niveis_preco_relativos"], dtype=float),
        semanas=int(config["numero_semanas"]),
        caixa_inicial=float(config["caixa_inicial"]),
        dividas=dividas,
        sigma_demanda=float(var["sigma_demanda"]),
        sigma_concorrente=float(var["sigma_preco_concorrente"]),
        sigma_logistica=float(var["sigma_custo_logistico"]),
        orcamento_semanal=float(config["orcamento_promocional_semanal"]) * p["ajuste_orcamento"],
        limite_risco_falha=float(config["limite_risco_falha"]),
        params=p,
    )


def catalogo_ampliado(catalogo, fator_skus=1, fator_estoque=1):
    """Escalabilidade: replica SKUs e/ou multiplica estoque e demanda (mantém a proporção)."""
    import pandas as pd
    partes = []
    for r in range(fator_skus):
        c = catalogo.copy()
        c["sku"] = c["sku"] + (f"-r{r}" if r else "")
        partes.append(c)
    c = pd.concat(partes, ignore_index=True)
    c["estoque_inicial"] = c["estoque_inicial"] * fator_estoque
    c["demanda_base_semanal"] = c["demanda_base_semanal"] * fator_estoque
    return c
