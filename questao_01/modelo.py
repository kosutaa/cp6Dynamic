"""
Questão 1 — modelo do problema (compartilhado por Brute Force, GA e Greedy).

Decisão: para cada SKU i, um índice k de preço em niveis_preco_relativos (preço = preco_atual * m_k),
ou o índice especial SUSPENDER (= K), que retira o SKU do canal nos 90 dias (vende 0, caixa 0).

Demanda (elasticidade constante):  d_i(m) = demanda_base_90d_i * m^(-elasticidade_i)
Unidades vendidas:                 u_i(m) = floor(min(estoque_i, d_i(m)))
Margem unitária (caixa por unid.): M_i(m) = preco_atual_i*m*(1 - taxa_i) - custo_i - custo_logistico_i
Caixa do SKU:                      V_i(m) = u_i(m) * M_i(m)
Custo do estoque parado:           C_i(m) = (estoque_i - u_i(m)) * custo_i * taxa_parado_i
    taxa_parado_i = taxa_carregamento_90d + obsolescencia_max * min(1, idade_estoque_i / 360)
Objetivo do SKU:                   W_i(m) = V_i(m) - C_i(m)

Função objetivo: maximizar sum_i W_i. O termo C representa capital empatado e perda de valor do estoque
envelhecido (giro de estoque); sem ele (taxas = 0) o ótimo é quase sempre "subir todos os preços",
o que contraria o objetivo de girar estoque. O caixa reportado é sempre sum_i V_i (caixa puro).

Restrições
  (R1) piso de margem por SKU:  M_i(m) / (preco_atual_i*m) >= margem_minima_percentual  (filtra níveis)
  (R2) capacidade logística:    sum_i u_i <= capacidade_unidades
       capacidade_unidades = fator_capacidade * sum_i min(estoque_i, demanda_base_90d_i)
  (R3) disciplina de desconto:  #SKUs com m < 1 <= max_skus_desconto = floor(frac_max_desconto * n)

(R2) e (R3) acoplam os SKUs: sem elas o problema seria separável e trivial. Com elas vira uma
mochila de múltipla escolha com duas restrições (NP-difícil), o que justifica BF (exato, pequeno)
e GA (heurístico, catálogo completo).
"""

from dataclasses import dataclass, field

import numpy as np

PARAMETROS_PADRAO = {
    "fator_capacidade": 1.00,       # CD/transportadora expedem no máximo as unidades do cenário base
    "frac_max_desconto": 0.20,      # no máximo 20% dos SKUs com preço abaixo do atual
    "taxa_carregamento_90d": 0.06,  # custo de capital/armazenagem do estoque que sobra (90 dias)
    "obsolescencia_max": 0.40,      # perda adicional de valor p/ estoque com 360+ dias
    # cenário-surpresa (multiplicadores / ajustes)
    "ajuste_meta": 1.00,            # meta de caixa +10% -> 1.10
    "ajuste_orcamento": 1.00,       # orçamento promocional -25% -> 0.75
    "ajuste_custo_logistico": 1.00, # custo logístico +15% -> 1.15
    "skus_removidos": [],           # ex.: ["Q1-005"]
}


@dataclass
class InstanciaQ1:
    skus: np.ndarray
    categorias: np.ndarray
    preco_atual: np.ndarray
    custo: np.ndarray
    estoque: np.ndarray
    demanda: np.ndarray
    elasticidade: np.ndarray
    taxa: np.ndarray
    custo_logistico: np.ndarray
    idade: np.ndarray
    niveis: np.ndarray
    margem_minima: float
    capacidade_unidades: float
    max_skus_desconto: int
    meta_caixa: float
    divida: float
    orcamento: float
    taxa_parado: np.ndarray = field(default=None, repr=False)
    # tabelas pré-computadas, shape (n, K+1); a última coluna é SUSPENDER
    U: np.ndarray = field(default=None, repr=False)
    M: np.ndarray = field(default=None, repr=False)
    V: np.ndarray = field(default=None, repr=False)
    C: np.ndarray = field(default=None, repr=False)
    W: np.ndarray = field(default=None, repr=False)
    permitido: np.ndarray = field(default=None, repr=False)
    desconto: np.ndarray = field(default=None, repr=False)

    @property
    def n(self):
        return len(self.skus)

    @property
    def K(self):
        return len(self.niveis)

    @property
    def SUSPENDER(self):
        return self.K

    def opcoes_permitidas(self, i):
        return np.flatnonzero(self.permitido[i])

    def subconjunto(self, indices, fator_capacidade, frac_max_desconto):
        """Nova instância só com os SKUs em 'indices' (restrições reescaladas ao subconjunto)."""
        idx = np.asarray(indices)
        sub = InstanciaQ1(
            skus=self.skus[idx], categorias=self.categorias[idx], preco_atual=self.preco_atual[idx],
            custo=self.custo[idx], estoque=self.estoque[idx], demanda=self.demanda[idx],
            elasticidade=self.elasticidade[idx], taxa=self.taxa[idx],
            custo_logistico=self.custo_logistico[idx], idade=self.idade[idx], niveis=self.niveis,
            margem_minima=self.margem_minima,
            capacidade_unidades=fator_capacidade * np.minimum(self.estoque[idx], self.demanda[idx]).sum(),
            max_skus_desconto=int(np.floor(frac_max_desconto * len(idx))),
            meta_caixa=self.meta_caixa, divida=self.divida, orcamento=self.orcamento,
            taxa_parado=self.taxa_parado[idx],
        )
        precomputar_tabelas(sub)
        return sub


def construir_instancia(catalogo, config, parametros=None):
    p = {**PARAMETROS_PADRAO, **(parametros or {})}
    cat = catalogo[~catalogo["sku"].isin(p["skus_removidos"])].reset_index(drop=True)
    if len(cat) == 0:
        raise ValueError("Catálogo vazio após remoção de SKUs.")

    estoque = cat["estoque"].to_numpy(dtype=float)
    demanda = cat["demanda_base_90d"].to_numpy(dtype=float)
    if (estoque <= 0).any() or (demanda <= 0).any():
        raise ValueError("Estoque e demanda devem ser positivos.")

    inst = InstanciaQ1(
        skus=cat["sku"].to_numpy(),
        categorias=cat["categoria"].to_numpy(),
        preco_atual=cat["preco_atual"].to_numpy(dtype=float),
        custo=cat["custo_unitario"].to_numpy(dtype=float),
        estoque=estoque,
        demanda=demanda,
        elasticidade=cat["elasticidade"].to_numpy(dtype=float),
        taxa=cat["taxa_marketplace"].to_numpy(dtype=float),
        custo_logistico=cat["custo_logistico"].to_numpy(dtype=float) * p["ajuste_custo_logistico"],
        idade=cat["idade_estoque_dias"].to_numpy(dtype=float),
        niveis=np.asarray(config["niveis_preco_relativos"], dtype=float),
        margem_minima=float(config["margem_minima_percentual"]),
        capacidade_unidades=p["fator_capacidade"] * np.minimum(estoque, demanda).sum(),
        max_skus_desconto=int(np.floor(p["frac_max_desconto"] * len(cat))),
        meta_caixa=float(config["meta_caixa_90d"]) * p["ajuste_meta"],
        divida=float(config["divida_total_90d"]),
        orcamento=float(config["orcamento_promocional"]) * p["ajuste_orcamento"],
    )
    inst.taxa_parado = p["taxa_carregamento_90d"] + p["obsolescencia_max"] * np.minimum(1.0, inst.idade / 360)
    precomputar_tabelas(inst)
    return inst


def precomputar_tabelas(inst):
    """Tabelas U, M, V, C, W (n x K+1). Custo O(n*K) uma única vez; depois cada avaliação é O(n)."""
    m = inst.niveis[None, :]                                  # (1, K)
    preco = inst.preco_atual[:, None] * m                     # (n, K)
    demanda = inst.demanda[:, None] * m ** (-inst.elasticidade[:, None])
    U = np.floor(np.minimum(inst.estoque[:, None], demanda))
    M = preco * (1 - inst.taxa[:, None]) - inst.custo[:, None] - inst.custo_logistico[:, None]
    permitido = (M / preco) >= inst.margem_minima

    zeros = np.zeros((inst.n, 1))
    inst.U = np.hstack([U, zeros])
    inst.M = np.hstack([M, zeros])
    inst.V = inst.U * inst.M
    inst.C = (inst.estoque[:, None] - inst.U) * (inst.custo * inst.taxa_parado)[:, None]
    inst.W = inst.V - inst.C
    inst.permitido = np.hstack([permitido, np.ones((inst.n, 1), dtype=bool)])  # SUSPENDER sempre permitido
    inst.desconto = np.append(inst.niveis < 1.0, False)                        # SUSPENDER não é desconto


def avaliar(inst, solucao):
    """Métricas de uma solução (vetor de índices de opção, tamanho n)."""
    s = np.asarray(solucao)
    linhas = np.arange(inst.n)
    unidades = inst.U[linhas, s]
    caixa = float((inst.V[linhas, s]).sum())
    custo_parado = float((inst.C[linhas, s]).sum())
    receita = float((unidades * inst.preco_atual * np.append(inst.niveis, 0.0)[s]).sum())
    n_desc = int(inst.desconto[s].sum())
    return {
        "objetivo": caixa - custo_parado,
        "caixa": caixa,
        "custo_estoque_parado": custo_parado,
        "estoque_remanescente": float((inst.estoque - unidades).sum()),
        "receita": receita,
        "unidades": float(unidades.sum()),
        "n_descontos": n_desc,
        "permitido": bool(inst.permitido[linhas, s].all()),
        "excesso_capacidade": max(0.0, float(unidades.sum()) - inst.capacidade_unidades),
        "excesso_descontos": max(0, n_desc - inst.max_skus_desconto),
    }


def viavel(inst, solucao):
    a = avaliar(inst, solucao)
    return a["permitido"] and a["excesso_capacidade"] == 0 and a["excesso_descontos"] == 0


def amplitude_objetivo(inst):
    """sum_i (max W_i - min W_i) entre opções permitidas. Penalidade do GA = isso + 1 garante que
    qualquer solução viável tenha fitness maior que qualquer inviável."""
    Wmax = np.where(inst.permitido, inst.W, -np.inf).max(axis=1)
    Wmin = np.where(inst.permitido, inst.W, np.inf).min(axis=1)
    return float((Wmax - Wmin).sum()) + 1.0


def descrever_solucao(inst, solucao):
    """Tabela por SKU (lista de dicts) para relatórios."""
    rotulos = [f"{m:.2f}" for m in inst.niveis] + ["SUSPENSO"]
    linhas = []
    for i, k in enumerate(np.asarray(solucao)):
        preco = inst.preco_atual[i] * (inst.niveis[k] if k < inst.K else 0.0)
        linhas.append({
            "sku": inst.skus[i],
            "categoria": inst.categorias[i],
            "nivel": rotulos[k],
            "preco": round(preco, 2),
            "unidades": int(inst.U[i, k]),
            "estoque_remanescente": int(inst.estoque[i] - inst.U[i, k]),
            "margem_unitaria": round(inst.M[i, k], 2),
            "caixa": round(inst.V[i, k], 2),
            "custo_estoque_parado": round(inst.C[i, k], 2),
        })
    return linhas


def catalogo_ampliado(catalogo, fator):
    """Replica o catálogo 'fator' vezes (SKUs renomeados) para medir escalabilidade."""
    import pandas as pd
    partes = []
    for r in range(fator):
        c = catalogo.copy()
        c["sku"] = c["sku"] + (f"-r{r}" if r else "")
        partes.append(c)
    return pd.concat(partes, ignore_index=True)
