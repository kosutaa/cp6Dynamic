"""
Questão 2 — Etapas A (estimação) e D (validação): Monte Carlo.

Etapa A  estimar_estatisticas(): N_c cenários de CONSTRUÇÃO (stream de seed [seed, 1]).
         Para cada SKU j, semana t e nível k produz a distribuição empírica (pmf) da demanda semanal
         e o multiplicador médio de logística. Essas pmfs são a ENTRADA da Programação Dinâmica.
Etapa D  simular_plano(): N_v cenários NOVOS de VALIDAÇÃO (stream [seed, 2], nunca usados na
         construção), aplicando a política da DP (preço depende do estoque realizado) e o plano
         promocional do Greedy. Mede probabilidade de pagar as dívidas, distribuição de caixa etc.

Complexidade (N = cenários, T = semanas, J = SKUs, K = níveis, S = estoque máximo):
  estimar   tempo O(N*T*J*K) (sorteio Poisson + bincount)   memória O(N*J*K + J*T*K*S) (lote semanal + pmfs)
  simular   tempo O(N*T*J)                                 memória O(N*J + N*T) (estado e trajetória de caixa)
"""

import numpy as np


def gerar_cenarios(inst, n, rng):
    """Fatores exógenos: eps (n,T,J), razão de preço do concorrente (n,T,J), logística (n,T)."""
    T, J = inst.T, inst.J
    p = inst.params
    sd, sc, sl = inst.sigma_demanda, inst.sigma_concorrente, inst.sigma_logistica

    z_mercado = rng.standard_normal((n, T, 1))
    z_sku = rng.standard_normal((n, T, J))
    rho = p["rho_mercado"]
    eps = np.exp(sd * (np.sqrt(rho) * z_mercado + np.sqrt(1 - rho) * z_sku) - sd ** 2 / 2)

    phi = p["phi_concorrente"]
    x = np.zeros((n, T, J))
    x_ant = sc * rng.standard_normal((n, J))
    for t in range(T):
        x_ant = phi * x_ant + sc * np.sqrt(1 - phi ** 2) * rng.standard_normal((n, J))
        x[:, t] = x_ant
    razao_conc = np.exp(x)

    logistica = np.exp(sl * rng.standard_normal((n, T)) - sl ** 2 / 2)
    return {"eps": eps, "razao_conc": razao_conc, "logistica": logistica}


def _efeito_preco(inst):
    """(J, K+1): m_k^(-e_j), com SUSPENDER = 0 (sem vendas)."""
    ef = inst.niveis[None, :] ** (-inst.elasticidade[:, None])
    return np.hstack([ef, np.zeros((inst.J, 1))])


def _efeito_concorrente(inst, razao_conc):
    """Preço do concorrente relativo à base -> multiplicador de demanda (pc/pc0)^(gama*e)."""
    return razao_conc ** (inst.params["gama_concorrente"] * inst.elasticidade)


def estimar_estatisticas(inst, n_cenarios=2000, seed=0):
    """Etapa A. Retorna pmf (J,T,K+1,Smax+1), logística média (T,) e resumo por nível de preço."""
    rng = np.random.default_rng([seed, 1])
    cen = gerar_cenarios(inst, n_cenarios, rng)
    J, T, K1 = inst.J, inst.T, inst.K + 1
    smax = int(inst.estoque.max())
    pmf = np.zeros((J, T, K1, smax + 1))
    ef_preco = _efeito_preco(inst)
    soma_d = np.zeros((J, K1))
    amostras_media = np.zeros((n_cenarios, J, K1))

    for t in range(T):
        mult = inst.demanda_base[None, :] * cen["eps"][:, t] * _efeito_concorrente(inst, cen["razao_conc"][:, t])
        lam = mult[:, :, None] * ef_preco[None, :, :]                      # (N, J, K+1)
        D = rng.poisson(lam)
        soma_d += D.sum(axis=0)
        amostras_media += D / T
        for j in range(J):
            S = int(inst.estoque[j])
            Dj = np.minimum(D[:, j, :], S)                                  # massa >= S acumulada em S
            for k in range(K1):
                pmf[j, t, k, :S + 1] = np.bincount(Dj[:, k], minlength=S + 1) / n_cenarios

    logistica_media = cen["logistica"].mean(axis=0)
    margem = inst.margem_unitaria(logistica_media.mean())                   # (J, K+1)
    demanda_media = soma_d / (n_cenarios * T)
    resumo = []
    for j in range(J):
        for k in range(inst.K):
            resumo.append({
                "sku": inst.skus[j], "nivel": float(inst.niveis[k]),
                "preco": round(inst.preco_atual[j] * inst.niveis[k], 2),
                "demanda_semanal_media": round(demanda_media[j, k], 3),
                "demanda_p05": float(np.percentile(amostras_media[:, j, k], 5)),
                "demanda_p95": float(np.percentile(amostras_media[:, j, k], 95)),
                "margem_unitaria_esperada": round(margem[j, k], 2),
                "caixa_semanal_esperado_sem_ruptura": round(demanda_media[j, k] * margem[j, k], 2),
            })
    return {"pmf": pmf, "logistica_media": logistica_media, "resumo": resumo, "n_cenarios": n_cenarios}


def simular_plano(inst, politica, plano_promo=None, acoes=None, n_cenarios=5000, seed=0, stream=2):
    """
    Etapa D. politica: (J, T, Smax+1) índice de nível por SKU/semana/estoque (saída da DP).
    plano_promo: (T, J) índice da ação promocional ou -1; acoes: lista de dicts (greedy.ACOES_PADRAO).
    """
    rng = np.random.default_rng([seed, stream])
    cen = gerar_cenarios(inst, n_cenarios, rng)
    J, T, N = inst.J, inst.T, n_cenarios
    precos = inst.precos()
    ef_preco = _efeito_preco(inst)
    linhas_j = np.arange(J)[None, :]

    estoque = np.tile(inst.estoque, (N, 1))
    saldo = np.full(N, inst.caixa_inicial)
    atraso = np.zeros(N)
    receita = np.zeros(N)
    gasto_promo = np.zeros(N)
    saldo_semana = np.zeros((N, T))
    caixa_operacional = np.zeros(N)
    prazos = sorted(inst.dividas)
    pago = np.zeros((N, len(prazos)), dtype=bool)
    estoque_medio = np.zeros((T + 1, J))
    estoque_medio[0] = inst.estoque
    preco_medio = np.zeros((T, J))

    for t in range(T):
        k = politica[linhas_j, t, estoque]                                  # (N, J) — política com feedback
        uplift = np.ones(J)
        custo_unit_promo = np.zeros(J)
        custo_fixo_promo = 0.0
        if plano_promo is not None:
            for j in range(J):
                a = plano_promo[t, j]
                if a >= 0:
                    uplift[j] = 1 + acoes[a]["uplift"]
                    custo_unit_promo[j] = acoes[a]["custo_por_unidade_frac_logistica"] * inst.logistica_base[j]
                    custo_fixo_promo += acoes[a]["custo_fixo"]

        lam = (inst.demanda_base[None, :] * ef_preco[linhas_j, k] * cen["eps"][:, t]
               * _efeito_concorrente(inst, cen["razao_conc"][:, t]) * uplift[None, :])
        vendas = np.minimum(estoque, rng.poisson(lam))
        p = precos[linhas_j, k]
        margem = (p * (1 - inst.taxa) - inst.custo - cen["logistica"][:, t:t + 1] * inst.logistica_base
                  - custo_unit_promo)
        caixa_t = (vendas * margem).sum(axis=1) - custo_fixo_promo
        gasto_promo += (vendas * custo_unit_promo).sum(axis=1) + custo_fixo_promo
        receita += (vendas * p).sum(axis=1)
        caixa_operacional += caixa_t
        saldo += caixa_t
        estoque = estoque - vendas
        estoque_medio[t + 1] = estoque.mean(axis=0)
        preco_medio[t] = p.mean(axis=0)

        semana = t + 1
        if semana in inst.dividas:
            idx = prazos.index(semana)
            exigido = atraso + inst.dividas[semana]
            ok = saldo >= exigido
            pago[:, idx] = ok
            atraso = np.where(ok, 0.0, exigido - np.maximum(saldo, 0))
            saldo = np.where(ok, saldo - exigido, np.minimum(saldo, 0))
        saldo_semana[:, t] = saldo

    return {
        "pago_no_prazo": pago,                       # (N, 3)
        "prazos": prazos,
        "todas_pagas": pago.all(axis=1),
        "saldo_final": saldo,                        # caixa após pagar as dívidas
        "divida_em_atraso": atraso,
        "caixa_operacional": caixa_operacional,      # gerado nas 12 semanas
        "receita": receita,
        "gasto_promocional": gasto_promo,
        "estoque_final": estoque,                    # (N, J)
        "saldo_semana": saldo_semana,
        "estoque_medio": estoque_medio,              # (T+1, J)
        "preco_medio": preco_medio,                  # (T, J)
        "choque_mercado_medio": cen["eps"].mean(axis=(1, 2)),
        "logistica_media": cen["logistica"].mean(axis=1),
    }


def politica_fixa(inst, nivel=1.0):
    """Política de referência: mesmo nível de preço em todas as semanas e estoques."""
    k = int(np.argmin(np.abs(inst.niveis - nivel)))
    return np.full((inst.J, inst.T, int(inst.estoque.max()) + 1), k, dtype=int)
