"""
Questão 2 — Etapa B: Programação Dinâmica (Bottom-Up) para a trajetória semanal de preços.

Formulação por SKU j (os SKUs são acoplados pelo caixa/dívidas; o acoplamento entra pelos pesos w_t,
calibrados por Monte Carlo no main — ver docs/decisoes.md):

  Estado     (t, s): semana t = 1..T e estoque remanescente s = 0..S_j
  Decisão    k em {níveis de preço permitidos} U {SUSPENDER}
             permitido = margem unitária esperada > 0 (não vender abaixo do custo)
  Transição  s' = s - min(s, D),  D ~ pmf_MC[j, t, k]   (pmf estimada pelo Monte Carlo)
  Recompensa r(t, s, k) = w_t * M_j(t, k) * E[min(s, D)]
             M_j(t,k) = p_k(1 - taxa) - custo - E[L_t]*logistica_base   (E[L_t] também vem do MC)
             w_t = 1 + lambda * (nº de dívidas ainda por vencer em t - 1)  -> caixa cedo vale mais
  Bellman    V_t(s) = max_k { r(t,s,k) + sum_d pmf(d) * V_{t+1}(s - min(s,d)) }
  Terminal   V_{T+1}(s) = valor_residual * custo * s

Reutilização de subproblemas: o vetor V_{t+1}(.) é calculado UMA vez e reutilizado por todos os
(S+1) estados e K decisões da semana t e por cada valor de demanda d — cada entrada V_{t+1}(s') é
consultada ~ (K+1) * (S+1) vezes. Sem memoização a recursão abriria (K+1)*(S+1) ramos por semana:
O(((K+1)(S+1))^T) chamadas. Ver contar_chamadas_sem_memo / dp_top_down_memo nos testes.

Complexidade por SKU (T = semanas, S = estoque, K = níveis):
  tempo   O(T * K * S^2)   — a esperança sum_d pmf(d) V(s - d) para todo s é uma convolução O(S^2)
  memória O(T * S)         — tabelas de política e de valor (T+1 x S+1)
Total no catálogo: tempo O(J*T*K*S^2), memória O(J*T*S) (política e V guardadas para o Greedy).
"""

from functools import lru_cache

import numpy as np


def pesos_semanais(inst, lam):
    prazos = sorted(inst.dividas)
    restantes = np.array([sum(1 for d in prazos if d >= t) for t in range(1, inst.T + 1)])
    return 1.0 + lam * np.maximum(restantes - 1, 0)


def margens_esperadas(inst, logistica_media):
    """(T, J, K+1): margem unitária esperada por semana."""
    return inst.margem_unitaria(logistica_media)


def dp_sku(pmf_j, margem_j, pesos, S, valor_residual_unit=0.0):
    """
    DP bottom-up de UM SKU.
    pmf_j: (T, K+1, >=S+1), massa >= S acumulada em S. margem_j: (T, K+1). Retorna (V (T+1,S+1), politica (T,S+1)).
    """
    T, K1 = margem_j.shape
    s = np.arange(S + 1)
    V = np.zeros((T + 1, S + 1))
    V[T] = valor_residual_unit * s
    politica = np.zeros((T, S + 1), dtype=int)

    for t in range(T - 1, -1, -1):
        prox = V[t + 1]                                       # subproblemas da semana t+1, reutilizados abaixo
        melhor = np.full(S + 1, -np.inf)
        for k in range(K1):
            if k < K1 - 1 and margem_j[t, k] <= 0:
                continue                                      # nível proibido: venderia abaixo do custo
            P = pmf_j[t, k, :S + 1]
            cauda = 1.0 - np.cumsum(P)                        # P(D > s)
            # E[min(s, D)] = sum_{d<s} P(d)*d + s*P(D >= s)
            e_vendas = np.concatenate([[0.0], np.cumsum(P * s)[:-1]]) + s * (cauda + P)
            # E[V(s - min(s,D))] = sum_{d<=s} P(d) V(s-d) + P(D>s) V(0)
            futuro = np.convolve(P, prox)[:S + 1] + np.clip(cauda, 0, None) * prox[0]
            q = pesos[t] * margem_j[t, k] * e_vendas + futuro
            melhorou = q > melhor + 1e-12
            melhor = np.where(melhorou, q, melhor)
            politica[t] = np.where(melhorou, k, politica[t])
        V[t] = melhor
    return V, politica


def resolver_dp(inst, estat, lam=0.0, valor_residual=0.0):
    """DP de todos os SKUs. Retorna politica (J, T, Smax+1), tabela V (J, T+1, Smax+1), V0 e margens."""
    pesos = pesos_semanais(inst, lam)
    margens = margens_esperadas(inst, estat["logistica_media"])      # (T, J, K+1)
    smax = int(inst.estoque.max())
    politica = np.full((inst.J, inst.T, smax + 1), inst.K, dtype=int)
    tabela_V = np.zeros((inst.J, inst.T + 1, smax + 1))   # V_t(s) de cada SKU, usado pelo Greedy
    valor = np.zeros(inst.J)
    for j in range(inst.J):
        S = int(inst.estoque[j])
        V, pol = dp_sku(estat["pmf"][j], margens[:, j, :], pesos, S, valor_residual * inst.custo[j])
        politica[j, :, :S + 1] = pol
        tabela_V[j, :, :S + 1] = V
        valor[j] = V[0, S]
    return {"politica": politica, "valor": valor, "V": tabela_V, "pesos": pesos, "margens": margens, "lambda": lam}


def trajetoria_esperada(inst, estat, politica):
    """Reconstrói o caminho ótimo seguindo a política com a demanda ESPERADA (arredondada)."""
    linhas = []
    estoque = inst.estoque.astype(int).copy()
    for t in range(inst.T):
        for j in range(inst.J):
            s = int(estoque[j])
            k = int(politica[j, t, s])
            P = estat["pmf"][j, t, k, :s + 1].copy()
            if s > 0:
                P[s] += 1.0 - P.sum()                          # massa acima de s conta como s
            e_vendas = float((np.arange(s + 1) * P).sum()) if s > 0 else 0.0
            preco = inst.preco_atual[j] * inst.niveis[k] if k < inst.K else 0.0
            linhas.append({"semana": t + 1, "sku": inst.skus[j], "estoque_inicio": s,
                           "nivel": float(inst.niveis[k]) if k < inst.K else "SUSPENSO",
                           "preco": round(preco, 2), "vendas_esperadas": round(e_vendas, 2)})
            estoque[j] = max(0, s - int(round(e_vendas)))
    return linhas


# ---------------------------------------------------------------------------------------------
# Versões didáticas (usadas nos testes e na demonstração de reutilização de subproblemas)
# ---------------------------------------------------------------------------------------------

def dp_top_down_memo(pmf_j, margem_j, pesos, S, valor_residual_unit=0.0):
    """Mesma recorrência em Top-Down com memoization (lru_cache). Retorna (valor, chamadas_unicas, consultas)."""
    T, K1 = margem_j.shape
    consultas = [0]

    @lru_cache(maxsize=None)
    def V(t, s):
        if t == T:
            return valor_residual_unit * s
        melhor = -np.inf
        for k in range(K1):
            if k < K1 - 1 and margem_j[t, k] <= 0:
                continue
            P = pmf_j[t, k]
            total = 0.0
            for d in range(len(P)):
                if P[d] == 0:
                    continue
                vendas = min(s, d)
                consultas[0] += 1
                total += P[d] * (pesos[t] * margem_j[t, k] * vendas + V(t + 1, s - vendas))
            melhor = max(melhor, total)
        return melhor

    valor = V(0, S)
    return valor, V.cache_info().currsize, consultas[0]


def contar_chamadas_sem_memo(pmf_j, margem_j, S):
    """Nº de chamadas que a recursão SEM memoização faria (contagem analítica por semana/estado)."""
    T, K1 = margem_j.shape

    @lru_cache(maxsize=None)
    def chamadas(t, s):
        if t == T:
            return 1
        total = 1
        for k in range(K1):
            if k < K1 - 1 and margem_j[t, k] <= 0:
                continue
            for d in np.flatnonzero(pmf_j[t, k]):
                total += chamadas(t + 1, s - min(s, int(d)))
        return total

    return chamadas(0, S)
