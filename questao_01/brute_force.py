"""
Questão 1 — Etapa A: Brute Force (benchmark ótimo em subconjunto pequeno).

Enumera TODAS as combinações de opções permitidas (níveis que respeitam o piso de margem + SUSPENDER)
e guarda a de maior objetivo (sum W) viável em (R2) capacidade e (R3) número de descontos.

Complexidade (n = SKUs do subconjunto, O = opções permitidas por SKU, O <= K+1 = 8):
  tempo   O(O^n * n)  — o laço sobre itertools.product domina; cada combinação soma n valores
  memória O(n)        — product gera combinações sob demanda; só guardamos a melhor
Viável até n ~ 7 (8^7 ~ 2,1 milhões de combinações); acima disso usamos o GA.
"""

import itertools

import numpy as np


def brute_force(inst):
    """Retorna (melhor_solucao np.ndarray, melhor_objetivo, combinacoes_avaliadas)."""
    opcoes = [inst.opcoes_permitidas(i) for i in range(inst.n)]
    W, U, desconto = inst.W, inst.U, inst.desconto
    cap, max_desc = inst.capacidade_unidades, inst.max_skus_desconto

    melhor_obj, melhor = -np.inf, None
    avaliadas = 0
    for combinacao in itertools.product(*opcoes):
        avaliadas += 1
        obj = unidades = 0.0
        n_desc = 0
        for i, k in enumerate(combinacao):
            obj += W[i, k]
            unidades += U[i, k]
            n_desc += desconto[k]
        if unidades <= cap and n_desc <= max_desc and obj > melhor_obj:
            melhor_obj, melhor = obj, combinacao

    return np.array(melhor), float(melhor_obj), avaliadas
