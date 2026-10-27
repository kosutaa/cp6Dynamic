"""
Questão 2 — Etapa C: Greedy semanal de recursos promocionais escassos, guiado pelo plano da DP.

Recursos por semana: no máximo 'slots_semana' ações e 'orcamento_semanal' em R$; no máximo 1 ação por SKU.
Ações (parâmetros em ACOES_PADRAO, custos fixos escalados pelo orçamento semanal do grupo):
  destaque           vitrine/home do marketplace: +12% de demanda, custo fixo
  midia              anúncio patrocinado:          +28% de demanda, custo fixo maior
  frete_promocional  frete reduzido ao cliente:    +18% de demanda, custo por unidade = 50% da logística base

Entrada vinda da DP: a trajetória esperada (estoque no início de cada semana e nível de preço escolhido),
a função de valor V_{t+1}(s) da DP e as pmfs do Monte Carlo para esse nível. Ganho esperado da ação a
no SKU j na semana t (vendas v = min(s, D) sem ação, v' = min(s, round((1+u)D)) com ação):
  Q(v)  = w_t * M * v + V_{t+1}(s - v)
  ganho = E[Q(v') - w_t * custo_unit * v'] - E[Q(v)] - w_t * custo_fixo
Usar V_{t+1} é essencial: a unidade vendida agora pela promoção deixa de ser vendida depois (o estoque é
finito). Sem esse termo o Greedy superestimava o ganho e piorava o caixa na validação.
Regra gulosa (por semana): ordena candidatos por ganho / custo_esperado e escolhe enquanto houver
slot e orçamento, com ganho > 0 e sem repetir SKU.

Complexidade (T = semanas, J = SKUs, A = ações, S = estoque):
  tempo   O(T * J * A * S) para os ganhos esperados + O(T * J*A log(J*A)) da ordenação
  memória O(J * A) por semana + plano O(T * J)
"""

import numpy as np


def acoes_padrao(orcamento_semanal):
    return [
        {"nome": "destaque", "uplift": 0.12, "custo_fixo": 0.06 * orcamento_semanal,
         "custo_por_unidade_frac_logistica": 0.0},
        {"nome": "midia", "uplift": 0.28, "custo_fixo": 0.16 * orcamento_semanal,
         "custo_por_unidade_frac_logistica": 0.0},
        {"nome": "frete_promocional", "uplift": 0.18, "custo_fixo": 0.0,
         "custo_por_unidade_frac_logistica": 0.50},
    ]


def _valor_esperado(P, s, fator, peso, margem, custo_unit, V_prox):
    """E[w*(M - custo_unit)*v + V_{t+1}(s - v)] com v = min(s, round(fator*D)); retorna (valor, E[v])."""
    d = np.arange(len(P))
    v = np.minimum(s, np.rint(fator * d)).astype(int)
    valor = (P * (peso * (margem - custo_unit) * v + V_prox[s - v])).sum()
    return float(valor), float((P * v).sum())


def greedy_promocional(inst, estat, dp, trajetoria, slots_semana=5, orcamento_semanal=None, acoes=None):
    """Retorna plano (T, J) com índice da ação ou -1, tabela de escolhas e uso de orçamento por semana."""
    orcamento_semanal = inst.orcamento_semanal if orcamento_semanal is None else orcamento_semanal
    acoes = acoes or acoes_padrao(inst.orcamento_semanal)
    plano = -np.ones((inst.T, inst.J), dtype=int)
    escolhas, uso = [], []
    por_semana = {}
    for linha in trajetoria:
        por_semana.setdefault(linha["semana"] - 1, {})[linha["sku"]] = linha
    indice_sku = {sku: j for j, sku in enumerate(inst.skus)}

    for t in range(inst.T):
        candidatos = []
        for sku, linha in por_semana[t].items():
            j = indice_sku[sku]
            s = int(linha["estoque_inicio"])
            k = int(dp["politica"][j, t, s])
            if s == 0 or k == inst.K:
                continue
            P = estat["pmf"][j, t, k, :s + 1]
            M, w = dp["margens"][t, j, k], dp["pesos"][t]
            V_prox = dp["V"][j, t + 1]
            base, _ = _valor_esperado(P, s, 1.0, w, M, 0.0, V_prox)
            for a, acao in enumerate(acoes):
                custo_unit = acao["custo_por_unidade_frac_logistica"] * inst.logistica_base[j]
                com, vendas = _valor_esperado(P, s, 1 + acao["uplift"], w, M, custo_unit, V_prox)
                custo = acao["custo_fixo"] + custo_unit * vendas
                ganho = com - base - w * acao["custo_fixo"]
                if ganho > 0 and custo > 0:
                    candidatos.append((ganho / custo, ganho, custo, j, a))

        candidatos.sort(reverse=True)
        gasto, usados, skus_usados = 0.0, 0, set()
        for razao, ganho, custo, j, a in candidatos:
            if usados >= slots_semana:
                break
            if j in skus_usados or gasto + custo > orcamento_semanal:
                continue
            plano[t, j] = a
            gasto += custo
            usados += 1
            skus_usados.add(j)
            escolhas.append({"semana": t + 1, "sku": inst.skus[j], "acao": acoes[a]["nome"],
                             "ganho_esperado": round(ganho, 2), "custo_esperado": round(custo, 2),
                             "razao_ganho_por_real": round(razao, 3)})
        uso.append({"semana": t + 1, "acoes": usados, "gasto_esperado": round(gasto, 2),
                    "orcamento": round(orcamento_semanal, 2)})
    return {"plano": plano, "acoes": acoes, "escolhas": escolhas, "uso_semanal": uso}
