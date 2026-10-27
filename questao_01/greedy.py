"""
Questão 1 — Etapa C: Greedy para distribuir o orçamento de incentivo comercial sobre a política do GA.

Entrada: instância + solução de preços do GA (vetor de opções). O preço de cada SKU fica FIXO.
Incentivos (por unidade vendida, valor v em R$):
  cupom            cliente percebe 100% de v; empresa paga 100% de v
  cashback         cliente percebe  80% de v; empresa paga  55% de v (nem todo cashback é resgatado)
  frete_subsidiado cliente percebe 150% de v (efeito "frete grátis"); empresa paga 100%; v <= custo_logistico
Efeito na demanda: preço percebido = p*(1 - delta), delta = percepcao*v/p, via a mesma elasticidade do modelo.

Benefício líquido de uma opção = W_com_incentivo - W_sem_incentivo (já desconta o custo do incentivo).
Regra gulosa: a cada passo escolhe a MELHOR razão  delta_beneficio / delta_investimento  entre todas as
"promoções de nível" possíveis (trocar a opção atual do SKU por outra mais cara), respeitando
orçamento, capacidade logística e piso de margem. Para quando nenhuma troca tem benefício > 0.

Complexidade (n = SKUs, O = opções de incentivo por SKU, R = passos executados <= n*O):
  tempo   O(n*O) para montar a tabela + O(R * n * O) no laço guloso (varredura das trocas) -> O(n^2 * O^2)
  memória O(n*O) — tabelas de benefício, investimento e unidades
"""

import numpy as np

TIPOS_INCENTIVO = {
    #                   percepção, custo efetivo, base do valor, frações
    "cupom":            (1.00, 1.00, "preco", (0.03, 0.06, 0.09)),
    "cashback":         (0.80, 0.55, "preco", (0.03, 0.06, 0.09)),
    "frete_subsidiado": (1.50, 1.00, "logistica", (0.50, 1.00)),
}


def tabela_incentivos(inst, solucao, tipos=TIPOS_INCENTIVO):
    """Para cada SKU e opção de incentivo: benefício líquido, investimento, unidades extras.
    Coluna 0 = 'sem incentivo'. Retorna dict de arrays (n, O+1) e lista de rótulos."""
    s = np.asarray(solucao)
    linhas = np.arange(inst.n)
    m = np.append(inst.niveis, 0.0)[s]
    preco = inst.preco_atual * m
    U0, M0, W0 = inst.U[linhas, s], inst.M[linhas, s], inst.W[linhas, s]
    ativo = s != inst.SUSPENDER

    rotulos = [("nenhum", 0.0)]
    ben, inv, du = [np.zeros(inst.n)], [np.zeros(inst.n)], [np.zeros(inst.n)]
    for nome, (percep, custo_ef, base, fracoes) in tipos.items():
        for f in fracoes:
            v = f * (preco if base == "preco" else inst.custo_logistico)
            delta = np.where(ativo, percep * v / np.where(preco > 0, preco, 1.0), 0.0)
            demanda = inst.demanda * (np.where(ativo, m, 1.0) * (1 - delta)) ** (-inst.elasticidade)
            U1 = np.where(ativo, np.floor(np.minimum(inst.estoque, demanda)), 0.0)
            custo_unit = custo_ef * v
            investimento = U1 * custo_unit
            W1 = U1 * M0 - investimento - (inst.estoque - U1) * inst.custo * inst.taxa_parado
            margem_ok = (M0 - custo_unit) / np.where(preco > 0, preco, 1.0) >= inst.margem_minima
            valido = ativo & margem_ok & (delta < 1)
            rotulos.append((nome, f))
            ben.append(np.where(valido, W1 - W0, -np.inf))
            inv.append(np.where(valido, investimento, np.inf))
            du.append(np.where(valido, U1 - U0, 0.0))
    return {"beneficio": np.column_stack(ben), "investimento": np.column_stack(inv),
            "unidades_extra": np.column_stack(du)}, rotulos


def greedy_incentivos(inst, solucao, orcamento=None, tipos=TIPOS_INCENTIVO):
    """Retorna dict com escolha por SKU (índice de rótulo), investimento total, benefício e passos."""
    orcamento = inst.orcamento if orcamento is None else orcamento
    tab, rotulos = tabela_incentivos(inst, solucao, tipos)
    B, I, dU = tab["beneficio"], tab["investimento"], tab["unidades_extra"]
    linhas = np.arange(inst.n)

    unidades_base = float(inst.U[linhas, np.asarray(solucao)].sum())
    folga_cap = inst.capacidade_unidades - unidades_base
    escolha = np.zeros(inst.n, dtype=int)
    gasto, passos = 0.0, []

    while True:
        atual_b = B[linhas, escolha][:, None]
        atual_i = I[linhas, escolha][:, None]
        atual_u = dU[linhas, escolha][:, None]
        with np.errstate(invalid="ignore"):   # inf - inf em opções inválidas vira nan e é descartado
            ganho = B - atual_b
            custo = I - atual_i
        extra_u = dU - atual_u
        ok = (ganho > 1e-9) & (custo > 1e-9) & (gasto + custo <= orcamento) & (extra_u <= folga_cap + 1e-9)
        if not ok.any():
            break
        razao = np.where(ok, ganho / np.where(ok, custo, 1.0), -np.inf)
        i, o = np.unravel_index(np.argmax(razao), razao.shape)
        passos.append({"sku": inst.skus[i], "de": rotulos[escolha[i]], "para": rotulos[o],
                       "razao_beneficio_por_real": float(razao[i, o]),
                       "ganho": float(ganho[i, o]), "custo": float(custo[i, o])})
        gasto += custo[i, o]
        folga_cap -= extra_u[i, o]
        escolha[i] = o

    beneficio = float(B[linhas, escolha].sum())
    return {
        "escolha": escolha,
        "rotulos": rotulos,
        "investimento": float(I[linhas, escolha].sum()),
        "beneficio_liquido": beneficio,
        "unidades_extra": float(dU[linhas, escolha].sum()),
        "orcamento": float(orcamento),
        "passos": passos,
        "tabela": tab,
    }


def aplicar_incentivos(inst, solucao, resultado_greedy):
    """Métricas finais por SKU (antes e depois do incentivo), para a tabela antes/depois."""
    s = np.asarray(solucao)
    linhas = np.arange(inst.n)
    tab, esc, rot = resultado_greedy["tabela"], resultado_greedy["escolha"], resultado_greedy["rotulos"]
    U0, M0 = inst.U[linhas, s], inst.M[linhas, s]
    U1 = U0 + tab["unidades_extra"][linhas, esc]
    inv = tab["investimento"][linhas, esc]
    preco = inst.preco_atual * np.append(inst.niveis, 0.0)[s]
    saida = []
    for i in range(inst.n):
        nome, f = rot[esc[i]]
        saida.append({
            "sku": inst.skus[i],
            "preco_ga": round(preco[i], 2),
            "incentivo": nome if nome == "nenhum" else f"{nome} {f:.0%}",
            "unidades_antes": int(U0[i]),
            "unidades_depois": int(U1[i]),
            "caixa_antes": round(U0[i] * M0[i], 2),
            "caixa_depois": round(U1[i] * M0[i] - inv[i], 2),
            "investimento": round(inv[i], 2),
            "estoque_rem_depois": int(inst.estoque[i] - U1[i]),
        })
    return saida
