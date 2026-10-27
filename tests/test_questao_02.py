import itertools

import numpy as np
import pytest

from questao_02.dynamic_programming import (dp_sku, dp_top_down_memo, pesos_semanais, resolver_dp,
                                            trajetoria_esperada)
from questao_02.greedy import greedy_promocional
from questao_02.modelo import construir_instancia
from questao_02.monte_carlo import estimar_estatisticas, politica_fixa, simular_plano


@pytest.fixture
def inst(catalogo_q2_pequeno, config_q2):
    return construir_instancia(catalogo_q2_pequeno, config_q2)


@pytest.fixture
def estat(inst):
    return estimar_estatisticas(inst, 500, seed=11)


def test_pmf_valida(inst, estat):
    for j in range(inst.J):
        S = inst.estoque[j]
        somas = estat["pmf"][j, :, :, :S + 1].sum(axis=2)
        assert np.allclose(somas, 1.0)
        assert (estat["pmf"][j, :, :, S + 1:] == 0).all()
        assert np.allclose(estat["pmf"][j, :, -1, 0], 1.0)  # SUSPENDER não vende


def test_mc_reprodutivel_por_seed(inst):
    a = estimar_estatisticas(inst, 300, seed=5)
    b = estimar_estatisticas(inst, 300, seed=5)
    c = estimar_estatisticas(inst, 300, seed=6)
    assert np.array_equal(a["pmf"], b["pmf"])
    assert not np.array_equal(a["pmf"], c["pmf"])


def test_dp_bottom_up_igual_top_down_memo(inst, estat):
    dp = resolver_dp(inst, estat, lam=0.25)
    j = 2
    T, S = 3, int(inst.estoque[j])
    pmf = estat["pmf"][j, :T, :, :S + 1]
    margem, pesos = dp["margens"][:T, j, :], dp["pesos"][:T]
    V, _ = dp_sku(pmf, margem, pesos, S)
    valor, estados, consultas = dp_top_down_memo(pmf, margem, pesos, S)
    assert V[0, S] == pytest.approx(valor)
    assert consultas > estados  # houve reutilização de subproblemas


def test_dp_igual_forca_bruta_em_politicas_abertas():
    """Instância minúscula com demanda determinística: DP = melhor sequência de preços enumerada."""
    T, S = 3, 5
    margem = np.array([[2.0, 5.0, 0.0]] * T)              # 2 níveis + SUSPENDER
    demandas = [3, 1, 0]                                   # nível barato vende 3, caro vende 1
    pmf = np.zeros((T, 3, S + 1))
    for k, d in enumerate(demandas):
        pmf[:, k, d] = 1.0
    V, pol = dp_sku(pmf, margem, np.ones(T), S)
    melhor = -1
    for seq in itertools.product(range(3), repeat=T):
        s, total = S, 0.0
        for t, k in enumerate(seq):
            v = min(s, demandas[k])
            total += v * margem[t, k]
            s -= v
        melhor = max(melhor, total)
    assert V[0, S] == pytest.approx(melhor)


def test_pesos_privilegiam_caixa_antecipado(inst):
    w = pesos_semanais(inst, 0.5)
    assert w[0] > w[4] > w[8] == 1.0
    assert np.allclose(pesos_semanais(inst, 0.0), 1.0)


def test_politica_dp_nunca_vende_abaixo_do_custo_esperado(inst, estat):
    dp = resolver_dp(inst, estat)
    for j in range(inst.J):
        for t in range(inst.T):
            for s in range(1, inst.estoque[j] + 1):
                k = dp["politica"][j, t, s]
                assert k == inst.K or dp["margens"][t, j, k] > 0


def test_dp_melhor_que_preco_fixo_em_valor_esperado(inst, estat):
    dp = resolver_dp(inst, estat)
    sim_dp = simular_plano(inst, dp["politica"], n_cenarios=3000, seed=1)
    sim_fixo = simular_plano(inst, politica_fixa(inst), n_cenarios=3000, seed=1)
    assert sim_dp["caixa_operacional"].mean() >= sim_fixo["caixa_operacional"].mean() * 0.99


def test_greedy_respeita_slots_orcamento_e_um_por_sku(inst, estat):
    dp = resolver_dp(inst, estat)
    traj = trajetoria_esperada(inst, estat, dp["politica"])
    g = greedy_promocional(inst, estat, dp, traj, slots_semana=2, orcamento_semanal=100.0)
    for u in g["uso_semanal"]:
        assert u["acoes"] <= 2 and u["gasto_esperado"] <= 100.0 + 1e-9
    assert ((g["plano"] >= 0).sum(axis=1) <= 2).all()


def test_simulacao_estoque_nunca_negativo_e_pagamentos_coerentes(inst, estat):
    dp = resolver_dp(inst, estat)
    sim = simular_plano(inst, dp["politica"], n_cenarios=1000, seed=2)
    assert (sim["estoque_final"] >= 0).all()
    assert (sim["todas_pagas"] == sim["pago_no_prazo"].all(axis=1)).all()
    assert ((sim["divida_em_atraso"] > 0) <= ~sim["pago_no_prazo"][:, -1]).all()


def test_validacao_usa_cenarios_diferentes_da_construcao(inst):
    pol = politica_fixa(inst)
    a = simular_plano(inst, pol, n_cenarios=200, seed=3, stream=1)
    b = simular_plano(inst, pol, n_cenarios=200, seed=3, stream=2)
    assert not np.allclose(a["caixa_operacional"], b["caixa_operacional"])


def test_dividas_impossiveis_falham(catalogo_q2_pequeno, config_q2):
    inst = construir_instancia(catalogo_q2_pequeno, config_q2, {"ajuste_dividas": 1000})
    sim = simular_plano(inst, politica_fixa(inst), n_cenarios=200, seed=0)
    assert sim["todas_pagas"].mean() == 0.0


def test_cenario_surpresa_e_entrada_invalida(catalogo_q2_pequeno, config_q2):
    alt = construir_instancia(catalogo_q2_pequeno, config_q2, {"choque_preco_concorrente": -0.08,
                                                               "skus_removidos": ["U-001"]})
    assert alt.J == 2
    assert alt.preco_concorrente[0] == pytest.approx(55.9 * 0.92)
    ruim = catalogo_q2_pequeno.copy()
    ruim.loc[0, "demanda_base_semanal"] = 0
    with pytest.raises(ValueError):
        construir_instancia(ruim, config_q2)
