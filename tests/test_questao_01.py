import itertools

import numpy as np
import pytest

from questao_01.brute_force import brute_force
from questao_01.genetic_algorithm import ConfigGA, algoritmo_genetico, fitness_laco, fitness_vetorizado
from questao_01.greedy import greedy_incentivos
from questao_01.modelo import amplitude_objetivo, avaliar, construir_instancia, viavel


def test_tabelas_respeitam_estoque_e_piso_de_margem(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    assert (inst.U[:, :-1] <= inst.estoque[:, None]).all()
    preco = inst.preco_atual[:, None] * inst.niveis[None, :]
    margem_pct = inst.M[:, :-1] / preco
    assert (margem_pct[inst.permitido[:, :-1]] >= inst.margem_minima).all()
    assert inst.permitido[:, -1].all()  # SUSPENDER sempre permitido


def test_demanda_cai_com_preco(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1, {"fator_capacidade": 10})
    assert (np.diff(inst.U[:, :-1], axis=1) <= 0).all()


def test_brute_force_bate_com_enumeracao_independente(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    sol, obj, avaliadas = brute_force(inst)
    melhor = max(
        (inst.W[np.arange(inst.n), list(c)].sum()
         for c in itertools.product(*[inst.opcoes_permitidas(i) for i in range(inst.n)])
         if viavel(inst, c)),
    )
    assert obj == pytest.approx(melhor)
    assert viavel(inst, sol)
    assert avaliadas == np.prod([len(inst.opcoes_permitidas(i)) for i in range(inst.n)])


def test_ga_atinge_otimo_do_brute_force_em_instancia_pequena(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    _, obj_bf, _ = brute_force(inst)
    res = algoritmo_genetico(inst, ConfigGA(seed=7))
    assert viavel(inst, res["solucao"])
    assert avaliar(inst, res["solucao"])["objetivo"] == pytest.approx(obj_bf, rel=1e-3)


def test_ga_reprodutivel_por_seed(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    a = algoritmo_genetico(inst, ConfigGA(seed=3))
    b = algoritmo_genetico(inst, ConfigGA(seed=3))
    assert (a["solucao"] == b["solucao"]).all()
    assert a["historico_melhor"] == b["historico_melhor"]


def test_historico_do_melhor_nunca_piora_elitismo(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    h = algoritmo_genetico(inst, ConfigGA(seed=1))["historico_melhor"]
    assert all(b >= a - 1e-9 for a, b in zip(h, h[1:]))


def test_fitness_otimizado_igual_ao_original(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    rng = np.random.default_rng(0)
    pop = np.array([[rng.choice(inst.opcoes_permitidas(i)) for i in range(inst.n)] for _ in range(50)])
    P = amplitude_objetivo(inst)
    assert np.allclose(fitness_vetorizado(inst, pop, P), fitness_laco(inst, pop, P))


def test_penalidade_coloca_viavel_acima_de_inviavel(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1, {"frac_max_desconto": 0.0})
    P = amplitude_objetivo(inst)
    tudo_suspenso = np.full((1, inst.n), inst.SUSPENDER)
    desconto_maximo = np.array([[0 if inst.permitido[i, 0] else inst.SUSPENDER for i in range(inst.n)]])
    assert not viavel(inst, desconto_maximo[0])
    assert fitness_vetorizado(inst, tudo_suspenso, P)[0] > fitness_vetorizado(inst, desconto_maximo, P)[0]


def test_greedy_respeita_orcamento_e_capacidade(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    sol, _, _ = brute_force(inst)
    for orc in [0.0, 50.0, 500.0, 5000.0]:
        g = greedy_incentivos(inst, sol, orcamento=orc)
        assert g["investimento"] <= orc + 1e-6
        unidades = avaliar(inst, sol)["unidades"] + g["unidades_extra"]
        assert unidades <= inst.capacidade_unidades + 1e-6
        assert g["beneficio_liquido"] >= 0


def test_greedy_orcamento_zero_nao_aloca(catalogo_q1_pequeno, config_q1):
    inst = construir_instancia(catalogo_q1_pequeno, config_q1)
    g = greedy_incentivos(inst, np.full(inst.n, 4), orcamento=0.0)
    assert (g["escolha"] == 0).all() and g["passos"] == []


def test_cenario_surpresa_por_parametro(catalogo_q1_pequeno, config_q1):
    base = construir_instancia(catalogo_q1_pequeno, config_q1)
    alt = construir_instancia(catalogo_q1_pequeno, config_q1, {
        "ajuste_meta": 1.10, "ajuste_orcamento": 0.75, "ajuste_custo_logistico": 1.15, "skus_removidos": ["T-002"]})
    assert alt.meta_caixa == pytest.approx(base.meta_caixa * 1.10)
    assert alt.orcamento == pytest.approx(base.orcamento * 0.75)
    assert alt.n == base.n - 1 and "T-002" not in alt.skus
    assert alt.custo_logistico[0] == pytest.approx(base.custo_logistico[0] * 1.15)


def test_entradas_invalidas(catalogo_q1_pequeno, config_q1):
    with pytest.raises(ValueError):
        construir_instancia(catalogo_q1_pequeno, config_q1, {"skus_removidos": list(catalogo_q1_pequeno["sku"])})
    ruim = catalogo_q1_pequeno.copy()
    ruim.loc[0, "estoque"] = 0
    with pytest.raises(ValueError):
        construir_instancia(ruim, config_q1)
