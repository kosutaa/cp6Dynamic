"""
Questão 1 — Etapa B: Algoritmo Genético para a política de preços do catálogo completo.

Cromossomo: vetor de n inteiros; gene i = opção de preço do SKU i (índice em niveis ou SUSPENDER).
Somente opções permitidas pelo piso de margem são geradas (inicialização e mutação).

Fitness = objetivo (sum W) - penalidade, onde penalidade = P*[violou] + P*(excesso_cap/cap + excesso_desc),
P = amplitude do objetivo (modelo.amplitude_objetivo). Assim qualquer solução viável vence qualquer
inviável, sem descartar material genético útil durante a busca.

Operadores (justificativa em docs/decisoes.md):
  seleção    torneio de tamanho 3 (pressão seletiva moderada, sem escalar fitness)
  crossover  uniforme, prob. pc (genes independentes; não há ordem entre SKUs)
  mutação    por gene, prob. pm (default 1/n), sorteia outra opção permitida
  elitismo   'elite' melhores copiados intactos (fitness do melhor nunca piora)
  parada     max_geracoes OU 'paciencia' gerações sem melhora do melhor

Complexidade (G = gerações, P = população, n = SKUs):
  tempo   O(G * P * n)  — avaliação da população (gather em V/U) e operadores são O(P*n) por geração
  memória O(P * n + n*K) — população + tabelas pré-computadas (+ histórico O(G))
"""

from dataclasses import dataclass

import numpy as np

from questao_01.modelo import amplitude_objetivo


@dataclass
class ConfigGA:
    populacao: int = 80
    max_geracoes: int = 400
    paciencia: int = 80
    pc: float = 0.9
    pm: float = None        # None -> 1/n
    torneio: int = 3
    elite: int = 2
    seed: int = 0
    avaliacao: str = "vetorizada"   # "vetorizada" (otimizada) ou "laco" (versão original)


def _tabela_permitidas(inst):
    """Matriz (n, Omax) com as opções permitidas de cada SKU (preenchida repetindo) e contagens."""
    listas = [inst.opcoes_permitidas(i) for i in range(inst.n)]
    contagem = np.array([len(l) for l in listas])
    tabela = np.zeros((inst.n, contagem.max()), dtype=int)
    for i, l in enumerate(listas):
        tabela[i] = np.resize(l, contagem.max())
    return tabela, contagem


def _sortear_opcoes(rng, tabela, contagem, shape):
    """Sorteia opção permitida uniforme para cada gene; shape = (linhas, n)."""
    r = (rng.random(shape) * contagem[None, :]).astype(int)
    return tabela[np.arange(tabela.shape[0])[None, :], r]


def fitness_vetorizado(inst, pop, P):
    """Avalia a população inteira (P x n) com indexação avançada nas tabelas: O(P*n) em numpy."""
    linhas = np.arange(inst.n)[None, :]
    obj = inst.W[linhas, pop].sum(axis=1)
    unidades = inst.U[linhas, pop].sum(axis=1)
    n_desc = inst.desconto[pop].sum(axis=1)
    exc_cap = np.maximum(0.0, unidades - inst.capacidade_unidades) / inst.capacidade_unidades
    exc_desc = np.maximum(0, n_desc - inst.max_skus_desconto)
    violou = (exc_cap > 0) | (exc_desc > 0)
    return obj - P * violou - P * (exc_cap + exc_desc)


def fitness_laco(inst, pop, P):
    """Versão ANTES da otimização: recalcula demanda/margem gene a gene em Python puro."""
    resultado = np.empty(len(pop))
    for j, individuo in enumerate(pop):
        obj = unidades = 0.0
        n_desc = 0
        for i, k in enumerate(individuo):
            u = margem = 0.0
            if k != inst.SUSPENDER:
                m = inst.niveis[k]
                preco = inst.preco_atual[i] * m
                u = np.floor(min(inst.estoque[i], inst.demanda[i] * m ** (-inst.elasticidade[i])))
                margem = preco * (1 - inst.taxa[i]) - inst.custo[i] - inst.custo_logistico[i]
                n_desc += m < 1.0
            obj += u * margem - (inst.estoque[i] - u) * inst.custo[i] * inst.taxa_parado[i]
            unidades += u
        exc_cap = max(0.0, unidades - inst.capacidade_unidades) / inst.capacidade_unidades
        exc_desc = max(0, n_desc - inst.max_skus_desconto)
        violou = exc_cap > 0 or exc_desc > 0
        resultado[j] = obj - P * violou - P * (exc_cap + exc_desc)
    return resultado


def algoritmo_genetico(inst, cfg=None):
    """Retorna dict com melhor solução, fitness, histórico (melhor/médio por geração) e gerações."""
    cfg = cfg or ConfigGA()
    rng = np.random.default_rng(cfg.seed)
    pm = cfg.pm if cfg.pm is not None else 1.0 / inst.n
    avaliar = fitness_vetorizado if cfg.avaliacao == "vetorizada" else fitness_laco
    P = amplitude_objetivo(inst)
    tabela, contagem = _tabela_permitidas(inst)

    pop = _sortear_opcoes(rng, tabela, contagem, (cfg.populacao, inst.n))
    fit = avaliar(inst, pop, P)

    hist_melhor, hist_medio = [], []
    melhor_fit, sem_melhora, geracao = -np.inf, 0, 0
    for geracao in range(1, cfg.max_geracoes + 1):
        ordem = np.argsort(fit)[::-1]
        if fit[ordem[0]] > melhor_fit + 1e-9:
            melhor_fit, sem_melhora = fit[ordem[0]], 0
        else:
            sem_melhora += 1
        hist_melhor.append(float(fit[ordem[0]]))
        hist_medio.append(float(fit.mean()))
        if sem_melhora >= cfg.paciencia:
            break

        n_filhos = cfg.populacao - cfg.elite
        # seleção por torneio: para cada pai, o melhor de 'torneio' sorteados
        competidores = rng.integers(0, cfg.populacao, size=(2 * n_filhos, cfg.torneio))
        vencedores = competidores[np.arange(2 * n_filhos), np.argmax(fit[competidores], axis=1)]
        pais_a, pais_b = pop[vencedores[:n_filhos]], pop[vencedores[n_filhos:]]

        # crossover uniforme
        mascara = rng.random((n_filhos, inst.n)) < 0.5
        cruza = rng.random(n_filhos) < cfg.pc
        filhos = np.where(mascara & cruza[:, None], pais_b, pais_a)

        # mutação por gene
        muta = rng.random(filhos.shape) < pm
        filhos = np.where(muta, _sortear_opcoes(rng, tabela, contagem, filhos.shape), filhos)

        # elitismo
        pop = np.vstack([pop[ordem[:cfg.elite]], filhos])
        fit = np.concatenate([fit[ordem[:cfg.elite]], avaliar(inst, filhos, P)])

    idx = int(np.argmax(fit))
    return {
        "solucao": pop[idx].copy(),
        "fitness": float(fit[idx]),
        "historico_melhor": hist_melhor,
        "historico_medio": hist_medio,
        "geracoes": geracao,
    }
