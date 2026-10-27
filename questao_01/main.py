"""
Questão 1 — Operação Caixa 90. Fluxo integrado:

    dados oficiais -> modelo (tabelas U/M/V/W)
        -> [A] Brute Force no subconjunto (ótimo exato)  ──┐
        -> [B] GA no MESMO subconjunto -> gap% vs BF   <───┘  (BF valida o GA)
        -> [B] GA no catálogo completo -> política de preços
        -> [C] Greedy recebe a política do GA e distribui o orçamento de incentivo
        -> relatório financeiro (caixa, margem, estoque, cobertura da dívida, uso do orçamento)

Exemplos:
    python questao_01/main.py --modo completo
    python questao_01/main.py --seed 14 --ajuste-meta 1.10 --ajuste-orcamento 0.75
    python questao_01/main.py --remover-sku Q1-005 --populacao 150 --pm 0.05
"""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import carregar_questao, grupo_padrao, verificar_integridade
from comum.medicao import medir, medir_tempo
from questao_01.brute_force import brute_force
from questao_01.genetic_algorithm import ConfigGA, algoritmo_genetico
from questao_01.greedy import aplicar_incentivos, greedy_incentivos
from questao_01.modelo import (avaliar, catalogo_ampliado, construir_instancia,
                               descrever_solucao, viavel)

PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description="Questão 1 — Operação Caixa 90")
    ap.add_argument("--grupo", default=None, help="Código do grupo (default: o de data/dados_gerados)")
    ap.add_argument("--seed", type=int, default=None, help="Seed dos componentes aleatórios (default: seed_final)")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="completo",
                    help="rapido: sem medições de escalabilidade pesadas")
    ap.add_argument("--n-subconjunto", type=int, default=7, help="SKUs no subconjunto do Brute Force (>= 6)")
    ap.add_argument("--reinicios", type=int, default=5, help="Execuções independentes do GA no catálogo completo")
    # GA (configuração B = escolhida)
    ap.add_argument("--populacao", type=int, default=80)
    ap.add_argument("--pm", type=float, default=None, help="Taxa de mutação por gene (default 1/n)")
    ap.add_argument("--pc", type=float, default=0.9)
    ap.add_argument("--geracoes", type=int, default=400)
    ap.add_argument("--paciencia", type=int, default=80)
    ap.add_argument("--elite", type=int, default=2)
    # modelo / cenário-surpresa
    ap.add_argument("--fator-capacidade", type=float, default=1.00)
    ap.add_argument("--frac-max-desconto", type=float, default=0.20)
    ap.add_argument("--ajuste-meta", type=float, default=1.0, help="ex.: 1.10 = meta +10%%")
    ap.add_argument("--ajuste-orcamento", type=float, default=1.0, help="ex.: 0.75 = orçamento -25%%")
    ap.add_argument("--ajuste-custo-logistico", type=float, default=1.0, help="ex.: 1.15 = +15%%")
    ap.add_argument("--remover-sku", nargs="*", default=[], help="SKUs indisponíveis")
    ap.add_argument("--saida", default=str(PASTA_RESULTADOS))
    return ap.parse_args(argv)


def parametros_modelo(args):
    return {
        "fator_capacidade": args.fator_capacidade,
        "frac_max_desconto": args.frac_max_desconto,
        "ajuste_meta": args.ajuste_meta,
        "ajuste_orcamento": args.ajuste_orcamento,
        "ajuste_custo_logistico": args.ajuste_custo_logistico,
        "skus_removidos": args.remover_sku,
    }


def ga_melhor_de(inst, cfg, reinicios):
    """Executa o GA 'reinicios' vezes (seeds cfg.seed, cfg.seed+1, ...) e devolve a melhor execução."""
    execucoes = [algoritmo_genetico(inst, replace(cfg, seed=cfg.seed + r)) for r in range(reinicios)]
    return max(execucoes, key=lambda e: e["fitness"]), execucoes


def etapa_bf_vs_ga(inst, args, cfg_a, cfg_b, seed):
    """[A] + [B] no subconjunto: o BF fornece o ótimo que valida o GA."""
    n_sub = max(6, args.n_subconjunto)
    # subconjunto: os SKUs com estoque mais antigo (maior urgência de giro), ordem determinística
    indices = np.argsort(-inst.idade, kind="stable")[:n_sub]
    sub = inst.subconjunto(indices, args.fator_capacidade, args.frac_max_desconto)

    (sol_bf, obj_bf, avaliadas), t_bf, mem_bf = medir(brute_force, sub)
    linhas = [{"metodo": "Brute Force", "seed": "-", "objetivo": obj_bf, "caixa": avaliar(sub, sol_bf)["caixa"],
               "gap_percentual": 0.0, "avaliacoes": avaliadas, "tempo_s": t_bf, "memoria_pico_KiB": mem_bf,
               "solucao": str(sol_bf.tolist())}]
    for nome, cfg in [("GA config A", cfg_a), ("GA config B", cfg_b)]:
        for r in range(5):
            res, t, mem = medir(algoritmo_genetico, sub, replace(cfg, seed=seed + r))
            a = avaliar(sub, res["solucao"])
            linhas.append({"metodo": nome, "seed": seed + r, "objetivo": a["objetivo"], "caixa": a["caixa"],
                           "gap_percentual": abs(100 * (obj_bf - a["objetivo"]) / abs(obj_bf)),
                           "avaliacoes": res["geracoes"] * cfg.populacao, "tempo_s": t,
                           "memoria_pico_KiB": mem, "solucao": str(res["solucao"].tolist())})
    return pd.DataFrame(linhas), sub


def etapa_configs_ga(inst, cfg_a, cfg_b, seed, reinicios):
    """[B] Comparação de duas configurações do GA no catálogo completo (convergência, tempo, memória)."""
    linhas, historicos = [], {}
    for nome, cfg in [("A", cfg_a), ("B", cfg_b)]:
        for r in range(reinicios):
            res, t, mem = medir(algoritmo_genetico, inst, replace(cfg, seed=seed + r))
            a = avaliar(inst, res["solucao"])
            linhas.append({"config": nome, "seed": seed + r, "populacao": cfg.populacao,
                           "pm": cfg.pm if cfg.pm is not None else round(1 / inst.n, 4),
                           "objetivo": a["objetivo"], "caixa": a["caixa"], "viavel": viavel(inst, res["solucao"]),
                           "geracoes": res["geracoes"], "tempo_s": t, "memoria_pico_KiB": mem})
            historicos[(nome, seed + r)] = res
    return pd.DataFrame(linhas), historicos


def grafico_fitness(historicos, caminho):
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = {"A": "#d97706", "B": "#2563eb"}
    for (nome, seed), res in historicos.items():
        ax.plot(res["historico_melhor"], color=cores[nome], alpha=0.8, lw=1.4,
                label=f"Config {nome} — melhor" if seed == min(s for n, s in historicos if n == nome) else None)
        ax.plot(res["historico_medio"], color=cores[nome], alpha=0.25, lw=0.8, ls="--",
                label=f"Config {nome} — média" if seed == min(s for n, s in historicos if n == nome) else None)
    melhores = [v for res in historicos.values() for v in res["historico_melhor"]]
    ax.set_ylim(min(melhores) - 0.1 * abs(min(melhores)), max(melhores) * 1.02)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness (objetivo R$, com penalidade)")
    ax.set_title("Questão 1 — GA: fitness por geração (catálogo completo)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def etapa_otimizacao(inst, seed):
    """Otimização demonstrada: avaliação de fitness em laço Python vs. tabelas + numpy vetorizado."""
    linhas = []
    for modo in ["laco", "vetorizada"]:
        cfg = ConfigGA(seed=seed, avaliacao=modo, max_geracoes=150, paciencia=10**9)
        res, t = medir_tempo(algoritmo_genetico, inst, cfg)
        linhas.append({"avaliacao_fitness": modo, "geracoes": res["geracoes"], "tempo_s": t,
                       "fitness_final": res["fitness"]})
    df = pd.DataFrame(linhas)
    df["speedup"] = df["tempo_s"].iloc[0] / df["tempo_s"]
    return df


def etapa_escalabilidade(catalogo, config, parametros, cfg_b, seed, completo):
    linhas = []
    base_inst = construir_instancia(catalogo, config, parametros)
    tamanhos_bf = [4, 5, 6, 7] if completo else [4, 5]
    for n in tamanhos_bf:
        idx = np.argsort(-base_inst.idade, kind="stable")[:n]
        sub = base_inst.subconjunto(idx, parametros["fator_capacidade"], parametros["frac_max_desconto"])
        (_, _, aval), t, mem = medir(brute_force, sub)
        linhas.append({"algoritmo": "Brute Force", "n_skus": n, "trabalho": aval, "tempo_s": t, "memoria_pico_KiB": mem})
    fatores = [1, 2, 4, 8] if completo else [1, 2]
    for f in fatores:
        inst = construir_instancia(catalogo_ampliado(catalogo, f), config, parametros)
        cfg = replace(cfg_b, seed=seed, max_geracoes=200, paciencia=10**9)   # G fixo para comparar
        res, t, mem = medir(algoritmo_genetico, inst, cfg)
        linhas.append({"algoritmo": "GA (G=200, P=80)", "n_skus": inst.n, "trabalho": res["geracoes"] * cfg.populacao,
                       "tempo_s": t, "memoria_pico_KiB": mem})
        (g, t, mem) = medir(greedy_incentivos, inst, res["solucao"])
        linhas.append({"algoritmo": "Greedy", "n_skus": inst.n, "trabalho": len(g["passos"]),
                       "tempo_s": t, "memoria_pico_KiB": mem})
    return pd.DataFrame(linhas)


def executar(args):
    grupo = args.grupo or grupo_padrao()
    integro, manifesto, divergentes = verificar_integridade(grupo)
    if not integro:
        print(f"ATENÇÃO: arquivos com hash divergente do manifesto: {divergentes}")
    catalogo, config = carregar_questao("q1", grupo)
    seed = args.seed if args.seed is not None else int(config["seed_final"])
    parametros = parametros_modelo(args)
    inst = construir_instancia(catalogo, config, parametros)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    completo = args.modo == "completo"

    cfg_b = ConfigGA(populacao=args.populacao, pm=args.pm, pc=args.pc, max_geracoes=args.geracoes,
                     paciencia=args.paciencia, elite=args.elite, seed=seed)
    cfg_a = ConfigGA(populacao=30, pm=0.5 / inst.n, pc=0.7, torneio=2, elite=1,
                     max_geracoes=args.geracoes, paciencia=40, seed=seed)

    print(f"Grupo {grupo} | seed {seed} | id_instancia {manifesto['id_instancia']} | integridade "
          f"{'OK' if integro else 'FALHOU'} | SKUs {inst.n}")

    # [A]+[B] subconjunto
    df_sub, sub = etapa_bf_vs_ga(inst, args, cfg_a, cfg_b, seed)
    df_sub.to_csv(saida / "bf_vs_ga_subconjunto.csv", index=False)
    print(f"\n[A/B] Brute Force vs GA no subconjunto ({sub.n} SKUs: {', '.join(sub.skus)})")
    print(df_sub.drop(columns=["solucao"]).round(3).to_string(index=False))

    # [B] duas configurações do GA no catálogo completo
    df_cfg, historicos = etapa_configs_ga(inst, cfg_a, cfg_b, seed, args.reinicios)
    df_cfg.to_csv(saida / "comparacao_configs_ga.csv", index=False)
    grafico_fitness(historicos, saida / "fitness_por_geracao.png")
    print("\n[B] Configurações do GA no catálogo completo (resumo)")
    print(df_cfg.groupby("config")[["objetivo", "caixa", "geracoes", "tempo_s", "memoria_pico_KiB"]]
          .agg(["mean", "std", "max"]).round(3).to_string())

    # política final = melhor execução da configuração B
    melhor = max((r for (nome, _), r in historicos.items() if nome == "B"), key=lambda r: r["fitness"])
    politica = melhor["solucao"]
    if not viavel(inst, politica):
        raise RuntimeError("GA não encontrou política viável; aumente população/gerações.")
    pd.DataFrame(descrever_solucao(inst, politica)).to_csv(saida / "politica_precos_ga.csv", index=False)

    # [C] Greedy alimentado pela política do GA
    g = greedy_incentivos(inst, politica)
    df_g = pd.DataFrame(aplicar_incentivos(inst, politica, g))
    df_g.to_csv(saida / "greedy_antes_depois.csv", index=False)
    pd.DataFrame(g["passos"]).to_csv(saida / "greedy_passos.csv", index=False)
    print("\n[C] Greedy — SKUs que receberam incentivo (antes/depois)")
    print(df_g[df_g["incentivo"] != "nenhum"].to_string(index=False))

    # resultado financeiro final
    antes = avaliar(inst, politica)
    caixa_final = float(df_g["caixa_depois"].sum())   # caixa GA + vendas extras - investimento em incentivo
    receita_final = float((df_g["unidades_depois"] * df_g["preco_ga"]).sum())
    base = avaliar(inst, np.full(inst.n, int(np.argmin(np.abs(inst.niveis - 1.0)))))
    resumo = {
        "grupo": grupo, "seed": seed, "id_instancia": manifesto["id_instancia"], "integridade_ok": integro,
        "parametros": parametros, "config_ga_escolhida": asdict(cfg_b),
        "n_skus": inst.n,
        "caixa_politica_atual_preco_1_00": round(base["caixa"], 2),
        "caixa_apos_ga": round(antes["caixa"], 2),
        "caixa_final_apos_greedy": round(caixa_final, 2),
        "objetivo_apos_ga": round(antes["objetivo"], 2),
        "objetivo_final_apos_greedy": round(antes["objetivo"] + g["beneficio_liquido"], 2),
        "custo_estoque_parado_final": round(antes["objetivo"] + g["beneficio_liquido"] - caixa_final, 2) * -1,
        "margem_caixa_sobre_receita": round(caixa_final / receita_final, 4) if receita_final else 0.0,
        "receita_final": round(receita_final, 2),
        "estoque_inicial": int(inst.estoque.sum()),
        "estoque_remanescente": int(df_g["estoque_rem_depois"].sum()),
        "unidades_vendidas": int(df_g["unidades_depois"].sum()),
        "capacidade_unidades": round(inst.capacidade_unidades, 1),
        "skus_com_desconto": antes["n_descontos"], "max_skus_desconto": inst.max_skus_desconto,
        "skus_suspensos": int((politica == inst.SUSPENDER).sum()),
        "meta_caixa_90d": round(inst.meta_caixa, 2),
        "meta_atingida": caixa_final >= inst.meta_caixa,
        "folga_meta": round(caixa_final - inst.meta_caixa, 2),
        "divida_total_90d": round(inst.divida, 2),
        "cobertura_divida": round(caixa_final / inst.divida, 4),
        "orcamento_promocional": round(inst.orcamento, 2),
        "orcamento_usado": round(g["investimento"], 2),
        "uso_orcamento_percentual": round(100 * g["investimento"] / inst.orcamento, 2) if inst.orcamento else 0.0,
        "gap_medio_ga_b_subconjunto_percentual": round(float(df_sub[df_sub.metodo == "GA config B"].gap_percentual.mean()), 4),
    }

    df_opt = etapa_otimizacao(inst, seed)
    df_opt.to_csv(saida / "otimizacao_fitness.csv", index=False)
    print("\nOtimização demonstrada (avaliação de fitness):")
    print(df_opt.round(4).to_string(index=False))

    df_perf = etapa_escalabilidade(catalogo, config, parametros, cfg_b, seed, completo)
    df_perf.to_csv(saida / "desempenho_escalabilidade.csv", index=False)
    print("\nDesempenho (tempo e memória por tamanho de entrada):")
    print(df_perf.round(4).to_string(index=False))

    (saida / "resumo_final.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    manifesto_exec = {"grupo": grupo, "seed": seed, "n_skus": inst.n, "meta_caixa_90d": inst.meta_caixa,
                      "divida_total_90d": inst.divida, "orcamento_promocional": inst.orcamento,
                      "id_instancia": manifesto["id_instancia"], "hashes_dataset": manifesto["arquivos"],
                      "integridade_ok": integro}
    (saida / "manifesto_execucao.json").write_text(json.dumps(manifesto_exec, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")

    print("\n=== RESULTADO FINAL — QUESTÃO 1 ===")
    for k in ["caixa_politica_atual_preco_1_00", "caixa_apos_ga", "caixa_final_apos_greedy", "objetivo_apos_ga",
              "objetivo_final_apos_greedy", "custo_estoque_parado_final", "margem_caixa_sobre_receita",
              "estoque_inicial", "estoque_remanescente", "meta_caixa_90d", "meta_atingida", "folga_meta",
              "divida_total_90d", "cobertura_divida", "orcamento_promocional", "orcamento_usado",
              "uso_orcamento_percentual", "gap_medio_ga_b_subconjunto_percentual"]:
        print(f"  {k}: {resumo[k]}")
    return resumo


if __name__ == "__main__":
    executar(argumentos())
