"""
Questão 2 — Plano de Sobrevivência em 12 Semanas. Fluxo integrado:

    [A] Monte Carlo (2.000 cenários de construção) -> pmf da demanda por SKU/semana/preço + logística média
    [B] Programação Dinâmica (estado = semana, estoque) consome as pmfs -> política e trajetória de preços
        (peso lambda do caixa antecipado calibrado nos cenários de CONSTRUÇÃO)
    [C] Greedy semanal recebe a trajetória da DP -> plano de destaques/mídia/frete
    [D] Monte Carlo (5.000 cenários NOVOS) valida DP + Greedy -> probabilidade de pagar as dívidas

Exemplos:
    python questao_02/main.py --cenarios 5000
    python questao_02/main.py --choque-concorrente -0.08 --ajuste-custo-logistico 1.15
    python questao_02/main.py --ajuste-orcamento 0.75 --remover-sku Q2-003 --ajuste-dividas 1.10
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import carregar_questao, grupo_padrao, verificar_integridade
from comum.medicao import medir
from questao_02.dynamic_programming import (contar_chamadas_sem_memo, dp_sku, dp_top_down_memo,
                                            resolver_dp, trajetoria_esperada)
from questao_02.greedy import greedy_promocional
from questao_02.modelo import catalogo_ampliado, construir_instancia
from questao_02.monte_carlo import estimar_estatisticas, politica_fixa, simular_plano

PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
GRADE_LAMBDA = [0.0, 0.1, 0.25, 0.5, 1.0]


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description="Questão 2 — Plano de Sobrevivência em 12 Semanas")
    ap.add_argument("--grupo", default=None)
    ap.add_argument("--seed", type=int, default=None, help="default: seed_final do grupo")
    ap.add_argument("--cenarios", type=int, default=5000, help="cenários de VALIDAÇÃO (>= 5000)")
    ap.add_argument("--cenarios-construcao", type=int, default=2000, help="cenários de CONSTRUÇÃO (>= 2000)")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    ap.add_argument("--lambda", dest="lam", type=float, default=None, help="fixa o peso do caixa antecipado")
    ap.add_argument("--slots", type=int, default=5, help="ações promocionais por semana")
    ap.add_argument("--valor-residual", type=float, default=0.0, help="fração do custo recuperada do estoque final")
    ap.add_argument("--percentil-risco", type=float, default=5.0)
    # cenário-surpresa
    ap.add_argument("--choque-concorrente", type=float, default=0.0, help="-0.08 = concorrente -8%%")
    ap.add_argument("--ajuste-custo-logistico", type=float, default=1.0)
    ap.add_argument("--ajuste-orcamento", type=float, default=1.0)
    ap.add_argument("--ajuste-dividas", type=float, default=1.0)
    ap.add_argument("--remover-sku", nargs="*", default=[])
    ap.add_argument("--saida", default=str(PASTA_RESULTADOS))
    return ap.parse_args(argv)


def parametros_modelo(args):
    return {
        "choque_preco_concorrente": args.choque_concorrente,
        "ajuste_custo_logistico": args.ajuste_custo_logistico,
        "ajuste_orcamento": args.ajuste_orcamento,
        "ajuste_dividas": args.ajuste_dividas,
        "skus_removidos": args.remover_sku,
    }


def planejar(inst, estat, lam, slots, valor_residual):
    """[B] DP -> trajetória -> [C] Greedy. Retorna (dp, trajetoria, greedy)."""
    dp = resolver_dp(inst, estat, lam, valor_residual)
    traj = trajetoria_esperada(inst, estat, dp["politica"])
    g = greedy_promocional(inst, estat, dp, traj, slots_semana=slots)
    return dp, traj, g


def metricas(inst, sim, percentil):
    folga_min = sim["saldo_semana"][:, [p - 1 for p in sim["prazos"]]].min(axis=1)
    d = {f"P_pagar_semana_{p}": float(sim["pago_no_prazo"][:, i].mean()) for i, p in enumerate(sim["prazos"])}
    d.update({
        "P_pagar_todas": float(sim["todas_pagas"].mean()),
        "P_falha": float(1 - sim["todas_pagas"].mean()),
        "caixa_operacional_medio": float(sim["caixa_operacional"].mean()),
        f"caixa_operacional_P{percentil:g}": float(np.percentile(sim["caixa_operacional"], percentil)),
        "saldo_final_medio": float(sim["saldo_final"].mean()),
        f"saldo_final_P{percentil:g}": float(np.percentile(sim["saldo_final"], percentil)),
        f"folga_minima_nos_prazos_P{percentil:g}": float(np.percentile(folga_min, percentil)),
        "margem_caixa_sobre_receita": float(sim["caixa_operacional"].sum() / max(sim["receita"].sum(), 1e-9)),
        "receita_media": float(sim["receita"].mean()),
        "gasto_promocional_medio": float(sim["gasto_promocional"].mean()),
        "estoque_final_medio": float(sim["estoque_final"].sum(axis=1).mean()),
        "estoque_inicial": int(inst.estoque.sum()),
    })
    return d


def calibrar_lambda(inst, estat, args, seed):
    """Escolhe lambda nos cenários de CONSTRUÇÃO (stream 1): máx P(todas pagas), desempate pela folga P5."""
    linhas = []
    for lam in GRADE_LAMBDA:
        dp, _, g = planejar(inst, estat, lam, args.slots, args.valor_residual)
        sim = simular_plano(inst, dp["politica"], g["plano"], g["acoes"], args.cenarios_construcao, seed, stream=1)
        m = metricas(inst, sim, args.percentil_risco)
        linhas.append({"lambda": lam, "P_pagar_todas": m["P_pagar_todas"],
                       "folga_P": m[f"folga_minima_nos_prazos_P{args.percentil_risco:g}"],
                       "caixa_operacional_medio": m["caixa_operacional_medio"]})
    df = pd.DataFrame(linhas)
    melhor = df.sort_values(["P_pagar_todas", "folga_P"], ascending=False).iloc[0]
    return float(melhor["lambda"]), df


def graficos(inst, sims, dp, traj, saida, percentil):
    # distribuição do saldo final
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = {"Preço fixo 1,00": "#9ca3af", "DP": "#d97706", "DP + Greedy": "#2563eb"}
    for nome, sim in sims.items():
        ax.hist(sim["caixa_operacional"] / 1000, bins=60, alpha=0.45, color=cores[nome], label=nome)
    total_dividas = sum(inst.dividas.values()) - inst.caixa_inicial
    ax.axvline(total_dividas / 1000, color="#dc2626", ls="--", label="Dívidas − caixa inicial")
    ax.set_xlabel("Caixa operacional gerado em 12 semanas (R$ mil)")
    ax.set_ylabel("Cenários de validação")
    ax.set_title("Questão 2 — Distribuição do caixa (Monte Carlo de validação)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "distribuicao_caixa.png", dpi=130)
    plt.close(fig)

    # trajetórias de preço e estoque de SKUs representativos (maior receita esperada, 1 por categoria)
    df = pd.DataFrame(traj)
    df["receita"] = df["preco"] * df["vendas_esperadas"]
    cat = dict(zip(inst.skus, inst.categorias))
    df["categoria"] = df["sku"].map(cat)
    reps = (df.groupby(["categoria", "sku"])["receita"].sum().reset_index()
            .sort_values("receita", ascending=False).drop_duplicates("categoria").head(4)["sku"].tolist())
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    sim = sims["DP + Greedy"]
    for sku in reps:
        j = list(inst.skus).index(sku)
        d = df[df.sku == sku]
        nivel = pd.to_numeric(d["nivel"], errors="coerce")
        axes[0].step(d["semana"], nivel, where="mid", label=f"{sku} ({cat[sku]})")
        axes[1].plot(range(0, inst.T + 1), sim["estoque_medio"][:, j] / inst.estoque[j], marker="o", ms=3, label=sku)
    axes[0].set_ylabel("Nível de preço (× preço atual)")
    axes[0].set_title("Trajetória ótima da DP (demanda esperada) e estoque médio na validação")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Estoque / estoque inicial")
    axes[1].set_xlabel("Semana")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "trajetorias_preco_estoque.png", dpi=130)
    plt.close(fig)
    return reps


def demo_memoizacao(inst, estat, dp):
    """Reutilização de subproblemas: Top-Down memo vs Bottom-Up vs recursão sem memo (instância reduzida)."""
    j = int(np.argmin(inst.estoque))
    T, S = 4, min(int(inst.estoque[j]), 12)
    pmf = estat["pmf"][j, :T, :, :].copy()
    pmf[:, :, S] = pmf[:, :, S:].sum(axis=2)
    pmf = pmf[:, :, :S + 1]
    margem = dp["margens"][:T, j, :]
    pesos = dp["pesos"][:T]
    V, _ = dp_sku(pmf, margem, pesos, S)
    valor_memo, estados, consultas = dp_top_down_memo(pmf, margem, pesos, S)
    sem_memo = contar_chamadas_sem_memo(pmf, margem, S)
    return {"sku": inst.skus[j], "semanas": T, "estoque": S, "valor_bottom_up": float(V[0, S]),
            "valor_top_down_memo": float(valor_memo), "estados_unicos_calculados": estados,
            "consultas_a_subproblemas": consultas, "chamadas_sem_memoizacao": sem_memo,
            "reutilizacao_media_por_estado": consultas / max(estados, 1)}


def escalabilidade(catalogo, config, parametros, seed, completo, n_construcao):
    tamanhos = [(8, 1), (24, 1), (48, 2), (96, 2)] if completo else [(8, 1), (24, 1)]
    linhas = []
    base = catalogo
    for n_sku, fator_estoque in tamanhos:
        fator_skus = max(1, n_sku // len(base))
        cat = catalogo_ampliado(base, fator_skus, fator_estoque).head(n_sku)
        inst = construir_instancia(cat, config, parametros)
        rotulo = f"J={inst.J}, S_max={int(inst.estoque.max())}"
        estat, t, mem = medir(estimar_estatisticas, inst, n_construcao, seed)
        linhas.append({"etapa": "Monte Carlo (estimação)", "tamanho": rotulo, "tempo_s": t, "memoria_pico_KiB": mem})
        dp, t, mem = medir(resolver_dp, inst, estat, 0.0)
        linhas.append({"etapa": "Programação Dinâmica", "tamanho": rotulo, "tempo_s": t, "memoria_pico_KiB": mem})
        traj = trajetoria_esperada(inst, estat, dp["politica"])
        g, t, mem = medir(greedy_promocional, inst, estat, dp, traj)
        linhas.append({"etapa": "Greedy semanal", "tamanho": rotulo, "tempo_s": t, "memoria_pico_KiB": mem})
        _, t, mem = medir(simular_plano, inst, dp["politica"], g["plano"], g["acoes"], 5000, seed)
        linhas.append({"etapa": "Monte Carlo (validação 5000)", "tamanho": rotulo, "tempo_s": t, "memoria_pico_KiB": mem})
    return pd.DataFrame(linhas)


def estresse_dividas(catalogo, config, parametros, estat_base, lam, args, seed):
    """Curva de risco: dívidas multiplicadas por f. Mostra onde a política deixa de ser segura."""
    linhas = []
    for f in [1.0, 1.4, 1.8, 2.0, 2.1, 2.2, 2.4]:
        inst = construir_instancia(catalogo, config, {**parametros, "ajuste_dividas": parametros["ajuste_dividas"] * f})
        n = min(args.cenarios, 2000)
        fixa = simular_plano(inst, politica_fixa(inst), n_cenarios=n, seed=seed)
        dp, _, g = planejar(inst, estat_base, lam, args.slots, args.valor_residual)
        sim = simular_plano(inst, dp["politica"], g["plano"], g["acoes"], n, seed)
        linhas.append({"fator_dividas": f, "P_todas_preco_fixo": fixa["todas_pagas"].mean(),
                       "P_todas_dp_greedy": sim["todas_pagas"].mean(),
                       **{f"P_semana_{p}_dp_greedy": sim["pago_no_prazo"][:, i].mean() for i, p in enumerate(sim["prazos"])}})
    return pd.DataFrame(linhas)


def executar(args):
    if args.cenarios < 5000 or args.cenarios_construcao < 2000:
        print("Aviso: o enunciado pede >= 2000 cenários de construção e >= 5000 de validação.")
    grupo = args.grupo or grupo_padrao()
    integro, manifesto, divergentes = verificar_integridade(grupo)
    if not integro:
        print(f"ATENÇÃO: arquivos com hash divergente do manifesto: {divergentes}")
    catalogo, config = carregar_questao("q2", grupo)
    seed = args.seed if args.seed is not None else int(config["seed_final"])
    parametros = parametros_modelo(args)
    inst = construir_instancia(catalogo, config, parametros)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    completo = args.modo == "completo"
    print(f"Grupo {grupo} | seed {seed} | id_instancia {manifesto['id_instancia']} | integridade "
          f"{'OK' if integro else 'FALHOU'} | SKUs {inst.J} | semanas {inst.T}")

    # [A] Monte Carlo de construção
    estat, t_mc, mem_mc = medir(estimar_estatisticas, inst, args.cenarios_construcao, seed)
    pd.DataFrame(estat["resumo"]).to_csv(saida / "mc_demanda_por_nivel.csv", index=False)
    print(f"\n[A] Monte Carlo: {args.cenarios_construcao} cenários de construção ({t_mc:.2f}s, {mem_mc:.0f} KiB)")

    # [B]+[C] calibração de lambda (somente construção) e plano final
    if args.lam is None:
        lam, df_lam = calibrar_lambda(inst, estat, args, seed)
        df_lam.to_csv(saida / "calibracao_lambda.csv", index=False)
        print("\nCalibração do peso do caixa antecipado (cenários de construção):")
        print(df_lam.round(4).to_string(index=False))
    else:
        lam = args.lam
    (dp, traj, g), t_plan, mem_plan = medir(planejar, inst, estat, lam, args.slots, args.valor_residual)
    pd.DataFrame(traj).to_csv(saida / "trajetoria_dp.csv", index=False)
    pd.DataFrame(g["escolhas"]).to_csv(saida / "greedy_plano_promocional.csv", index=False)
    pd.DataFrame(g["uso_semanal"]).to_csv(saida / "greedy_uso_semanal.csv", index=False)
    df_traj = pd.DataFrame(traj)
    print(f"\n[B] DP com lambda = {lam} — trajetória de preços (nível por semana, demanda esperada):")
    print(df_traj.pivot(index="sku", columns="semana", values="nivel").to_string())
    print(f"\n[C] Greedy: {len(g['escolhas'])} ações promocionais em {inst.T} semanas")
    print(pd.DataFrame(g["uso_semanal"]).to_string(index=False))

    # [D] validação em cenários novos
    sims = {
        "Preço fixo 1,00": simular_plano(inst, politica_fixa(inst), n_cenarios=args.cenarios, seed=seed),
        "DP": simular_plano(inst, dp["politica"], n_cenarios=args.cenarios, seed=seed),
        "DP + Greedy": simular_plano(inst, dp["politica"], g["plano"], g["acoes"], args.cenarios, seed),
    }
    df_val = pd.DataFrame({nome: metricas(inst, s, args.percentil_risco) for nome, s in sims.items()})
    df_val.to_csv(saida / "validacao_monte_carlo.csv")
    print(f"\n[D] Validação em {args.cenarios} cenários independentes:")
    print(df_val.round(4).to_string())

    final = sims["DP + Greedy"]
    falhas = np.flatnonzero(~final["todas_pagas"])
    df_falhas = pd.DataFrame({
        "cenario": falhas,
        **{f"pagou_semana_{p}": final["pago_no_prazo"][falhas, i] for i, p in enumerate(final["prazos"])},
        "divida_em_atraso": final["divida_em_atraso"][falhas],
        "caixa_operacional": final["caixa_operacional"][falhas],
        "choque_demanda_medio": final["choque_mercado_medio"][falhas],
        "logistica_media": final["logistica_media"][falhas],
    })
    df_falhas.to_csv(saida / "cenarios_falha.csv", index=False)
    pd.DataFrame(final["caixa_operacional"], columns=["caixa_operacional"]).describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95]).to_csv(saida / "distribuicao_caixa_tabela.csv")
    reps = graficos(inst, sims, dp, traj, saida, args.percentil_risco)

    memo = demo_memoizacao(inst, estat, dp)
    print("\nReutilização de subproblemas (instância reduzida):", json.dumps(memo, default=str))

    df_estresse = estresse_dividas(catalogo, config, parametros, estat, lam, args, seed)
    df_estresse.to_csv(saida / "estresse_dividas.csv", index=False)
    print("\nEstresse de dívidas (probabilidade de pagar todas):")
    print(df_estresse.round(4).to_string(index=False))

    df_perf = escalabilidade(catalogo, config, parametros, seed, completo, args.cenarios_construcao)
    df_perf.to_csv(saida / "desempenho_escalabilidade.csv", index=False)
    print("\nDesempenho por tamanho de entrada:")
    print(df_perf.round(4).to_string(index=False))

    m = metricas(inst, final, args.percentil_risco)
    resumo = {
        "grupo": grupo, "seed": seed, "id_instancia": manifesto["id_instancia"], "integridade_ok": integro,
        "parametros": parametros, "lambda": lam, "slots_semana": args.slots,
        "cenarios_construcao": args.cenarios_construcao, "cenarios_validacao": args.cenarios,
        "caixa_inicial": inst.caixa_inicial, "dividas": inst.dividas,
        "orcamento_promocional_semanal": inst.orcamento_semanal,
        "limite_risco_falha": inst.limite_risco_falha,
        "dentro_do_limite_de_risco": m["P_falha"] <= inst.limite_risco_falha,
        "n_cenarios_falha": int(len(falhas)),
        "skus_representativos": reps,
        "tempo_mc_construcao_s": t_mc, "tempo_dp_greedy_s": t_plan,
        **m,
    }
    (saida / "resumo_final.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    manifesto_exec = {"grupo": grupo, "seed": seed, "n_skus": inst.J, "semanas": inst.T,
                      "dividas": inst.dividas, "caixa_inicial": inst.caixa_inicial,
                      "id_instancia": manifesto["id_instancia"], "hashes_dataset": manifesto["arquivos"],
                      "integridade_ok": integro}
    (saida / "manifesto_execucao.json").write_text(json.dumps(manifesto_exec, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print("\n=== RESULTADO FINAL — QUESTÃO 2 (DP + Greedy, validação) ===")
    for k, v in resumo.items():
        if k.startswith(("P_", "caixa_", "saldo_", "folga_", "margem_", "estoque_", "gasto_", "dentro", "n_cen", "lambda")):
            print(f"  {k}: {v}")
    return resumo


if __name__ == "__main__":
    executar(argumentos())
