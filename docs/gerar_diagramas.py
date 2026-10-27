"""Gera docs/arquitetura.png: fluxo de dados entre os algoritmos das duas questões."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SAIDA = Path(__file__).resolve().parent / "arquitetura.png"
CORES = {"dados": "#e5e7eb", "bf": "#fde68a", "ga": "#bfdbfe", "greedy": "#bbf7d0", "mc": "#fbcfe8",
         "dp": "#ddd6fe", "saida": "#fecaca"}


def caixa(ax, x, y, w, h, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=cor, ec="#374151", lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=8.5)


def seta(ax, a, b, texto=None, curva=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#374151",
                                 connectionstyle=f"arc3,rad={curva}"))
    if texto:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, texto, ha="center", fontsize=7.2, color="#1f2937")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))

    ax = axes[0]
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Questão 1 — Operação Caixa 90: Brute Force → GA → Greedy", fontsize=11, loc="left")
    caixa(ax, 0.1, 1.5, 2.0, 1.0, "q1_catalogo + q1_config\n(seed_final)\nmodelo.py: tabelas\nU, M, V, W", CORES["dados"])
    caixa(ax, 2.9, 2.7, 2.3, 1.0, "[A] Brute Force\nsubconjunto (7 SKUs)\nótimo exato", CORES["bf"])
    caixa(ax, 2.9, 0.3, 2.3, 1.0, "[B] GA — subconjunto\nmesma instância", CORES["ga"])
    caixa(ax, 6.0, 1.5, 2.4, 1.0, "[B] GA — catálogo\ncompleto (30 SKUs)\npolítica de preços", CORES["ga"])
    caixa(ax, 9.1, 1.5, 2.0, 1.0, "[C] Greedy\norçamento de\nincentivos", CORES["greedy"])
    caixa(ax, 11.5, 1.5, 1.4, 1.0, "Relatório\ncaixa, meta,\ndívida", CORES["saida"])
    seta(ax, (2.1, 2.2), (2.9, 3.1))
    seta(ax, (2.1, 1.8), (2.9, 0.8))
    seta(ax, (4.05, 2.7), (4.05, 1.3), "gap % (valida GA)")
    seta(ax, (5.2, 0.8), (6.0, 1.8), "config validada")
    seta(ax, (8.4, 2.0), (9.1, 2.0), "preços")
    seta(ax, (11.1, 2.0), (11.5, 2.0))

    ax = axes[1]
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Questão 2 — Sobrevivência 12 semanas: Monte Carlo → DP → Greedy → validação Monte Carlo",
                 fontsize=11, loc="left")
    caixa(ax, 0.1, 1.5, 2.0, 1.0, "q2_catalogo + q2_config\nσ demanda, σ concorr.,\nσ logística", CORES["dados"])
    caixa(ax, 2.7, 1.5, 2.3, 1.0, "[A] Monte Carlo\n2.000 cenários\n(construção)", CORES["mc"])
    caixa(ax, 5.6, 1.5, 2.3, 1.0, "[B] Prog. Dinâmica\nestado (semana, estoque)\npolítica π(t, s)", CORES["dp"])
    caixa(ax, 8.5, 1.5, 2.0, 1.0, "[C] Greedy semanal\ndestaque / mídia /\nfrete (slots, R$)", CORES["greedy"])
    caixa(ax, 11.0, 1.5, 1.9, 1.0, "[D] Monte Carlo\n5.000 cenários\nNOVOS", CORES["mc"])
    caixa(ax, 8.5, 0.05, 4.4, 0.8, "P(pagar sem. 4, 8, 12), caixa esperado, P5, estoque final, falhas", CORES["saida"])
    seta(ax, (2.1, 2.0), (2.7, 2.0))
    seta(ax, (5.0, 2.0), (5.6, 2.0))
    ax.text(5.3, 1.2, "pmf demanda\nE[logística]", ha="center", fontsize=7.2)
    seta(ax, (7.9, 2.0), (8.5, 2.0), "trajetória")
    seta(ax, (10.5, 2.0), (11.0, 2.0), "plano")
    seta(ax, (7.9, 2.4), (11.0, 2.4), "política π(t,s)", curva=-0.35)
    seta(ax, (11.95, 1.5), (11.95, 0.85))

    fig.tight_layout()
    fig.savefig(SAIDA, dpi=140)
    print(f"Diagrama salvo em {SAIDA}")


if __name__ == "__main__":
    main()
