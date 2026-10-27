"""
Executa o notebook oficial Gerador_Inputs_Checkpoint_6.ipynb sem alterar sua lógica.

Apenas o Bloco 02 (GRUPO e MATRICULAS) é substituído, como permitido no enunciado.
Todos os demais blocos são executados exatamente como estão no notebook, em ordem.

Uso:
    python data/gerar_dados.py --grupo G05 --matriculas RM562162 RM563236 RM563366

Saída (gerada pelo próprio notebook, relativa a esta pasta data/):
    data/dados_gerados/<GRUPO>/q1_catalogo_<GRUPO>.csv ... manifesto_dados.json
Também copia manifesto_dados.json para data/manifesto_dados.json (cópia byte a byte).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

PASTA_DATA = Path(__file__).resolve().parent
NOTEBOOK = PASTA_DATA / "Gerador_Inputs_Checkpoint_6.ipynb"
ID_BLOCO_02 = "41006db1"


def montar_bloco_02(grupo, matriculas):
    linhas = [
        "# Bloco 02 — Unico Bloco para editar",
        f"GRUPO = {grupo!r}",
        f"MATRICULAS = {list(matriculas)!r}",
        'print("Grupo:", GRUPO)',
        'print("Quantidade de integrantes:", len(MATRICULAS))',
    ]
    return "\n".join(linhas)


def executar_notebook(grupo, matriculas):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celulas = [c for c in nb["cells"] if c["cell_type"] == "code"]

    if not any(c.get("id") == ID_BLOCO_02 for c in celulas):
        raise RuntimeError("Bloco 02 não encontrado no notebook oficial.")

    namespace = {"display": print, "__name__": "__gerador__"}
    cwd_original = os.getcwd()
    os.chdir(PASTA_DATA)
    try:
        for celula in celulas:
            if celula.get("id") == ID_BLOCO_02:
                codigo = montar_bloco_02(grupo, matriculas)
            else:
                codigo = "".join(celula["source"])
            exec(compile(codigo, f"<notebook:{celula.get('id')}>", "exec"), namespace)
    finally:
        os.chdir(cwd_original)

    pasta_grupo = PASTA_DATA / "dados_gerados" / grupo
    shutil.copyfile(pasta_grupo / "manifesto_dados.json", PASTA_DATA / "manifesto_dados.json")
    return pasta_grupo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grupo", required=True, help="Código do grupo, ex.: G07")
    parser.add_argument("--matriculas", nargs="+", required=True, help="Matrículas dos integrantes")
    args = parser.parse_args()

    pasta = executar_notebook(args.grupo, args.matriculas)
    print(f"\nArquivos oficiais em: {pasta}")


if __name__ == "__main__":
    main()
