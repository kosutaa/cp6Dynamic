"""Leitura dos arquivos oficiais e verificação de integridade (SHA-256)."""

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data" / "dados_gerados"


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_integridade(grupo, pasta=PASTA_DADOS):
    """Recalcula os hashes e compara com manifesto_dados.json. Retorna (ok, manifesto, divergentes)."""
    pasta_grupo = Path(pasta) / grupo
    manifesto = json.loads((pasta_grupo / "manifesto_dados.json").read_text(encoding="utf-8"))
    divergentes = [
        nome for nome, hash_esperado in manifesto["arquivos"].items()
        if sha256_arquivo(pasta_grupo / nome) != hash_esperado
    ]
    return not divergentes, manifesto, divergentes


def carregar_questao(questao, grupo, pasta=PASTA_DADOS):
    """Carrega (catalogo DataFrame, config dict) da questão 'q1' ou 'q2'."""
    pasta_grupo = Path(pasta) / grupo
    catalogo = pd.read_csv(pasta_grupo / f"{questao}_catalogo_{grupo}.csv", encoding="utf-8-sig")
    config = json.loads((pasta_grupo / f"{questao}_config_{grupo}.json").read_text(encoding="utf-8"))
    return catalogo, config


def grupo_padrao(pasta=PASTA_DADOS):
    """Primeiro grupo encontrado em data/dados_gerados (há apenas um por repositório)."""
    grupos = sorted(p.name for p in Path(pasta).iterdir() if p.is_dir())
    if not grupos:
        raise FileNotFoundError("Nenhuma instância encontrada. Rode data/gerar_dados.py primeiro.")
    return grupos[0]
