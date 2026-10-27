import json

from comum.dados import PASTA_DADOS, carregar_questao, grupo_padrao, sha256_arquivo, verificar_integridade


def test_instancia_oficial_integra():
    ok, manifesto, divergentes = verificar_integridade(grupo_padrao())
    assert ok, f"Arquivos alterados após a geração: {divergentes}"
    for chave in ["grupo", "seed_base", "seed_final", "id_instancia", "arquivos"]:
        assert chave in manifesto


def test_seed_final_regra_do_enunciado():
    grupo = grupo_padrao()
    _, cfg = carregar_questao("q1", grupo)
    numero = int("".join(c for c in grupo if c.isdigit()))
    assert cfg["seed_final"] == 1000 * numero + cfg["seed_base"]


def test_catalogos_tem_tamanho_esperado():
    grupo = grupo_padrao()
    q1, _ = carregar_questao("q1", grupo)
    q2, _ = carregar_questao("q2", grupo)
    assert len(q1) == 30 and len(q2) == 24


def test_alteracao_muda_hash(tmp_path):
    grupo = grupo_padrao()
    origem = PASTA_DADOS / grupo
    destino = tmp_path / grupo
    destino.mkdir()
    for arq in origem.iterdir():
        (destino / arq.name).write_bytes(arq.read_bytes())
    alvo = destino / f"q1_config_{grupo}.json"
    cfg = json.loads(alvo.read_text(encoding="utf-8"))
    cfg["meta_caixa_90d"] = 1.0
    alvo.write_text(json.dumps(cfg), encoding="utf-8")
    ok, _, divergentes = verificar_integridade(grupo, pasta=tmp_path)
    assert not ok and divergentes == [alvo.name]
    assert sha256_arquivo(origem / alvo.name) != sha256_arquivo(alvo)
