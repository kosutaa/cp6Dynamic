import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


@pytest.fixture
def catalogo_q1_pequeno():
    return pd.DataFrame([
        # sku, categoria, custo, preço, estoque, demanda, elasticidade, taxa, logística, idade
        ["T-001", "Moda", 50.0, 120.9, 100, 80, 1.5, 0.10, 10.0, 200],
        ["T-002", "Beleza", 20.0, 59.9, 200, 220, 2.0, 0.12, 6.0, 30],
        ["T-003", "Eletronicos", 900.0, 1199.9, 30, 25, 1.2, 0.15, 40.0, 300],
        ["T-004", "Moveis", 400.0, 650.9, 20, 18, 0.9, 0.10, 80.0, 90],
        ["T-005", "Casa_Utilidades", 30.0, 69.9, 150, 120, 1.1, 0.09, 9.0, 150],
        ["T-006", "Informatica", 300.0, 459.9, 40, 50, 1.8, 0.14, 20.0, 10],
    ], columns=["sku", "categoria", "custo_unitario", "preco_atual", "estoque", "demanda_base_90d",
                "elasticidade", "taxa_marketplace", "custo_logistico", "idade_estoque_dias"])


@pytest.fixture
def config_q1():
    return {"meta_caixa_90d": 10000.0, "divida_total_90d": 8000.0, "orcamento_promocional": 2000.0,
            "margem_minima_percentual": 0.06, "niveis_preco_relativos": [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]}


@pytest.fixture
def catalogo_q2_pequeno():
    return pd.DataFrame([
        ["U-001", "Moda", 50.0, 120.9, 40, 5, 1.5, 0.10, 10.0, 125.9],
        ["U-002", "Beleza", 20.0, 59.9, 60, 8, 2.0, 0.12, 6.0, 55.9],
        ["U-003", "Eletronicos", 900.0, 1199.9, 12, 2, 1.2, 0.15, 40.0, 1250.9],
    ], columns=["sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial", "demanda_base_semanal",
                "elasticidade", "taxa_marketplace", "custo_logistico_base", "preco_concorrente_base"])


@pytest.fixture
def config_q2():
    return {"numero_semanas": 12, "caixa_inicial": 1000.0,
            "dividas": {"semana_4": 1500.0, "semana_8": 2000.0, "semana_12": 2500.0},
            "variabilidade": {"sigma_demanda": 0.1, "sigma_preco_concorrente": 0.05, "sigma_custo_logistico": 0.1},
            "orcamento_promocional_semanal": 300.0, "niveis_preco_relativos": [0.82, 0.88, 0.94, 1.0, 1.06, 1.12],
            "limite_risco_falha": 0.10}
