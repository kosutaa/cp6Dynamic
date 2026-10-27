# Desempenho — medições reais

Todas as medições usam `comum/medicao.py`: `time.perf_counter()` para tempo e `tracemalloc` para o pico de
memória alocada pelo Python/numpy durante a chamada. Máquina: Windows 11, Python 3.14, numpy 2.5.
Instância G05 (seed 5014). Os números vêm dos CSVs em `questao_0X/resultados/` (rodar os `main.py`
regenera tudo).

## Questão 1 — `questao_01/resultados/desempenho_escalabilidade.csv`

| Algoritmo | Entrada | Trabalho | Tempo (s) | Pico de memória (KiB) |
|---|---|---|---|---|
| Brute Force | 4 SKUs | 768 combinações | 0,015 | 2,4 |
| Brute Force | 5 SKUs | 4.608 | 0,104 | 2,4 |
| Brute Force | 6 SKUs | 36.864 | 0,937 | 3,4 |
| Brute Force | 7 SKUs | 294.912 | 8,490 | 3,7 |
| GA (G=200, P=80) | 30 SKUs (pequena) | 16.000 avaliações | 0,052 | 159 |
| GA (G=200, P=80) | 60 SKUs (média) | 16.000 | 0,061 | 296 |
| GA (G=200, P=80) | 120 SKUs (grande) | 16.000 | 0,078 | 556 |
| GA (G=200, P=80) | 240 SKUs | 16.000 | 0,114 | 1.024 |
| Greedy | 30 SKUs | 14 passos | 0,003 | 29 |
| Greedy | 120 SKUs | 60 passos | 0,007 | 103 |
| Greedy | 240 SKUs | 42 passos | 0,006 | 177 |

Leitura: o Brute Force multiplica o tempo por ~6–9× a cada SKU (exponencial: aqui quase todos os SKUs do
subconjunto têm 8 opções permitidas), com memória constante. O GA tem memória linear em n e tempo quase
constante nesses tamanhos porque a avaliação é vetorizada.

### Otimização demonstrada (antes/depois) — `otimizacao_fitness.csv`

**Antes** (`genetic_algorithm.py::fitness_laco`): para cada indivíduo e cada gene, recalculava em Python
puro a demanda `d·m^(-e)`, a margem e o custo de estoque parado.
**Depois** (`fitness_vetorizado`): as tabelas U, M, C e W (n × K+1) são pré-computadas uma vez em
`modelo.precomputar_tabelas` e a população inteira é avaliada com indexação avançada do numpy
(`W[linhas, pop].sum(axis=1)`).

| Avaliação do fitness | Gerações | Tempo (s) | Fitness final | Speedup |
|---|---|---|---|---|
| laço Python (antes) | 150 | 0,700 | 482.869,94 | 1,0× |
| tabelas + numpy (depois) | 150 | 0,016 | 482.869,94 | **43,2×** |

O resultado é idêntico (mesma seed → mesma busca), e o teste `test_fitness_otimizado_igual_ao_original`
garante que as duas funções produzem o mesmo fitness.

### Duas configurações do GA — `comparacao_configs_ga.csv`, `bf_vs_ga_subconjunto.csv`, `fitness_por_geracao.png`

| Config | População | pm | pc | Torneio | Elite | Gap vs BF no subconjunto (5 seeds) | Objetivo no catálogo (5 seeds) | Gerações (média) | Tempo médio | Memória |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 30 | 0,5/n | 0,7 | 2 | 1 | 0% em 2 seeds, **0,072% em 3** | 482.870 em todas | 160 | 0,035 s | 70 KiB |
| B (escolhida) | 80 | 1/n | 0,9 | 3 | 2 | **0% em todas** | 482.870 em todas | 130 | 0,034 s | 156 KiB |

A config B converge em menos gerações (a paciência de 80 encerra mais cedo) e nunca ficou presa em ótimo
local. A config A gasta metade da memória, mas no subconjunto parou num ótimo local em 3 das 5 seeds.
As duas usam muito menos avaliações que o Brute Force (2–7 mil contra 295 mil).

## Questão 2 — `questao_02/resultados/desempenho_escalabilidade.csv`

Três tamanhos (mais um extra): J SKUs e S_max = maior estoque inicial (o catálogo foi replicado e,
nos maiores, estoque e demanda multiplicados por 2).

| Etapa | J=8, S=474 (pequena) | J=24, S=474 (média — real) | J=48, S=948 (grande) | J=96, S=948 |
|---|---|---|---|---|
| Monte Carlo — estimação (2.000) | 0,09 s / 9,3 MiB | 0,30 s / 27,1 MiB | 0,60 s / 68,4 MiB | 1,14 s / 136,4 MiB |
| Programação Dinâmica | 0,05 s / 0,9 MiB | 0,15 s / 2,3 MiB | 0,37 s / 9,0 MiB | 0,71 s / 17,7 MiB |
| Greedy semanal | 0,014 s / 25 KiB | 0,037 s / 40 KiB | 0,075 s / 64 KiB | 0,145 s / 94 KiB |
| Monte Carlo — validação (5.000) | 0,06 s / 16,3 MiB | 0,16 s / 46,2 MiB | 0,33 s / 91,1 MiB | 0,66 s / 180,8 MiB |

Leitura:
- Todas as etapas escalam ~linearmente em J (dobrar J ≈ dobrar tempo).
- Ao dobrar S (24 → 48 SKUs com estoque ×2) a DP cresceu 2,5× em tempo e 3,9× em memória — acima de
  2× por causa do termo S (estados) e da convolução. O gargalo de memória é o Monte Carlo (pmfs e lote de
  cenários), não a DP.
- O pipeline completo da instância real (MC 2.000 + calibração de λ com 5 DPs + validação de 3 estratégias
  em 5.000 cenários) roda em poucos segundos.

### Correção demonstrada na Q2 (antes/depois)

Na primeira versão, o Greedy calculava o ganho da promoção só pelas vendas extras da semana. Na validação
do G05 isso **piorou** o caixa: DP R$ 460,1 mil contra DP + Greedy R$ 456,1 mil. O motivo é que a unidade
vendida agora pela promoção deixaria de ser vendida depois (o estoque é finito). A versão atual usa a
função de valor da DP, `V_{t+1}(s − vendas)`, no cálculo do ganho. Resultado: DP + Greedy R$ 462,5 mil
contra DP R$ 460,3 mil, e folga mínima P5 de R$ 155,1 mil contra R$ 145,5 mil. Ver histórico do Git.
