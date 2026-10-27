# Big O — tempo e memória

Variáveis usadas em todo o documento:

| Símbolo | Significado | Valor na instância G05 |
|---|---|---|
| `n` | SKUs da Questão 1 (catálogo completo) | 30 |
| `n_s` | SKUs do subconjunto do Brute Force | 7 |
| `K` | níveis de preço (`niveis_preco_relativos`) | Q1: 7, Q2: 6 |
| `O` | opções permitidas por SKU (níveis que passam no piso de margem + SUSPENDER), `O ≤ K+1` | 1 a 8 |
| `P` | tamanho da população do GA | 80 (config B) |
| `G` | gerações executadas pelo GA | ~120–190 (parada por paciência) |
| `I` | opções de incentivo por SKU no Greedy da Q1 | 8 (+ "nenhum") |
| `R` | passos executados pelo Greedy da Q1 (`R ≤ n·I`) | 14 |
| `J` | SKUs da Questão 2 | 24 |
| `T` | semanas | 12 |
| `S` | estoque inicial máximo de um SKU (estados de estoque) | 474 |
| `N_c`, `N_v` | cenários Monte Carlo de construção / validação | 2.000 / 5.000 |
| `A` | tipos de ação promocional (Q2) | 3 |

---

## Questão 1

### Brute Force — `questao_01/brute_force.py::brute_force`

| | Expressão | Trecho que domina |
|---|---|---|
| Tempo | **O(O^n_s · n_s)** | `for combinacao in itertools.product(*opcoes)` — cada combinação percorre `n_s` SKUs somando `W`, `U` e descontos |
| Memória | **O(n_s)** | `product` gera combinações sob demanda; só a melhor é guardada |

- No subconjunto de 7 SKUs foram avaliadas **294.912** combinações em **8,5 s** (`desempenho_escalabilidade.csv`).
- Cada SKU a mais multiplica o tempo por ≈ `O` (≈ 6–9×): medido 0,015 s (4 SKUs) → 0,10 s (5) → 0,94 s (6) → 8,5 s (7).
- **Deixa de ser viável** a partir de `n_s ≈ 8–9` (≈ 35 mil combinações por segundo: 8^8 ≈ 17 milhões ≈ 8 min; 8^9 ≈ 134 milhões ≈ 1 h). Para os 30 SKUs seriam até 8^30 ≈ 1,2·10^27 combinações. **Alternativa usada:** o GA, que o próprio Brute Force valida no subconjunto (gap = 0% na config B).

### Algoritmo Genético — `questao_01/genetic_algorithm.py::algoritmo_genetico`

| | Expressão | Trecho que domina |
|---|---|---|
| Pré-processamento | O(n · K) | `modelo.precomputar_tabelas` (uma única vez) |
| Tempo | **O(G · P · n)** | por geração: `fitness_vetorizado` (gather `W[linhas, pop]` de P×n), torneio O(P·t), crossover e mutação O(P·n) |
| Memória | **O(P · n + n · K + G)** | população + filhos (P×n inteiros), tabelas U/M/V/C/W (n×(K+1)), histórico de fitness (G) |

- Medição com `G = 200` fixo e `P = 80`: n = 30 → 0,052 s / 159 KiB; n = 60 → 0,061 s / 296 KiB; n = 120 → 0,078 s / 556 KiB; n = 240 → 0,114 s / 1.024 KiB. A memória cresce linearmente em `n`, como previsto. O tempo cresce menos que linearmente porque, nesses tamanhos, o custo fixo de cada chamada numpy domina (o termo `P·n` é vetorizado).
- **Limite prático:** o número de gerações necessário cresce com `n`; para catálogos de milhares de SKUs seria preciso aumentar `P` e `G`, ou decompor por categoria. Como as restrições acopladas são só duas (capacidade e nº de descontos), uma relaxação lagrangiana também seria alternativa.

### Greedy de incentivos — `questao_01/greedy.py::greedy_incentivos`

| | Expressão | Trecho que domina |
|---|---|---|
| Tabela de opções | O(n · I) | `tabela_incentivos`: recalcula demanda/caixa para cada SKU × incentivo |
| Laço guloso | **O(R · n · I)**, com `R ≤ n·I` ⇒ pior caso **O(n² · I²)** | `while True:` — a cada passo varre a matriz n×(I+1) de trocas possíveis e escolhe a de maior razão benefício/custo |
| Memória | **O(n · I)** | matrizes de benefício, investimento e unidades extras |

- Medido: n = 30 → 2,7 ms (14 passos); n = 120 → 7,0 ms (60 passos); n = 240 → 6,0 ms (42 passos). Uma fila de prioridade (heap) reduziria o laço para O(R · log(n·I)); não foi necessária no tamanho real.

---

## Questão 2

### Monte Carlo — `questao_02/monte_carlo.py`

| Função | Tempo | Memória | Trecho que domina |
|---|---|---|---|
| `estimar_estatisticas` (construção) | **O(N_c · T · J · K)** | **O(N_c · J · K + J · T · K · S)** | `rng.poisson(lam)` com `lam` de forma (N_c, J, K+1) por semana; depois um `bincount` por (j, t, k) para montar as pmfs |
| `simular_plano` (validação) | **O(N_v · T · J)** | **O(N_v · J + N_v · T)** | laço semanal: consulta da política `politica[j, t, estoque]`, sorteio Poisson e atualização de caixa/estoque para todos os cenários (vetorizado) |

- Medido (N_c = 2.000): J = 8 → 0,09 s / 9,3 MiB; J = 24 → 0,30 s / 27 MiB; J = 48 (S×2) → 0,60 s / 68 MiB; J = 96 (S×2) → 1,14 s / 136 MiB. O tempo é linear em J; a memória é dominada pelo lote semanal N_c × J × (K+1) e pelas pmfs J × T × (K+1) × (S+1).
- **Limite:** as pmfs ocupam `8·J·T·(K+1)·(S+1)` bytes — com J = 1.000 e S = 2.000 seriam ~1,3 GB. Alternativa: guardar só os momentos e usar uma Poisson/Binomial Negativa paramétrica, ou agrupar estoque em faixas.

### Programação Dinâmica — `questao_02/dynamic_programming.py::dp_sku` / `resolver_dp`

| | Expressão | Trecho que domina |
|---|---|---|
| Tempo por SKU | **O(T · K · S²)** | `np.convolve(P, prox)` dentro de `for k in range(K1)` dentro de `for t in ...`: calcula `Σ_d pmf(d)·V_{t+1}(s−d)` para todos os s (convolução S×S) |
| Tempo total | **O(J · T · K · S²)** | `resolver_dp` repete para cada SKU |
| Memória | **O(J · T · S)** | tabelas de política e de valor (J × (T+1) × (S+1)); a de valor é reutilizada pelo Greedy |

- Medido: J = 24, S = 474 → 0,15 s / 2,3 MiB; J = 96, S = 948 → 0,71 s / 17,7 MiB.
- **Reutilização de subproblemas (medida em `demo_memoizacao`)**: num SKU reduzido (Q2-004, 4 semanas, 12 unidades), a versão Top-Down memoizada calcula **50 estados únicos** e faz **696 consultas** a eles (≈ 14 reutilizações por estado). A mesma recursão **sem memoização faria 129.543 chamadas** — crescimento O(((K+1)(S+1))^T), exponencial em T. As versões Bottom-Up e Top-Down dão o mesmo valor (teste `test_dp_bottom_up_igual_top_down_memo`).
- **Limite:** o termo S² cresce rápido para estoques grandes. Alternativa: FFT na convolução (O(S log S)) ou discretizar o estoque em lotes.
- O estado não inclui o caixa acumulado (estado (t, s, caixa) com 24 SKUs juntos seria intratável: S^J estados). O acoplamento pelas dívidas entra pelos pesos `w_t` (ver `decisoes.md`) e pela validação Monte Carlo.

### Greedy semanal — `questao_02/greedy.py::greedy_promocional`

| | Expressão | Trecho que domina |
|---|---|---|
| Tempo | **O(T · J · A · S + T · J·A · log(J·A))** | `_valor_esperado` (soma sobre a pmf de tamanho S, com consulta a V_{t+1}) para cada SKU × ação × semana; depois `candidatos.sort` |
| Memória | **O(J · A + T · J)** | candidatos da semana + plano T×J |

- Medido: J = 24 → 37 ms / 40 KiB; J = 96 (S×2) → 145 ms / 94 KiB.
