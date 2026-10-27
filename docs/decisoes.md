# Decisões de modelagem e de projeto

## Gerais

- **Dados oficiais intocados.** `data/gerar_dados.py` lê o notebook oficial, substitui só o Bloco 02
  (grupo e matrículas) e executa os demais blocos exatamente como estão. Os arquivos ficam em
  `data/dados_gerados/<GRUPO>/`. Cada `main.py` recalcula os SHA-256 e compara com
  `manifesto_dados.json` antes de rodar (`comum/dados.py::verificar_integridade`). O `.gitattributes`
  marca esses arquivos como binários para o Git não trocar fins de linha (o que mudaria o hash).
- **Seed.** Por padrão os dois `main.py` usam `seed_final` do config; `--seed` sobrescreve. Todo
  componente aleatório recebe seed explícita (`np.random.default_rng`).
- **Cenário-surpresa por parâmetro**, sem mexer em código: `--ajuste-meta`, `--ajuste-orcamento`,
  `--ajuste-custo-logistico`, `--remover-sku`, `--populacao`, `--pm` (Q1); `--choque-concorrente`,
  `--ajuste-custo-logistico`, `--ajuste-orcamento`, `--ajuste-dividas`, `--remover-sku`, `--slots`,
  `--lambda` (Q2).

## Questão 1

### Modelo (`questao_01/modelo.py`)
- **Demanda com elasticidade constante:** `d(m) = demanda_base_90d · m^(−elasticidade)`, vendas =
  `min(estoque, d)`. É o modelo mais simples coerente com o campo `elasticidade` do gerador.
- **Caixa por unidade:** `preço·(1 − taxa) − custo − logística` — a mesma fórmula que o gerador usa para
  calcular a meta de caixa, então meta e resultado são comparáveis.
- **Piso de margem** (`margem_minima_percentual`) filtra os níveis de preço de cada SKU. Se nenhum nível
  passa, a única opção é **SUSPENDER** o SKU (não vender abaixo do piso). No G05, 1 SKU fica suspenso.
- **Custo do estoque parado.** Maximizar só o caixa leva a "subir todos os preços" (com essas
  elasticidades quase todo SKU dá mais caixa a 1,10). Isso contraria o objetivo de girar estoque.
  Então o objetivo é `caixa − custo do estoque que sobra`, com
  `taxa = 6% (capital/armazenagem em 90 dias) + 40% × min(1, idade/360)` (obsolescência). O caixa
  puro continua sendo reportado separadamente.
- **Restrições acopladas** (sem elas o problema seria separável e trivial por SKU):
  1. capacidade logística: total de unidades ≤ unidades do cenário base (`--fator-capacidade`);
  2. disciplina de desconto: no máximo 20% dos SKUs abaixo do preço atual (`--frac-max-desconto`) —
     evita "descontos indiscriminados".
  Com elas o problema vira uma mochila de múltipla escolha (NP-difícil).

### Brute Force
- Subconjunto = os 7 SKUs com **estoque mais antigo** (mais urgentes; escolha determinística). O
  enunciado pede ≥ 6; 7 roda em ~8 s no G05 (8 levaria ~8 min).
- Restrições reescaladas ao subconjunto (mesmo fator de capacidade e mesma fração de descontos).

### Algoritmo Genético
| Escolha | Justificativa |
|---|---|
| Codificação inteira (gene = opção de preço do SKU) | representação direta; só opções permitidas são sorteadas, então o piso de margem nunca é violado |
| População 80 | com 30 (config A) o GA parou num ótimo local em 3 de 5 seeds no subconjunto; com 80 acertou o ótimo do BF em todas; memória ainda pequena (156 KiB) |
| Torneio k=3 | pressão seletiva moderada e independente da escala do fitness (que pode ser negativo por causa da penalidade) |
| Crossover uniforme, pc=0,9 | os genes não têm ordem/adjacência (SKUs independentes), então ponto de corte não faz sentido |
| Mutação por gene pm=1/n | em média 1 gene muda por filho: explora sem destruir boas soluções |
| Elitismo 2 | garante que o melhor nunca piora (testado em `test_historico_do_melhor_nunca_piora_elitismo`) |
| Parada: 400 gerações ou 80 sem melhora | as curvas estabilizam em < 70 gerações; a paciência evita gerações inúteis |
| Penalidade = amplitude do objetivo | qualquer solução viável tem fitness maior que qualquer inviável, sem descartar genes úteis |
| 5 reinícios (seeds consecutivas) | reduz o risco de ótimo local; a política final é a melhor execução da config B |

### Greedy de incentivos
- Recebe a política de preços do GA (preço fica fixo) e escolhe **um incentivo por SKU** entre cupom,
  cashback e frete subsidiado, em intensidades diferentes.
- Premissas (parâmetros em `TIPOS_INCENTIVO`): o cliente percebe 100% do cupom, 80% do cashback e
  150% do frete (efeito "frete grátis"); a empresa paga 100%, 55% (nem todo cashback é resgatado) e 100%.
- Regra: maior **benefício líquido por real investido** (`ΔW / Δinvestimento`), permitindo trocar para
  uma opção mais forte do mesmo SKU (alocação marginal). Respeita orçamento, capacidade e piso de margem.
- Resultado G05: usou R$ 27,3 mil dos R$ 111,9 mil (24%), em 11 SKUs. O restante **não foi gasto de
  propósito**: nenhum outro incentivo tinha benefício líquido positivo. Gastar o orçamento inteiro
  destruiria caixa. O incentivo troca R$ 7,2 mil de caixa por menos estoque envelhecido (objetivo
  +R$ 4,4 mil); com o peso de estoque parado zerado, o Greedy gastaria bem menos.

## Questão 2

### Monte Carlo
- **Demanda:** Poisson com média `base · m^(−e) · (pc/pc0)^(γ·e) · ε`. ε é lognormal com média 1, com
  um choque de mercado comum a todos os SKUs (ρ = 0,3) — semanas ruins atingem o catálogo inteiro, o que
  é o que de fato ameaça o pagamento.
- **Concorrente:** log-preço AR(1) (φ = 0,6), dispersão `sigma_preco_concorrente`. Elasticidade cruzada
  γ·e com γ = 0,5.
- **Logística:** multiplicador lognormal semanal comum a todos os SKUs (`sigma_custo_logistico`).
- **Construção e validação independentes:** streams `default_rng([seed, 1])` (2.000 cenários) e
  `default_rng([seed, 2])` (5.000 cenários). O teste `test_validacao_usa_cenarios_diferentes_da_construcao`
  verifica isso.

### Programação Dinâmica (estado, decisão, transição, objetivo)
- **Estado:** (semana t, estoque remanescente s) por SKU.
- **Decisão:** nível de preço (ou SUSPENDER); níveis com margem esperada ≤ 0 são proibidos.
- **Transição:** `s' = s − min(s, D)`, com D vindo da pmf do Monte Carlo de (SKU, semana, nível).
- **Objetivo:** `max E[Σ_t w_t · margem · vendas_t]`, com `w_t = 1 + λ·(dívidas restantes − 1)`:
  caixa antes da semana 4 pesa `1+2λ`, antes da 8 pesa `1+λ`, depois pesa 1.
- **λ é calibrado por Monte Carlo** (só cenários de construção): para λ ∈ {0; 0,1; 0,25; 0,5; 1} roda
  DP → Greedy → simulação, e escolhe o maior P(pagar todas), com desempate pela folga mínima no percentil
  de risco. No G05 todos os λ dão P = 100%; o desempate escolheu λ = 1,0 (folga P5 de R$ 154,5 mil
  contra R$ 137,7 mil com λ = 0). O preço disso é um caixa médio um pouco menor nos cenários de
  construção (R$ 462,2 mil contra R$ 468,9 mil): o plano troca caixa esperado por segurança nos prazos.
  `calibracao_lambda.csv` mostra a tabela inteira; `--lambda 0` força o plano de caixa máximo.
- **Por que não colocar o caixa no estado:** com 24 SKUs o estado conjunto teria S^24 combinações de
  estoque. A decomposição por SKU + pesos + validação conjunta por Monte Carlo é a aproximação tratável.
- **Política com feedback:** a DP entrega π(t, s) para todo s, não só um caminho. Na validação, o preço
  de cada semana depende do estoque realmente observado naquele cenário. A trajetória "ótima" exibida
  (`trajetoria_dp.csv`) segue a demanda esperada.
- **Leitura da trajetória:** preço alto (1,12) nas primeiras 8 semanas e redução (0,82–1,06) no fim em
  SKUs com estoque sobrando — típico de liquidação. Q2-014 e Q2-024 ficam suspensos as 12 semanas porque
  a margem esperada é negativa em todos os níveis (vender só daria prejuízo). Os "SUSPENSO" isolados no
  fim aparecem quando o estoque esperado é de 1–2 unidades: vender agora ou depois dá praticamente o
  mesmo valor (indiferença numérica).

### Greedy semanal
- Recurso escasso: **5 ações por semana** (`--slots`) e o orçamento semanal. Ações: destaque (+12% de
  demanda, custo fixo = 6% do orçamento semanal), mídia (+28%, 16%) e frete promocional (+18%, 50% da
  logística por unidade).
- Usa a trajetória da DP (estoque esperado e preço escolhido em cada semana) e a **função de valor
  V_{t+1}** da DP para calcular o ganho esperado de cada ação; ordena por ganho/custo e preenche os slots.
- Por que V_{t+1}: a unidade vendida agora pela promoção deixa de ser vendida depois. A primeira versão
  ignorava isso e, na validação do G05, o caixa com Greedy ficou **menor** que o da DP sozinha
  (R$ 456,1 mil contra R$ 460,1 mil). Com V_{t+1} o Greedy passou a somar caixa (R$ 462,5 mil contra
  R$ 460,3 mil) e folga (P5 de R$ 155,1 mil contra R$ 145,5 mil).
- No G05 os slots são a restrição ativa (5/5 todas as semanas); o orçamento usado é de 13–25% por semana,
  porque o frete promocional tem a melhor razão ganho/custo.

### Risco
- Percentil de risco escolhido: **P5** (`--percentil-risco`) — "em 95% dos cenários o caixa é pelo menos isso".
- `limite_risco_falha = 10%` do config: o plano é aceito se P(falhar alguma dívida) ≤ 10%.
- No G05 a probabilidade de falha é 0% nos 5.000 cenários. Por isso há um **teste de estresse** das
  dívidas (`estresse_dividas.csv`): o preço fixo já falha com dívidas ×1,4 (P = 66,7%) e falha sempre com
  ×1,8. O plano DP + Greedy aguenta até ×2,0 com 100% e só cai com ×2,1 (72,3%) e ×2,2 (1%).
  A falha sempre ocorre na semana 12, a maior parcela.

## Limitações conhecidas
- Elasticidades e efeitos de incentivo/promoção são premissas (parâmetros), não estimados de dados reais.
- A DP trata os SKUs separadamente; a interação de caixa entre SKUs só aparece na calibração de λ e na validação.
- No Greedy da Q2 o ganho é calculado na trajetória esperada da DP; V_{t+1} considera o estoque das
  semanas seguintes, mas não as promoções que serão escolhidas nelas.
