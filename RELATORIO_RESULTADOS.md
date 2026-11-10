# Relatório de resultados experimentais

**Data:** 9 de novembro de 2026  
**Base:** American Express Default Prediction  
**Clientes disponíveis:** 458.913  
**Seed:** 42  
**Divisão:** 64% treino, 16% validação e 20% teste, com separação por cliente

## 1. Escopo executado

Foram executados:

- baseline tabular LightGBM;
- GCN, GraphSAGE e GAT;
- janelas temporais de 3, 6 e 13 meses;
- avaliação GNN indutiva com grafo KNN;
- Hybrid OOF GraphSAGE com três folds;
- execuções em GPU (`cuda:0`) quando indicado.

As execuções completas de 25% e 50% terminaram com sucesso. A execução de
50% usou 229.457 clientes.

## 2. Melhor resultado por tamanho de amostra

| Amostra | Cobertura | Melhor modelo | Janela | AMEX | ROC-AUC |
|---:|---:|---|---:|---:|---:|
| 10.000 | 2,18% | GraphSAGE | 13m | 0,7320 | 0,9444 |
| 100.000 | 21,79% | GraphSAGE | 6m | 0,7122 | 0,9371 |
| 114.728 | 25,00% | GraphSAGE | 6m | 0,7058 | 0,9343 |
| 229.457 | 50,00% | GraphSAGE | 13m | 0,7084 | 0,9399 |

O GraphSAGE foi o melhor GNN em todas as escalas avaliadas. A janela ótima
variou entre 6 e 13 meses, sem evidência de ganho monotônico ao aumentar a
amostra.

## 3. Comparação com o baseline

| Amostra | Modelo | AMEX | ROC-AUC | PR-AUC | Brier | ECE |
|---:|---|---:|---:|---:|---:|---:|
| 25% | LightGBM | **0,7798** | **0,9573** | 0,8895 | 0,0716 | 0,0076 |
| 25% | Melhor GNN | 0,7058 | 0,9343 | — | — | — |
| 50% | LightGBM | **0,7819** | **0,9591** | 0,8912 | 0,0701 | 0,0048 |
| 50% | Melhor GNN | 0,7084 | 0,9399 | — | — | — |

O LightGBM superou o melhor GNN por 0,0739 AMEX e 0,0192 ROC-AUC em 25%.
Em 50%, a vantagem foi de 0,0734 AMEX e 0,0191 ROC-AUC. O resultado não
sustenta ganho preditivo adicional do grafo sobre as features tabulares neste
protocolo.

## 4. Resultado do Hybrid OOF

| Amostra | Janela | Folds | AMEX | ROC-AUC | Situação |
|---:|---:|---:|---:|---:|---|
| 25% | 13m | 3 | 0,6186 | 0,9198 | Plausível, mas inferior ao baseline |
| 50% | 13m | 3 | -0,2297 | 0,2439 | **Inválido/anômalo** |

O resultado de 50% não deve ser usado na conclusão. Ele apresenta 34.007 falsos
positivos, somente 2 verdadeiros negativos e ROC-AUC abaixo de 0,5. Os logs
também registram tentativas de alocação CUDA de aproximadamente 10 e 11 GB em
uma GPU de 6 GB. Apesar de o processo ter concluído, essa execução sofreu
pressão de memória e requer investigação antes de qualquer interpretação.

## 5. Conclusão científica

Os experimentos são suficientes para uma pesquisa empírica baseada em amostras
de até 50% da base. O protocolo inclui separação por cliente, múltiplas janelas
temporais, três arquiteturas GNN, baseline tabular e avaliação OOF.

A conclusão suportada pelos dados é:

> Neste protocolo, as representações relacionais e temporais com GCN,
> GraphSAGE e GAT não superaram o baseline LightGBM. GraphSAGE foi o GNN mais
> consistente, mas a estrutura relacional não demonstrou ganho preditivo
> adicional sobre as features tabulares.

Não é adequado apresentar os resultados como uma afirmação definitiva para
100% da base, pois o protocolo GNN full-batch não foi executado com os 458.913
clientes. Também não se deve utilizar o Hybrid OOF de 50% até corrigir sua
instabilidade.

## 6. Artefatos

- [Resultados GNN de 25%](./artifacts/experiments/25pct/metrics.json)
- [Hybrid OOF de 25%](./artifacts/experiments/25pct/hybrid_oof.json)
- [Baseline de 25%](./artifacts/experiments/25pct/baseline/metrics.json)
- [Resultados GNN de 50%](./artifacts/experiments/50pct/metrics.json)
- [Hybrid OOF de 50%](./artifacts/experiments/50pct/hybrid_oof.json)
- [Baseline de 50%](./artifacts/experiments/50pct/baseline/metrics.json)
