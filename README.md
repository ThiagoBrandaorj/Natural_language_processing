# Climate Stance Detection with BERT 🌍

Projeto de Processamento de Linguagem Natural (PLN) que treina um classificador baseado em **BERT** (`bert-base-cased`) para identificar o **posicionamento (stance)** de tweets sobre mudanças climáticas, usando o dataset [`cardiffnlp/tweet_eval` — subset `stance_climate`](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

O repositório contém duas versões do mesmo experimento, representando duas etapas de decisão de modelagem diante de um problema real de **desbalanceamento severo de classes**.

## Grupo

- Thiago Corrêa Brandão
- Isabella Vieira
- João Pedro Menezes
- Breno Alcaraz
- Fabiano Amorim

## Arquivos

### `AP2_PLN.ipynb` — versão original (3 classes)

Classificação de tweets em **3 categorias**: `none` (neutro), `against` (contra ações climáticas) e `favor` (a favor de ações climáticas).

Etapas do notebook:
1. **Carregamento dos dados** — download do dataset `tweet_eval/stance_climate` via 🤗 `datasets`, com visualização da distribuição entre treino/teste/validação.
2. **Pré-processamento** — mapeamento de labels numéricos para texto, limpeza dos tweets (lowercase, remoção de URLs, menções, hashtags e ruídos específicos), tokenização com NLTK.
3. **Modelagem** — tokenização com `AutoTokenizer` do `bert-base-cased`, fine-tuning com `Trainer` da 🤗 `transformers`, usando:
   - `class_weight` balanceado para compensar o desbalanceamento das classes;
   - uma implementação de **Focal Loss** que foi testada e **abandonada** (fica no notebook como registro, mas não é usada no treino final);
   - `CustomTrainer` para logar curvas de loss por época.
4. **Avaliação** — métricas por classe (accuracy, F1 macro, F1 por classe), matriz de confusão, exemplos de predições certas/erradas.
5. **Conclusão** — o modelo performa muito bem em `favor` e `none` (F1 ~94-95), mas **não consegue acertar nenhum exemplo de `against`**, já que essa classe representa apenas 3.7% do treino (13 exemplos).

### `AP2_PLN_sem_against.ipynb` — versão ajustada (2 classes)

Mesma estrutura e pipeline do notebook anterior, mas com uma mudança de modelagem: a classe `against` é **removida** do dataset (filtrada), reduzindo o problema a uma classificação binária `none` vs `favor`, com classes bem mais balanceadas (~44% / 56% no treino).

Resultado: F1-scores de ~94-95 em ambas as classes, com um treinamento mais estável e métricas mais confiáveis — a solução prática encontrada pelo grupo para o problema de subrepresentação identificado na primeira versão.

## Dataset

- **Fonte**: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), subset `stance_climate`
- **Splits**: treino, teste e validação (tamanhos pequenos — poucas centenas de exemplos no total)
- **Tarefa**: stance detection (identificar a posição do autor do tweet em relação a um tópico)

## Modelo

- **Base**: `bert-base-cased` ([Hugging Face](https://huggingface.co/bert-base-cased))
- **Fine-tuning**: `Trainer` da biblioteca `transformers`, com `class weights` para lidar com desbalanceamento
- **Métricas**: accuracy, F1 macro e F1 por classe

## Como rodar

Os notebooks foram desenvolvidos no Google Colab. Principais dependências:

```bash
pip install transformers datasets torch scikit-learn nltk matplotlib seaborn
```

Depois é só rodar as células em ordem — o próprio notebook baixa o dataset e os recursos do NLTK necessários (`punkt`, `stopwords`, `wordnet`).

## Referências

[1] https://lume.ufrgs.br/handle/10183/259959 — artigo usado como referência para a abordagem de Focal Loss em cenários de desbalanceamento severo.

## Principais aprendizados

- Desbalanceamento extremo de classes (<5%) pode inviabilizar o aprendizado de uma classe minoritária mesmo com técnicas de compensação como class weights.
- Simplificar o problema (reduzir de 3 para 2 classes) pode ser uma solução mais robusta do que insistir em técnicas mais sofisticadas (como Focal Loss) quando os dados simplesmente não sustentam a classe.
