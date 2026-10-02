# Previsão de Doenças Cardiovasculares com Machine Learning

Projeto acadêmico de classificação supervisionada que compara **Regressão Logística** e **K-Nearest Neighbors (KNN)** para identificar a presença ou ausência de doença cardiovascular a partir de características registradas em um conjunto de dados.

A análise inclui exploração dos dados, limpeza, engenharia de atributos, padronização, otimização de hiperparâmetros e avaliação dos modelos.

## Objetivo

Comparar o desempenho dos dois classificadores utilizando Accuracy, Precision, Recall, F1-Score e ROC-AUC, analisando também matrizes de confusão, curvas ROC e coeficientes da Regressão Logística.

A variável-alvo é `cardio`:

- **0:** ausência de doença cardiovascular no registro.
- **1:** presença de doença cardiovascular no registro.

## Conjunto de dados

Foi utilizado o **Cardiovascular Disease Dataset**, disponível no Kaggle, com **70.000 registros e 13 colunas** na base analisada. O arquivo de entrada é `cardio_train.csv`, com campos separados por ponto e vírgula (`;`).

| Coluna original | Descrição |
|---|---|
| `id` | Identificador do registro; não utilizado no treinamento |
| `age` | Idade em dias |
| `gender` | Sexo codificado na base |
| `height` | Altura em centímetros |
| `weight` | Peso em quilogramas |
| `ap_hi` | Pressão arterial sistólica |
| `ap_lo` | Pressão arterial diastólica |
| `cholesterol` | Categoria de colesterol |
| `gluc` | Categoria de glicose |
| `smoke` | Indicador de tabagismo |
| `alco` | Indicador de consumo de álcool |
| `active` | Indicador de atividade física |
| `cardio` | Classe a ser prevista |

O dataset deve ser obtido separadamente e colocado na mesma pasta do script. Ele não está incluído neste repositório.

## Metodologia

### 1. Análise exploratória

O script inspeciona dimensões, tipos das colunas, valores ausentes, linhas completamente duplicadas, estatísticas descritivas e distribuição das classes.

Na análise registrada no projeto, a base original apresentou 35.021 registros da classe 0 e 34.979 da classe 1.

### 2. Limpeza

Foram adotados os seguintes filtros para selecionar os registros utilizados no estudo:

| Variável | Critério mantido |
|---|---|
| Altura | De 120 a 220 cm |
| Peso | De 30 a 200 kg |
| Pressão sistólica | De 80 a 250 |
| Pressão diastólica | De 40 a 150 |
| Relação entre pressões | Sistólica maior que diastólica |

Esses intervalos são decisões de pré-processamento do projeto. Na execução relatada, foram removidos **1.394 registros**, restando **68.606**: 34.665 da classe 0 e 33.941 da classe 1.

### 3. Engenharia de atributos

Foram criadas duas variáveis:

- `age_years`: idade em anos, calculada por `age / 365.25`.
- `bmi`: índice de massa corporal, calculado por `weight / (height / 100) ** 2`.

Os modelos recebem 12 atributos: `age_years`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active` e `bmi`.

### 4. Treinamento e validação

- Divisão estratificada: **80% para treinamento e 20% para teste**.
- Na base limpa relatada: 54.884 registros de treino e 13.722 de teste.
- Semente da divisão e da validação cruzada: `random_state=42`.
- Validação cruzada estratificada com **5 folds**, com embaralhamento.
- Padronização com `StandardScaler` dentro de um `Pipeline` para cada modelo.
- Busca de hiperparâmetros com `GridSearchCV`, usando **F1-Score** como critério de seleção e `n_jobs=-1`.

O scaler é ajustado dentro de cada etapa de treinamento da validação cruzada. O conjunto de teste é usado na avaliação posterior à seleção dos hiperparâmetros.

### 5. Modelos e hiperparâmetros

| Modelo | Valores pesquisados |
|---|---|
| Regressão Logística | `C`: 0.01, 0.1, 1 e 10; `solver`: liblinear e lbfgs; `max_iter=2000` |
| KNN | `n_neighbors`: 5, 11, 21 e 31; `weights`: uniform e distance; `p`: 1 e 2 |

Configurações selecionadas na execução documentada:

- **Regressão Logística:** `C=0.1`, `solver="liblinear"`.
- **KNN:** `n_neighbors=31`, `weights="distance"`, `p=2` (distância Euclidiana).

O script realiza a busca novamente a cada execução; essas configurações não estão fixadas como resultado obrigatório.

## Resultados

Os valores abaixo foram registrados na conversa **“Previsão De Doença Cardíaca”** e se referem à avaliação do projeto no conjunto de teste. Eles não foram recalculados durante a elaboração deste README.

| Métrica | Regressão Logística | KNN |
|---|---:|---:|
| Accuracy | **73,20%** | 72,74% |
| Precision | **75,78%** | 73,67% |
| Recall | 67,34% | **69,86%** |
| F1-Score | 71,31% | **71,72%** |
| ROC-AUC | **0,7956** | 0,7848 |

Precision, Recall e F1-Score consideram a classe positiva `cardio=1`. A ROC-AUC é apresentada na escala de 0 a 1; o script a multiplica por 100 na tabela impressa, exibindo 79,56 e 78,48, respectivamente.

A Regressão Logística apresentou maior Accuracy, Precision e ROC-AUC. O KNN apresentou maior Recall e F1-Score. Assim, os modelos tiveram desempenhos próximos, com diferenças no tipo de erro cometido.

### Matrizes de confusão

Contagens registradas na execução do projeto:

| Resultado | Regressão Logística | KNN |
|---|---:|---:|
| Verdadeiros negativos | 5.472 | 5.238 |
| Falsos positivos | 1.461 | 1.695 |
| Falsos negativos | 2.217 | 2.046 |
| Verdadeiros positivos | 4.572 | 4.743 |

O KNN identificou mais registros positivos e apresentou menos falsos negativos, mas também mais falsos positivos.

## Tecnologias

- Python 3
- Pandas e NumPy
- Matplotlib
- Scikit-learn

O script foi exportado do Google Colab e também pode ser executado localmente.

## Como executar

### 1. Obter o projeto

```bash
git clone https://github.com/KauaGhizzo/Predi-oDoen-asCardiovascularesML.git
cd Predi-oDoen-asCardiovascularesML
```

### 2. Instalar as dependências

Com Python 3 instalado, execute, preferencialmente em um ambiente virtual:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

### 3. Adicionar os dados

Extraia `cardio_train.csv` do dataset e coloque-o na pasta do projeto:

```text
Predi-oDoen-asCardiovascularesML/
├── README.md
├── cardiovascular_ml.py
└── cardio_train.csv          # fornecido separadamente
```

Mantenha o nome do arquivo e o separador `;`, conforme esperado pelo código.

### 4. Executar a análise

No terminal, dentro da pasta do projeto:

```bash
python cardiovascular_ml.py
```

A busca de hiperparâmetros, especialmente para o KNN, pode demorar e utiliza os núcleos de processamento disponíveis. Em execução local, feche cada janela de gráfico para permitir que o script continue, caso a exibição esteja bloqueando a execução.

No Google Colab, envie `cardiovascular_ml.py` e `cardio_train.csv` para o diretório de trabalho e execute em uma célula:

```python
%run cardiovascular_ml.py
```

## Saídas da análise

O script imprime informações dos dados, melhores hiperparâmetros, F1 da validação cruzada e tabela de métricas do teste. Também exibe:

1. Distribuição das classes antes da limpeza.
2. Distribuição das classes após a limpeza.
3. Matriz de confusão da Regressão Logística.
4. Matriz de confusão do KNN.
5. Comparação das métricas dos modelos.
6. Curvas ROC dos dois modelos.
7. Coeficientes da Regressão Logística.

Os gráficos são exibidos com `plt.show()`. O código atual não salva automaticamente imagens, modelos treinados ou relatórios em disco. Expressões como `df.head()` são úteis em células interativas, mas não imprimem uma tabela quando o arquivo é executado como script.

## Limitações

Este é um estudo acadêmico de classificação dos rótulos presentes no dataset. Não há validação clínica ou avaliação em uma população externa documentada, e o projeto não constitui uma ferramenta de diagnóstico nem uma previsão validada de eventos futuros.

Os resultados dependem da versão do dataset, dos filtros e das versões das bibliotecas, que não estão fixadas no projeto. A exclusão de registros altera a população avaliada. Os coeficientes descrevem associações aprendidas pelo modelo e não demonstram causalidade.

## Possíveis melhorias

- Registrar versões das dependências e a origem exata da versão do dataset utilizada.
- Avaliar outros classificadores sob o mesmo protocolo de validação.
- Investigar diferentes limiares de decisão e seus efeitos sobre Precision e Recall.
- Avaliar calibração das probabilidades e desempenho por subgrupos.
- Salvar modelos, gráficos e métricas para facilitar a reprodução da análise.
