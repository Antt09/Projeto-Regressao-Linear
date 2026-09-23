# 🏠 Regressão Linear para Previsão de Valores de Aluguel

<p align="center">
  <strong>📊 Análise Exploratória • 🤖 Machine Learning • 📈 Regressão Linear</strong>
</p>

<p align="center">
  Projeto desenvolvido em Python para analisar características de imóveis e construir modelos capazes de estimar o <strong>valor do aluguel</strong>.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-data%20analysis-150458?logo=pandas">
  <img alt="Scikit--learn" src="https://img.shields.io/badge/Scikit--learn-machine%20learning-F7931E?logo=scikit-learn">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter">
  <img alt="Status" src="https://img.shields.io/badge/Status-Estudo%20%2F%20Projeto%20Acadêmico-4CAF50">
</p>

---

## ✨ Sobre o projeto

Este projeto apresenta um estudo de **regressão linear aplicada à previsão de valores de aluguel de imóveis**.

A análise começa com uma exploração da base de dados, passa pelo tratamento dos dados e termina com a construção e avaliação de dois modelos:

- 📌 **Regressão Linear Simples** — utilizando apenas `Metragem`;
- 🧠 **Regressão Linear Múltipla** — utilizando todas as variáveis independentes disponíveis na base.

O objetivo é entender quanto das variações no valor do aluguel pode ser explicado pelas características observadas nos imóveis.

---

## 🎯 Objetivos

- 🔎 Explorar e compreender a base de dados;
- 🧹 Verificar qualidade, tipos e valores ausentes;
- 📊 Analisar a distribuição das variáveis;
- 🚨 Investigar valores extremos;
- 🔗 Avaliar correlações entre as variáveis;
- 🏗️ Construir um modelo de regressão linear simples;
- 🧠 Construir um modelo de regressão linear múltipla;
- 📏 Avaliar os modelos utilizando o coeficiente de determinação `R²`;
- ⚖️ Comparar o desempenho das duas abordagens.

---

## 🗂️ Estrutura da base

**Arquivo:** `ALUGUEL_MOD12.csv`

A base possui **7.203 registros** e **7 variáveis**, todas carregadas como `int64` no notebook.

✅ Não foram identificados valores nulos nas colunas.

### 📋 Dicionário de dados

| Variável | Descrição |
|---|---|
| `Valor_Aluguel` 💰 | Valor total pago pelo aluguel do imóvel |
| `Valor_Condominio` 🏢 | Valor do condomínio |
| `Metragem` 📐 | Área do apartamento, em m² |
| `N_Quartos` 🛏️ | Número de quartos |
| `N_banheiros` 🚿 | Número de banheiros |
| `N_Suites` 🛁 | Número de suítes |
| `N_Vagas` 🚗 | Número de vagas |

### 🎯 Variável-alvo

A variável dependente do projeto é:

```text
Valor_Aluguel
```

As demais variáveis são utilizadas como variáveis independentes nos modelos.

---

## 📊 Análise exploratória

A análise descritiva registrada no notebook apresenta:

| Variável | Média | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| `Valor_Aluguel` 💰 | 2.966,60 | 2.000 | 480 | 25.000 |
| `Valor_Condominio` 🏢 | 811,54 | 592 | 0 | 9.500 |
| `Metragem` 📐 | 88,51 | 67 | 30 | 880 |
| `N_Quartos` 🛏️ | 2,30 | 2 | 1 | 10 |
| `N_banheiros` 🚿 | 2,10 | 2 | 1 | 8 |
| `N_Suites` 🛁 | 1,02 | 1 | 0 | 5 |
| `N_Vagas` 🚗 | 1,44 | 1 | 0 | 9 |

💡 A diferença entre média e mediana, principalmente em `Valor_Aluguel`, `Valor_Condominio` e `Metragem`, evidencia a presença de valores extremos relevantes.

---

## 🧹 Pré-processamento

### ✅ Verificação dos dados

O notebook utiliza:

```python
df.info()
df.isna().sum()
```

Resultados:

- ✅ Todas as 7 colunas foram carregadas como `int64`;
- ✅ Nenhuma coluna apresentou valores ausentes;
- ✅ Não foi necessária imputação de valores.

---

## 🚨 Tratamento de outliers

Foi utilizado o método do **Intervalo Interquartil (IQR)**.

### 📌 Limites calculados

#### `Valor_Aluguel`

- Q1 = `1.350`
- Q3 = `3.200`
- Limite superior = **5.975**

#### `Metragem`

- Q1 = `52`
- Q3 = `100`
- Limite superior = **172 m²**

O notebook aplica:

```python
df_clean = df[
    (df['Valor_Aluguel'] <= lim_superior_aluguel) &
    (df['Metragem'] >= lim_superior_metragem)
].copy()
```

Com essa regra, a base passa de:

```text
7.203 registros
        ↓
227 registros
```

### ⚠️ Ponto importante

A condição utilizada para `Metragem` é:

```python
Metragem >= 172
```

Ou seja, ela **mantém imóveis com metragem igual ou superior ao limite**, em vez de remover os valores acima dele.

Portanto, `df_clean` representa um **subconjunto específico de imóveis de grande metragem**, e não simplesmente uma base convencionalmente “sem outliers”.

> 🔎 **Importante:** os modelos apresentados no notebook são treinados com `df`, e não com `df_clean`. Assim, o tratamento de outliers dessa etapa exploratória não é aplicado diretamente à modelagem apresentada.

---

## 🔗 Análise de relações entre variáveis

O notebook apresenta análises bivariadas envolvendo:

### 📐 Metragem × Valor do Aluguel

O gráfico de dispersão indica uma tendência positiva: em geral, imóveis maiores aparecem associados a valores de aluguel maiores.

### 🛏️ Número de quartos × Valor do Aluguel

O boxplot mostra diferenças na distribuição dos valores de aluguel entre grupos de quantidade de quartos.

### 🏢 Valor do condomínio × Valor do Aluguel

O gráfico com reta de tendência mostra associação positiva entre as duas variáveis no recorte analisado.

---

## 🧩 Matriz de correlação

As correlações com `Valor_Aluguel`, calculadas em `df_clean`, são aproximadamente:

| Variável | Correlação com `Valor_Aluguel` |
|---|---:|
| `N_Vagas` 🚗 | `0,21` |
| `N_Suites` 🛁 | `0,19` |
| `N_banheiros` 🚿 | `0,17` |
| `Valor_Condominio` 🏢 | `0,11` |
| `N_Quartos` 🛏️ | `0,11` |
| `Metragem` 📐 | `-0,08` |

Nesse recorte específico, as maiores correlações positivas com o valor do aluguel são observadas em `N_Vagas`, `N_Suites` e `N_banheiros`.

### 🔄 Relação entre variáveis explicativas

O notebook também identifica correlações relevantes entre algumas variáveis independentes:

| Variáveis | Correlação aproximada |
|---|---:|
| `N_banheiros` × `N_Suites` | `0,67` |
| `N_Suites` × `N_Vagas` | `0,54` |
| `Valor_Condominio` × `N_Vagas` | `0,50` |

⚠️ Essas relações podem indicar **multicolinearidade**, isto é, variáveis explicativas carregando informações parcialmente semelhantes.

---

# 📈 Regressão Linear Simples

Na primeira modelagem, a única variável independente utilizada é:

```text
Metragem
```

O modelo utiliza `LinearRegression()` do `scikit-learn`.

### 🧮 Equação

O notebook registra:

```text
Valor_Aluguel = -96,999 + (34,474 × Metragem)
```

Aproximadamente:

```text
Valor_Aluguel ≈ -97 + 34,47 × Metragem
```

📌 Dentro do modelo ajustado, uma unidade adicional de metragem está associada a um aumento estimado de aproximadamente **R$ 34,47** no valor do aluguel.

> Isso representa a relação estimada pelo modelo e não deve ser interpretado isoladamente como uma relação causal.

### 📊 Desempenho

| Conjunto | R² |
|---|---:|
| 🏋️ Treinamento | `0,5214` |
| 🧪 Teste | `0,5698` |

O `R²` de teste corresponde a aproximadamente **56,98%**.

---

# 🧠 Regressão Linear Múltipla

Na segunda abordagem, o modelo considera simultaneamente:

```text
Valor_Condominio
Metragem
N_Quartos
N_banheiros
N_Suites
N_Vagas
```

### 🧮 Coeficientes

| Variável | Coeficiente |
|---|---:|
| `Valor_Condominio` | `0,7840` |
| `Metragem` | `20,6814` |
| `N_Quartos` | `-649,1219` |
| `N_banheiros` | `223,7115` |
| `N_Suites` | `340,3379` |
| `N_Vagas` | `501,2463` |

**Intercepto:**

```text
435,2927
```

### 📐 Equação aproximada

```text
Valor_Aluguel =
435,29
+ 0,7840 × Valor_Condominio
+ 20,6814 × Metragem
- 649,1219 × N_Quartos
+ 223,7115 × N_banheiros
+ 340,3379 × N_Suites
+ 501,2463 × N_Vagas
```

Cada coeficiente representa a associação estimada de uma variável **mantendo as demais constantes dentro do modelo**.

---

## 🏆 Comparação dos modelos

Os resultados registrados no notebook são:

| Modelo | R² Treino | R² Teste |
|---|---:|---:|
| 📈 Regressão Linear Simples | `0,5214` | `0,5698` |
| 🧠 Regressão Linear Múltipla | `0,5958` | `0,6401` |

### 📌 Leitura dos resultados

No experimento registrado, a regressão múltipla apresenta `R²` superior ao da regressão simples tanto no conjunto de treinamento quanto no conjunto de teste.

Isso indica que, **dentro dessa avaliação**, considerar simultaneamente as características disponíveis fornece mais informação para explicar a variação do valor do aluguel do que utilizar somente a metragem.

---

## 📌 Principais conclusões

### 🔎 O que foi observado?

- ✅ A base possui **7.203 imóveis**;
- ✅ Não foram encontrados valores ausentes;
- 📊 `Valor_Aluguel` varia de **480 a 25.000**;
- 📐 A metragem apresenta associação positiva com o valor do aluguel nas análises exploratórias;
- 📈 A regressão simples alcançou `R² = 0,5214` no treino;
- 🧪 A regressão simples alcançou `R² = 0,5698` no teste;
- 🧠 A regressão múltipla alcançou `R² = 0,5958` no treino;
- 🧪 A regressão múltipla alcançou `R² = 0,6401` no teste;
- 🔗 Algumas variáveis independentes apresentam correlações relevantes entre si.

---

## 🛠️ Tecnologias

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-Statistics-76B900" alt="Seaborn">
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly" alt="Plotly">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter" alt="Jupyter">
</p>

---

## ▶️ Como executar

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
```

### 3. Posicione a base

O notebook carrega:

```python
pd.read_csv("ALUGUEL_MOD12.csv", delimiter=";")
```

Por isso, mantenha o arquivo `ALUGUEL_MOD12.csv` no diretório de execução do notebook ou ajuste o caminho no código.

### 4. Execute o Jupyter

```bash
jupyter notebook
```

ou:

```bash
jupyter lab
```

Depois, abra:

```text
Projeto_Regressao_Linear.ipynb
```

---

## 📁 Estrutura do repositório

### Estrutura atual

```text
.
├── 📄 ALUGUEL_MOD12.csv
├── 📓 Projeto_Regressao_Linear.ipynb
└── 📘 README.md
```

### Estrutura sugerida para uma versão futura

```text
.
├── 📂 data/
│   └── ALUGUEL_MOD12.csv
├── 📂 notebooks/
│   └── Projeto_Regressao_Linear.ipynb
├── 📂 src/
│   └── ...
├── 📄 requirements.txt
├── 📘 README.md
└── 📄 LICENSE
```

---

## 📚 O que este projeto demonstra

Este projeto reúne, em um único fluxo:

```text
📥 Dados
   ↓
🔎 Exploração
   ↓
🧹 Pré-processamento
   ↓
🚨 Outliers
   ↓
🔗 Correlação
   ↓
✂️ Treino / Teste
   ↓
📈 Regressão Simples
   ↓
🧠 Regressão Múltipla
   ↓
📊 Avaliação
   ↓
📌 Conclusões
```

---

## 👤 Antônio Pedro Rosa Crespilho

**Projeto de estudo em Regressão Linear**

Desenvolvido com foco em:

`Python` • `Análise de Dados` • `Machine Learning` • `Regressão Linear`

---

<p align="center">
</p>
