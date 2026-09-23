# 📊 Análise e Modelagem de Risco de Crédito & Previsão de Imóveis

Este repositório reúne projetos práticos voltados para a análise de dados, pipeline de pré-processamento, engenharia de *features* e construção de modelos preditivos com **Python**, **Pandas** e **Scikit-Learn**.

---

## 📌 Visão Geral do Projeto

O objetivo principal deste projeto é aplicar técnicas de *Machine Learning* e análise estatística para resolver problemas de negócio, abrangendo desde o tratamento de dados brutos até a avaliação de métricas de desempenho dos modelos.

### 📑 Etapas do Pipeline de Dados
1. **Tratamento e Limpeza:** Identificação de valores ausentes (*NaN*), remoção de inconsistências e formatação de tipos de dados.
2. **Engenharia de Features & Encodings:**
   - **Variáveis Ordinais (`Education`, `Credit Score`):** Mapeamento hierárquico manual para preservar a relação de ordem dos dados sem ruídos de ordenação alfabética.
   - **Variáveis Nominais (`Gender`, `Marital Status`, `Home Ownership`):** Aplicação de *One-Hot Encoding* (`pd.get_dummies`) com remoção da primeira categoria (`drop_first=True`) para evitar multicolinearidade.
3. **Análise Exploratória & Previsão:**
   - Implementação de modelos de **Regressão Linear (Simples e Múltipla)**.
   - Avaliação da capacidade explicativa do modelo utilizando o coeficiente de determinação ($R^2$).

---

## 📈 Principais Resultados (Modelo de Regressão)

Durante a avaliação do modelo de Regressão Linear Simples para previsão de preços de aluguel em função da área:

- **$R^2$ Treino:** `0.5214`
- **$R^2$ Teste:** `0.5698`

### 💡 Diagnóstico do Modelo
- **Capacidade Explicativa:** O modelo explica entre **52% e 57%** da variabilidade do preço com base apenas na metragem do imóvel.
- **Generalização:** O modelo não apresentou *overfitting*, performando de maneira consistente nos dados de teste.
- **Evolução:** Como a metragem isolada não captura toda a complexidade do preço de um imóvel (*underfitting* leve), o próximo passo do projeto é a expansão para **Regressão Linear Múltipla** incluindo variáveis explicativas adicionais.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Linguagem:** Python 3.12+
- **Manipulação de Dados:** `pandas`, `numpy`
- **Machine Learning & Pré-processamento:** `scikit-learn`
- **Visualização de Dados:** `matplotlib`, `seaborn`

---

## 📂 Estrutura do Repositório

```text
├── Projeto_Regressao_Linear.ipynb  # Notebook principal com análises e modelos
├── ALUGUEL_MOD12.csv               # Dataset utilizado
├── .gitignore                      # Arquivos ignorados pelo Git
└── README.md                       # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto Localmente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/Antt09/Projeto-Regressao-Linear.git
   cd Projeto-Regressao-Linear
   ```

2. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv .venv
   # No Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instale as dependências:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
   ```

4. **Abra o Jupyter Notebook ou execute pelo PyCharm:**
   ```bash
   jupyter notebook
   ```

---

## ✉️ Contato

Desenvolvido por **Antônio Pedro Rosa Crespilho**  
- **GitHub:** [@Antt09](https://github.com/Antt09)
