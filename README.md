# Análise da Razão População/Empresas por UF

## Descrição do Projeto

Este projeto tem como objetivo analisar a relação entre a população estimada e o número de empresas ativas por Unidade Federativa (UF) no Brasil. Utiliza-se dados da API SIDRA e arquivos de projeção populacional para realiza análises exploratórias, modelagem de séries temporais, clustering e visualização de tendências.

## Funcionalidades
- Coleta de dados sobre empresas ativas via API SIDRA do IBGE.
- Leitura de projeções populacionais a partir de arquivo .xlsx.
- Cálculo da razão entre população e empresas ativas por UF e ano.
- Teste de estacionariedade das séries temporais (ADF Test).
- Modelagem preditiva com Exponential Smoothing e Auto-ARIMA.
- Agrupamento das UFs por comportamento similar usando K-Means.
- Identificação de estados mais saturados e com maiores oportunidades.
- Geração de gráficos para análise e visualização dos resultados.

## Bibliotecas Utilizadas

* requests - Requisição de dados via API.
* pandas - Manipulação de dados tabulares.
* numpy - Operações matemáticas e arrays.
* matplotlib e seaborn - Visualização de dados.
* sklearn.preprocessing - Padronização de dados.
* sklearn.cluster - Algoritmo K-Means para clustering.
* sklearn.metrics - Métricas para avaliação de clustering.
* statsmodels.tsa.holtwinters - Modelo de Suavização Exponencial.
* statsmodels.tsa.stattools - Teste ADF.
* pmdarima - Modelo Auto-ARIMA para previsão de séries temporais.

## Como Executar

### Instale as dependências necessárias:

```bash
pip install -r requeriments.txt
```

### Certifique-se de ter os dados necessários:

- A API SIDRA deve estar acessível.

- O arquivo projecoes_2024_tab1_idade_simples.xlsx deve estar no diretório de execução.

- Execute o script principal.

python case.py

## Principais Resultados

- Previsão da razão população/empresas para os anos de 2021 e 2022.

- Identificação de estados com maior ou menor tendência de crescimento econômico.

- Visualização das tendências e clusterização dos estados conforme seu comportamento econômico.