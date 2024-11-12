# Modelo de Rede Neural para Classificação Bancária

Este projeto utiliza uma rede neural para prever a resposta de um cliente bancário em relação a uma oferta de produto financeiro, baseado em dados históricos de clientes. A rede neural é treinada com um conjunto de dados de treinamento e testada com um conjunto de dados de teste para avaliar a performance do modelo.

## Descrição

O modelo de rede neural é implementado usando a biblioteca `scikit-learn`, e é treinado para classificar se um cliente responderá afirmativamente (`y_yes`) a uma oferta bancária. O processo envolve pré-processamento dos dados, transformação de variáveis categóricas em variáveis dummies, normalização das características, e treinamento de uma rede neural multilayer perceptron (MLP).

### Funcionalidade:
- Carregar dados de treinamento e teste.
- Pré-processar os dados: transformação de variáveis categóricas em dummies.
- Normalizar as características para garantir uma melhor performance do modelo.
- Treinamento do modelo de rede neural (MLP).
- Avaliação do modelo utilizando métricas como acurácia, precisão, recall e F1 score.

## Requisitos

Para rodar este projeto, você precisará das seguintes bibliotecas Python:

- `pandas`: Para manipulação e análise de dados.
- `numpy`: Para operações matemáticas.
- `scikit-learn`: Para a implementação da rede neural e métricas de avaliação.

Você pode instalar as dependências usando o `pip`:

```bash
pip install -r requirements.txt
