# 🍎 Classificação de Frutas com Inteligência Artificial

Projeto de classificação de imagens de frutas utilizando **Python, TensorFlow e Redes Neurais Convolucionais (CNN)**. O modelo é treinado para identificar diferentes categorias de frutas a partir de imagens, utilizando técnicas de aumento de dados, regularização e avaliação de desempenho.

## 📌 Sobre o Projeto

Este projeto implementa um modelo de Deep Learning capaz de classificar imagens de frutas em diferentes categorias. A solução utiliza uma Rede Neural Convolucional (CNN), uma arquitetura amplamente utilizada em tarefas de visão computacional e reconhecimento de imagens.

O treinamento é realizado a partir de um conjunto de imagens organizado em diretórios por classe. Durante o processo, o modelo aprende características visuais que ajudam a distinguir as categorias presentes no dataset.

## 🚀 Funcionalidades

- **Classificação de imagens:** previsão da categoria de uma fruta a partir de uma imagem.
- **Rede Neural Convolucional (CNN):** extração e aprendizado de características visuais por meio de múltiplas camadas convolucionais.
- **Data Augmentation:** aplicação de transformações nas imagens, como rotação, zoom, deslocamento e inversão horizontal.
- **Validação do modelo:** separação de 20% das imagens para validação.
- **Prevenção de overfitting:** utilização de Dropout e regularização L2.
- **Otimização do treinamento:** implementação de Early Stopping e redução automática da taxa de aprendizado.
- **Visualização de métricas:** geração de gráficos de acurácia e perda durante o treinamento.
- **Avaliação de classificação:** geração de relatório de classificação e matriz de confusão.
- **Salvamento do modelo:** exportação do modelo treinado para reutilização posterior.

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem de programação |
| TensorFlow / Keras | Construção, treinamento e avaliação da rede neural |
| NumPy | Manipulação de arrays e processamento das previsões |
| Matplotlib | Visualização de gráficos |
| Seaborn | Visualização da matriz de confusão |
| Scikit-learn | Métricas de avaliação e relatório de classificação |
| Google Colab | Ambiente de execução utilizado pelos caminhos de arquivos do código |
| Google Drive | Armazenamento do dataset e das imagens |

## 🧠 Arquitetura do Modelo

A arquitetura implementada utiliza uma rede neural sequencial composta por:

1. Quatro camadas convolucionais, com 32, 64, 128 e 256 filtros.
2. Camadas de Max Pooling para redução das dimensões espaciais.
3. Camadas de Dropout para ajudar a reduzir o overfitting.
4. Uma camada Flatten para transformar os mapas de características em um vetor.
5. Uma camada densa com 128 neurônios, ativação ReLU e regularização L2.
6. Uma camada de saída com ativação Softmax, dimensionada conforme o número de classes de frutas.

As imagens são redimensionadas para **100 × 100 pixels**, com três canais de cores (RGB), e normalizadas para valores entre 0 e 1.

O treinamento utiliza o otimizador Adam e a função de perda Categorical Crossentropy, adequada à classificação multiclasse com rótulos categóricos.

## 📂 Organização do Dataset

O código espera que as imagens estejam organizadas em subdiretórios, sendo cada subdiretório correspondente a uma classe.

Exemplo de estrutura:

```text
frutas/
├── maca/
│   ├── imagem_001.jpg
│   ├── imagem_002.jpg
│   └── ...
├── banana/
│   ├── imagem_001.jpg
│   └── ...
├── laranja/
│   ├── imagem_001.jpg
│   └── ...
└── ...
```

Os nomes das pastas são utilizados automaticamente como rótulos das classes.

O dataset é dividido em 80% para treinamento e 20% para validação. O código utiliza aumento de dados durante o carregamento das imagens.

## ⚙️ Instalação e Configuração

### 1. Preparar o ambiente

O projeto pode ser executado no Google Colab, que permite executar código Python em um ambiente de computação hospedado.

Acesse: [Google Colab](https://colab.research.google.com/)

### 2. Instalar as dependências

Em um ambiente Python com suporte às bibliotecas necessárias, instale:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

### 3. Configurar o Google Drive

No Google Colab, monte o Drive para disponibilizar o dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Organize as imagens no diretório esperado pelo código:

```text
/content/drive/MyDrive/frutas/
```

Se necessário, altere a variável `base_dir` para apontar para a localização correta do dataset.

## ▶️ Como Executar

1. Abra o notebook Python no Google Colab.
2. Monte o Google Drive, caso esteja utilizando esse ambiente.
3. Organize o dataset em pastas separadas por categoria.
4. Configure os caminhos do dataset e da imagem de teste.
5. Execute as células do código.
6. Acompanhe o treinamento e os gráficos de desempenho.
7. Consulte o relatório de classificação e a matriz de confusão.
8. Verifique a classe prevista para a imagem de teste.
9. Utilize o modelo salvo para futuras previsões.

O treinamento está configurado para até 30 épocas, podendo terminar antes caso o Early Stopping identifique que a perda de validação deixou de melhorar.

## 📊 Avaliação do Modelo

O projeto utiliza diferentes recursos para analisar o comportamento do modelo:

- **Acurácia:** proporção de previsões corretas.
- **Perda (Loss):** medida do erro utilizado durante o treinamento.
- **Classification Report:** apresenta métricas como precisão, recall e F1-score por classe.
- **Matriz de confusão:** permite visualizar os acertos e os erros de classificação entre as categorias.

Os gráficos de treinamento e validação permitem observar a evolução do aprendizado e identificar possíveis sinais de overfitting.

> Os resultados numéricos de acurácia e demais métricas dependem do dataset utilizado e devem ser obtidos por meio da execução do treinamento.

## 🔎 Predição de Novas Imagens

Após o treinamento, o código carrega uma imagem de teste, redimensiona-a para 100 × 100 pixels, normaliza seus valores e utiliza o modelo para prever sua categoria.

Exemplo de saída:

```text
Classe prevista: banana
```

*Exemplo ilustrativo; a classe exibida depende da imagem fornecida e do modelo treinado.*

## 💾 Salvamento do Modelo

Ao final da execução, o modelo é salvo no arquivo:

```text
fruits_model.h5
```

Esse arquivo pode ser utilizado para preservar o modelo treinado e permitir sua reutilização sem precisar realizar todo o treinamento novamente, desde que o ambiente tenha as dependências e a compatibilidade necessárias.

## 🔮 Possíveis Melhorias Futuras

- Implementar uma interface web para classificação de imagens.
- Disponibilizar uma API para receber imagens e retornar previsões.
- Comparar a CNN desenvolvida com arquiteturas pré-treinadas, como MobileNet ou EfficientNet.
- Avaliar o desempenho em um conjunto de teste independente.
- Implementar o carregamento do modelo salvo para realizar previsões em novas sessões.
- Comparar diferentes hiperparâmetros e arquiteturas para avaliar possíveis ganhos de desempenho.

## 👨‍💻 Objetivo Profissional

Este projeto demonstra a aplicação prática de Python, Deep Learning, visão computacional e avaliação de modelos de aprendizado de máquina, reunindo etapas de preparação de dados, construção de uma rede neural, treinamento, análise de métricas e predição de imagens.

Desenvolvido como projeto de estudo e portfólio em Inteligência Artificial.

---

**Tecnologias:** Python · TensorFlow · Keras · CNN · NumPy · Scikit-learn · Matplotlib · Seaborn
