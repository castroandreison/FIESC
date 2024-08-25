# FIESC
Projeto analise de descarga parcial transformadores

Fraquezas ou defeitos no isolamento elétrico frequentemente resultam em descargas elétricas localizadas, conhecidas como Descargas Parciais (DP). Esse fenômeno ocorre quando o isolamento entre condutores é parcialmente comprometido, gerando uma conexão parcial que pode ser prejudicial ao sistema elétrico. A Descarga Parcial é um evento comum e, devido à sua natureza, a análise desse fenômeno tem se mostrado uma ferramenta valiosa para monitorar a condição de diversos equipamentos elétricos.

Diversos estudos têm proposto técnicas para a medição de DP, utilizando abordagens que capturam uma ou mais dessas formas de energia liberada. No entanto, devido à dificuldade de acesso a equipamentos de alta tensão, a utilização de algoritmos de inteligência artificial tem se mostrado uma alternativa promissora para a análise e monitoramento das Descargas Parciais, oferecendo uma maneira eficiente de diagnosticar e prever falhas em sistemas elétricos complexos.

O projeto consiste na aprendizgem de maquina através de um banco de dados analisado que apresenta as caracterisitcas de falhas em nucleos de transformadores, com o modelo já criado um segundo sofware recebe o banco de dados provindo de sensores no nucleo do transformador e armazenados neste banco são analisados novamente através de um modelo embarcado de AI que fará a analise das falhas e gravará em um novo banco o qual será realizado a apresentação na tela de monitoramento geral, e pode ser integrado a sofwares comuns da industria como SCADA.

Utilizar as bibliotecas e funções mencionadas é fundamental para construir, treinar, avaliar e salvar modelos de inteligência artificial em Python. Aqui está o motivo de usar cada uma delas:

1. import pandas as pd:
Pandas é uma biblioteca poderosa para manipulação e análise de dados. Usá-la permite carregar, explorar e pré-processar datasets de forma eficiente, facilitando a preparação dos dados para o treinamento de modelos de IA.
2. from sklearn.model_selection import train_test_split:
train_test_split é uma função que divide o dataset em conjuntos de treino e teste. Isso é essencial para avaliar o desempenho do modelo em dados não vistos durante o treinamento, evitando overfitting e garantindo uma avaliação mais realista.
3. from sklearn.preprocessing import StandardScaler:
StandardScaler é usado para padronizar os dados, ou seja, ajustá-los para que tenham média zero e desvio padrão um. Isso é importante para modelos de IA que são sensíveis à escala dos dados, como redes neurais e métodos baseados em distância.
4. from sklearn.ensemble import RandomForestClassifier:
RandomForestClassifier é um algoritmo de aprendizado de máquina que cria múltiplas árvores de decisão e as combina para melhorar a precisão e robustez do modelo. Ele é popular devido à sua capacidade de lidar com datasets complexos e reduzir o risco de overfitting.
5. from sklearn.metrics import classification_report, accuracy_score:
classification_report e accuracy_score são funções que fornecem métricas para avaliar o desempenho do modelo, como precisão, recall, F1-score e acurácia geral. Essas métricas ajudam a interpretar a eficácia do modelo e identificar áreas de melhoria.
6. import joblib:
Joblib é uma biblioteca utilizada para salvar e carregar modelos treinados. Isso é útil para reutilizar modelos sem precisar treiná-los novamente, economizando tempo e recursos, especialmente em projetos que envolvem grandes volumes de dados e modelos complexos.
Resumo:
Essas bibliotecas e funções são fundamentais em projetos de inteligência artificial porque permitem carregar e preparar os dados, dividir o dataset para treinamento e teste, padronizar os dados, treinar modelos eficazes como Random Forests, avaliar o desempenho do modelo com métricas detalhadas, e finalmente, salvar o modelo treinado para uso futuro.
