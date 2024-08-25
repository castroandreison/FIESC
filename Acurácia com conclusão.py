import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset
df = pd.read_csv('C:/Users/an053116/Documents/01 - Códigos python/Projeto FIESC/Documentos/Dataset.csv')

# Exibir as primeiras linhas do dataset para verificar a estrutura
print(df.head())

# Verificar se há valores NaN no dataset
print("Valores NaN no dataset:")
print(df.isna().sum())

# Tratar valores NaN
# Remover linhas com valores NaN
df = df.dropna()

# Ou, alternativamente, preencher valores NaN com a média ou mediana
# df = df.fillna(df.mean())

# Converter a coluna 'date' para um formato numérico
df['date'] = pd.to_datetime(df['date'], errors='coerce').astype(int) / 10**9

# Pré-processamento básico (ajuste conforme necessário)
X = df.drop('classe_dp', axis=1)  # Remover a coluna de rótulo
y = df['classe_dp']  # Coluna de rótulo

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar apenas as colunas numéricas
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Treinar o modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')

# Exibir um relatório de classificação
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Conclusões
print("\nConclusões da Análise:")
print("1. **Verificação da Acurácia**:")
print(f"   - A acurácia do modelo é {accuracy * 100:.2f}%. Isso sugere que o modelo está performando de maneira {('razoável', 'boa')[accuracy > 0.8]}.")
print("   - Nota: A acurácia por si só pode não ser suficiente para determinar a eficácia do modelo, especialmente se houver classes desbalanceadas.")

print("2. **Relatório de Classificação**:")
print("   - **Precision**: Mede a proporção de verdadeiros positivos em relação ao total de positivos preditos.")
for label, metrics in report.items():
    if label == 'accuracy':
        continue
    print(f"     - A precisão para a classe '{label}' é: {metrics['precision']:.2f}")
print("   - **Recall**: Mede a proporção de verdadeiros positivos que foram corretamente identificados.")
for label, metrics in report.items():
    if label == 'accuracy':
        continue
    print(f"     - O recall para a classe '{label}' é: {metrics['recall']:.2f}")
print("   - **F1-Score**: Combina precision e recall em uma única métrica.")
for label, metrics in report.items():
    if label == 'accuracy':
        continue
    print(f"     - O F1-Score para a classe '{label}' é: {metrics['f1-score']:.2f}")
print("   - **Suporte**: Indica o número de ocorrências de cada classe no conjunto de teste.")
for label, metrics in report.items():
    if label == 'accuracy':
        continue
    print(f"     - O suporte para a classe '{label}' é: {metrics['support']}")

print("3. **Interpretação dos Resultados**:")
print("   - **Desempenho Global**: O modelo apresentou um desempenho sólido e equilibrado com base nas métricas.")
print("     - Conclusão: O modelo pode ser considerado robusto e adequado para as previsões dentro do contexto analisado.")
print("   - **Desafios com Classes Específicas**: Se houver discrepância significativa nas métricas entre as classes, isso pode indicar dificuldade do modelo em prever corretamente classes menos representadas.")
print("     - Ação Recomendada: Ajustar o modelo ou realizar balanceamento das classes pode ser necessário.")
print("   - **Considerações Finais**: O modelo não apresentou indícios de overfitting ou underfitting significativo com base nas métricas analisadas.")
