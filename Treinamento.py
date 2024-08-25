import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Passo 1: Carregar o dataset
df = pd.read_csv('C:/Users/an053116/Documents/01 - Códigos python/Projeto FIESC/Documentos/Dataset.csv')

# Amostragem (opcional) para acelerar o treinamento
df = df.sample(frac=0.1, random_state=42)  # Usar 10% do dataset

# Verifique as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Identificar valores NaN
print("\nValores NaN por coluna:")
print(df.isna().sum())

# Passo 2: Tratar valores NaN
df = df.dropna()  # Remover linhas com valores NaN

# Verifique novamente se há valores NaN
print("\nValores NaN por coluna após tratamento:")
print(df.isna().sum())

# Passo 3: Pré-processar os dados
X = df[['fase', 'amplitude']]
y = df['classe_dp']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Passo 4: Treinar o modelo
model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)  # Menos árvores e uso de múltiplos núcleos
model.fit(X_train, y_train)

# Passo 5: Avaliar o modelo
y_pred = model.predict(X_test)
print("\nAcurácia no Conjunto de Teste:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Salvar o modelo treinado
joblib.dump(model, 'model_descarga.pkl')
joblib.dump(scaler, 'scaler_descarga.pkl')

print("\nModelo treinado e salvo com sucesso!")
