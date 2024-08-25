import pandas as pd
import joblib

# Carregar o modelo treinado e o escalonador
model = joblib.load('model_descarga.pkl')
scaler = joblib.load('scaler_descarga.pkl')

# Carregar a nova tabela com o delimitador correto e tratar tipos mistos
df_novo = pd.read_csv('C:/Users/an053116/Documents/01 - Códigos python/Projeto FIESC/Documentos/Não_reconhecido.CSV', delimiter=';', low_memory=False)

# Verificar as primeiras linhas e os nomes das colunas do novo dataset
print(df_novo.head())
print("Nomes das colunas:", df_novo.columns)

# Tratar valores mal formatados e garantir que as colunas sejam numéricas
# Convertendo as colunas para o tipo numérico, forçando erros para NaN
df_novo['fase'] = pd.to_numeric(df_novo['fase'], errors='coerce')
df_novo['amplitude'] = pd.to_numeric(df_novo['amplitude'], errors='coerce')

# Verificar se há valores NaN após a conversão
print("Valores NaN por coluna:")
print(df_novo[['fase', 'amplitude']].isna().sum())

# Tratar valores NaN (por exemplo, removendo linhas com NaN)
df_novo = df_novo.dropna(subset=['fase', 'amplitude'])

# Preparar os dados para previsão
X_novo = df_novo[['fase', 'amplitude']]

# Normalizar os dados
X_novo = scaler.transform(X_novo)

# Fazer previsões
df_novo['tipo_descarga_previsto'] = model.predict(X_novo)

# Salvar o resultado em um novo arquivo CSV
df_novo.to_csv('C:/Users/an053116/Documents/01 - Códigos python/Projeto FIESC/Documentos/Resultados_Descarga_Previsto.csv', index=False)

print("Previsões realizadas e resultados salvos com sucesso!")
