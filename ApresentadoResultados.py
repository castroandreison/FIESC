import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Carregar o dataset
df = pd.read_csv('C:/Users/an053116/Documents/01 - Códigos python/Projeto FIESC/Documentos/Dataset.csv')

# Converter a coluna 'date' para datetime com tratamento de erros
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

# Remover datas inválidas que não puderam ser convertidas
df = df.dropna(subset=['date'])

# Plotando gráfico de fase vs amplitude
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='fase', y='amplitude', hue='dataset_number', palette='viridis')
plt.title('Gráfico de Fase vs Amplitude')
plt.xlabel('Fase (graus)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Criando a interface Tkinter
root = tk.Tk()
root.title("Correlação entre Tipo de Falhas e Data")

# Função para calcular e mostrar a correlação
def show_correlation():
    # Agrupando por data e tipo de falha, contando ocorrências
    correlation_data = df.groupby(['date', 'classe_dp']).size().unstack(fill_value=0)
    
    # Plotando a correlação
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_data.plot(ax=ax)
    ax.set_title('Correlação entre Tipo de Falhas e Data')
    ax.set_xlabel('Data')
    ax.set_ylabel('Número de Falhas')
    ax.grid(True)
    
    # Adicionando o gráfico à interface Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Botão para mostrar a correlação
btn_show = ttk.Button(root, text="Mostrar Correlação", command=show_correlation)
btn_show.pack(pady=10)

root.mainloop()
