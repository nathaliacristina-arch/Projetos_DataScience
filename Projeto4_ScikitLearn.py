#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv ('dataset.csv')


# In[13]:


data.info()


# ## Análise Exploratória - Resumo Estatístico

# In[4]:


data.isnull().sum()


# In[5]:


data.shape


# In[6]:


data.columns


# In[14]:


data.corr ()


# In[7]:


data.describe


# In[15]:


# Resumo estatístico da variável preditora
data ["horas_estudo_mes"].describe()


# In[49]:


sns.histplot (data = data, x = 'horas_estudo_mes', kde = True)


# #### Preparação dos Dados

# In[20]:


# Prepara a variável de entrada X
X = np.array(data['horas_estudo_mes'])


# In[21]:


# Ajusta o shape de X
X = X.reshape(-1, 1)


# In[23]:


# Prepara a variável alvo
y = data['salario']


# In[47]:


# Gráfico de dispersão entre X e y
plt.scatter(X, y, color = "green", label = "Dados Reais Históricos")
plt.xlabel("Horas de Estudo")
plt.ylabel("Salário")
plt.legend()
plt.show()


# In[25]:


# Dividir dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[26]:


X_treino.shape


# In[27]:


X_teste.shape


# In[28]:


y_treino.shape


# In[29]:


y_teste.shape


# #### Modelagem Preditiva (Machine Learning)
# 

# In[35]:


# Cria o modelo de regressão linear simples
modelo = LinearRegression() #uma classe


# In[36]:


# Treina o modelo
modelo.fit(X_treino, y_treino)


# In[46]:


# Visualiza a reta de regressão linear (previsões) e os dados reais usados no treinamento
plt.scatter(X, y, color = "green", label = "Dados Reais Históricos")
plt.plot(X, modelo.predict(X), color = "red", label = "Reta de Regressão com as Previsões do Modelo")
plt.xlabel("Horas de Estudo")
plt.ylabel("Salário")
plt.legend()
plt.show()


# In[38]:


# Avalia o modelo nos dados de teste
score = modelo.score(X_teste, y_teste)
print(f"Coeficiente R^2: {score:.2f}")


# In[39]:


# Intercepto - parâmetro w0
modelo.intercept_


# In[40]:


# Slope - parâmetro w1
modelo.coef_


# ![image.png](attachment:image.png)
# 

# ## Deploy do Modelo
# 
# Usaremos o modelo para prever o salário com base nas horas de estudo.

# In[42]:


# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[48]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)


# In[43]:


# Mesmo resultado anterior usando os parâmetros (coeficientes) aprendidos pelo modelo
# y_novo = w0 + w1 * X
salario = modelo.intercept_ + (modelo.coef_ * horas_estudo_novo)
print(salario)


# In[44]:


# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[65]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)


# In[45]:


# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[73]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)


# In[50]:


# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[89]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)


# In[ ]:




