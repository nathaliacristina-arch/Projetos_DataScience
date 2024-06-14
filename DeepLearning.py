#!/usr/bin/env python
# coding: utf-8

# ###### Construir um modelo de Inteligência Artificial capaz de classificar imagens considerando 10 categorias: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']. Dada uma nova imagem de uma dessas categorias o modelo deve ser capaz de classificar e indicar o que é a imagem.

# In[56]:


# Instala o TF
get_ipython().system('pip install -q tensorflow==2.12')


# In[57]:


# Silencia mensagens do TF
get_ipython().run_line_magic('env', 'TF_CPP_MIN_LOG_LEVEL=3')


# In[58]:


# Imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# In[59]:


# Carrega o dataset CIFAR-10
(imagens_treino, labels_treino), (imagens_teste, labels_teste) = datasets.cifar10.load_data()


# In[60]:


# Clases das imagens
nomes_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# #### Pré-Processamento e Visualização das Imagens
# 

# In[34]:


# Normaliza os valores dos pixels para que os dados fiquem na mesma escala
imagens_treino = imagens_treino / 255.0
imagens_teste = imagens_teste / 255.0


# In[35]:


# Função para exibir as imagens
def visualiza_imagens(images, labels):
    plt.figure(figsize = (10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.binary)
        plt.xlabel(nomes_classes[labels[i][0]])
    plt.show()


# In[36]:


# Executa a função
visualiza_imagens(imagens_treino, labels_treino)


# ## Construção do Modelo
# 
# 
# ![DSA](imagens/convnet.jpg)

# In[76]:


# Modelo

# Cria o objeto de sequência de camadas
modelo_dsa = models.Sequential()

# Adiciona o primeiro bloco de convolução e max pooling (camada de entrada)
modelo_dsa.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))

# Adiciona o segundo bloco de convolução e max pooling (camada intermediária)
modelo_dsa.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))

# Adiciona o terceiro bloco de convolução e max pooling (camada intermediária)
modelo_dsa.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))


# In[77]:


# Adicionar camadas de classificação
modelo_dsa.add(layers.Flatten())
modelo_dsa.add(layers.Dense(64, activation = 'relu'))
modelo_dsa.add(layers.Dense(10, activation = 'softmax'))


# In[78]:


# Sumário do modelo
modelo_dsa.summary()


# In[79]:


# Compilação do modelo
modelo_dsa.compile(optimizer = 'adam', 
                   loss = 'sparse_categorical_crossentropy', 
                   metrics = ['accuracy'])


# In[41]:


get_ipython().run_cell_magic('time', '', 'history = modelo_dsa.fit(imagens_treino, \n                         labels_treino, \n                         epochs = 10, \n                         validation_data = (imagens_teste, labels_teste))')


# ## Avaliação do Modelo

# In[80]:


# Avalia o modelo
erro_teste, acc_teste = modelo_dsa.evaluate(imagens_teste, labels_teste, verbose = 2)


# In[81]:


print('\nAcurácia com Dados de Teste:', acc_teste)


# ## Deploy do Modelo

# In[62]:


# Carrega uma nova imagem
nova_imagem = Image.open("dados/nova_imagem.jpg")


# In[63]:


# Dimensões da imagem (em pixels)
nova_imagem.size


# In[64]:


# Obtém largura e altura da imagem
largura = nova_imagem.width
altura = nova_imagem.height


# In[65]:


print("A largura da imagem é: ", largura)
print("A altura da imagem é: ", altura)


# In[66]:


# Redimensiona para 32x32 pixels
nova_imagem = nova_imagem.resize((32, 32))


# In[67]:


# Exibir a imagem
plt.figure(figsize = (1,1))
plt.imshow(nova_imagem)
plt.xticks([])
plt.yticks([])
plt.show()


# In[68]:


# Converte a imagem para um array NumPy e normaliza
nova_imagem_array = np.array(nova_imagem) / 255.0


# In[69]:


# Expande a dimensão do array para que ele tenha o formato (1, 32, 32, 3)
nova_imagem_array = np.expand_dims(nova_imagem_array, axis = 0) 


# In[70]:


# Previsões
previsoes = modelo_dsa.predict(nova_imagem_array)


# In[71]:


print(previsoes)


# In[72]:


# Obtém a classe com maior probabilidade e o nome da classe
classe_prevista = np.argmax(previsoes)
nome_classe_prevista = nomes_classes[classe_prevista]


# In[73]:


print("A nova imagem foi classificada como:", nome_classe_prevista)


# In[ ]:




