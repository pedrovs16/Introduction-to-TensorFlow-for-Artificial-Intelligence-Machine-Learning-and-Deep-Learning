# -*- coding: utf-8 -*-

import tensorflow as tf

# Criando banco de dados
mnist = tf.keras.datasets.fashion_mnist

# Separando em treino e teste
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Ver como são os dados acima 
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(test_images[30])
#print(training_labels[0])
#print(training_images[0])

'''
Entretanto se você ver na imagem como (label) os valores são de 0 a 255
então se divide tudo por 255 para virar de 0 a 1 para facilitar o trabalho 
do computador
'''
training_images  = training_images / 255.0
test_images = test_images / 255.0
#print(training_labels[0])
#print(training_images[0])

# Rede neural para identificar quais são as roupas das imagens teste
# Dividida em 2 camadas uma de 128 neuroneos outra de 10
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(), 
     tf.keras.layers.Dense(128, activation=tf.nn.relu), 
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

'''
Sequencial : Isso define uma SEQUÊNCIA de camadas na rede neural

Flatten(Achatar) : Lembra-se antes, onde nossas imagens eram um quadrado,
 quando você as imprimiu? Flatten apenas pega aquele quadrado e o transforma 
 em um conjunto unidimensional. Em vez de escrever 28x28

Dense : adiciona uma camada de neurônios

Cada camada de neurônios precisa de uma função de ativação para dizer a eles o 
que fazer. Existem muitas opções, mas apenas use-as por enquanto.

Relu significa efetivamente "Se X> 0 retornar X, então retorne 0" - então o 
que ele faz, ele apenas passa valores 0 ou maiores para a próxima camada na rede.

Softmax pega um conjunto de valores e efetivamente escolhe o maior, então, 
por exemplo, se a saída da última camada for semelhante a 
[0,1, 0,1, 0,05, 0,1, 9,5, 0,1, 0,05, 0,05, 0,05], ele salva você pescar por 
ele procurando o maior valor e transformá-lo em [0,0,0,0,1,0,0,0,0] -
 O objetivo é economizar muito código!
 
A última camada de neuroneos tem que ser igual ao número de respostas
'''

# Aqui ocorre a compilação para depois ocorrer o teste
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5 )

# Para medir a perda e precisão final
model.evaluate(test_images, test_labels)

# Para parar quando alcançar uma % de acertos

'''
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
'''

# Para fazer um teste específico
#print(np.argmax(model.predict(np.array([test_images[30]]))))
