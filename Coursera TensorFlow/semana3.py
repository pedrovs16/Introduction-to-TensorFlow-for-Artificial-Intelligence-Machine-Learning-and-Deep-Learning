# -*- coding: utf-8 -*-
import tensorflow as tf 

# Banco de dados
mnist = tf.keras.datasets.fashion_mnist

# Divisão do banco em gurpo treino e grupo teste
(training_image, training_label), (test_image, test_label) = mnist.load_data()

# Padronizando imagens em 28x28 pixels e as cores representadas em 0 a 1
training_image = training_image.reshape(60000, 28, 28, 1)
training_image = training_image/255
test_image = test_image.reshape(10000, 28, 28, 1)
test_image = test_image/255

# Criação da rede neural
model = tf.keras.Sequential([
    # Executar 64 convoluções(alteração do indice em um padrão) em 3x3 pixels
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    # A cada 2x2 pixel se extrai o com maior indice
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Transformar imagem em uma linha
    tf.keras.layers.Flatten(),
    # Identificar padroes
    tf.keras.layers.Dense(128, activation='relu'),
    # Resultado final tendo 10 resultados finais
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

# Compilar rede neural para fazer a medida de perda e precisão
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Sumário das etapas da rede neural e como fica a imagem em cada etapa
model.summary()

# Treinar 
model.fit(training_image, training_label, epochs=5)

# Media da perda
test_loss = model.evaluate(test_image, test_label)



'''
É difícil de enxergar a modificação da imagem e como isso vai beneficiar a 
máquina para entender então esse programa abaixo serve para mostrar o resultado
final de uma área da imagem e perceber que quando é o mesmo material o final
é parecido senão igual
'''
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_image[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_image[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_image[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

