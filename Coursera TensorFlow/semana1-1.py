# -*- coding: utf-8 -*-

#Importando módulos
import tensorflow as tf
import numpy as np

# Criando rede neural
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Varíaveis
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Reconhecendo (epochs = numero de tentativas)
model.fit(xs, ys, epochs=500)

# Resultado de ys em relação ao valor de xs
print(model.predict([10.0]))    

"""
Created on Sat Nov  7 18:50:20 2020

@author: pedro
"""

