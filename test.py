import NN
import auxiliary as aux
import random
import numpy as np

n = 10
Xlist = []
true_model = np.zeros((n,1))
for p in range(0, 3000):
    x = np.random.normal((random.randint(-10, 10)),(random.randint(1, 20)),size=(n,1))
    Xlist.append(x)
    true_model = true_model + x**2
# x_1 = np.random.normal(5,3,size=(n,1))
# x_2 = np.random.normal(1,3,size=(n,1))
error = np.random.normal(0,1,size=(n,1))
Y = true_model + error
X = aux.comb_arrays(Xlist)

threshold = 1e-24
eta = 0.005
hlayers = [100,50,50,30,20,5,30,5]
active_fnct = [
    'tanh',
    'Arctan',
    'Softsign',
    'Sigmoid',
    'tanh',
    'tanh',
    'tanh',
    'tanh',
    'linear'
]

[zreg,w,b,loss,iterations] = NN.neural_sifu(Y,X,threshold,eta,hlayers,active_fnct)
print('zreg = ',zreg,' | w = ',w,' | b = ',b,' | loss = ',loss,' | iterations = ',iterations)
