import numpy as np
import math

def comb_arrays(x):
    
    z = np.zeros((len(x[0][:]),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x[0])):
            z[j,i] = x[i][j]
    return(z)

def activation(x_i, arg):
    if arg == 'tanh':
        e = math.e
        z = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z[j,i] = (e**(x_i[j,i])-e**(-x_i[j,i]))/(e**(x_i[j,i])+e**(-x_i[j,i]))
    elif arg == 'Sigmoid':
        e = math.e
        z = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z[j,i] = 1/(1+e**(-x_i[j,i]))
    elif arg == 'Softsign':
        z = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z[j,i] = (x_i[j,i])/(1+abs(x_i[j,i]))
    elif arg == 'Softplus':
        e = math.e
        z = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z[j,i] = np.log(1+e**(x_i[j,i]))
    elif arg == 'Arctan':
        z = np.arctan(x_i)
    else:
        z = x_i
    return(z)

def gradient(x_i, arg):
    if arg == 'tanh':
        e = math.e
        z = 0
        g = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z = (e**(x_i[j,i])-e**(-x_i[j,i]))/(e**(x_i[j,i])+e**(-x_i[j,i]))
                g[j,i] = 1- z**2
    elif arg == 'Sigmoid':
        e = math.e
        z = 0
        g = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                z = 1/(1+e**(-x_i[j,i]))
                g[j,i] = z * (1- z)
    elif arg == 'Softsign':
        g = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                g[j,i] = 1/((1+abs(x_i[j,i]))**2)
    elif arg == 'Softplus':
        e = math.e
        g = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                g[j,i] = 1/(1+e**(-x_i[j,i]))
    elif arg == 'Arctan':
        g = np.zeros((len(x_i[:,0]),len(x_i[0,:])))
        for i in range(0,len(x_i[0,:])):
            for j in range(0,len(x_i[:,0])):
                g[j,i] = 1/((x_i[j,i])**2 + 1)
    else:
        g = np.ones(x_i.shape)
    return(g)

def normalize(x):
    for i in range(0,len(x[0,:])):
        for j in range(0,len(x[:,0])):
            x[j,i] = (x[j,i] - np.mean(x[:,i]))/np.std(x[:,i])
    return(x)


#The real deal
def neurons(x_i,w,b,hlayers=[2],active_fnct=['tanh','linear']):
    z = []
    active_z = []
    for i in range(0,(len(hlayers)+1)):
        if i == 0:
            z.append(np.dot(x_i,w[i])+b[i])
            active_z.append(activation(z[i],active_fnct[i]))
        elif not(i == (len(hlayers))):
            z.append(np.dot(active_z[i-1],w[i])+b[i])
            active_z.append(activation(z[i],active_fnct[i]))
        else:
            z.append(np.dot(active_z[i-1],w[i])+b[i])
            active_z.append(activation(z[i],active_fnct[len(active_fnct)-1]))
    return(list([z,active_z]))

def delta_rule(target,z,w,hlayers=[2],active_fnct=['tanh','linear']):
    deltas = []
    i = len(hlayers)
    j = 0
    while not(i<0):
        if i == len(hlayers):
            deltas.append(gradient(z[0][i],active_fnct[i])*(z[1][i]-target))
        elif i < 0:
            break
        else:
            o = gradient(z[0][i],active_fnct[i])
            o = o.T
            e = np.dot(w[i+1],deltas[j-1])
            deltas.append(e*o)
        i = i - 1
        j = j + 1

    return(deltas)

def update(x,z,w,b,deltas,eta=0.005,hlayers=[2]):
    new_w = []
    new_b = []
    j = len(hlayers)
    for i in range(0,len(hlayers)+1):
        if i == 0:
            gw = np.dot(deltas[j],x)
        else:
            gw = np.dot(deltas[j],z[1][i-1])
        new_w.append(w[i]-eta*gw.T)
        gb = deltas[j]
        new_b.append(b[i]-eta*gb.T)
        j = j - 1
    return(list([new_w,new_b]))

def pred_loss(target,z):
    loss = 0.5 * (np.sum(z-target))**2
    return(loss)

def extraction(pred,loss):
    loss = np.min(loss)
    weights = pred[loss][1]
    biases = pred[loss][2]
    return([weights,biases])

def predict(new_x,w,b,hlayers,active_fnct):
    predictor = np.zeros((len(new_x[:,0]),1))
    for i in range(0,len(new_x[:,0])):
        x_i = new_x[i,:]
        x_i.shape = (1,len(x_i))
        z=neurons(x_i,w,b,hlayers,active_fnct)
        predictor[i] = z[1][:]
    return(predictor)
