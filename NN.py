import auxiliary as aux
import numpy as np

def neural_sifu(target,x,threshold,eta,hlayers,active_fnct):
    x = aux.normalize(x)
    b = []
    w = []
    l = 10e+50
    loss = []
    iterations = 0
    for h in range(0,(len(hlayers)+1)):
        if h == 0:
            b.append(np.array(np.random.rand(1,hlayers[0])))
            w.append(np.array(np.random.rand(len(x[0,:]),hlayers[0])))
        elif h == (len(hlayers)):
            b.append(np.array(np.random.rand(1,1)))
            w.append(np.array(np.random.rand(hlayers[len(hlayers)-1],1)))
        else:
            b.append(np.array(np.random.rand(1,hlayers[h])))
            w.append(np.array(np.random.rand(hlayers[h-1],hlayers[h])))
    while l > threshold:
        print(l)
        new_z = np.zeros((len(x),1))
        for i in range(0,len(x[:,0])):
            x_i = x[i,:]
            x_i.shape = (1,len(x_i))
            target_i = target[i,:]
            z = aux.neurons(x_i,w,b,hlayers,active_fnct)
            deltas = aux.delta_rule(target_i,z,w,hlayers,active_fnct)
            [w,b] = aux.update(x_i,z,w,b,deltas,eta,hlayers)
            z = aux.neurons(x_i,w,b,hlayers,active_fnct)
            new_z[i] = z[1][len(hlayers)]
        l = aux.pred_loss(target,new_z)
        loss.append(l)
        iterations = iterations + 1
    return([new_z,w,b,loss,iterations])
