# Feed_forward_FC
Feed forward fully connected neural network from scratch.
### Dependencies
   * Python3
   * Numpy
## Feedforward Fully connected Neural Network
![](https://i.stack.imgur.com/epElm.png)

### Pipeline
![Image](pipeline.jpg)

### Dimensions
```
l=layer number
* W[l] : ( n[l] ,n[l-l] ) 
* b[l] : ( n[l] ,1 ) 
* dW[l] : ( n[l] , n[l-1] )
* db[l] : (n[l] , 1)
* a[l] : g[l] ( Z[l] )
```

