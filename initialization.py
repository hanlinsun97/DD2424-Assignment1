import numpy as np

def initialization():
    W = np.random.normal(0,0.01,[10,3072])
    b = np.random.normal(0,0.01,[10,1])
    return W, b