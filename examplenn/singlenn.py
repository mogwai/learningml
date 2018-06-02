import numpy as np

# https://iamtrask.github.io/2015/07/12/basic-python-network/

def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print('X\n', X)
# output dataset
y = np.array([[0, 0, 1, 1]]).T
print('y\n', y)
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3, 3)) - 1
print('syn0\n', syn0)
for _ in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0)) 
    
    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
