import numpy as np
from csv import reader
from math import exp


def sig(x, deriv=False):
    if(deriv == True):
        return x*(1-x)

    return 1/(1+np.exp(-x))


LEARNING_RATE = 0.1


def load_csv():
    data = []
    with open("./data/iris.csv", 'r') as csvfile:
        r = reader(csvfile)
        for row in r:
            data.append(row)

    del data[0]  # Delete Headers
    return data


data = load_csv()

# Map labels
labels = []
out = []
for j in range(len(data)):
    row = data[j]
    l = row.pop()
    try:
        i = labels.index(l)
    except:
        labels.append(l)
        i = len(labels) - 1

    out.append(i)
    data[j] = list(map(float, row))


idx = np.random.randint(len(data), size=50)
X = np.array(data)[idx, :]
y = np.array([out]).T[idx, :]
np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = np.random.random((4, 8))
syn1 = np.random.random((8, 3))

for j in range(9999999):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sig(np.dot(l0, syn0))
    l2 = sig(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*sig(l2, True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sig(l1, True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
