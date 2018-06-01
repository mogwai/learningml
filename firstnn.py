# First attempt at a raw no library NN
from csv import reader
from math import exp
from random import random, sample


LEARNING_RATE = 0.1


def load_csv():
    data = []
    with open("./data/iris.csv", 'r') as csvfile:
        r = reader(csvfile)
        for row in r:
            data.append(row)

    del data[0]  # Delete Headers
    return data


def forward_prop():
    for l in range(len(layers)-1):
        cur = layers[l]
        nxt = layers[l+1]
        w = weights[l]

        for i in range(len(nxt)):
            n = 0
            for j in range(len(cur)):
                n += w[j][i]*cur[j]

            nxt[i] = sig(n)


def toVector(label):
    return list(map(lambda x: int(x == label), range(len(labels))))



# Our one line sigmoid function
sig = lambda x, d=False: sig(x)*(1-sig(x)) if d else 1/(1+exp(-x))

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


# Configuration of the network (Size)
features = len(data[0])
layer_config = [features, 8, 8, len(labels)]

# Initialise layers and weights
weights = []
layers = []

# Create layers and random weights
for i in range(len(layer_config)):
    cur = layer_config[i]

    if i+1 < len(layer_config):
        nxt = layer_config[i+1]
        # next.length = row; cur.length = col;
        weights.append([[random()]*nxt]*cur)

    layers.append([0]*cur)


# Begin training
for _ in range(100000):
    newWeights = weights

    for dataindex in range(len(data)):
        # Set Input Layer
        layers[0] = data[dataindex]
        answer = toVector(out[dataindex])
        forward_prop()

        for L in range(len(weights)-1, 0, -1):
            w = weights[L]
            nw = newWeights[L]

            for i in range(len(w)-1, 0, -1):
                for j in range(len(w[i])):
                    error = 0
                    # Calculate the error based on what layer we're on
                    if (L == len(weights)-1):
                        for k in range(len(layers[L+1])):
                            error += 2*(layers[L+1][k] - answer[k])
                    else:
                        for k in range(len(layers[L+1])):
                            error += w[i][k] - nw[i][k]

                    delta = layers[L][j] * \
                        sig(layers[L+1][j], True)*error*LEARNING_RATE
                    nw[i][j] = w[i][j] - delta
