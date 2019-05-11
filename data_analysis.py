import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model


def load_data(dataFile, samples = 50000):
    print("LOADING DATA")
    df = open(dataFile, mode = 'r')
    iterations = 0
    inputs = []
    outputs = []
    while True:
        data = df.read(1735)
        if (len(data)== 0):
            break
        if (data[0] == "-"):
            continue
        if (iterations > samples):
            break
        else:
            iterations += 1
            line = list(map(int, data))
            inputs.append(line[:561])
            outputs.append(line[561:581])
    df.close()
    print("LOADING DONE")
    return np.array(inputs), np.array(outputs)

def split_data(gamestates, labels):
    data = [[] for i in range(20)]
    for g, l in zip(gamestates, labels):
        index = np.argmax(l)
        data[index].append(g)
    return data

def plot_distribution(data):
    total = 0
    for x in data:
        total += len(x)
    dist = [len(x)/total for x in data]
    plt.bar(range(20), dist)
    plt.title("Data Distribution")
    plt.xlabel("Action Label")
    plt.ylabel("Number of samples")
    plt.show()

#state, action = load_data("data/iggi.txt")
#data = split_data(state, action)

#plot_distribution(data)
model_name = "models/bidirectional/"
model = load_model(model_name + "discriminator.h5")
#model.summary()
i = 0
for layer in model.layers:
    config = layer.get_config()
    for l in config['layers']:
        print(l)


