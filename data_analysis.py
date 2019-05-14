import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.layers import Flatten, Dense, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad, Adadelta, Adamax
from keras.callbacks import Callback
from keras.utils import to_categorical


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
        if (samples and iterations > samples):
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
"""
state, action = load_data("data/{0}.txt".format("iggi"), samples = None)
data = split_data(state, action)
plot_distribution(data)

"""
agents = [ "outer"]
for agent in agents:
    state, action = load_data("data/{0}.txt".format(agent), samples = 600000)
    data = split_data(state, action)

    test_size = 10000
    train_sizes = [-1]
    for train in train_sizes:
        test_state = state[:test_size]
        test_action = action[:test_size]
        train_state = state[test_size:train]
        train_action = action[test_size:train]

        K.clear_session()
        model = Sequential([
            Dense(256, input_dim=561),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(128),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(64, activation = 'sigmoid'),
            BatchNormalization(momentum=0.8),
            Dense(20, activation='softmax'),
        ])
        opt = Adam(0.001, 0.5, clipvalue=5)
        model.compile(loss=['categorical_crossentropy'], optimizer = opt, metrics = ['accuracy'])

        model.fit(train_state, train_action, epochs = 40)
        loss, acc = model.evaluate(test_state, test_action)
        predicted = model.predict(test_state)
        model.save("{0}{1}.h5".format(agent, train))
        
        real = [0 for i in range(20)]
        gen = [0 for i in range(20)]
        for lab in predicted:
            real[np.argmax(lab)] += 1
        for lab in test_action:
            gen[np.argmax(lab)] += 1
        plt.bar(np.arange(0,20, step = 1), np.array(real) / test_size, label = "Real", width = 0.5)
        plt.bar(np.arange(0.5,20.5, step = 1), np.array(gen) / test_size, label = "Predicted", width = 0.5)
        plt.xlabel('Label')
        plt.ylabel('Samples')
        plt.legend()
        plt.title('Supervised {0}, Test Acc: {1}'.format(agent, acc))
        plt.grid(True)
        plt.savefig("Supervised/{0}{1}".format(agent, train))
        plt.close()
