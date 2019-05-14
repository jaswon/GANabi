from keras.models import load_model
import numpy as np
import random
from deap import base, creator, tools, algorithms, cma
import matplotlib.pyplot as plt
from keras.optimizers import Adam

#Loads data from a file
def load_data(dataFile, samples = 50000):
    print("LOADING DATA")
    df = open(dataFile, mode = 'r', newline = '\n')
    iterations = 0
    inputs = []
    outputs = []
    while True:
        data = df.read(1735)
        #data = df.read(592)
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
            #inputs.append(line[:572])
            #outputs.append(line[572:])
    df.close()
    print("LOADING DONE")
    return np.array(inputs), np.array(outputs)

#Loads generator model and compiles
def get_model(modelName):
    model = load_model(modelName)
    model.compile(optimizer = Adam(lr = 0.001), loss=['categorical_crossentropy'], metrics = ['accuracy'])
    return model

#Fitness Function
def evaluate(individaul):
    latent_space = np.tile(np.array(individaul), (inputs.shape[0], 1))
    score = model.evaluate([latent_space, inputs], outputs)
    return [score[1]]

class Evolution():
    def __init__(self, tournamentSize = 3, independence = 0.1):
        self.latent_dim = 50
        self.toolbox = None
        self.stats = None
        self.generations = 0
        self.avg_fitness = []
        self.max_fitness = []
        self.min_fitness = []
        self.hof = tools.HallOfFame(1)

        #Individual to evolve.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=self.latent_dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #evolution strategy
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=independence)
        toolbox.register("select", tools.selTournament, tournsize=tournamentSize)
        toolbox.register("evaluate", evaluate)

        #cma strategy, can add lambda_= pop_size
        strategy = cma.Strategy(centroid=np.random.normal(0,1, (self.latent_dim,)), sigma=1.0)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        self.toolbox = toolbox
        #Statistics gathering
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        self.stats = stats

    def evolve(self, crossover_prob = 0.5, mutation_prob = 0.1, num_generation = 20, pop_size = 20, elitism = 5):
        #clear stats for graphing
        self.max_fitness = []
        self.min_fitness = []
        self.avg_fitness = []
        self.generations = num_generation
        #initialize population
        pop = self.toolbox.population(n=pop_size)
        #assign initial fitness
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        #Record the best result
        #evolve 
        for g in range(num_generation):
            print("Generation {}".format(g + 1))

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = map(self.toolbox.clone, offspring)
            # Apply both crossover and mutation
            offspring = algorithms.varAnd(pop, self.toolbox, crossover_prob, mutation_prob)

            # Evaluate the new individuals for fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Record Fitness statistics
            record = self.stats.compile(pop)
            self.avg_fitness.append(record['avg'])
            self.max_fitness.append(record['max'])
            self.min_fitness.append(record['min'])
            print("Average Fitness {}".format(record['avg']))

            #Update the best gene so far
            self.hof.update(offspring)

            pop[:] = self.toolbox.select(pop, elitism) + self.toolbox.select(offspring, len(pop) - elitism)

        # returns the best gene
        return hof

    def evolve_preset(self, crossover_prob = 0.5, mutation_prob = 0.4, num_generation = 20, pop_size = 20, mu = 100, lam = 30):
        pop = self.toolbox.population(n=pop_size)
        self.generations = num_generation + 1
        #individuals, stats = algorithms.eaSimple(pop, self.toolbox, crossover_prob, mutation_prob, num_generation, stats = self.stats, verbose = True)
        #individuals, stats = algorithms.eaMuPlusLambda(pop, self.toolbox, mu, lam, crossover_prob, mutation_prob, num_generation, stats = self.stats, verbose = True)
        #individuals, stats = algorithms.eaMuCommaLambda(pop, self.toolbox, mu, lam, crossover_prob, mutation_prob, num_generation, stats = self.stats, verbose = True)
        #CMA-ES
        individuals, stats = algorithms.eaGenerateUpdate(self.toolbox, num_generation, halloffame= self.hof, stats = self.stats,verbose=True)
        #Differential Evolution
        self.avg_fitness, self.max_fitness, self.min_fitness = stats.select('avg', 'max', 'min')
        return individuals, self.hof

    def plot(self, name):
        plt.plot(range(len(self.avg_fitness)), self.avg_fitness, label="Average Fitness")
        plt.plot(range(len(self.max_fitness)), self.max_fitness,  label="Maximum Fitness")
        plt.plot(range(len(self.min_fitness)), self.min_fitness, label="Minimum Fintess")
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('GAN Evaluation')
        plt.grid(True)
        plt.savefig("Evolution/" + name)
        plt.close()

    def predict(self, model, state, action, name, total):
        latent_space = np.tile(np.array(self.hof[0]), (state.shape[0], 1))
        print(state.shape[0])
        labels = model.predict([latent_space, state])
        sample_size = 1000
        gen = [0 for i in range(20)]
        for lab in labels:
            gen[np.argmax(lab)] += 1
        real = [0 for i in range(20)]
        for lab in action:
            real[np.argmax(lab)] += 1
        plt.bar(np.arange(0,20, step = 1), np.array(real) / total, label = "Real", width = 0.5)
        plt.bar(np.arange(0.5,20.5, step = 1), np.array(gen) / total, label = "Predicted", width = 0.5)
        plt.xlabel("Label")
        plt.ylabel("Samples")
        plt.title("Evolved Generator")
        plt.legend()
        plt.grid(True)
        plt.savefig("Evolution/" + name)
        plt.close()

if __name__ == '__main__':
    total = 10000
    state, action = load_data("data/iggi.txt", samples = total)
    inputs = state
    outputs = action
    x = ["model"]
    for name in x:
        model = get_model(name + "/generator.h5")
        ### Uses Tournament Selection right now randomized - (Best of [tournamentsize]) 
        ev = Evolution(tournamentSize = 5, independence = 0.1) 
        #individuals = ev.evolve(crossover_prob = 0.5, mutation_prob = 0.5, num_generation = 150, pop_size = 30)
        individuals, hof  = ev.evolve_preset(crossover_prob = 0.9, mutation_prob = 0.1, num_generation = 100, pop_size = 100)
        ev.predict(model, state, action, name + "Distribution", total)
        ev.plot(name)

