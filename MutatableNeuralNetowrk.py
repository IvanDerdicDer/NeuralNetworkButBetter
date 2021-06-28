from NeuralNetworkButBetter import NeuralNetwork
from random import random, randint

class MutatableNeuralNetwork(NeuralNetwork):
    def __init__(self, mutationRate: float = 0.05):
        super(MutatableNeuralNetwork, self).__init__()
        self.mutationRate = mutationRate

    def mutate(self):
        for layer in range(len(self.neuralNetworkLayers)):
            for node in range(len(self.neuralNetworkLayers[layer])):
                mutation = random() * randint(-100, 100) / 100
                if abs(mutation) < self.mutationRate:
                    self.neuralNetworkLayers[layer][node].weight *= 1 - mutation
                    self.neuralNetworkLayers[layer][node].bias *= 1 - mutation
