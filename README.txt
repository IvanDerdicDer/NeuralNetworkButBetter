# Python 3.9

A better version of my previous neural network.

# Nodes.py #
Contains the type of nodes the neural network uses.
    Node() - box standard node: weight, bias, and output(weight*x + bias)
    RELNode() - uses a rectified liner function for the output: output if output > 0 else 0

# NeuralNetworkButBetter.py #
Contains the bare bones version of a neural network.
Intended to be inherited by other models and modified to work as intended for that model.

class NeuralNetwork:
    Methods:
        softmaxActivation(outputOfTheLastLayer: list[float]) -> list[float]
            - converts the output of the given layer (should be the last layer) to a probability chart
            (sum of the return list is 1)
        addLayer(self, numberOfNodes: int) - add a layer of numberOfNodes nodes
        layerSum(self, inputValues: list[float], numberOfLayers: int) -> list[float]
            - runs one layer
        runNetwork(self, inputList: list[float]) -> list[float]
            - checks if the network can be ran with the given input list
            - runs the network

# MutatableNeuralNetwork.py #
Inherits the NeuralNetwork class. Expands it for genetic algorithm learning.

class MutatableNeuralNetwork(NeuralNetwork):
    Overridden methods:
        __init__(self, mutationRate: float = 0.05)
            - added mutation rate variable

    New methods:
        mutate(self)
            - mutates the network based on the mutation rate

# Example.py #
To be added.