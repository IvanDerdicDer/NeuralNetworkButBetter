from math import exp
from Nodes import *
from random import random

class NeuralNetwork:
    def __init__(self, nodeType = RELNode):
        self.node = nodeType
        self.neuralNetworkLayers:list[list[nodeType]] = []

    def __repr__(self) -> str:
        output = ''
        for layer in self.neuralNetworkLayers:
            for node in layer:
                output += str(node)
            output += '\n'
        return output

    @staticmethod
    def softmaxActivation(outputOfTheLastLayer: list[float]) -> list[float]:
        sumOfTheEXPOutput: float = 0
        copyList: list[float] = outputOfTheLastLayer.copy()
        largestOutput: float = max(copyList)

        #Normalasing the output
        for i in range(len(copyList)):
            copyList[i] -= largestOutput

        for i in copyList:
            sumOfTheEXPOutput += exp(i)

        for i in range(len(copyList)):
            copyList[i] = exp(copyList[i]) / sumOfTheEXPOutput

        return copyList

    def addLayer(self, numberOfNodes: int):
        if type(numberOfNodes) != type(int()):
            raise TypeError("'numberOfNodes' should be an int")

        self.neuralNetworkLayers.append([self.node(random(), random()) for _ in range(numberOfNodes)])

    def layerSum(self, inputValues: list[float], numberOfLayers: int) -> list[float]:
        indexOfCurrentLayer = numberOfLayers - 1
        lengthOfCurrentLayer = len(self.neuralNetworkLayers[indexOfCurrentLayer])
        lengthOfNextLayer = len(self.neuralNetworkLayers[(indexOfCurrentLayer + 1) % len(self.neuralNetworkLayers)])
        if not indexOfCurrentLayer:
            toReturn = 0
            for i in range(lengthOfCurrentLayer):
                toReturn += self.neuralNetworkLayers[indexOfCurrentLayer][i].output(inputValues[i])

            return [toReturn] * lengthOfNextLayer

        previousLayerOutput: list[float] = self.layerSum(inputValues, numberOfLayers - 1)

        if indexOfCurrentLayer == len(self.neuralNetworkLayers) - 1:
            return self.softmaxActivation([self.neuralNetworkLayers[indexOfCurrentLayer][i].output(previousLayerOutput[i]) for i in range(lengthOfCurrentLayer)])

        return [sum([self.neuralNetworkLayers[indexOfCurrentLayer][i].output(previousLayerOutput[i]) for i in range(lengthOfCurrentLayer)])] * lengthOfNextLayer

    def runNetwork(self, inputList: list[float]) -> list[float]:
        if len(inputList) != len(self.neuralNetworkLayers[0]):
            raise IndexError("Length of the 'inputList' does not match the number of nodes in the first layer")

        return self.layerSum(inputList, len(self.neuralNetworkLayers))
