from math import exp
from Nodes import *
from random import random, randint

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

        self.neuralNetworkLayers.append([self.node(random() * randint(-100, 100) / 100,
                                                   random() * randint(-100, 100) / 100) for _ in range(numberOfNodes)])

    @staticmethod
    def layerSum(inputValues: list[float], layer: list) -> list[float]:
        return [sum([node.output(x) for x in inputValues]) for node in layer]

    def runNetwork(self, inputList: list[float]) -> list[float]:
        if len(inputList) != len(self.neuralNetworkLayers[0]):
            raise IndexError("Length of the 'inputList' does not match the number of nodes in the first layer")

        nextInput = inputList.copy()
        for layer in self.neuralNetworkLayers:
            nextInput = self.layerSum(nextInput, layer)

        return self.softmaxActivation(nextInput)


"""if __name__ == '__main__':
    nn = NeuralNetwork()
    for _ in range(4):
        nn.addLayer(4)

    inputList = [randint(1, 5) for _ in range(4)]

    print(nn)

    print(f"Output of layer(4): {nn.runNetwork(inputList)}")"""