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

        self.neuralNetworkLayers.append([self.node(random(), random()) for _ in range(numberOfNodes)])

    def layerSum(self, inputValues: list[float], numberOfLayers: int) -> list[float]:
        indexOfCurrentLayer = numberOfLayers - 1
        lengthOfCurrentLayer = len(self.neuralNetworkLayers[indexOfCurrentLayer])
        if not indexOfCurrentLayer:
            return [self.neuralNetworkLayers[indexOfCurrentLayer][i].output(inputValues[i]) for i in range(lengthOfCurrentLayer)]

        previousLayerOutput: list[float] = self.layerSum(inputValues, numberOfLayers - 1)

        #print(f"Output of layer({indexOfCurrentLayer - 1}): {previousLayerOutput}")

        if indexOfCurrentLayer == len(self.neuralNetworkLayers) - 1:
            return self.softmaxActivation([sum([node.output(x) for x in previousLayerOutput]) for node in self.neuralNetworkLayers[indexOfCurrentLayer]])

        return [sum([node.output(x) for x in previousLayerOutput]) for node in self.neuralNetworkLayers[indexOfCurrentLayer]]

    def runNetwork(self, inputList: list[float]) -> list[float]:
        if len(inputList) != len(self.neuralNetworkLayers[0]):
            raise IndexError("Length of the 'inputList' does not match the number of nodes in the first layer")

        return self.layerSum(inputList, len(self.neuralNetworkLayers))


"""if __name__ == '__main__':
    nn = NeuralNetwork()
    for _ in range(4):
        nn.addLayer(4)

    inputList = [randint(1, 5) for _ in range(4)]

    print(nn)

    print(f"Output of layer(4): {nn.runNetwork(inputList)}")"""