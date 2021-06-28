class Node:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __repr__(self) -> str:
        return f"({self.weight} * x + {self.bias})"

    def output(self, x: float) -> float:
        return self.weight * x + self. bias

class RELNode(Node):
    def output(self, x:float) -> float:
        out = super().output(x)
        return out if out > 0 else 0