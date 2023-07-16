from abc import abstractclassmethod
import numpy as np

class Activation():
    _repository = {}

    def __init_subclass__(cls, prefix, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._repository[prefix] = cls

    def __new__(cls, **kwargs):
        cls = cls._repository[kwargs['prefix']]

        obj = object.__new__(cls)
        return obj

    @abstractclassmethod
    def function(self, x):
        pass

    @abstractclassmethod
    def derivative(self, x):
        pass


class Sigmoid(Activation, prefix='sigmoid'):
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Hyperbolic(Activation, prefix='htan'):
    def function(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x):
        return 1 - (
            np.power(np.exp(x) - np.exp(-x), 2) /
            np.power(np.exp(x) + np.exp(-x), 2)
        )

class LReLU(Activation, prefix='lrelu'):
    def function(self, x):
        return np.where(x > 0, x, x * 0.01)

    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)

class ReLU(Activation, prefix='relu'):
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Tan(Activation, prefix='tan'):
    def function(self, x):
        return np.tan(x)

    def derivative(self, x):
        return 1 / np.power(np.cos(x), 2)