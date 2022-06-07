from cantor_set import CantorSet
import matplotlib.pyplot as plt
import numpy as np


class DevilStaircase:
    def __init__(self, p = 0.5):
        self.cantorSet = CantorSet(p)


    def calculate(self, x, cantorSetLevel = 10):
        intrs, measures = self.cantorSet._getLevel(cantorSetLevel)
        
        def func(x):
            value = 0

            for intr, measure in zip(intrs, measures):
                if x > intr[0].sup:
                    value += measure
                elif x in intr:
                    value += measure * (x - intr[0].inf) / (intr[0].sup - intr[0].inf)    
                else:
                    break

            return value

        return func(x)


    def plot(self, cantorSetLevel=10):
        if cantorSetLevel < 10:
            num = 3**cantorSetLevel
        else:
            num = 3**10

        x = np.linspace(0, 1, num)
        y = np.asarray([self.calculate(x_val, cantorSetLevel) for x_val in x])

        plt.figure('Devil Staircase')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.suptitle('Devil Staircase')
        plt.plot(x, y)
        plt.show()


if __name__=='__main__':
    dv = DevilStaircase()
    dv.plot()