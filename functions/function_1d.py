import numpy as np
import matplotlib.pyplot as plt

class Func:
    def __init__(self, x0, x1, width, k0=-1, k1=1):
        self.x0 = x0
        self.x1 = x1
        self.width = width
        self.k0 = k0
        self.k1 = k1

    def eval_point(self, x):
        sum0 = np.abs(x-self.x0)**(0.4)
        sum1 = np.exp(-(x-self.x1)**2/(2 * self.width**2))
        return self.k0 * sum0 + self.k1 * sum1

    def eval_range(self, x_min, x_max, num):
        xs = np.linspace(x_min, x_max, num)

        output = np.empty((num))
        for i,x in enumerate(xs):
            output[i] = self.eval_point(x)

        return xs, output


if __name__=='__main__':
    func = Func(1.7, 0.5, 1, k0=-1.1, k1=1.7)
    _, eval = func.eval_range(0, 2, 2000)
    
    plt.title('Function')
    plt.plot(np.linspace(0, 2, 2000),eval)
    plt.show()
