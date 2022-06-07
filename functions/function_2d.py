import numpy as np
import matplotlib.pyplot as plt

class Func:
    def __init__(self, x_x0, y_x0, x_x1, y_x1, width, a=1, b=-1):
        self.x_x0 = x_x0
        self.y_x0 = y_x0
        self.x_x1 = x_x1
        self.y_x1 = y_x1
        self.a = a
        self.b = b
        self.width = width


    def eval_point(self, x, y):
        mod1 = (x-self.x_x1)**2 + (y-self.y_x1)**2
        mod2 = np.sqrt((x-self.x_x0)**2 + (y-self.y_x0)**2)
        gauss = np.exp(-mod1/(2 * self.width**2))
        return self.a * gauss + self.b * mod2**(0.3)


    def eval_range(self, x_min, x_max, y_min, y_max, num):
        xs = np.linspace(x_min, x_max, num)
        ys = np.linspace(y_min, y_max, num)
       
        output_points = np.empty((num, num), tuple)
        output_eval = np.empty((num, num))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                output_eval[j][i] = self.eval_point(x,y)
                output_points[j][i] = (x, y)

        return output_points, output_eval


if __name__=='__main__':
    func = Func(-128, -128, 128, 128, 64)
    _, eval = func.eval_range(-256, 256, -256, 256, 513)

    plt.title('Function')
    plt.imshow(eval, cmap='gray', vmin=np.min(eval), vmax=np.max(eval), origin='lower', extent=[-256, 256, -256, 256])
    plt.show()