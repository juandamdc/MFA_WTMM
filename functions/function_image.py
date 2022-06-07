import matplotlib.pyplot as plt
import numpy as np


class FunctionImage:
    def __init__(self, path, shape = (2048,2048), dtype = np.dtype('>u2')):
        img_file = open(path, 'rb')
        self.img = np.fromfile(img_file, dtype).reshape(shape)
        self.rows = shape[0]
        self.columns = shape[1]


    def reduce_size_toHalf(self, n=1):
        for _ in range(n):
            new_img = np.empty((self.rows//2, self.columns//2))

            for i in range(new_img.shape[0]):
                for j in range(new_img.shape[1]):
                    new_img[i,j] = (self.img[2*i, 2*j] + self.img[2*i, 2*j+1] + self.img[2*i+1, 2*j] + self.img[2*i+1, 2*j+1]) / 4

            self.img = new_img
            self.rows = new_img.shape[0]
            self.columns = new_img.shape[1]
            

    def eval_point(self, x, y):
        return self.img[y][x]


    def eval_range(self, x_min, x_max, y_min, y_max, num):
        xs = np.linspace(x_min, x_max, x_max-x_min+1, dtype=int)
        ys = np.linspace(y_min, y_max, y_max-y_min+1, dtype=int)

        output_points = np.empty((len(ys), len(xs)), tuple)
        output_eval = np.empty((len(ys), len(xs)))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                output_eval[j][i] = self.eval_point(x,y)
                output_points[j][i] = (x, y)

        return output_points, output_eval


    def plot(self):
        plt.imshow(self.img, cmap='Greys')
        plt.show()

