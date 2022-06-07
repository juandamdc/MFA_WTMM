import numpy as np


def select_wavelet(num):
    if num == 0:
        return gaussian_2d
    if num == 1:
        return ricker_2d


def gaussian_2d():
    def gaussian_dx(points, scale):
        xs = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        ys = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        
        output = np.empty((len(ys), len(xs)))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                mod = x**2 + y**2
                output[j][i] = -x/scale**2 * np.exp(-mod/(2*scale**2)) 

        return output

    def gaussian_dy(points, scale):
        xs = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        ys = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        
        output = np.empty((len(ys), len(xs)))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                mod = x**2 + y**2
                output[j][i] = -y/scale**2 * np.exp(-mod/(2*scale**2)) 

        return output

    return gaussian_dx, gaussian_dy



def ricker_2d():
    def ricker_dx(points, scale):
        xs = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        ys = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        
        output = np.empty((len(ys), len(xs)))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                mod = x**2 + y**2
                output[j][i] = -x/scale**2 * (4 - mod/scale**2) * np.exp(-mod/(2*scale**2)) 

        return output

    def ricker_dy(points, scale):
        xs = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        ys = np.arange(0, np.sqrt(points)) - (np.sqrt(points) - 1.0) / 2
        
        output = np.empty((len(ys), len(xs)))
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                mod = x**2 + y**2
                output[j][i] = -y/scale**2 * (4 - mod/scale**2) * np.exp(-mod/(2*scale**2)) 

        return output

    return ricker_dx, ricker_dy
