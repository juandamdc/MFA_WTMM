import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from functions.cantor_set import CantorSet
from functions.function_2d import Func as func2d
from functions.function_1d import Func as func1d
from functions.function_image import FunctionImage
from wavelet_2d import ricker_2d, gaussian_2d


def _mod(inpt1, inpt2):
    if inpt1.shape != inpt2.shape:
        raise ValueError('inpt1 and inpt2 should have the same shape')
    
    output = np.empty(inpt1.shape)
    for i in range(inpt1.shape[0]):
        for j in range(inpt1.shape[1]):
            output[i][j] = np.sqrt(inpt1[i][j]**2 + inpt2[i][j]**2)
    
    return output


def _angle(inpt1, inpt2):
    if inpt1.shape != inpt2.shape:
        raise ValueError('inpt1 and inpt2 should have the same shape')
    
    return np.arctan2(inpt1, inpt2) * 180 / np.pi


def cwt_1d(points, wavelet, scales):
    if points.ndim != 1:
        raise ValueError('points should have dimension 1')

    return signal.cwt(points, wavelet, scales)


def cwt_2d(points, wavelet_x, wavelet_y, scales):
    if points.ndim != 2:
        raise ValueError('points should have dimension 2')

    output = list()
    for scale in scales:
        N = np.min([10 * scale, points.shape[0], points.shape[1]])**2
        
        wavelet_x_vals = np.conj(wavelet_x(N, scale))[::-1]
        wavelet_y_vals = np.conj(wavelet_y(N, scale))[::-1]

        output_dx = signal.convolve2d(points, wavelet_x_vals, mode='same', boundary='symm')
        output_dy = signal.convolve2d(points, wavelet_y_vals, mode='same', boundary='symm')
        output_mod = _mod(output_dx, output_dy)
        output_angle = _angle(output_dy, output_dx)

        output.append((output_dx, output_dy, output_mod, output_angle))
    
    return output



if __name__=='__main__':
    ### test func
    # func = func2d(-128, -128, 128, 128, 64)
    # _, eval = func.eval_range(-256, 256, -256, 256, 512)

    # wavelet_x, wavelet_y = gaussian_2d()
    # out = cwt_2d(eval, wavelet_x, wavelet_y, [5])

    # plt.title('Function')
    # plt.imshow(out[-1][0], cmap='Greys', vmin=np.min(out[-1][0]), vmax=np.max(out[-1][0]), origin='lower', extent=[-256, 256, -256, 256])
    # plt.show()
    # plt.imshow(out[-1][1], cmap='Greys', vmin=np.min(out[-1][1]), vmax=np.max(out[-1][1]), origin='lower', extent=[-256, 256, -256, 256])
    # plt.show()
    # plt.imshow(out[-1][2], cmap='Greys', vmin=np.min(out[-1][2]), vmax=np.max(out[-1][2]), origin='lower', extent=[-256, 256, -256, 256])
    # plt.show()
    # plt.imshow(np.abs(out[-1][3]), cmap='Greys', vmin=np.min(np.abs(out[-1][3])), vmax=np.max(np.abs(out[-1][3])), origin='lower', extent=[-256, 256, -256, 256])
    # plt.show()



    ### test image
    func = FunctionImage('JSRT_Database/Nodule/JPCLN022.IMG')
    func.reduce_size_toHalf(2)
    _, eval = func.eval_range(0, func.columns-1, 0, func.rows-1, 0)
    
    wavelet_x, wavelet_y = gaussian_2d()
    out = cwt_2d(eval, wavelet_x, wavelet_y, [3])

    func.plot()

    plt.title('Function')
    plt.imshow(out[0][0], cmap='Greys', vmin=np.min(out[-1][0]), vmax=np.max(out[-1][0]), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()
    plt.imshow(out[0][1], cmap='Greys', vmin=np.min(out[-1][1]), vmax=np.max(out[-1][1]), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()
    plt.imshow(out[0][2], cmap='Greys', vmin=np.min(out[-1][2]), vmax=np.max(out[-1][2]), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()
    plt.imshow(np.abs(out[0][3]), cmap='Greys', vmin=np.min(np.abs(out[-1][3])), vmax=np.max(np.abs(out[-1][3])), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()
