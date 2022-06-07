from time import time
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

from wtmm import wtmm_2d
from wavelet_2d import gaussian_2d
from functions.function_2d import Func as func2d
from functions.function_image import FunctionImage
from utils.time_convert import time_convert


def _compute_holder(cwt_mod, scales, wavelet_vanishing_moments):
    ln_scales = np.log(scales)
    
    exp_holder = np.empty(cwt_mod[0].shape)
    for indy in range(exp_holder.shape[0]):
        for indx in range(exp_holder.shape[1]):
            mod_values = [cwt_mod[indscale][indy][indx] for indscale in range(len(scales))]
            ln_mod = np.log(mod_values)

            c, _, _, _ = lstsq(np.c_[ln_scales], ln_mod)
            exp_holder[indy][indx] = c[0]

    return exp_holder


def _compute_partition_function(q, wtmmm_lines):
    return np.sum(np.power(wtmmm_lines, q))


def _compute_tao_q(q, scales, wtmmm_lines):
    partition_functions = np.empty(len(scales))
    for ind in range(len(scales)):
        partition_functions[ind] = _compute_partition_function(q, wtmmm_lines[ind])

    ln_partition_functions = np.log(partition_functions)
    ln_scales = np.log(scales)
    
    c, _,_,_ = lstsq(np.c_[ln_scales], ln_partition_functions)
    return c[0]


def _computer_multifractal_spectrum(exp_holder, qs, tao_qs):
    dh = np.empty(exp_holder.shape)
    for indy in range(dh.shape[0]):
        for indx in range(dh.shape[1]):
            dh[indy][indx] = np.min([q*exp_holder[indy][indx] - tao_q for q, tao_q in zip(qs, tao_qs)])

    return dh


def mfa_wtmm_2d(x_min, x_max, y_min, y_max, num_points, function, wavelet, min_scale, max_scale, num_scales, min_q, max_q, num_q):
    scales = np.linspace(min_scale, max_scale, num_scales)
    qs = np.linspace(min_q, max_q, num_q)

    cwt, _, _, _, wtmmm_sup_lines = wtmm_2d(x_min, x_max, y_min, y_max, num_points, function, wavelet, scales)
    
    tao_q = np.empty(len(qs))
    for ind, q in enumerate(qs):
        tao_q[ind] = _compute_tao_q(q, scales, wtmmm_sup_lines)

    cwt_mod = [cwt_scale[2] for cwt_scale in cwt]
    exp_holder = _compute_holder(cwt_mod, scales, 0)

    dh = _computer_multifractal_spectrum(exp_holder, qs, tao_q)

    return exp_holder, dh



if __name__=='__main__':
    ### 2d test function
    # func = func2d(-128, -128, 128, 128, 64)

    # start_time = time()
    # mfa_wtmm_2d(-256, 256, -256, 256, 1024, func.eval_range, gaussian_2d, 3, 5, 10, 2, 4, 5)
    # end_time = time()

    # time_convert(end_time - start_time)


    ### image test
    func = FunctionImage('JSRT_Database/Nodule/JPCLN022.IMG')
    # func = FunctionImage('JSRT_Database/Normal/JPCNN001.IMG')
    func.reduce_size_toHalf(2)

    start_time = time()
    holder, dh = mfa_wtmm_2d(0, func.columns-1, 0, func.rows-1, 0, func.eval_range, gaussian_2d, 3, 5, 10, 2, 4, 5)
    end_time = time()

    time_convert(end_time - start_time)

    plt.imshow(holder, cmap='Greys', vmin=np.min(holder), vmax=np.max(holder), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()
    plt.imshow(dh, cmap='Greys', vmin=np.min(dh), vmax=np.max(dh), extent=[0, func.columns-1, 0, func.rows-1])
    plt.show()