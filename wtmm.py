import numpy as np
from scipy.signal import ricker

from cwt import cwt_1d, cwt_2d
from wavelet_2d import gaussian_2d, ricker_2d
from functions.function_1d import Func as func1d
from functions.function_2d import Func as func2d
from utils.utils import neighborhood_1d, is_max_1d, neighborhood_2d, is_max_2d, neighborhood_2d_check, _2d_euclidian
from utils.disjointSet import DisjointSet


def _build_chains_2d(mod_max, epsilon, useThreshold = False, threshold = 1, dist_function = _2d_euclidian):
    if useThreshold:
        mod_max = [elm for elm in mod_max if elm[2] > threshold]

    ds = DisjointSet([i for i in range(len(mod_max))])
    
    for i in range(len(mod_max)):
        for j in range(i+1, len(mod_max)):
            if dist_function(mod_max[i][1], mod_max[j][1]) <= epsilon:
                ds.joint(i,j)

    chains, max_chains = dict(), dict()
    for i in range(len(mod_max)):
        parent = ds.getParent(i)
        
        if parent in chains.keys():
            chains[parent].append(mod_max[i])

            if mod_max[i][2] > max_chains[parent][0][2]:
                max_chains[parent] = [mod_max[i]]
            elif mod_max[i][2] == max_chains[parent][0][2]:
                max_chains[parent].append(mod_max[i])
        
        else:
            chains[parent] = [mod_max[i]]
            max_chains[parent] = [mod_max[i]]

    return list(chains.values()), list(max_chains.values())


def _build_lines_2d(wtmmm_alphamin, wtmmm_alphamax, dist_function = _2d_euclidian):
    connect = dict()

    for alphamin in wtmmm_alphamin:
        min_dist = np.Infinity
        for alphamax in wtmmm_alphamax:
            if dist_function(alphamin[0][1], alphamax[0][1]) < min_dist:
                min_dist = dist_function(alphamin[0][1], alphamax[0][1])
                connect[alphamin[0]] = alphamax[0]
    
    return connect


def _build_wtmmm_max_lines(max_lines):
    wtmmm_max_lines = np.empty((len(max_lines)+1, len(max_lines[0])))

    for ind, max in enumerate(max_lines[0].keys()):
        wtmmm_max_lines[0][ind] = max[2]

    lower_alpha = list(max_lines[0].keys())
    for ind_line in range(1, wtmmm_max_lines.shape[0]):
        for ind_column, val_alpha in enumerate(lower_alpha):
            wtmmm_max_lines[ind_line][ind_column] = np.max([wtmmm_max_lines[ind_line-1][ind_column], val_alpha[2]])
            lower_alpha[ind_column] = max_lines[ind_line-1][val_alpha]

    return wtmmm_max_lines


def wtmm_1d(x_min, x_max, num, function, wavelet, scales):
    points, eval_points = function(x_min, x_max, num)
    eval_cwt = cwt_1d(eval_points, wavelet, scales)

    wtmm = np.empty(len(scales), list)
    for ind, scale in enumerate(scales):
        wtmm[ind] = list()

        for point_ind in range(len(points)):
            neighborhood = neighborhood_1d(point_ind, points, np.abs(points[1] - points[0]), scale)

            if is_max_1d(point_ind, neighborhood, eval_cwt[ind]):
                wtmm[ind].append((point_ind, points[point_ind], eval_cwt[ind][point_ind]))

    return wtmm


def wtmm_2d(x_min, x_max, y_min, y_max, num, function, wavelet, scales):
    points, eval_points = function(x_min, x_max, y_min, y_max, num)
    C = np.abs(points[0][1][0] - points[0][0][0])
    
    wavelet_x, wavelet_y = wavelet()
    eval_cwt = cwt_2d(eval_points, wavelet_x, wavelet_y, scales)

    memo = dict()
    wtmm = np.empty(len(scales), list)
    for ind, scale in enumerate(scales[::-1]):
        wtmm[ind] = list()
        _, _, cwt_mod, cwt_angle = eval_cwt[ind]

        for y in range(points.shape[0]):
            for x in range(points.shape[1]):
                if ind == 0:
                    memo[(x,y)] = neighborhood_2d(x, y, cwt_angle[y][x], points, C, scale, alpha_error=0.1)
                else:
                    memo[(x,y)] = neighborhood_2d_check(x, y, memo[(x,y)], points, C, scale)

                if is_max_2d(x, y, memo[(x,y)], cwt_mod):
                    wtmm[ind].append(((x, y), points[y][x], cwt_mod[y][x], cwt_angle[y][x]))
   
    wtmm = wtmm[::-1]

    chains = np.empty(len(scales), list)
    wtmmm = np.empty(len(scales), list)
    for ind, scale in enumerate(scales):
        chains[ind], wtmmm[ind] = _build_chains_2d(wtmm[ind], C*scale)
    
    max_lines = np.empty(len(scales)-1, dict)
    for i in range(len(scales)-1):
        max_lines[i] = _build_lines_2d(wtmmm[i], wtmmm[i+1])

    wtmmm_max_lines = _build_wtmmm_max_lines(max_lines)

    return eval_cwt, wtmm, wtmmm, max_lines, wtmmm_max_lines



if __name__=='__main__':
    # func = func1d(1.7, 0.5, 1, k0=-1.1, k1=1.7)
    # print(wtmm_1d(0, 2, 2000, func.eval_range, ricker,  np.linspace(3,5, 10)))
    
    func = func2d(-128, -128, 128, 128, 64)
    cwt, wtmm, max_chains, max_lines, wtmmm_max_lines = wtmm_2d(-256, 256, -256, 256, 513, func.eval_range, gaussian_2d, [3,3.3])
    