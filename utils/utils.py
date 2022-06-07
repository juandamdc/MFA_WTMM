import numpy as np


def _1d_euclidian(point1, point2):
    return np.abs(point1 - point2)


def _2d_euclidian(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def _right_direction(point1, point2, alpha, alpha_error):
    return np.abs(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / np.pi - alpha) <= alpha_error



def neighborhood_1d(point_ind, points, C, scale, dist_function=_1d_euclidian):
    left_neighborhood, right_neighborhood = list(), list()

    for ind in range(point_ind - 1, -1, -1):
        if dist_function(points[point_ind], points[ind]) <= C * scale:
           left_neighborhood.append((ind, points[ind]))
        else:
            break

    for ind in range(point_ind + 1, len(points), 1):
        if dist_function(points[point_ind], points[ind]) <= C * scale:
           right_neighborhood.append((ind, points[ind]))
        else:
            break

    return left_neighborhood, right_neighborhood


def neighborhood_2d(pointx, pointy, alpha, points, C, scale, dist_function=_2d_euclidian, alpha_error=1):
    neighborhood = list()

    if alpha == 0:
        for indx in range(pointx + 1, points.shape[1], 1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                neighborhood.append(((indx, pointy), points[pointy][indx]))
            else:
                break
   
    elif 0 < alpha < 90:
        for indx in range(pointx + 1, points.shape[1], 1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                for indy in range(pointy + 1, points.shape[1], 1):
                    if dist_function(points[pointy][pointx], points[indy][indx]) <= C * scale:
                        if _right_direction(points[pointy][pointx], points[indy][indx], alpha, alpha_error):
                            neighborhood.append(((indx, indy), points[indy][indx]))
                    else:
                        break
            else:
                break   
    
    elif alpha == 90:
        for indy in range(pointy + 1, points.shape[1], 1):
            if dist_function(points[pointy][pointx], points[indy][pointx]) <= C * scale:
                neighborhood.append(((pointx, indy), points[indy][pointx]))
            else:
                break

    elif 90 < alpha < 180:
        for indx in range(pointx - 1, -1, -1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                for indy in range(pointy + 1, points.shape[1], 1):
                    if dist_function(points[pointy][pointx], points[indy][indx]) <= C * scale:
                        if _right_direction(points[pointy][pointx], points[indy][indx], alpha, alpha_error):
                            neighborhood.append(((indx, indy), points[indy][indx]))
                    else:
                        break
            else:
                break

    elif alpha == 180 or alpha == -180:
        for indx in range(pointx - 1, -1, -1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                neighborhood.append(((indx, pointy), points[pointy][indx]))
            else:
                break

    elif -180 < alpha < -90:
        for indx in range(pointx - 1, -1, -1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                for indy in range(pointy - 1, -1, -1):
                    if dist_function(points[pointy][pointx], points[indy][indx]) <= C * scale:
                        if _right_direction(points[pointy][pointx], points[indy][indx], alpha, alpha_error):
                            neighborhood.append(((indx, indy), points[indy][indx]))
                    else:
                        break
            else:
                break

    elif alpha == -90:
        for indy in range(pointy - 1, -1, -1):
            if dist_function(points[pointy][pointx], points[indy][pointx]) <= C * scale:
                neighborhood.append(((pointx, indy), points[indy][pointx]))
            else:
                break

    elif -90 < alpha < 0:
        for indx in range(pointx + 1, points.shape[1], 1):
            if dist_function(points[pointy][pointx], points[pointy][indx]) <= C * scale:
                for indy in range(pointy - 1, -1, -1):
                    if dist_function(points[pointy][pointx], points[indy][indx]) <= C * scale:
                        if _right_direction(points[pointy][pointx], points[indy][indx], alpha, alpha_error):
                            neighborhood.append(((indx, indy), points[indy][indx]))
                    else:
                        break
            else:
                break

    return neighborhood


def neighborhood_2d_check(pointx, pointy, neighborhood, points, C, scale, dist_function=_2d_euclidian):
    new_neighborhood = list()

    for (indx, indy), _ in neighborhood:
        if dist_function(points[pointy][pointx], points[indy][indx]) <= C * scale:
            new_neighborhood.append(((indx, indy), points[indy][indx]))

    return new_neighborhood


def is_max_1d(point_ind, neighborhood, eval_wavelet):
    left, right = neighborhood

    cond_one, cond_two = True, True

    for ind, _ in left:
        if eval_wavelet[point_ind] <= eval_wavelet[ind]:
            cond_one = False
        
        if eval_wavelet[point_ind] < eval_wavelet[ind]:
            cond_two = False

        if not(cond_one or cond_two):
            return False
    
    for ind, _ in right:
        if eval_wavelet[point_ind] < eval_wavelet[ind]:
            cond_one = False

        if eval_wavelet[point_ind] <= eval_wavelet[ind]:
            cond_two = False
        
        if not(cond_one or cond_two):
            return False

    return True


def is_max_2d(point_x, point_y, neighborhood, eval_wavelet):
    if len(neighborhood) == 0:
        return False

    for (indx, indy), _ in neighborhood:
        if eval_wavelet[point_y][point_x] <= eval_wavelet[indy][indx]:
            return False

    return True
