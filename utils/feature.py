import numpy as np

def feature_est1(holder, dh):
    holder_min = np.min(holder)
    holder_max = np.max(holder)
    holder_mean = np.mean(holder)

    dh_min = np.min(dh)
    dh_max = np.max(dh)
    dh_mean = np.mean(dh)

    return [holder_min, holder_max, holder_mean, dh_min, dh_max, dh_mean]


def feature_est2(holder, dh):
    holder.resize((holder.shape[0] * holder.shape[1], ))
    dh.resize((dh.shape[0] * dh.shape[1], ))
    
    return list(holder) + list(dh)
