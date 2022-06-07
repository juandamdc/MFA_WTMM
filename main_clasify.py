import sys
from joblib import load

from mfa_wtmm import mfa_wtmm_2d
from wavelet_2d import gaussian_2d, select_wavelet
from functions.function_image import FunctionImage
from utils.feature import feature_est1, feature_est2



def clasify_image(imageLocation, modelLocation, wavelet=gaussian_2d):
    func = FunctionImage(imageLocation)
    func.reduce_size_toHalf(2)
    holder, dh = mfa_wtmm_2d(0, func.columns-1, 0, func.rows-1, 0, func.eval_range, wavelet, 3, 5, 10, 0, 2, 5)
    X = [feature_est2(holder, dh)]

    clf = load(modelLocation)
    return clf.predict(X)[0]



if __name__=='__main__':
    usage = '''   '''
    
    try:
        clasification = clasify_image(sys.argv[1], sys.argv[2], select_wavelet(int(sys.argv[3])))

        if clasification == 0:
            print('Normal lung')
        else:
            print('Lung with nodule')

    except:
        Exception(usage)
