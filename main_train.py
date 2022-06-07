import sys
from os import listdir
from random import sample
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, fbeta_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
from joblib import dump

from utils.feature import feature_est1, feature_est2
from mfa_wtmm import mfa_wtmm_2d
from wavelet_2d import select_wavelet
from functions.function_image import FunctionImage



def train_model(imageLocation, saveLocation, wavelet, do_metrics = True):
    print(wavelet, do_metrics)
    normal_chestRx = [ imageLocation+'/Normal/'+file for file in listdir(imageLocation+'/Normal')]    
    nodule_chestRx = [ imageLocation+'/Nodule/'+file for file in listdir(imageLocation+'/Nodule')]
    
    sample_normal = sample(normal_chestRx, 47)
    sample_nodule = sample(nodule_chestRx, 77)

    Y = [0] * len(sample_normal) + [1] * len(sample_nodule)
    X = list()
    for file in sample_normal+sample_nodule:
        print(f'train:  {file}')

        func = FunctionImage(file)
        func.reduce_size_toHalf(2)
        holder, dh = mfa_wtmm_2d(0, func.columns-1, 0, func.rows-1, 0, func.eval_range, wavelet, 3, 5, 10, 0, 2, 5)
        
        X.append(feature_est2(holder, dh))

    clf =  make_pipeline(StandardScaler(), SVC(cache_size=1024))
    clf.fit(X,Y)

    dump(clf, saveLocation)


    if do_metrics:
        metrics_sample_normal = [chestRx for chestRx in normal_chestRx if chestRx not in sample_normal]
        metrics_sample_nodule = [chestRx for chestRx in nodule_chestRx if chestRx not in sample_nodule]

        Y_real = [0] * len(metrics_sample_normal) + [1] * len(metrics_sample_nodule)
        X = list()
        for file in metrics_sample_normal+metrics_sample_nodule:
            print(f'test:  {file}')
           
            func = FunctionImage(file)
            func.reduce_size_toHalf(2)
            holder, dh = mfa_wtmm_2d(0, func.columns-1, 0, func.rows-1, 0, func.eval_range, wavelet, 3, 5, 10, 0, 2, 5)
           
            X.append(feature_est2(holder, dh))

        Y_predict = clf.predict(X)

        precision = precision_score(Y_real, Y_predict)
        print(f'precision: {precision}')

        recall = recall_score(Y_real, Y_predict)
        print(f'recall: {recall}')
        
        accuracy = accuracy_score(Y_real, Y_predict)
        print(f'accuracy: {accuracy}')
        
        roc_auc = roc_auc_score(Y_real, Y_predict)
        print(f'roc_auc: {roc_auc}')
        
        matthews = matthews_corrcoef(Y_real, Y_predict)
        print(f'matthews_corrcoef:  {matthews}')

        youden = balanced_accuracy_score(Y_real, Y_predict)
        print(f'youden:  {youden}')

        cohen_kappa = cohen_kappa_score(Y_real, Y_predict)
        print(f'cohen kappa:  {cohen_kappa}')

        fbeta_2 = fbeta_score(Y_real, Y_predict, beta=2) 
        print(f'fbeta: {fbeta_2}')

        fbeta_05 = fbeta_score(Y_real, Y_predict, beta=0.5) 
        print(f'fbeta: {fbeta_05}')

        fbeta_1 = fbeta_score(Y_real, Y_predict, beta=1) 
        print(f'fbeta: {fbeta_1}')



if __name__=='__main__':
    usage = '''  '''
    
    try:
        if sys.argv[-1] == '--metrics':
            doMetrics = True
        else:
            doMetrics = False
    except:
        Exception(usage)

    train_model(sys.argv[1], sys.argv[2], select_wavelet(int(sys.argv[3])), do_metrics=doMetrics)