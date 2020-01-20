import os
import numpy as np
import pickle
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA


# global
pca_temp_file = 'PCA_temp.sav'
pca_down_factor = 100


def run_PCA(PCA_dir, pca_components, data):
    pca_model = PCA(pca_components)
    data_pca = pca_model.fit_transform(data)

    pickle.dump(pca_model, open(PCA_dir, 'wb'))   # save PCA model
    return data_pca


def vector_AR(maxlag, ts):
    var_model = VAR(ts)
    model_fit = var_model.fit(maxlags = maxlag, ic = 'aic', verbose = 1)     # fit on the given time series

    pred = model_fit.forecast(ts[-maxlag:], 1)     # given lastest (maxlag) observations to predict the next one
    return pred


def rolling_fit_pred(data_pca, pca_components, train_len, test_len, timestep):
    pred_len = test_len - timestep
    data_pca_pred = np.zeros((pred_len, pca_components))     # initiate
    for i in range(pred_len):
        ts = data_pca[i : i+train_len]     # get time series
        print('Processing time series #', i)
        data_pca_pred[i] = vector_AR(6, ts)
    return data_pca_pred


def run_VAR(out_dir, trainSet, testSet, timestep):
    # parameters
    PCA_dir = os.path.join(out_dir, pca_temp_file)
    H, W, C = trainSet.shape[1], trainSet.shape[2], trainSet.shape[3]
    train_len, test_len = trainSet.shape[0], testSet.shape[0]

    # reshape inputs
    train2D = trainSet.reshape((train_len, -1))     # (T, 1200)
    test2D = testSet.reshape((test_len, -1))

    # PCA
    data2D = np.concatenate([train2D, test2D], axis=0)
    pca_components = int(data2D.shape[-1] / pca_down_factor)
    dataPCA = run_PCA(PCA_dir, pca_components, data2D)

    dataPCA_pred = rolling_fit_pred(dataPCA, pca_components, train_len, test_len, timestep)
    pca_model = pickle.load(open(PCA_dir, 'rb'))
    data_pred = pca_model.inverse_transform(dataPCA_pred)

    y_pred = data_pred.reshape((-1, H, W, C))
    y_true = testSet[timestep:]

    return y_true, y_pred
