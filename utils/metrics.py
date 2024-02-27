import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(pred, true):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred)))

def MdAPE(pred, true):
    return np.median(np.abs((pred - true) / true))

def ADJ_MAPE(pred, true, cutoff = 10):
    true_cut = true[true > cutoff]
    pred_cut = pred[true > cutoff]

    if len(true_cut) > 0:
        return np.mean(np.abs((pred_cut - true_cut) / true_cut))
    else:
        return np.nan

def MASE(y,t) :
    n = len(y)
    d = np.abs(np.diff(y)).sum() / (n-1)
    error = np.abs(y-t)

    return error.mean(axis=None)/d

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    corr = Corr(pred, true)

    return mae,mse,rmse,mape,mspe,corr
