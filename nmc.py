import numpy as np


def nmc(model, ref, dt, t1, alpha=1):
    """
    Use NMC method to estimate background error covariance.
    
    Parameter:
    ---------
    model: callable function
        It should include two parameters: the first one is initial condition, and 
        the second one is time. This model should return the computing result at 
        each time stamp and shape=(Ndim, Nt), where `Ndim` is the dimension of
        model and Nt is the number of time stamp.
        For example:
            x0 = np.array([1, 1, 1])
            ts = np.arange(0, 10, 0.01)  # shape=(1000,)
            model_result = model(x0, ts)
            print(model_result.shape)    # (3, 1000)
            print(model_result[:,0])     # array([1, 1, 1])
    ref: ndarray, shape = (ndim, nt)
        Reference for estimating forecast error. Its shape should be (ndim, nt),
        where ndim is the dimension and nt is the total time stamp number.
    dt: scalar
        Time interval for model integrate.
    t1: scalar
        To determine the time length of forecasting comparison. The i'th forecast
        would start from time T, integrate to T+2*t1, and compare to the (i+1)'th
        forecast result at time T+2*t1 which start from time T+t1.
    alpha: scalar
        Adjustment amount for the final result. The estimated background error
        background covariance would multiply alpha before return.
        
    Return:
    ------
    Estimate background error covariance.
    """
    ndim, length = ref.shape
    x0 = ref[:,0]
    t1_nstep = int(t1 / dt)
    num = length // t1_nstep - 1
    
    result = np.zeros((num, ndim, 2*t1_nstep))
    for i in range(num):
        idx = i * t1_nstep
        scale = ref[:,idx].mean()
        x0 = ref[:,idx] + np.random.randn(ndim) * scale
        ts = np.arange(0, 2*t1_nstep*dt, dt)  # shape=(2*t1_step,)
        x_forecast = model(x0, ts)
        result[i,:] = x_forecast
    
    xfcst_short = []
    xfcst_long = []
    for xfcst in result:
        # x_f.shape = (ndim, 2*t1_nstep)
        xfcst_short.append(xfcst[:,t1_nstep-1])
        xfcst_long.append(xfcst[:,2*t1_nstep-1])
    
    Pb = np.zeros((ndim, ndim))
    for xshort, xlong in zip(xfcst_short[1:], xfcst_long[:-1]):
        # xshort.shape = xlong.shape = (ndim,)
        # convert to column vector
        xshort = xshort[:,np.newaxis]
        xlong = xlong[:,np.newaxis]
        Pb = Pb + (xlong-xshort) @ (xlong-xshort).T
    return alpha * Pb / (num-1)


def cycle_nmc(da, x0):
    pass