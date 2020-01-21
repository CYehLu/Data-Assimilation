import sys
import warnings
import numpy as np
from scipy.optimize import minimize


def progressbar(i, tol, prefix='', size=60, file=sys.stdout):
    def show(j):
        x = int(size*j/tol)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, tol))
        file.flush()
    show(i)
    
    if i == tol:
        file.write("\n")
        file.flush()


class DAbase:
    def __init__(self, model, dt, store_history=False):
        self._isstore = store_history
        self._params = {'alpha': 0, 'inflat': 1}
        self.model = model
        self.dt = dt
        self.X_ini = None
        
    def set_params(self, param_list, **kwargs):
        for key, value in kwargs.items():
            if key in param_list:
                self._params[key] = kwargs.get(key)
            else:
                raise ValueError(f'Invalid parameter: {key}')
        
    def _check_params(self, param_list):
        missing_params = []
        for var in param_list:
            if self._params.get(var) is None:
                missing_params.append(var)
        return missing_params


class ExtendedKF(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
            'H', 
            'M', 
            'alpha', 
            'inflat'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
        if self._params.get('H') is None:
            H = np.eye(self._params.get('R').shape[0])
            self._params['H'] = H
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, Pb, R, H_func=None, H=None):
        if H_func is None:
            K = Pb @ np.linalg.inv(Pb + R)
            xa = xb + K @ (yo - xb)
            Pa = (np.eye(len(xb)) - K) @ Pb
        else:
            K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
            xa = xb + K @ (yo - H_func(xb))
            Pa = (np.eye(len(xb)) - K @ H) @ Pb
        return (xa, Pa)
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini']
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        H = self._params['H']
        alpha = self._params['alpha']
        inflat = self._params['inflat']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.linspace(t_start, (cycle_len-1)*dt, cycle_len)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa, Pa = self._analysis(xb, obs[:,[nc]], Pb, R, H_func, H)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            M = self._params['M'](xb.ravel())
            Pb = alpha * Pb + (1-alpha) * M @ Pa @ M.T
            Pb *= inflat
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.linspace(t_start, t_start+(cycle_len-1)*dt, cycle_len)
            
        self.background = background
        self.analysis = analysis
        
        
class OI(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
            'H'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
        if self._params.get('H') is None:
            H = np.eye(self._params.get('R').shape[0])
            self._params['H'] = H
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, Pb, R, H_func=None, H=None):
        if H_func is None:
            K = Pb @ np.linalg.inv(Pb + R)
            xa = xb + K @ (yo - xb)
            Pa = (np.eye(len(xb)) - K) @ Pb
        else:
            K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
            xa = xb + K @ (yo - H_func(xb))
            Pa = (np.eye(len(xb)) - K @ H) @ Pb
        return (xa, Pa)
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini']
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        H = self._params['H']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.linspace(t_start, (cycle_len-1)*dt, cycle_len)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa, _ = self._analysis(xb, obs[:,[nc]], Pb, R, H_func, H)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.linspace(t_start, t_start+(cycle_len-1)*dt, cycle_len)
            
        self.background = background
        self.analysis = analysis
        

class M3DVar(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ini', 
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _3dvar_costfunction(self, x, xb, yo, invPb, invR, H_func=None, H=None):
        """
        x and xb is 1d array with shape (n,), the other is 2d matrix
        """
        x = x[:,np.newaxis]
        xb = xb[:,np.newaxis]

        if H_func is None:
            innovation = yo - x
        else:
            innovation = yo - H_func(x) 

        return 0.5 * (xb-x).T @ invPb @ (xb-x) + 0.5 * innovation.T @ invR @ innovation

    def _analysis(self, xb, yo, Pb, R, H_func=None):    
        if H_func is None:
            innovation = yo - xb
        else:
            innovation = yo - H_func(xb)

        invPb = np.linalg.inv(Pb)
        invR = np.linalg.inv(R)
        cost_func = lambda x: self._3dvar_costfunction(x, xb.ravel(), yo, invPb, invR, H_func)

        return minimize(cost_func, xb.ravel(), method='BFGS').x
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini']
        obs = self._params['obs']
        Pb = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        
        background = np.zeros((xb.size, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.linspace(t_start, (cycle_len-1)*dt, cycle_len)
        
        for nc in range(cycle_num):
            # analysis and forecast
            xa = self._analysis(xb, obs[:,[nc]], Pb, R, H_func)
            x_forecast = model(xa.ravel(), ts)
            
            # store result of background and analysis field
            idx1 = nc*cycle_len
            idx2 = (nc+1)*cycle_len
            analysis[:,idx1:idx2] = x_forecast
            background[:,[idx1]] = xb
            background[:,(idx1+1):idx2] = x_forecast[:,1:]
            
            # for next cycle
            xb = x_forecast[:,[-1]]
            t_start = int(ts[-1] + dt)
            ts = np.linspace(t_start, t_start+(cycle_len-1)*dt, cycle_len)
            
        self.background = background
        self.analysis = analysis
        
        
class EnKF(DAbase):
    def __init__(self, model, dt, store_history=False):
        super().__init__(model, dt, store_history)
        self._param_list = [
            'X_ens_ini', 
            'obs', 
            'obs_interv', 
            'R', 
            'H_func', 
            'alpha', 
            'inflat'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _analysis(self, xb, yo, R, H_func=None):
        """xb.shape = (n_dim, n_ens)"""
        if H_func is None:
            H_func = lambda arr: arr
        
        N_ens = xb.shape[1]
        yo_ens = np.random.multivariate_normal(yo.ravel(), R, size=N_ens).T  # (ndim_yo, N_ens)
        xb_mean = xb.mean(axis=1)[:,np.newaxis]  # (ndim_xb, 1)
        
        xa_ens = np.zeros((xb.shape[0], N_ens))
        for iens in range(N_ens):
            xb_mean = xb.mean(axis=1)[:,np.newaxis]
            Xb_perturb = xb - xb_mean
            HXb_perturb = H_func(Xb_perturb) - H_func(Xb_perturb).mean(axis=1)[:,np.newaxis]
            
            PfH_T = Xb_perturb @ HXb_perturb.T / (N_ens-1)
            HPfH_T = HXb_perturb @ HXb_perturb.T / (N_ens-1)
            K = PfH_T @ np.linalg.inv(HPfH_T + R)
            xa_ens[:,[iens]] = xb[:,[iens]] + K @ (yo_ens[:,[iens]] - H_func(xb[:,[iens]]))
            
        return xa_ens
    
    def cycle(self):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ens_ini'].copy()
        obs = self._params['obs']
        R = self._params['R']
        H_func = self._params['H_func']
        alpha = self._params['alpha']
        inflat = self._params['inflat']
        
        ndim, N_ens = xb.shape
        background = np.zeros((N_ens, ndim, cycle_len*cycle_num))
        analysis = np.zeros_like(background)
        
        t_start = 0
        ts = np.linspace(t_start, (cycle_len-1)*dt, cycle_len)
        
        for nc in range(cycle_num):
            # analysis
            xa = self._analysis(xb, obs[:,[nc]], R, H_func)
            
            # inflat
            xa_perturb = xa - xa.mean(axis=1)[:,np.newaxis]
            xa_perturb *= inflat
            xa = xa.mean(axis=1)[:,np.newaxis] + xa_perturb
            
            # ensemble forecast
            for iens in range(N_ens):
                x_forecast = model(xa[:,iens], ts)   # (ndim, ts.size)
                
                idx1 = nc*cycle_len
                idx2 = (nc+1)*cycle_len
                analysis[iens,:,idx1:idx2] = x_forecast
                background[iens,:,[idx1]] = xb[:,iens]
                background[iens,:,(idx1+1):idx2] = x_forecast[:,1:]
                
                # xb for next cycle
                xb[:,iens] = x_forecast[:,-1]
                
            # for next cycle
            t_start = int(ts[-1] + dt)
            ts = np.linspace(t_start, t_start+(cycle_len-1)*dt, cycle_len)
            
        self.background = background
        self.analysis = analysis
        

class M4DVar(DAbase):
    def __init__(self, model, dt):
        super().__init__(model, dt)
        self._param_list = [
            'X_ini', 
            'obs', 
            'window_num',
            'window_len', 
            'forecast_len',
            'Pb', 
            'R', 
            'H_func', 
            'H', 
            'M', 
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
        if self._params.get('H') is None:
            H = np.eye(self._params.get('R').shape[0])
            self._params['H'] = H
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def _find_window_range(self):
        isobs = np.any(self._params['obs'] != 0, axis=0) + 0
        win_start = []
        win_end = []
        for idx, iso in enumerate(isobs):
            if idx == 0 and isobs[idx] == 1:
                # start window
                win_start.append(idx)
                continue
            if idx == len(isobs)-1:
                if isobs[idx]:
                    # end window
                    win_end.append(idx)
                    continue
                else:
                    continue

            d = isobs[idx+1] - isobs[idx-1]
            if d == 1 and isobs[idx] == 1:
                # start window
                win_start.append(idx)
            elif d == -1 and isobs[idx] == 1:
                # end window
                win_end.append(idx)

        win_range = list(zip(win_start, win_end))
        self._win_range = win_range
        
    def _4dvar_costfunction(self, x, xb, xtrac, yo, Pb, R, H_func=None):
        """
        x, xb: (ndim, 1)
        x_trac: (ndim, window)
        Pb: (ndim, ndim)
        yo: (N, window)
        R: (window, N, N)
        model, H_func: (window,)
        """
        assim_window_len = yo.shape[1]

        if H_func is None:
            H_func = [lambda arr: arr for _ in range(assim_window_len)]

        part1 = (xb-x).T @ np.linalg.inv(Pb) @ (xb-x)
        part2 = np.zeros_like(part1)
        for j in range(assim_window_len):
            yj = yo[:,[j]]
            Rj = R[j,:,:]
            H_j = H_func[j]
            xtracj = xtrac[:,[j]]
            part2 += (yj-H_j(xtracj)).T @ np.linalg.inv(Rj) @ (yj-H_j(xtracj))
        return 0.5 * (part1 + part2)

    def _gradient_4dvar_costfunction(self, x, xb, xtrac, yo, Pb, R, M, H_func=None, H=None):
        """
        x, xb: (ndim, 1)
        xtrac: (ndim, window)
        Pb: (ndim, ndim)
        yo: (N, window)
        R: (window, N, N)
        H_func: (window,)
        M: (window, ndim, ndim)
        H: (window, N, ndim)
        """
        No, assim_window_len = yo.shape

        if H_func is None:
            H_func = [lambda arr: arr for _ in range(assim_window_len)]
            H = np.concatenate([np.eye(No)[np.newaxis] for i in range(assim_window_len)])

        part1 = np.linalg.inv(Pb) @ (x-xb)
        part2 = np.zeros_like(part1)
        for j in range(assim_window_len):
            yj = yo[:,[j]]
            Mj = M[j,:,:]
            Hj = H[j,:,:]
            Rj = R[j,:,:]
            H_fj = H_func[j]
            xtracj = xtrac[:,[j]]
            part2 += Mj.T @ Hj.T @ np.linalg.inv(Rj) @ (yj - H_fj(xtracj))
        return part1 - part2
    
    def _analysis(self, xb, obs, Pb, R, ts, model, M, H_func, H, win_len, r=0.01, maxiter=1000, epsilon=0.0001):
        """find the x that minimizes the cost function"""
        x = xb.copy()
        
        # prepare H_func and H
        if not isinstance(H_func, list) and callable(H_func):
            H_func = [H_func for _ in range(win_len)]
        if isinstance(H, np.ndarray) and H.ndim == 2:
            H = np.stack([H for _ in range(win_len)])
        
        # gradient descent
        for _ in range(maxiter):
            # calculate trajectory
            x_forecast = model(x.ravel(), ts)
            
            # prepare tangent linear model of every point on the trajectory
            Ms = []
            m = np.eye(3)
            for i in range(win_len):
                m = M(x_forecast[:,i]) @ m
                Ms.append(m)
            Ms = np.stack(Ms)
            
            # find the gradient and perform gradient descent
            gradient = self._gradient_4dvar_costfunction(x, xb, x_forecast, obs, Pb, R, Ms, H_func, H)
            x_new = x - r * gradient
            
            # stop criteria
            if np.linalg.norm(gradient) <= epsilon:
                x = x_new   # x is minize result
                break
            else:
                x = x_new
                
        if _ == maxiter - 1:
            warnings.warn('Iteration of gradient did not converge.')
            
        return x
    
    def cycle(self, r=0.01, maxiter=1000, epsilon=0.0001, showbar=True):
        self._check_params()
        self._find_window_range()
        
        xb = self._params['X_ini'].copy()
        dt = self.dt
        Pb = self._params['Pb']
        R = self._params['R']
        X_obs = self._params['obs']
        model = self.model
        M = self._params['M']
        H_func = self._params['H_func']
        H = self._params['H']
        
        analysis = np.zeros((xb.size, X_obs.shape[1]))
        background = np.zeros_like(analysis)

        for iwin, (start_win, end_win) in enumerate(self._win_range):
            if showbar:
                progressbar(iwin+1, len(self._win_range), 'Assimilation: ')

            ### assimilation stage
            win_len = end_win - start_win + 1
            ts = np.linspace(0, (win_len-1)*dt, win_len)
            x = xb.copy()
            
            if R.ndim == 2:
                R = np.concatenate([R[np.newaxis,:,:] for i in range(win_len)])

            # find observations in the assimilation window
            obs = X_obs[:,start_win:end_win+1]

            # gradient descent to minimize cost function
            x = self._analysis(
                xb, obs, Pb, R, ts, model, M, H_func, H, win_len, 
                r, maxiter, epsilon
            )

            # analysis trajectory and analysis point (at the end of window)
            xa_trajectory = model(x.ravel(), ts)
            xa = xa_trajectory[:,[-1]]
            analysis[:, start_win:end_win+1] = xa_trajectory
            
            ### forecast stage
            if iwin != len(self._win_range)-1:
                forecast_start = end_win + 1
                forecast_end = self._win_range[iwin+1][0] - 1
                forecast_len = forecast_end - forecast_start + 1
                ts = np.linspace(0, (forecast_len-1)*dt, forecast_len)
                xa_forecast = model(xa.ravel(), ts)
                xb = xa_forecast[:,[-1]]   # the background of next assimilation
                analysis[:, forecast_start:forecast_end+1] = xa_forecast
            else:
                # if it is the last assimilation window
                forecast_start = end_win + 1
                forecast_end = X_obs.shape[1] - 1
                forecast_len = forecast_end - forecast_start + 1
                ts = np.linspace(0, (forecast_len-1)*dt, forecast_len)
                xa_forecast = model(xa.ravel(), ts)
                analysis[:, forecast_start:forecast_end+1] = xa_forecast
               
        self.background = background
        self.analysis = analysis
        
        
class Hybrid3DEnVar(DAbase):
    def __init__(self, model, dt):
        super().__init__(model, dt)
        self._param_list = [
            'X_ini', 
            'X_ens_ini',
            'obs', 
            'obs_interv', 
            'Pb', 
            'R', 
            'H_func', 
            'H', 
            'alpha', 
            'inflat',
            'beta'
        ]
        
    def list_params(self):
        return self._param_list
        
    def set_params(self, **kwargs):
        super().set_params(self._param_list, **kwargs)
        
    def _check_params(self):
        if self._params.get('H_func') is None:
            H_func = lambda arr: arr
            self._params['H_func'] = H_func
            
        missing_params = super()._check_params(self._param_list)
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
    def cycle(self, recenter=True):
        self._check_params()
        
        model = self.model
        dt = self.dt
        cycle_len = self._params['obs_interv']
        cycle_num = self._params['obs'].shape[1]
        
        xb = self._params['X_ini'].copy()
        xb_ens = self._params['X_ens_ini'].copy()
        X_obs = self._params['obs']
        Pb_3dvar = self._params['Pb']
        R = self._params['R']
        H_func = self._params['H_func']
        alpha = self._params['alpha']
        inflat = self._params['inflat']
        beta = self._params['beta']
        
        ndim, N_ens = xb_ens.shape
        background_ens = np.zeros((N_ens, ndim, cycle_len*cycle_num))
        analysis_ens = np.zeros_like(background_ens)
        background_3dvar = np.zeros((ndim, cycle_len*cycle_num))
        analysis_3dvar = np.zeros_like(background_3dvar)
        
        t_start = 0 
        ts = np.linspace(t_start, (cycle_len-1)*dt, cycle_len)
        
        for nc in range(cycle_num):
            ### analysis
            obs = X_obs[:,[nc]]
            
            ens_mean = xb_ens.mean(axis=1)[:,np.newaxis]
            Pb_EnKF = (xb_ens - ens_mean) @ (xb_ens - ens_mean).T / (xb_ens.shape[1] - 1)
            Pb = (1-beta) * Pb_3dvar + beta * Pb_EnKF
            
            xa_3dvar = M3DVar(model, dt)._analysis(xb, obs, Pb, R, H_func)
            xa_3dvar = xa_3dvar[:,np.newaxis]
            xa_ens = EnKF(model, dt)._analysis(xb_ens, obs, R, H_func)  # (ndim, N_ens)
            
            # inflat
            xa_ens_pertb = xa_ens - xa_ens.mean(axis=1)[:,np.newaxis]
            xa_ens_pertb *= inflat
            xa_ens = xa_ens.mean(axis=1)[:,np.newaxis] + xa_ens_pertb
            
            if recenter:
                xa_ensmean = xa_ens.mean(axis=1)[:,np.newaxis]
                xa_ens_pertb = xa_ens - xa_ensmean
                xa_ens = xa_ens + xa_3dvar
                
            ### forecast
            ts = np.linspace(0, (cycle_len-1)*dt, cycle_len)
            x_forecast_3dvar = model(xa_3dvar.ravel(), ts)
            start_idx = nc * ts.size
            end_idx = start_idx + ts.size
            analysis_3dvar[:,start_idx:end_idx] = x_forecast_3dvar
            background_3dvar[:,start_idx:end_idx] = x_forecast_3dvar
            background_3dvar[:,start_idx] = xb.ravel()
            
            
            x_forecast_ens = np.zeros((N_ens, ndim, ts.size))
            for iens in range(N_ens):
                x_forecast_ens[iens,:,:] = model(xa_ens[:,iens], ts)
                analysis_ens[iens,:,start_idx:end_idx] = x_forecast_ens[iens,:,:]
                background_ens[iens,:,start_idx:end_idx] = x_forecast_ens[iens,:,:]
                background_ens[iens,:,start_idx] = xb_ens[:,iens]
                
            ### for next cycle
            xb = x_forecast_3dvar[:,[-1]]
            xb_ens = x_forecast_ens[:,:,-1].T
            
        self.analysis_3dvar = analysis_3dvar
        self.background_3dvar = background_3dvar
        self.analysis_ens = analysis_ens
        self.background_ens = background_ens
        
        
        