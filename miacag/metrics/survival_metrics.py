
import warnings
import numpy as np
import pandas as pd
import numba
import scipy
from miacag.model_utils.predict_utils import predict_surv_df
from miacag.plots.plot_utils import compute_mean_lower_upper
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
import os

def idx_at_times(index_surv, times, steps='pre', assert_sorted=True):
    """Gives index of `index_surv` corresponding to `time`, i.e. 
    `index_surv[idx_at_times(index_surv, times)]` give the values of `index_surv`
    closet to `times`.
    
    Arguments:
        index_surv {np.array} -- Durations of survival estimates
        times {np.array} -- Values one want to match to `index_surv`
    
    Keyword Arguments:
        steps {str} -- Round 'pre' (closest value higher) or 'post'
          (closest value lower) (default: {'pre'})
        assert_sorted {bool} -- Assert that index_surv is monotone (default: {True})
    
    Returns:
        np.array -- Index of `index_surv` that is closest to `times`
    """
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == 'pre':
        idx = np.searchsorted(index_surv, times)
    elif steps == 'post':
        idx = np.searchsorted(index_surv, times, side='right') - 1
    return idx.clip(0, len(index_surv)-1)

@numba.njit(parallel=True)
def _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                     idx_tt_censor, scores, weights, n_times, n_indiv, max_weight):
    def _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores, weights, n_indiv, max_weight):
        min_g = 1./max_weight
        for i in range(n_indiv):
            tt = durations[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            g_ts = censor_surv[idx_ts_censor_i, i]
            g_tt = censor_surv[idx_tt_censor[i], i]
            g_ts = max(g_ts, min_g)
            g_tt = max(g_tt, min_g)
            score, w = func(ts, tt, s, g_ts, g_tt, d)
            #w = min(w, max_weight)
            scores[i] = score * w
            weights[i] = w

    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        idx_ts_censor_i = idx_ts_censor[i]
        scores_i = scores[i]
        weights_i = weights[i]
        _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores_i, weights_i, n_indiv, max_weight)

def _inverse_censoring_weighted_metric(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor, max_weight=np.inf,
               reduce=True, steps_surv='post', steps_censor='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is type(censor_surv) is
                type(index_surv) is type(index_censor) is np.ndarray), 'Need all input to be np.ndarrays'
        n_times = len(time_grid)
        n_indiv = len(durations)
        scores = np.zeros((n_times, n_indiv))
        weights = np.zeros((n_times, n_indiv))
        idx_ts_surv = idx_at_times(index_surv, time_grid, steps_surv, assert_sorted=True)
        idx_ts_censor = idx_at_times(index_censor, time_grid, steps_censor, assert_sorted=True)
        idx_tt_censor = idx_at_times(index_censor, durations, 'pre', assert_sorted=True)
        if steps_censor == 'post':
            idx_tt_censor  = (idx_tt_censor - 1).clip(0)
            #  This ensures that we get G(tt-)
        _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                         idx_tt_censor, scores, weights, n_times, n_indiv, max_weight)
        if reduce is True:
            return np.sum(scores, axis=1) / np.sum(weights, axis=1)
        return scores, weights
    return metric


@numba.njit()
def _brier_score(ts, tt, s, g_ts, g_tt, d):
    if (tt <= ts) and d == 1:
        return np.power(s, 2), 1./g_tt
    if tt > ts:
        return np.power(1 - s, 2), 1./g_ts
    return 0., 0.

@numba.njit()
def _binomial_log_likelihood(ts, tt, s, g_ts, g_tt, d, eps=1e-7):
    s = eps if s < eps else s
    s = (1-eps) if s > (1 - eps) else s
    if (tt <= ts) and d == 1:
        return np.log(1 - s), 1./g_tt
    if tt > ts:
        return np.log(s), 1./g_ts
    return 0., 0.

brier_score = _inverse_censoring_weighted_metric(_brier_score)
binomial_log_likelihood = _inverse_censoring_weighted_metric(_binomial_log_likelihood)

def _integrated_inverce_censoring_weighed_metric(func):
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
               max_weight=np.inf, steps_surv='post', steps_censor='post'):
        scores = func(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
                      max_weight, True, steps_surv, steps_censor)
        integral = scipy.integrate.simps(scores, time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric

integrated_brier_score = _integrated_inverce_censoring_weighed_metric(brier_score)
integrated_binomial_log_likelihood = _integrated_inverce_censoring_weighed_metric(binomial_log_likelihood)


@numba.njit
def _group_loop(n, surv_idx, durations, events, di, ni):
    idx = 0
    for i in range(n):
        idx += durations[i] != surv_idx[idx]
        di[idx] += events[i]
        ni[idx] += 1
    return di, ni


def kaplan_meier(durations, events, start_duration=0):
    """A very simple Kaplan-Meier fitter. For a more complete implementation
    see `lifelines`.
    
    Arguments:
        durations {np.array} -- durations array
        events {np.arrray} -- events array 0/1
    
    Keyword Arguments:
        start_duration {int} -- Time start as `start_duration`. (default: {0})
    
    Returns:
        pd.Series -- Kaplan-Meier estimates.
    """
    n = len(durations)
    assert n == len(events)
    if start_duration > durations.min():
        warnings.warn(f"start_duration {start_duration} is larger than minimum duration {durations.min()}. "
            "If intentional, consider changing start_duration when calling kaplan_meier.")
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    surv_idx = np.unique(durations)
    ni = np.zeros(len(surv_idx), dtype='int')
    di = np.zeros_like(ni)
    di, ni = _group_loop(n, surv_idx, durations, events, di, ni)
    ni = n - ni.cumsum()
    ni[1:] = ni[:-1]
    ni[0] = n
    survive = 1 - di / ni
    zero_survive = survive == 0
    if zero_survive.any():
        i = np.argmax(zero_survive)
        surv = np.zeros_like(survive)
        surv[:i] = np.exp(np.log(survive[:i]).cumsum())
        # surv[i:] = surv[i-1]
        surv[i:] = 0.
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration < surv_idx.min():
        tmp = np.ones(len(surv)+ 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx)+ 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv


@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))

@numba.jit(nopython=True)
def _is_comparable_antolini(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

@numba.jit(nopython=True)
def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j: 
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True)
def _is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable_antolini(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True, parallel=True)
def _sum_comparable(t, d, is_comparable_func):
    n = t.shape[0]
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += is_comparable_func(t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant(s, t, d):
    n = len(t)
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += _is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc(s, t, d, s_idx, is_concordant_func):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                count += is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count


def concordance_td(durations, events, surv, surv_idx, method='adj_antolini'):
    """Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
    If 'method' is 'adj_antolini' (default) we have made a small modifications
    for ties in predictions and event times.
    We have followed step 3. in Sec 5.1. in Random Survial Forests paper, except for the last
    point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
    See '_is_concordant'.

    Arguments:
        durations {np.array[n]} -- Event times (or censoring times.)
        events {np.array[n]} -- Event indicators (0 is censoring).
        surv {np.array[n_times, n]} -- Survival function (each row is a duraratoin, and each col
            is an individual).
        surv_idx {np.array[n_test]} -- Mapping of survival_func s.t. 'surv_idx[i]' gives index in
            'surv' corresponding to the event time of individual 'i'.

    Keyword Arguments:
        method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

    Returns:
        float -- Time dependent concordance index.
    """
    if np.isfortran(surv):
        surv = np.array(surv, order='C')
    assert durations.shape[0] == surv.shape[1] == surv_idx.shape[0] == events.shape[0]
    assert type(durations) is type(events) is type(surv) is type(surv_idx) is np.ndarray
    if events.dtype in ('float', 'float32'):
        events = events.astype('int32')
    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                (_sum_comparable(durations, events, is_comparable) + 1e-8))
    elif method == 'antolini':
        is_concordant = _is_concordant_antolini
        is_comparable = _is_comparable_antolini
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                _sum_comparable(durations, events, is_comparable))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")


class EvalSurv:
    """Class for evaluating predictions.
    
    Arguments:
        surv {pd.DataFrame} -- Survival predictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, EvalSurv} -- Censoring distribution.
            If provided data frame (survival function for censoring) or EvalSurv object,
            this will be used. 
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
        censor_durations {np.array}: -- Administrative censoring times. (default: {None})
        steps {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. For a visualization see `help(EvalSurv.steps)`. (default: {'post'})
    """
    def __init__(self, surv, durations, events, censor_surv=None, censor_durations=None, steps='post'):
        assert (type(durations) == type(events) == np.ndarray), 'Need `durations` and `events` to be arrays'
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        self.censor_durations = censor_durations
        self.steps = steps
       # assert pd.Series(self.index_surv).is_monotonic

    @property
    def censor_surv(self):
        """Estimated survival for censorings. 
        Also an EvalSurv object.
        """
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, EvalSurv):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {censor_surv}. Use e.g. 'km'")
        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        """How to handle predictions that are between two indexes in `index_surv`.

        For a visualization, run the following:
            ev = EvalSurv(pd.DataFrame(np.linspace(1, 0, 7)), np.empty(7), np.ones(7), steps='pre')
            ax = ev[0].plot_surv()
            ev.steps = 'post'
            ev[0].plot_surv(ax=ax, style='--')
            ax.legend(['pre', 'post'])
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        """Add censoring estimates so one can use inverse censoring weighting.
        `censor_surv` are the survival estimates trained on (durations, 1-events),
        
        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.

    Keyword Arguments:
        round {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. If `None` use `self.steps` (default: {None})
        """
        if not isinstance(censor_surv, EvalSurv):
            censor_surv = self._constructor(censor_surv, self.durations, 1-self.events, None,
                                            steps=steps)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        """Add censoring estimates obtained by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        km = kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def censor_durations(self):
        """Administrative censoring times."""
        return self._censor_durations
    
    @censor_durations.setter
    def censor_durations(self, val):
        if val is not None:
            assert (self.durations[self.events == 0] == val[self.events == 0]).all(),\
                'Censored observations need same `durations` and `censor_durations`'
            assert (self.durations[self.events == 1] <= val[self.events == 1]).all(),\
                '`durations` cannot be larger than `censor_durations`'
            if (self.durations == val).all():
                warnings.warn("`censor_durations` are equal to `durations`." +
                              " `censor_durations` are likely wrong!")
            self._censor_durations = val
        else:
            self._censor_durations = val

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice) :
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(surv, durations, events, None, steps=self.steps)
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        """Plot survival estimates. 
        kwargs are passed to `self.surv.plot`.
        """
        if len(self.durations) > 50:
            raise RuntimeError("We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return idx_at_times(self.index_surv, times, self.steps)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    # def prob_alive(self, time_grid):
    #     return self.surv_at_times(time_grid).values

    def concordance_td(self, method='adj_antolini'):
        """Time dependent concorance index from
        Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A time-dependent discrimination
        index for survival data. Statistics in Medicine 24:3927–3944.

        If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
        If 'method' is 'adj_antolini' (default) we have made a small modifications
        for ties in predictions and event times.
        We have followed step 3. in Sec 5.1. in Random Survival Forests paper, except for the last
        point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
        See 'metrics._is_concordant'.

        Keyword Arguments:
            method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

        Returns:
            float -- Time dependent concordance index.
        """
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    def brier_score(self, time_grid, max_weight=np.inf):
        """Brier score weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute Brier score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = brier_score(time_grid, self.durations, self.events, self.surv.values,
                              self.censor_surv.surv.values, self.index_surv,
                              self.censor_surv.index_surv, max_weight, True, self.steps,
                              self.censor_surv.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        """Integrated Brier score weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `brier_score(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, self.steps,
                                           self.censor_surv.steps)



def compute_concordance(df_target, base_haz, bch, config_task):
    survival_etimates = predict_surv_df(df_target, base_haz, bch, config_task)
    survival_etimates = survival_etimates.reset_index()
    # merge two pandas dataframes
    survival_etimates = pd.merge(survival_etimates, df_target, on=config_task['labels_names'][0], how='inner')
    #df_new = pd.DataFrame({i: survival_etimates.loc[i, i] for i in survival_etimates.index}, index=[0])
    surv_preds_observed = pd.DataFrame({i: survival_etimates.loc[i, i] for i in survival_etimates.index}, index=[0])
    #auc_1_year = get_roc_auc_ytest_1_year_surv(survival_etimates, config_task, len(surv_preds_observed.columns))
    if config_task['debugging']:
        try:
            c_index = concordance_index_censored(df_target['event'].values.astype(bool), df_target[config_task['labels_names'][0]].values, np.squeeze(surv_preds_observed.values))
        except:
            c_index = [0.5]
    else:
        c_index = concordance_index_censored(df_target['event'].values.astype(bool), df_target[config_task['labels_names'][0]].values, np.squeeze(surv_preds_observed.values))
    c_index = c_index[0]


    return c_index, None

def compute_brier(df_target, base_haz, bch, config_task):
    survival_etimates = predict_surv_df(df_target, base_haz, bch, config_task)

    ev = EvalSurv(survival_etimates, df_target[config_task['labels_names'][0]].to_numpy(), df_target['event'].to_numpy(), censor_surv='km')
    #concordance = ev.concordance_td()


    time_grid = np.linspace(df_target[config_task['labels_names'][0]].min(), df_target[config_task['labels_names'][0]].max(), 100)
    brier_scores = ev.brier_score(time_grid)
    ibs = ev.integrated_brier_score(time_grid)
    return ibs, brier_scores

def compute_bootstrapped_scores(df_target, base_haz, bch, config_task, flag='concordance'):
    n_bootstraps = 1000
    rng_seed = 42 # control reproducibility

    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range (n_bootstraps): 
        #bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(df_target), len(df_target))
       # if flag == 'concordance':
        if len(np.unique(df_target['event'].iloc[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # # to be defined: reject the sample
            continue
        if flag == 'concordance':
          #  if len(np.unique(df_target['event'].iloc[indices])) < 2:
                
            scores, _ = compute_concordance(df_target.iloc[indices], base_haz, bch, config_task)
        else:
            scores, _ = compute_brier(df_target.iloc[indices], base_haz, bch, config_task)
        bootstrapped_scores.append(scores)

    


    return bootstrapped_scores


def confidences_upper_lower_survival(df_target, base_haz, bch, config_task):
    compute_bootstrapped_scores_conc = compute_bootstrapped_scores(
        df_target, base_haz, bch, config_task, flag='concordance')
    compute_bootstrapped_scores_ibs = compute_bootstrapped_scores(
        df_target, base_haz, bch, config_task, flag='brier')
    _, brier_scores = compute_brier(df_target, base_haz, bch, config_task)
    mean_conc, upper_conc, lower_conc = compute_mean_lower_upper(compute_bootstrapped_scores_conc)
    mean_brier, upper_brier, lower_brier = compute_mean_lower_upper(compute_bootstrapped_scores_ibs)
    variable_dict = {
        k: v for k, v in locals().items() if k in [
            "mean_conc", "upper_conc", "lower_conc",
            "mean_brier", "upper_brier", "lower_brier", "brier_scores"]}
    return variable_dict

def plot_auc_surv(variable_dict_1_year, variable_dict_5_year, output_plots):

    #"mean_auc", "upper_auc", "lower_auc",
       #     "yprobs", "ytest"]
   # plt.figure()
    fig = plt.figure(figsize=(16,14))
    count = 0
    legende = ['1 year survival', '5 year survival']
    for variable_dict in [variable_dict_1_year, variable_dict_5_year]:
        fpr, tpr, _ = roc_curve(variable_dict["ytest"],  variable_dict["yprobs"])
        plt.plot(fpr, 
            tpr, 
            label="{}, AUC={:.3f} ({:.3f}-{:-3f})".format(
                legende[count],
                variable_dict["mean_auc"], variable_dict["lower_auc"], variable_dict_1_year["upper_auc"]))
        count += 1
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis for survival', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':10}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, 'survival_1_5_year' + '.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, 'survival_1_5_year'  + '.pdf'), dpi=100,
                bbox_inches='tight')
