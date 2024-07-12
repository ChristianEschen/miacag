from miacag.utils.sql_utils import add_columns
from miacag.utils.sql_utils import getDataFromDatabase
import yaml
from miacag.utils.sql_utils import update_cols
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from miacag.models.modules import get_loss_names_groups
from miacag.dataloader.get_dataloader import get_data_from_loader, to_device, convert_numpy_to_torch
import os
import torch

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

def make_cuts(n_cuts, scheme, durations, events, min_=0., dtype='float64'):
    if scheme == 'equidistant':
        cuts = cuts_equidistant(durations.max(), n_cuts, min_, dtype)
    elif scheme == 'quantiles':
        cuts = cuts_quantiles(durations, events, n_cuts, min_, dtype)
    else:
        raise ValueError(f"Got invalid `scheme` {scheme}.")
    if (np.diff(cuts) == 0).any():
        raise ValueError("cuts are not unique.")
    return cuts

def _values_if_series(x):
    if type(x) is pd.Series:
        return x.values
    return x

def cuts_equidistant(max_, num, min_=0., dtype='float64'):
    return np.linspace(min_, max_, num, dtype=dtype)


def cuts_quantiles(durations, events, num, min_=0., dtype='float64'):
    """
    If min_ = None, we will use durations.min() for the first cut.
    """
    km = kaplan_meier(durations, events)
    surv_est, surv_durations = km.values, km.index.values
    s_cuts = np.linspace(km.values.min(), km.values.max(), num)
    cuts_idx = np.searchsorted(surv_est[::-1], s_cuts)[::-1]
    cuts = surv_durations[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if len(cuts) != num:
        warnings.warn(f"cuts are not unique, continue with {len(cuts)} cuts instead of {num}")
    cuts[0] = durations.min() if min_ is None else min_
    assert cuts[-1] == durations.max(), 'something wrong...'
    return cuts.astype(dtype)


def _is_monotonic_increasing(x):
    assert len(x.shape) == 1, 'Only works for 1d'
    return (x[1:] >= x[:-1]).all()

def bin_numerical(x, right_cuts, error_on_larger=False):
    """
    Discretize x into bins defined by right_cuts (needs to be sorted).
    If right_cuts = [1, 2], we have bins (-inf, 1], (1, 2], (2, inf).
    error_on_larger results in a ValueError if x contains larger
    values than right_cuts.
    
    Returns index of bins.
    To optaine values do righ_cuts[bin_numerica(x, right_cuts)].
    """
    assert _is_monotonic_increasing(right_cuts), 'Need `right_cuts` to be sorted.'
    bins = np.searchsorted(right_cuts, x, side='left')
    if bins.max() == right_cuts.size:
        if error_on_larger:
            raise ValueError('x contrains larger values than right_cuts.')
    return bins

def discretize(x, cuts, side='right', error_on_larger=False):
    """Discretize x to cuts.
    
    Arguments:
        x {np.array} -- Array of times.
        cuts {np.array} -- Sortet array of discrete times.
    
    Keyword Arguments:
        side {str} -- If we shold round down or up (left, right) (default: {'right'})
        error_on_larger {bool} -- If we shold return an error if we pass higher values
            than cuts (default: {False})
    
    Returns:
        np.array -- Discretized values.
    """
    if side not in ['right', 'left']:
        raise ValueError('side argument needs to be right or left.')
    bins = bin_numerical(x, cuts, error_on_larger)
    if side == 'right':
        cuts = np.concatenate((cuts, np.array([np.inf])))
        return cuts[bins]
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = - np.inf
    return vals

class _OnlyTransform:
    """Abstract class for sklearn preprocessing methods.
    Only implements fit and fit_transform.
    """
    def fit(self, *args):
        return self
    
    def transform(self, *args):
        raise NotImplementedError
    
    def fit_transform(self, *args):
        return self.fit(*args).transform(*args)

def duration_idx_map(duration):
    duration = np.unique(duration)
    duration = np.sort(duration)
    idx = np.arange(duration.shape[0])
    return {d: i for i, d in zip(idx, duration)}



class Duration2Idx(_OnlyTransform):
    def __init__(self, durations=None):
        self.durations = durations
        if durations is None:
            raise NotImplementedError()
        if self.durations is not None:
            self.duration_to_idx = self._make_map(self.durations)
    
    @staticmethod
    def _make_map(durations):
        return np.vectorize(duration_idx_map(durations).get)
    
    def transform(self, duration, y=None):
        if duration.dtype is not self.durations.dtype:
            raise ValueError('Need `time` to have same type as `self.durations`.')
        idx = self.duration_to_idx(duration)
        if np.isnan(idx).any():
            raise ValueError('Encountered `nans` in transformed indexes.')
        return idx


class DiscretizeUnknownC(_OnlyTransform):
    """Implementation of scheme 2.
    
    cuts should be [t0, t1, ..., t_m], where t_m is right sensored value.
    """
    def __init__(self, cuts, right_censor=False, censor_side='left'):
        self.cuts = cuts
        self.right_censor = right_censor
        self.censor_side = censor_side
    
    def transform(self, duration, event):
        dtype_event = event.dtype
        event = event.astype('bool')
        if self.right_censor:
            duration = duration.copy()
            censor = duration > self.cuts.max()
            duration[censor] = self.cuts.max()
            event[censor] = False
        if duration.max() > self.cuts.max():
            raise ValueError("`duration` contains larger values than cuts. Set `right_censor`=True to censor these")
        td = np.zeros_like(duration)
        c = event == False
        td[event] = discretize(duration[event], self.cuts, side='right', error_on_larger=True)
        if c.any():
            td[c] = discretize(duration[c], self.cuts, side=self.censor_side, error_on_larger=True)
        return td, event.astype(dtype_event)


class IdxDiscUnknownC:
    """Get indexed for discrete data using cuts.

        Arguments:
            cuts {np.array} -- Array or right cuts.
        
        Keyword Arguments:
            label_cols {tuple} -- Name of label columns in dataframe (default: {None}).
    """
    def __init__(self, cuts, label_cols=None, censor_side='left'):
        self.cuts = cuts
        self.duc = DiscretizeUnknownC(cuts, right_censor=True, censor_side=censor_side)
        self.di = Duration2Idx(cuts)
        self.label_cols= label_cols
    
    def transform(self, time, d):
        time, d = self.duc.transform(time, d)
        idx = self.di.transform(time)
        return idx, d
    
    def transform_df(self, df):
        if self.label_cols is None:
            raise RuntimeError("Need to set 'label_cols' to use this. Use 'transform instead'")
        col_duration, col_event = self.label_cols
        time = df[col_duration].values
        d = df[col_event].values
        return self.transform(time, d)


class LabTransDiscreteTime:
    """
    Discretize continuous (duration, event) pairs based on a set of cut points.
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    The discretization learned from fitting to data will move censorings to the left cut point,
    and events to right cut point.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.
    
    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.
    """
    def __init__(self, cuts, scheme='equidistant', min_=0., dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, '__iter__'):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True

    def fit(self, durations, events):
        if self._predefined_cuts:
            warnings.warn("Calling fit method, when 'cuts' are already defined. Leaving cuts unchanged.")
            return self
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype('float64')
        durations = durations.astype(self._dtype)
        self.cuts = make_cuts(self._cuts, self._scheme, durations, events, self._min, self._dtype)
        self.idu = IdxDiscUnknownC(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        idx_durations, events = self.transform(durations, events)
        return idx_durations, events

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        idx_durations, events = self.idu.transform(durations, events)
        return idx_durations.astype('int64'), events.astype('float32')

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        
        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)
    
def create_cols_survival(config, output_table_name, trans_label, event):
    for i in range(0, len(trans_label)):
        event_i = event[i]
        trans_label_i = trans_label[i]

        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [event_i],
                    ['int8'] * len(trans_label_i))
        
        
        create_event_col({'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name,
            'query': config['query_transform']},
                        event_i)
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i],
                    ['float8'] * len(trans_label_i))
    
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i +'_confidences'],
                    ['VARCHAR'] * len(trans_label_i))
        
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i +'_predictions'],
                    ['float8'] * len(trans_label_i))
        
        create_duration_col({'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name,
            'query': config['query_transform']},
                            trans_label_i)
    return None

def create_duration_col(config, duration):
    
    
    df, conn = getDataFromDatabase(config)
    mask = (df['status'] == 90)

    df['individual_end_dates'] = df['end_of_data_date']
    df.loc[mask, 'individual_end_dates'] = df.loc[mask, 'status_date']

    df[duration] = (df['individual_end_dates'] - df['TimeStamp']).dt.days
    # value counts
   # print('value counts', df[duration].value_counts())
    update_cols(
                df.to_dict('records'),
                config,
                [duration],)
    return None

def create_event_col(config, event):
    df, conn = getDataFromDatabase(config)
    conn.close()
    df[event] = df['status']
    df[event] = df[[event]].applymap(lambda x: 1 if x == 90 else 0)
    update_cols(
                df.to_dict('records'),
                config,
                [event],)
    return None

if __name__ == '__main__':
    config = "/home/alatar/miacag/my_configs/pretrain_downstream/config.yaml"
    output_table_name = "dicom_table2x_v2"
    trans_label =  'duration'
    event = 'event'
    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['table_name'] = output_table_name
    create_cols_survival(config, output_table_name, trans_label, event)