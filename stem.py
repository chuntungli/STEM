"""
Created on Thu Sep 13 16:03:30 2019

@author: Dan
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
from skimage import filters

from SIMPADV2 import utility

_EPS = 1e-14

def stem(seq, sub_len, refine=True, gamma=0.15, delta=1, omega=1):
    """ STEM: Scalable Template Extraction Method based on mSTOMP

    Parameters
    ----------
    seq : numpy matrix, shape (n_dim, seq_len)
        input sequence
    sub_len : int
        subsequence length
    return_dimension : bool
        if True, also return the matrix profile dimension. It takses O(d^2 n)
        to store and O(d^2 n^2) to compute. (default is False)

    Returns
    -------
    matrix_profile : numpy matrix, shape (n_dim, sub_num)
        matrix profile
    profile_index : numpy matrix, shape (n_dim, sub_num)
        matrix profile index
    profile_dimension : list, optional, shape (n_dim)
        matrix profile dimension, this is only returned when return_dimension
        is True

    Notes
    -----
    This method is modified from the code provided in the following URL
    C.-C. M. Yeh, N. Kavantzas, and E. Keogh, "Matrix Profile VI: Meaningful
    Multidimensional Motif Discovery," IEEE ICDM 2017.
    https://sites.google.com/view/mstamp/
    http://www.cs.ucr.edu/~eamonn/MatrixProfile.html
    """
    if sub_len < 4:
        raise RuntimeError('Subsequence length (sub_len) must be at least 4')
    exc_zone = sub_len // 2
    seq = np.array(seq, dtype=float, copy=True)

    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)

    seq_len = seq.shape[1]
    sub_num = seq.shape[1] - sub_len + 1
    n_dim = seq.shape[0]
    # threshold = omega * utility.computeThreshold(n_dim * sub_len)

    skip_loc = np.zeros(sub_num, dtype=bool)
    for i in range(sub_num):
        if not np.all(np.isfinite(seq[:, i:i + sub_len])):
            skip_loc[i] = True
    seq[~np.isfinite(seq)] = 0

    drop_val = 0
    matrix_profile = np.empty((n_dim, sub_num))
    matrix_profile[:] = np.inf
    profile_index = -np.ones((n_dim, sub_num), dtype=int)
    seq_freq = np.empty((n_dim, seq_len * 2), dtype=np.complex128)
    seq_mu = np.empty((n_dim, sub_num))
    seq_sig = np.empty((n_dim, sub_num))
    for i in range(n_dim):
        seq_freq[i, :], seq_mu[i, :], seq_sig[i, :] = \
            _mass_pre(seq[i, :], sub_len)

    dist_profile = np.empty((n_dim, sub_num))
    last_product = np.empty((n_dim, sub_num))
    first_product = np.empty((n_dim, sub_num))

    dist_template = np.empty(sub_num)
    dist_template[:] = np.inf
    index_template = []

    drop_val = np.empty(n_dim)
    que_sum = np.empty(n_dim)
    que_sq_sum = np.empty(n_dim)
    que_sig = np.empty(n_dim)
    tic = time.time()
    # print('\nComputing STEM on %d' % sub_len)
    for i in range(sub_num):
        cur_prog = (i + 1.0) / sub_num
        time_left = ((time.time() - tic) / (i + 1)) * (sub_num - i - 1)
        print('\rProgress [STEM] [{0:<50s}] {1:5.1f}% {2:8.1f} sec'
              .format('#' * int(cur_prog * 50),
                      cur_prog * 100, time_left), end="")

        # Update Distance Profile
        for j in range(n_dim):
            que = seq[j, i:i + sub_len]
            if i == 0:
                (dist_profile[j, :], last_product[j, :],
                 que_sum[j], que_sq_sum[j], que_sig[j]) = \
                    _mass(seq_freq[j, :], que, seq_len, sub_len,
                          seq_mu[j, :], seq_sig[j, :])
                first_product[j, :] = last_product[j, :].copy()
            else:
                que_sum[j] = que_sum[j] - drop_val[j] + que[-1]
                que_sq_sum[j] = que_sq_sum[j] - drop_val[j]**2 + que[-1]**2
                que_mu = que_sum[j] / sub_len
                que_sig_sq = que_sq_sum[j] / sub_len - que_mu**2
                if que_sig_sq < _EPS:
                    que_sig_sq = _EPS
                que_sig[j] = np.sqrt(que_sig_sq)
                last_product[j, 1:] = (last_product[j, 0:-1] -
                                       seq[j, 0:seq_len - sub_len] *
                                       drop_val[j] +
                                       seq[j, sub_len:seq_len] * que[-1])
                last_product[j, 0] = first_product[j, i]
                dist_profile[j, :] = \
                    (2 * (sub_len - (last_product[j, :] -
                                     sub_len * seq_mu[j, :] * que_mu) /
                          (seq_sig[j, :] * que_sig[j])))
                dist_profile[j, dist_profile[j, :] < _EPS] = 0
            drop_val[j] = que[0]

        if skip_loc[i] or np.any(que_sig < _EPS):
            continue

        # Excluding invalid zone
        exc_zone_st = max(0, i - exc_zone)
        exc_zone_ed = min(sub_num, i + exc_zone)
        dist_profile[:, exc_zone_st:exc_zone_ed] = np.inf
        dist_profile[:, skip_loc] = np.inf
        dist_profile[seq_sig < _EPS] = np.inf

        # Exact Algorithm based on MWIS
        # dp = dist_profile[n_dim - 1]
        # plt.plot(dp);plt.show();plt.close()
        # vl, el, al = _generateGraph(dp, sub_len)
        # opt_x = utility.MWIS(vl, al)

        # Greedy Approach to search the template candidates for the given dist_profile
        prun_zone = int((1 - gamma) * sub_len)
        dist_sum = 0
        peaks = [i]
        # Drop INF and NAN values
        dp = pd.Series(dist_profile[n_dim-1])
        dp.interpolate()
        dp[np.isnan(dp)] = np.inf
        dp = np.sqrt(dp)
        dp /= np.sqrt(n_dim * sub_len)
        threshold = filters.threshold_otsu(dp[~np.isinf(dp)])
        dp = np.array(dp)

        while not np.all(np.isinf(dp)):
            peak = np.argmin(dp)
            dist = dp[peak]
            if (dist >= threshold):
                if (len(peaks) == 1):
                    dist_sum = dist
                break
            dist_sum += dist
            peaks.append(peak)
            dp[max(0, peak - prun_zone): min(sub_num, peak + prun_zone)] = np.inf
        dist_template[i] = dist_sum / len(peaks)
        index_template.append(peaks)

    dist_template = dist_template[dist_template != np.inf]
    peaks = index_template[np.argmin(dist_template)]
    is_continuous = True

    if (len(peaks) == 0):
        print('Warning: No Template Candidates Found')
        return None, None, None, None, None, None, None

    raw_template = None
    raw_idx = None
    raw_candidates = None
    refine_template = None
    refine_idx = None
    refine_candidates = None
    is_continuous = None

    # Extract Template Directly
    candidates = []
    for peak in peaks:
        candidate = seq[:, peak:peak + sub_len]
        candidates.append(candidate)
    raw_candidates = np.array(candidates)
    del (peak, candidate, candidates)
    raw_template = utility.znorm(np.nanmedian(raw_candidates, axis=0))
    raw_idx = np.array(peaks)

    # Two-step approach for refinement
    # Extract Buffered Candidates
    buffer = int(gamma * sub_len)
    pad_temp_len = sub_len + 2 * buffer
    est_temp_idx = np.array(peaks) - buffer
    candidates = np.empty((len(est_temp_idx), n_dim, pad_temp_len))
    candidates[:] = np.nan
    for i in range(len(est_temp_idx)):
        idx = est_temp_idx[i]
        ext_data = seq[:, max(0, idx): min(seq_len, idx + pad_temp_len)]
        candidates[i, :, max(0, -idx): pad_temp_len - max(0, (idx + pad_temp_len) - seq_len)] = ext_data

    # Largest Pointwise SD
    std_threshold = delta * (np.sum(np.nanstd(seq, axis=1)) / n_dim)
    pw_std = np.sum(np.nanstd(candidates, axis=0), axis=0) / n_dim
    is_continuous = (max(pw_std[:2 * buffer]) < std_threshold) & (max(pw_std[-2 * buffer:]) < std_threshold)

    # Refining Template
    dp = np.zeros(2 * buffer)
    ip = np.zeros(2 * buffer, dtype=int)
    if not is_continuous:
        # Refined by Variability
        for i in range(2 * buffer):
            std_cumsum = np.cumsum(pw_std[i:min(pad_temp_len, i + sub_len + buffer)])
            std_avg = std_cumsum / (np.arange(len(std_cumsum)) + 1)
            std_avg = std_avg[sub_len - buffer:]
            dp[i] = np.min(std_avg)
            ip[i] = np.argmin(std_avg) + (sub_len - buffer) + i
    else:
        # Refined by Nearest Point
        template = np.nanmedian(candidates, axis=0)

        for i in range(2 * buffer):
            que = template[:, i]
            start_idx = i + (sub_len - buffer)
            stop_idx = i + sub_len + buffer
            dist = np.sqrt(np.sum(np.square(template[:, start_idx:stop_idx].T - que), axis=1))
            dp[i] = np.nanmin(dist)
            ip[i] = start_idx + np.argmin(dist)

    start_idx = np.argmin(dp)
    end_idx = ip[start_idx]
    refine_candidates = candidates[:, :, start_idx:end_idx]

    # Compute template from the candidates
    refine_template = utility.znorm(np.nanmedian(refine_candidates, axis=0))
    refine_idx = est_temp_idx + start_idx

    return raw_template, raw_idx, raw_candidates, refine_template, refine_idx, refine_candidates, is_continuous

    # # Extract Template Directly
    # if not refine:
    #     candidates = []
    #     for peak in peaks:
    #         candidate = seq[:, peak:peak + sub_len]
    #         candidates.append(candidate)
    #     candidates = np.array(candidates)
    #     del (peak, candidate)
    #     template = np.nanmedian(candidates, axis=0)
    #
    #     return template, peaks, False
    # else:
    #     # Two-step approach for refinement
    #
    #     # Extract Buffered Candidates
    #     buffer = int(gamma * sub_len)
    #     pad_temp_len = sub_len + 2 * buffer
    #     est_temp_idx = np.array(peaks) - buffer
    #     candidates = np.empty((len(est_temp_idx), n_dim, pad_temp_len))
    #     candidates[:] = np.nan
    #     for i in range(len(est_temp_idx)):
    #         idx = est_temp_idx[i]
    #         ext_data = seq[:, max(0, idx): min(seq_len, idx + pad_temp_len)]
    #         candidates[i, :, max(0, -idx): pad_temp_len - max(0, (idx + pad_temp_len) - seq_len)] = ext_data
    #
    #     # Largest Pointwise SD
    #     std_threshold = delta * (np.sum(np.nanstd(seq, axis=1)) / n_dim)
    #     pw_std = np.sum(np.nanstd(candidates, axis=0), axis=0) / n_dim
    #     is_continuous = (max(pw_std[:2*buffer]) < std_threshold) & (max(pw_std[-2*buffer:]) < std_threshold)
    #
    #     # Refining Template
    #     dp = np.zeros(2 * buffer)
    #     ip = np.zeros(2 * buffer, dtype=int)
    #     if not is_continuous:
    #         # Refined by Variability
    #         for i in range(2 * buffer):
    #             std_cumsum = np.cumsum(pw_std[i:min(pad_temp_len, i + sub_len + buffer)])
    #             std_avg = std_cumsum / (np.arange(len(std_cumsum)) + 1)
    #             std_avg = std_avg[sub_len - buffer:]
    #             dp[i] = np.min(std_avg)
    #             ip[i] = np.argmin(std_avg) + (sub_len - buffer) + i
    #     else:
    #         # Refined by Nearest Point
    #         template = np.nanmedian(candidates, axis=0)
    #
    #         for i in range(2 * buffer):
    #             que = template[:, i]
    #             start_idx = i + (sub_len - buffer)
    #             stop_idx = i + sub_len + buffer
    #             dist = np.sqrt(np.sum(np.square(template[:, start_idx:stop_idx].T - que), axis=1))
    #             dp[i] = np.nanmin(dist)
    #             ip[i] = start_idx + np.argmin(dist)
    #
    #     start_idx = np.argmin(dp)
    #     end_idx = ip[start_idx]
    #     candidates = candidates[:, :, start_idx:end_idx]
    #
    #     # Compute template from the candidates
    #     template = np.nanmedian(candidates, axis=0)
    #
    #     index = est_temp_idx + start_idx
    #
    #     return utility.znorm(template), candidates, index, is_continuous
        # return template, index, is_continuous

def _generateGraph(_c,_l):
    # Construct VertexList
    _vertexList = np.array(_c)

    # Construct EdgeList
    _edgeList = []
    for i in np.arange(len(_vertexList)):
        for j in np.arange(max(0,i-(_l-1)), min(len(_vertexList), i+_l)):
            if (j != i):
                _edgeList.append((i,j))
    _edgeList = np.array(_edgeList)

    # Construct Adjacency List
    _adjacencyList = [[] for vertex in _vertexList]
    for edge in _edgeList:
        _adjacencyList[edge[0]].append(edge[1])
    _adjacencyList = np.array(_adjacencyList)
    return (_vertexList, _edgeList, _adjacencyList)

def _mass_pre(seq, sub_len):
    """ pre-computation for iterative call to MASS

    Parameters
    ----------
    seq : numpy array
        input sequence
    sub_len : int
        subsequence length

    Returns
    -------
    seq_freq : numpy array
        sequence in frequency domain
    seq_mu : numpy array
        each subsequence's mu (mean)
    seq_sig : numpy array
        each subsequence's sigma (standard deviation)

    Notes
    -----
    This functions is modified from the code provided in the following URL
    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    seq_len = len(seq)
    seq_pad = np.zeros(seq_len * 2)
    seq_pad[0:seq_len] = seq
    seq_freq = np.fft.fft(seq_pad)
    seq_cum = np.cumsum(seq_pad)
    seq_sq_cum = np.cumsum(np.square(seq_pad))
    seq_sum = (seq_cum[sub_len - 1:seq_len] -
               np.concatenate(([0], seq_cum[0:seq_len - sub_len])))
    seq_sq_sum = (seq_sq_cum[sub_len - 1:seq_len] -
                  np.concatenate(([0], seq_sq_cum[0:seq_len - sub_len])))
    seq_mu = seq_sum / sub_len
    seq_sig_sq = seq_sq_sum / sub_len - np.square(seq_mu)
    seq_sig = np.sqrt(seq_sig_sq)
    return seq_freq, seq_mu, seq_sig


def _mass(seq_freq, que, seq_len, sub_len, seq_mu, seq_sig):
    """ iterative call of MASS

    Parameters
    ----------
    seq_freq : numpy array
        sequence in frequency domain
    que : numpy array
        query
    seq_len : int
        sequence length
    sub_len : int
        subsequence length
    seq_mu : numpy array
        each subsequence's mu (mean)
    seq_sig : numpy array
        each subsequence's sigma (standard deviation)

    Returns
    -------
    dist_profile : numpy array
        distance profile
    last_product : numpy array
        cross term
    que_sum : float64
        query's sum
    que_sq_sum : float64
        query's squre sum
    que_sig : float64
        query's sigma (standard deviation)

    Notes
    -----
    This functions is modified from the code provided in the following URL
    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    que = que[::-1]
    que_pad = np.zeros(seq_len * 2)
    que_pad[0:sub_len] = que
    que_freq = np.fft.fft(que_pad)
    product_freq = seq_freq * que_freq
    product = np.fft.ifft(product_freq)
    product = np.real(product)

    que_sum = np.sum(que)
    que_sq_sum = np.sum(np.square(que))
    que_mu = que_sum / sub_len
    que_sig_sq = que_sq_sum / sub_len - que_mu**2
    if que_sig_sq < _EPS:
        que_sig_sq = _EPS
    que_sig = np.sqrt(que_sig_sq)

    dist_profile = (2 * (sub_len - (product[sub_len - 1:seq_len] -
                                    sub_len * seq_mu * que_mu) /
                         (seq_sig * que_sig)))
    last_product = product[sub_len - 1:seq_len]
    return dist_profile, last_product, que_sum, que_sq_sum, que_sig
