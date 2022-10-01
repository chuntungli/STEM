"""
Created on Thu Apr 11 20:56:59 2019

@author: Dan
"""

import copy
import time
import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy import stats
from scipy.signal import find_peaks

from SIMPADV2 import fusion
# from SIMPADV2.stem import stem

# _L = np.arange(50, 101, 25)
_L = np.arange(40, 81, 20)
_m_factor = 5
_omega = 1.0
_is_refine = True
_gamma = 0.15
_delta = 1
_ecdf_wins = [15,30,45]
_lamb = 0.5
_max_warp = 20

acc_features = ['acc_x', 'acc_y', 'acc_z']
gyro_features = ['gyro_x', 'gyro_y', 'gyro_z']
mag_features = ['mag_x', 'mag_y', 'mag_z']
fused_features = ['fused_x', 'fused_y', 'fused_z']
noisy_features = ['noisy_acc_x', 'noisy_acc_y', 'noisy_acc_z']
magnitude_feature = ['magnitude']
rotation_features = ['pitch', 'roll']
filtered_features = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z']

# 90percentile = 1.282
# 95pecerntile = 1.645
# 99percentile = 2.326

# ================ Custom Methods ================

def safeDiv(a,b):
    if (b == 0):
        return np.nan
    return a / b

# array([-0.6238828 , -2.45931277,  1.32757688])
# array([-0.28145646,  0.25820461, -0.01561631])

def computeThreshold(x):
    k_mu = -0.6238828
    a_mu = -2.45931277
    epsilon_mu = 1.32757688

    k_sigma = -0.28145646
    a_sigma = 0.25820461
    epsilon_sigma = -0.01561631
    return (epsilon_mu + x ** k_mu * a_mu) - max(0, (1.645 * (epsilon_sigma + x ** k_sigma * a_sigma)))

# def computeStd(x):
#     k_sigma = -0.21974122
#     a_sigma = 0.29791919
#     epsilon_sigma = -0.06172607
#     return (epsilon_sigma + x ** k_sigma * a_sigma)

def znorm(seq):
    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)
    seq = np.array(seq, copy=True)
    ndim = seq.shape[0]
    seq_len = seq.shape[1]

    for i in range(ndim):
        seq[i, :] = (seq[i, :] - np.nanmean(seq[i, :])) / np.nanstd(seq[i, :])
    return seq

''' 
# ============= Autocorrelation Function ==============
# Inpput: x (1D array)
#         length (interested lags from 1 to length)
'''
def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]
        for i in range(1, length)])

'''
# ========= Low Pass Filter Accelerometer Data ========= 
# Input: _data (3 x n accelerometer data)
#        order
#        cutoff (cutoff frequency)
'''
def filterAccelerometer(_data, fs, cutoff=20, order=5):
    # Design the Buterworth filter
    B,A = signal.butter(order, cutoff / (0.5 * fs), btype='low', output='ba', analog=False)
    return signal.lfilter(B,A, _data)


# Compute TWED between seq1 and seq2 assuming two sequences have the same dimensions
def twed(seq1, seq2, lamb=0.5, max_warp=np.inf, aligned=True):
    # Convert values as numpy array
    seq1 = np.array(seq1, dtype=float, copy=True)
    seq2 = np.array(seq2, dtype=float, copy=True)

    if (seq1.ndim == 1):
        seq1 = np.expand_dims(seq1, axis=0)
    if (seq2.ndim == 1):
        seq2 = np.expand_dims(seq2, axis=0)

    if seq1.shape[0] != seq2.shape[0]:
        raise RuntimeError('Different Number of Dimensions between two sequences.')

    n_dim = seq1.shape[0]
    lamb *= n_dim

    long_seq = seq1 if seq1.shape[1] >= seq2.shape[1] else seq2
    short_seq = seq1 if seq1.shape[1] < seq2.shape[1] else seq2

    # align the sequence by maximizing the cross correlation
    if aligned:
        pad_seq = np.concatenate((long_seq, long_seq), axis=1)
        xcorr = np.zeros((n_dim, pad_seq.shape[1] - short_seq.shape[1] + 1))
        for d in range(n_dim):
            xcorr[d, :] = np.correlate(pad_seq[d, :], short_seq[d, :])
        xcorr = np.sum(xcorr, axis=0)
        long_seq[:] = np.roll(long_seq, -np.argmax(xcorr))

    n = long_seq.shape[1]+1
    m = short_seq.shape[1]+1

    # Add padding
    a = np.zeros((n_dim, n))
    a[:, 1:] = long_seq
    b = np.zeros((n_dim, m))
    b[:, 1:] = short_seq

    DP = np.empty((n, m))
    DP[:] = np.inf
    DP[0,0] = 0

    Di1 = np.zeros(n)
    for i in np.arange(1, n):
        Di1[i] = Dlp(a[:, i], a[:, i-1])
    Dj1 = np.zeros(m)
    for i in np.arange(1, m):
        Dj1[i] = Dlp(b[:, i], b[:, i-1])

    for i in np.arange(1, n):
        # for j in np.arange(1, m):
        for j in np.arange(max(1, i - max_warp), min(m, i + max_warp)):
            C = np.ones(3) * np.inf
            # Deletion in A
            C[0] = DP[i-1,j] + Di1[i] + lamb
            # Deletion in B
            C[1] = DP[i,j-1] + Dj1[j] + lamb
            # Keep data points in both time series
            C[2] = DP[i-1,j-1] + Dlp(a[:,i], b[:,j]) + Dlp(a[:,i-1], b[:,j-1])

            DP[i,j] = min(C)

    return DP[n-1, m-1]

def Dlp(a, b, degree=2):
    dist = 0
    for d in range(a.shape[0]):
        dist += abs(a[d] - b[d]) ** degree
    return dist ** 1/degree
    # return np.sum(np.abs(a-b) ** degree, axis=0) ** 1/degree

def compute_dist_matrix(X, Y=None, lamb=1, max_warp=np.inf):
    """
    Compute the M x N distance matrix between the set of X and Y

    :param X: list of multivariate time series [m_samples: array(d_dimension, l_timepoints)]
    :param Y: list of multivariate time series [n_samples: array(d_dimension, l_timepoints)]
    :return: Distance matrix between item of X and Y with shape [training_m_samples, testing_n_samples]
    """
    is_pairwise = False
    if Y is None:
        is_pairwise = True
        Y = X

    m = len(X)
    n = len(Y)

    dm = np.empty((m, n))
    dm[:] = np.inf
    dm_size = m * n

    tic = time.time()
    for i in range(m):
        if is_pairwise:
            start_index = i+1
        else:
            start_index = 0
        for j in range(start_index, n):
            iterration = (i*n) + j
            cur_prog = (iterration + 1.0) / dm_size
            time_left = ((time.time() - tic) / (iterration + 1)) * (dm_size - iterration - 1)
            print('\rProgress [Computing Distance Matrix] [{0:<50s}] {1:5.1f}% {2:8.1f} sec'
                  .format('#' * int(cur_prog * 50),
                          cur_prog * 100, time_left), end="")

            # Either X or Y is non-repetitive
            if (X[i].size == 0) | (Y[j].size == 0):
                # Both are non-repetitive
                if X[i].size == Y[j].size:
                    dm[i,j] = 0
                    break
                # Otherwise, leave dm[i,j] as infinite distance
                continue

            # If the length displacement is larger than max_warp, leave dm[i,j] as infinite distance
            if abs(X[i].shape[1] - Y[j].shape[1]) > max_warp:
                continue

            dm[i, j] = twed(X[i], Y[j], lamb, max_warp)
    dm[np.where(np.isinf(dm))] = np.nan
    return dm


# # Aligned TWED
# def alignedTWED(seq1, seq2, nu=0, lamb=2, ts1=None, ts2=None):
#     # Convert values as numpy array
#     seq1 = np.array(seq1, dtype=float, copy=True)
#     seq2 = np.array(seq2, dtype=float, copy=True)
#
#     if (seq1.ndim == 1):
#         seq1 = np.expand_dims(seq1, axis=0)
#     if (seq2.ndim == 1):
#         seq2 = np.expand_dims(seq2, axis=0)
#
#     if seq1.shape[0] != seq2.shape[0]:
#         raise RuntimeError('Different Number of Dimensions between two sequences.')
#     n_dim = seq1.shape[0]
#
#     long_seq = seq1
#     short_seq = seq2
#     if seq2.shape[1] > seq1.shape[1]:
#         short_seq = seq1
#         long_seq = seq2
#
#     pad_seq = np.concatenate((long_seq, long_seq), axis=1)
#     xcorr = np.zeros((n_dim, pad_seq.shape[1] - short_seq.shape[1] + 1))
#     for d in range(n_dim):
#         xcorr[d, :] = np.correlate(pad_seq[d, :], short_seq[d, :])
#     xcorr = np.sum(xcorr, axis=0)
#
#     return twed(long_seq, np.roll(short_seq, np.argmax(xcorr), axis=1), nu, lamb, ts1, ts2)


# '''
# # ========= Time Wrap Edit Distance (TWED) =========
# # Input: seq1
# #        seq2
# #        nu
# #        lamb
# '''
# def twed(seq1, seq2, nu=0, lamb=2, ts1=None, ts2=None):
#
#     # Convert values as numpy array
#     seq1 = np.array(seq1, dtype=float, copy=True)
#     seq2 = np.array(seq2, dtype=float, copy=True)
#
#     if (seq1.ndim == 1):
#         seq1 = np.expand_dims(seq1, axis=0)
#     if (seq2.ndim == 1):
#         seq2 = np.expand_dims(seq2, axis=0)
#
#     if (seq1.shape[0] != seq2.shape[0]):
#         raise RuntimeError('Different Number of Dimensions between two sequences.')
#     n_dim = seq1.shape[0]
#     lamb = lamb * n_dim
#
#     # Determine length of seq1 and seq2 as r and c
#     r = seq1.shape[1]
#     c = seq2.shape[1]
#
#     # Check time stamps
#     if ts1 is not None:
#         if len(ts1) != r:
#             raise RuntimeError('Time stamp does not match with sequence length.')
#     else:
#         ts1 = np.arange(r)
#     if ts2 is not None:
#         if (len(ts2) != c):
#             raise RuntimeError('Time stamp does not match with sequence length.')
#     else:
#         ts2 = np.arange(c)
#
#     # Allocate distance matrix
#     D = np.zeros((r+1,c+1))
#     Di1 = np.zeros(r+1)
#     Dj1 = np.zeros(c+1)
#
#     for j in np.arange(1,c+1):
#         distj1 = 0
#         for k in np.arange(n_dim):
#             if j > 1:
#                 distj1 += (seq2[k,j-2]-seq2[k,j-1]) ** 2
#             else:
#                 distj1 += seq2[k,j-1] ** 2
#         Dj1[j] = distj1 ** 0.5
#
#     for i in np.arange(1,r+1):
#         disti1 = 0
#         for k in np.arange(n_dim):
#             if i > 1:
#                 disti1 += (seq1[k,i-2] - seq1[k,i-1]) ** 2
#             else:
#                 disti1 += seq1[k,i-1] ** 2
#         Di1[i] = disti1 ** 0.5
#
#         for j in np.arange(1,c+1):
#             dist = 0
#             for k in np.arange(n_dim):
#                 dist += (seq1[k,i-1] - seq2[k,j-1]) ** 2
#                 if (i > 1) & (j > 1):
#                     dist += (seq1[k,i-2] - seq2[k,j-2]) ** 2
#             D[i,j] = dist ** 0.5
#
#     D[0,0] = 0
#     for i in np.arange(1,r+1):
#         D[i,0] = D[i-1,0] + Di1[i]
#     for j in np.arange(1,c+1):
#         D[0,j] = D[0,j-1] + Dj1[j]
#
#     dmin = 0
#     htrans = 0
#     dist0 = 0
#     iback = 0
#
#     for i in np.arange(1,r+1):
#         for j in np.arange(1,c+1):
#             htrans = np.abs(ts1[i-1] - ts2[j-1])
#             if (j > 1) & (i > 1):
#                 htrans += np.abs(ts1[i-2] - ts2[j-2])
#             dist0 = D[i-1,j-1] + nu * htrans + D[i,j]
#             dmin = dist0
#             if i > 1:
#                 htrans = ts1[i-1] - ts1[i-2]
#             else:
#                 htrans = ts1[i-1]
#             dist = Di1[i] + D[i-1,j] + lamb + nu * htrans
#             if dmin > dist:
#                 dmin = dist
#             if j > 1:
#                 htrans = ts2[j-1] - ts2[j-2]
#             else:
#                 htrans = ts2[j-1]
#             dist = Dj1[j] + D[i,j-1] + lamb + nu * htrans
#             if dmin > dist:
#                 dmin = dist
#             D[i,j] = dmin
#
#     dist = D[r,c]
#     return dist

# Common Functions

def preprocessData(data, samplingRate):
    # Interpolate missing data
    data = data.interpolate()

    # Noise Filter with Low-Pass Butter-worth Filtering Data at 20Hz
    data.values[:] = filterAccelerometer(data.T, fs=samplingRate, cutoff=20).T

    # Geting the gravity
    grav = filterAccelerometer(data[acc_features].T, fs=samplingRate, cutoff=0.3).T

    # Cancel Out Gravity effect
    filtered_data = pd.DataFrame(data[acc_features].values - grav, columns=filtered_features, index=data.index)

    # Compute Magnitude
    magnitude_data = pd.DataFrame(np.sqrt(np.sum(np.square(data[acc_features]), axis=1)), columns=magnitude_feature, index=data.index)
    # magnitude_data = pd.DataFrame(np.sqrt(np.sum(np.square(filtered_data), axis=1)), columns=magnitude_feature, index=data.index)

    # Generate Noisy Features
    noisy_data = pd.DataFrame(znorm(data[acc_features]) + np.random.normal(0, 0.1, data[acc_features].shape), columns=noisy_features, index=data.index)

    data = pd.concat([data, filtered_data, magnitude_data, noisy_data], axis=1)

    if set(gyro_features).issubset(data.columns):
        # Perform Sensor Fusion
        fused_accel = []
        rotation_data = []
        accel = data[acc_features].values
        # accel = filtered_data.values
        gyro = data[gyro_features].values

        mad_filter = fusion.Fusion(1 / samplingRate)
        for i in range(len(accel)):
            mad_filter.update_nomag(accel[i, :], gyro[i, :])
            fused_accel.append(np.dot(mad_filter.to_rotation_matrix(), accel[i, :]))
            rotation_data.append([mad_filter.pitch, mad_filter.roll])
        fused_accel = np.array(fused_accel)
        fused_accel[:, 2] -= 1 # Remove Gravity Effect
        # fused_data = np.array([np.sqrt(np.sum(np.square(accel), axis=1)), fused_accel[:, 2]]).T
        fused_data = pd.DataFrame(fused_accel, columns=fused_features, index=data.index)
        rotation_data = pd.DataFrame(rotation_data, columns=rotation_features, index=data.index)

        data = pd.concat([data, fused_data, rotation_data], axis=1)

    return data


def entropy(x):
    pX = x / x.sum()
    return -np.nansum(pX * np.log2(np.abs(pX)))


# def ecdf_rep(seq, n=None):
#     representation = []
#     D = seq.shape[0]
#     data_len = seq.shape[1]
#     for d in range(D):
#         x = seq[d,:]
#
#         # create a sorted series of unique data
#         cdfx = np.sort(np.unique(x))
#         if n is None:
#             n = len(cdfx)
#         # x-data for the ECDF: evenly spaced sequence of the uniques
#         x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
#
#         # y-data for the ECDF:
#         y_values = []
#         for i in x_values:
#             # all the values in raw data less than the ith value in x_values
#             temp = seq[seq <= i]
#             # fraction of that value with respect to the size of the x_values
#             value = temp.size / data_len
#             # pushing the value in the y_values
#             y_values.append(value)
#             # return both x and y values
#
#         # Calculate the inverse of y
#         y_values = np.power(y_values, -1)
#
#         # Cubic Interpolate Data of n points
#         f = interp1d(x_values, y_values, kind='cubic')
#         representation += f(np.linspace(start=min(cdfx), stop=max(cdfx), num=n)).tolist()
#     return representation

def ecdf_rep(seq, n=None):
    m = np.nanmean(seq, axis=1)
    data = np.sort(seq, axis=1)
    data = data[:, ~np.isnan(data).any(axis=0)]
    data = data[:, np.int32(np.around(np.linspace(0, data.shape[1] - 1, num=n)))]
    data = data.flatten('C')
    return np.hstack((data, m))

# Compute Statistical Features
def extStatFeature(seq):
    stat_features = []

    '''
    ================= Calculate Staistical Features ==================
    # [Mean, STD] on each channel of x,y,z     (1-12)
    # Correlation between three channels (e.g. xy, xz, yz)      (13-15)
    # [First 10 FFT coefficient on each channel                 (16-45)

    Unsupervised learning for human activity recognition using smartphone sensors
    '''
    # for sensor in range(seq.shape[0] // 5):
    #     x = seq[sensor * 5, :]
    #     y = seq[sensor * 5 + 1, :]
    #     z = seq[sensor * 5 + 2, :]
    #     pitches = seq[sensor * 5 + 3, :]
    #     rolls = seq[sensor * 5 + 4, :]
    #
    #     stat_features += [np.nanmean(x), np.nanstd(x), np.sum(np.square(x)), entropy(x)]
    #     stat_features += [np.nanmean(y), np.nanstd(y), np.sum(np.square(y)), entropy(y)]
    #     stat_features += [np.nanmean(z), np.nanstd(z), np.sum(np.square(z)), entropy(z)]
    #     stat_features += [np.nanmean(pitches), np.nanstd(pitches), np.sum(np.square(pitches)), entropy(pitches)]
    #     stat_features += [np.nanmean(rolls), np.nanstd(rolls), np.sum(np.square(rolls)), entropy(rolls)]
    #     stat_features += [np.correlate(x, y)[0], np.correlate(x, z)[0], np.correlate(y, z)[0]]
    #     sp = np.fft.fft(x)[:win_len // 2].real
    #     stat_features += sp[:10].tolist()
    #     sp = np.fft.fft(y)[:win_len // 2].real
    #     stat_features += sp[:10].tolist()
    #     sp = np.fft.fft(z)[:win_len // 2].real
    #     stat_features += sp[:10].tolist()

    '''
    ================= Calculate Staistical Features ==================
    # [Mean, STD, min, max, mode, range] on each channel        (1-6)
    # mean crossing rate, direct component                      (7-15)
    # [First 10 FFT coefficient on each channel                 (16-45)

    A Novel Feature Incremental Learning Method for Sensor-Based Activity Recognition
    '''
    for d in range(seq.shape[0]):
        seq_len = seq.shape[1]
        seq_d = seq[d, :]

        stat_features += [np.nanmean(seq_d), np.nanstd(seq_d), np.nanmin(seq_d), np.nanmax(seq_d), stats.mode(seq_d)[0][0],
                          np.nanmax(seq_d) - np.nanmin(seq_d)]
        # Mean Crossing Rate
        mean_cross = np.sum(np.diff((seq_d - np.nanmean(seq_d)) > 0))
        stat_features.append(mean_cross / seq_len)
        # Frequency Domain Features
        sp = np.fft.fft(seq_d).real[:seq_len // 2]
        freq = np.fft.fftfreq(seq_len)[:seq_len // 2] * 50 # Sampling rate at 50 Hz
        # Direct Component
        stat_features.append(sp[0])
        sp = sp[1:]
        freq = freq[1:]
        # First 5 Peaks
        peaks = find_peaks(sp)[0]
        peak_spec = np.zeros(5)
        peak_freq = np.zeros(5)
        for j in range(5):
            if len(peaks) == 0:
                break
            idx = np.argmax(sp[peaks])
            peak_spec[j] = sp[idx]
            peak_freq[j] = freq[idx]
        stat_features += peak_spec.tolist() + peak_freq.tolist()
        # Energy of signal
        stat_features.append(np.sum(np.square(seq_d)) / seq_len)
        # Four Shape Features of Spectrum
        stat_features += [np.nanmean(sp), np.nanstd(sp), stats.skew(sp), stats.kurtosis(sp)]
        # Four Amplitude Features
        amplitude = np.abs(sp)
        stat_features += [np.nanmean(amplitude), np.nanstd(amplitude), stats.skew(amplitude), stats.kurtosis(amplitude)]

    return stat_features

# def extTemp(seq, detections, start_idx, stop_idx):
#     temp = []
#     isOverlap = (stop_idx >= detections.start_idx) & (detections.stop_idx >= start_idx)
#     selected_detections = detections[isOverlap].copy()
#     if len(selected_detections) > 0:
#         selected_detections['range'] = 0
#         for index, row in selected_detections.iterrows():
#             selected_detections.loc[index, 'range'] = min(row.stop_idx, stop_idx) - max(row.start_idx, start_idx)
#         detection = selected_detections.loc[selected_detections.range.idxmax()]
#         temp, temp_idx, is_continuous = stem.stem(seq, detection.length, refine=_is_refine, gamma=_gamma, delta=_delta)
#     return temp
#
# def extTestTemp(raw_seq, fused_seq, omega=_omega):
#     test_temp = []
#     candidates = np.array([])
#     detections = msimpad(raw_seq, L=_L, m_factor=_m_factor, w=omega)
#     if len(detections) > 0:
#         detections['range'] = detections.stop_idx - detections.start_idx
#         if np.sum(detections.range) > fused_seq.shape[1] * 0.5:
#             detection = detections.loc[detections.range.idxmax()]
#             test_temp, candidates, temp_idx, is_continuous = stem.stem(fused_seq, detection.length, refine=_is_refine, gamma=_gamma, delta=_delta)
#     return test_temp, candidates
#
# def extTrainTemp(raw_seq, fused_seq):
#     train_temp = []
#     candidates = np.array([])
#     detections = msimpad(raw_seq, L=_L, m_factor=_m_factor, w=_omega)
#     if len(detections) <= 0:
#         train_temp, candidates, temp_idx, is_continuous = stem.stem(fused_seq, _ecdf_wins[0], refine=_is_refine, gamma=_gamma, delta=_delta)
#     else:
#         detections['range'] = detections.stop_idx - detections.start_idx
#         detection = detections.loc[detections.range.idxmax()]
#         train_temp, candidates, temp_idx, is_continuous = stem.stem(fused_seq, detection.length, refine=_is_refine, gamma=_gamma, delta=_delta)
#     return train_temp, candidates
