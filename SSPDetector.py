#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import copy
import warnings
import numpy as np
import pandas as pd
from functools import reduce

from SIMPADV2 import utility
from SIMPADV2.RCMP import rcmstomp

def _generateGraph(var_len_valleys):
    vertexList = []
    for l in var_len_valleys:
        vertexList = vertexList + var_len_valleys[l]
    vertexList = np.array(vertexList)

    edgeList = []
    for i in np.arange(len(vertexList)):
        overlap = \
        np.where((~(vertexList[i, 0] > vertexList[:, 1]) & ~(vertexList[i, 1] < vertexList[:, 0])) == True)[0]
        overlap = np.delete(overlap, np.where(overlap == i)[0])
        for j in overlap:
            edgeList.append((i, j))
    edgeList = np.array(edgeList)

    adjacencyList = [[] for vartex in vertexList]
    for edge in edgeList:
        adjacencyList[edge[0]].append(edge[1])
    adjacencyList = np.array(adjacencyList)

    return vertexList, edgeList, adjacencyList


def _generateSubgraphs(vertextList, adjacencyList):
    subgraphs = []
    freeVertices = list(np.arange(len(vertextList)))
    while freeVertices:
        freeVertex = freeVertices.pop()
        subgraph = _constructSubgraph(freeVertex, adjacencyList, [freeVertex])
        freeVertices = [vertex for vertex in freeVertices if vertex not in subgraph]
        subgraphs.append(subgraph)
    return subgraphs


def _constructSubgraph(vertex, adjacencyList, subgraph):
    neighbors = [vertex for vertex in adjacencyList[vertex] if vertex not in subgraph]
    if (len(neighbors) == 0):
        return subgraph
    else:
        subgraph = subgraph + neighbors
        for vertex in neighbors:
            subgraph = _constructSubgraph(vertex, adjacencyList, subgraph)
        return subgraph


def _incumb(vertexWeight, adjacencyList):
    N = len(vertexWeight)

    X = np.zeros(N, dtype=bool)
    for i in range(N):
        if (len(adjacencyList[i]) == 0):
            X[i] = True

    Z = np.zeros(N)
    for i in range(N):
        Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])

    freeVertices = np.where(X == 0)[0]
    while True:
        if len(freeVertices) == 0:
            break;
        imin = freeVertices[np.argmax(Z[freeVertices])]
        X[imin] = True
        freeVertices = freeVertices[freeVertices != imin]
        X[adjacencyList[imin]] = False
        freeVertices = freeVertices[~np.isin(freeVertices, adjacencyList[imin])]
        for i in freeVertices:
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[np.intersect1d(freeVertices, adjacencyList[i])])
    return X

def _calculateLB(X, vertexWeight, adjacencyList, visitedVertices=[]):
    neighbors = np.array([], dtype=int)
    if (len(adjacencyList[np.where(X == 1)[0]]) > 0):
        neighbors = reduce(np.union1d, adjacencyList[np.where(X == 1)[0]])
    if (len(visitedVertices) > 0):
        neighbors = np.append(neighbors, visitedVertices[np.where(X[visitedVertices] == False)])
    neighbors = np.unique(neighbors)
    neighbors = np.array(neighbors, dtype=int)
    wj = np.sum(vertexWeight[neighbors])
    return -1 * (np.sum(vertexWeight) - wj)

def _BBND(vertexWeight, adjacencyList, LB, OPT_X):
    N = len(vertexWeight)
    X = np.zeros(N)
    X[:] = np.nan
    visitedVertices = np.array([], dtype=int)
    OPT = np.sum(vertexWeight[OPT_X == 1])
    prob = {'X': [], 'visitedVertices': []}
    sub_probs = []

    while True:
        if (np.sum(np.isnan(X)) == 0):
            if (np.sum(vertexWeight[np.where(X == 1)[0]]) > OPT):
                OPT = np.sum(vertexWeight[np.where(X == 1)[0]])
                OPT_X = X
            if (len(sub_probs) > 0):
                prob = sub_probs.pop()
                X = prob['X']
                visitedVertices = prob['visitedVertices']
            else:
                break

        for i in range(N):
            if (~np.any(X[list(adjacencyList[i])])):
                X[i] = 1
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)

        Z = np.zeros(N)
        for i in range(N):
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])
        if (len(visitedVertices) > 0):
            Z[visitedVertices] = np.inf
        imin = np.argmin(Z)

        visitedVertices = np.append(visitedVertices, imin)

        X[imin] = 0
        LB0 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        X[imin] = 1
        LB1 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        if (LB0 < LB1):
            if (LB1 < LB):
                X[imin] = 1
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()

                prob['X'][list(adjacencyList[imin])] = 0
                neighbors = adjacencyList[imin]
                for i in neighbors:
                    if (not i in prob['visitedVertices']):
                        prob['visitedVertices'] = np.append(prob['visitedVertices'], i)
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())

            X[imin] = 0
        else:
            if (LB0 < LB):
                X[imin] = 0
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())
            X[imin] = 1
            X[list(adjacencyList[imin])] = 0
            neighbors = adjacencyList[imin]
            for i in neighbors:
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)
    return OPT_X


def MWIS(vertexWeight, adjacencyList):
    '''
    :param vertexWeight: List of real-valued vertex weight
    :param adjacencyList: List of adjacency vertices
    :return: Maximum sum of weights of the independent set
    :Note:
        This is the implementation of the follow publication:

        Pardalos, P. M., & Desai, N. (1991). An algorithm for finding a maximum weighted independent set in an arbitrary graph.
        International Journal of Computer Mathematics, 38(3-4), 163-175.
    '''
    X = _incumb(vertexWeight, adjacencyList)
    LB = _calculateLB(X, vertexWeight, adjacencyList)
    return _BBND(vertexWeight, adjacencyList, LB, X)


'''
# ========= Find Vallies from Matrix Profile ========= 
# Input: _mp (1d matrix profile)
#        _l (subsequence length)
#        _sigma (threshold)
'''
def _findVallies(_mp, _l, _sigma):
    vallies = []
    N = len(_mp)

    rough_detect = _mp < _sigma
    indices = np.where(np.diff(rough_detect))[0]
    if (len(indices) == 0):
        indices = [0,N]
    if (indices[0] > 0):
        indices = np.insert(indices, 0, 0)
    if (indices[len(indices)-1] < N):
        indices = np.insert(indices, len(indices), N-1)

    lengths = np.diff(indices)
    while True:
        if (np.all(lengths >= (2 * _l))):
            break
        rm_idx = np.argmin(lengths)
        if (rm_idx == 0):
            indices = np.delete(indices, 1)
            lengths[0] += lengths[1]
            lengths = np.delete(lengths, 1)
        elif rm_idx == len(lengths)-1:
            indices = np.delete(indices, len(indices)-2)
            lengths[-1] += lengths[-2]
            lengths = np.delete(lengths, len(lengths)-2)
        else:
            indices = np.delete(indices, [rm_idx, rm_idx+1])
            lengths[rm_idx-1] += lengths[rm_idx] + lengths[rm_idx+1]
            lengths = np.delete(lengths, [rm_idx, rm_idx+1])

    detection = np.zeros(N)
    for i in range(len(indices)-1):
        start_idx = indices[i]
        end_idx = indices[i+1]
        result = np.mean(rough_detect[start_idx:end_idx]) >= 0.5
        detection[start_idx:end_idx] = result
        if (result):
            # vallies.append([start_idx, min(end_idx + _l, N), _l, np.sum(1.4 - _mp[start_idx:end_idx]), np.std(_mp[start_idx:end_idx])])
            vallies.append([start_idx, end_idx + _l, _l, np.sum(_sigma - _mp[start_idx:end_idx]), np.std(_mp[start_idx:end_idx])])

    return vallies

def znorm(seq):
    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)
    seq = np.array(seq, copy=True)
    ndim = seq.shape[0]

    for i in range(ndim):
        seq[i, :] = (seq[i, :] - np.nanmean(seq[i, :])) / np.nanstd(seq[i, :])
    return seq

def SIMPAD(data, l, m, omega=1.0):
    '''
    :param data: d x n real-valued array - Input Time Series
    :param l: int - Target pattern length
    :param m: int - Maximum displacement between patterns
    :param dimensions: int (OPTIONAL) - The number of dimensions to be compared. It calculate the MP by all dimensions if not sepcified.
    :return: List of bool - Indicate SSP, True for identified SSP
    '''

    df_valleys = pd.DataFrame(columns=['start_idx', 'stop_idx', 'length'], dtype=int)

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    ndim = data.shape[0]
    seq_len = data.shape[1]

    if l + m >= seq_len:
        warnings.warn('Sequence length too short for computing RCMP - l+m: %d seq_len: %d' % (l + m, seq_len), RuntimeWarning)
        return df_valleys

    # mp, ip = rcmstomp(znorm(data) + np.random.normal(0, 0.1, data.shape), l, m)
    mp, ip = rcmstomp(data, l, m)

    mp = pd.Series(mp[mp.shape[0] - 1,:])

    mp[np.isinf(mp)] = np.nan
    mp = mp.interpolate()

    # Normalize RCMP
    mp = mp / np.sqrt(ndim * l)

    N = len(mp)

    # sigma = filters.threshold_otsu(mp.dropna())
    sigma = omega * utility.computeThreshold(ndim * l)
    valleys = _findVallies(mp, l, sigma)

    if len(valleys) > 0:
        df_valleys = pd.DataFrame(valleys[:, :3], columns=['start_idx', 'stop_idx', 'length'], dtype=int)

    return df_valleys


def mSIMPAD(data, L, m_factor, omega=1.0):
    '''
    :param data: d x n real-valued array - Input Time Series
    :param L: List of int - Target pattern lengths
    :param m_factor: int - A factor of target pattern length to determine the maximum displacement between patterns
    :param dimensions: int (OPTIONAL) - The number of dimensions to be compared. It calculate the MP by all dimensions if not sepcified.
    :return: List of bool - Indicate SSP, True for identified SSP
    '''

    var_len_valleys = {}

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    ndim = data.shape[0]
    seq_len = data.shape[1]

    for l in L:
        m = m_factor * l
        if l + m >= seq_len:
            warnings.warn('Sequence length too short for computing RCMP - l+m: %d seq_len: %d' % (l+m, seq_len), RuntimeWarning)
            continue

        # mp, ip = rcmstomp(znorm(data) + np.random.normal(0, 0.1, data.shape), l, m)
        mp, ip = rcmstomp(data, l, m)

        mp = pd.Series(mp[mp.shape[0] - 1])

        mp[np.isinf(mp)] = np.nan
        mp = mp.interpolate()

        # Normalize RCMP
        mp = mp / np.sqrt(ndim * l)

        # sigma = filters.threshold_otsu(mp.dropna())
        sigma = omega * utility.computeThreshold(ndim * l)
        var_len_valleys[l] = _findVallies(mp, l, sigma)

    vertexList, edgeList, adjacencyList = _generateGraph(var_len_valleys)
    subgraphs = _generateSubgraphs(vertexList, adjacencyList)

    solution = np.zeros(len(vertexList), dtype=bool)
    for subgraph in subgraphs:
        vl = np.array(copy.deepcopy(vertexList[subgraph, 3]))
        al = np.array(copy.deepcopy(adjacencyList[subgraph]))
        for i in range(len(al)):
            for j in range(len(al[i])):
                al[i][j] = np.where(subgraph == al[i][j])[0][0]
        OPT_X = MWIS(vl, al)
        solution[subgraph] = OPT_X

    valleys = vertexList[solution]

    if len(valleys) == 0:
        df_valleys = pd.DataFrame(columns=['start_idx', 'stop_idx', 'length'], dtype=int)
    else:
        df_valleys = pd.DataFrame(valleys[:, :3], columns=['start_idx', 'stop_idx', 'length'], dtype=int)
    return df_valleys