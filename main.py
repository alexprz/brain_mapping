"""Perform metaanalysis on keyword related studies."""
import numpy as np
import pandas as pd
import copy
import pickle
import scipy
import nilearn
from matplotlib import pyplot as plt
import seaborn as sns
import nibabel as nib

import meta_analysis
from meta_analysis.meta_analysis import threshold as thr
from meta_analysis.meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, \
    inv_affine, pickle_path, gray_mask, atlas, template
from tools import build_activity_map_from_pmid, build_df_from_keyword, \
    build_avg_map_corpus

from benchmark import benchmark, pearson_distance, RMS

if __name__ == '__main__':
    keyword = 'prosopagnosia'
    sigma = 2.
    N_sim = 5

    df = build_df_from_keyword(keyword)

    maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask,
                atlas=atlas, verbose=True)
    maps.smooth(sigma=sigma, inplace=True)

    avg = maps.avg()
    var = maps.var()

    # 1: Atlas maps
    plotting.plot_activity_map(avg.to_img_atlas(), title='Avg atlas')
    plotting.plot_activity_map(var.to_img_atlas(), title='Var atlas')
    plt.show()

    # 2: Thresholded maps
    avg_smoothed, var_smoothed = maps.iterative_smooth_avg_var(sigma=sigma)

    n_peaks = len(df.index)
    threshold = thr.threshold_MC(n_peaks, maps.n_maps, maps._Ni, maps._Nj,
                                 maps._Nk, N_sim=N_sim, sigma=sigma,
                                 mask=gray_mask, verbose=True)

    plotting.plot_activity_map(avg_smoothed.to_img(),
                               title='Avg smoothed thresholded',
                               threshold=threshold['avg'])
    plotting.plot_activity_map(var_smoothed.to_img(),
                               title='Var smoothed thresholded',
                               threshold=threshold['var'])
    plt.show()

    # 3: Covariance between atlas areas
    cov, labels = maps.cov()
    nilearn.plotting.plot_matrix(cov, labels=labels)
    nilearn.plotting.show()
