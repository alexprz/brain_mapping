"""
Generate and benchmark atlases.

Generate custom keyword related atlases and compare them to non specific
ones acquired on resting states.
"""
from nilearn import datasets
from matplotlib import pyplot as plt
from copy import copy
import numpy as np
import pandas as pd
import seaborn as sns


from meta_analysis.meta_analysis import Maps, plotting, print_percent

from tools import build_df_from_keyword, pickle_load, pickle_dump, \
    get_dump_token
from globals import template, gray_mask, pickle_path
from clustering import fit_CanICA, fit_DictLearning, fit_Kmeans, fit_Wards, \
    fit_GroupICA, fit_Model

save_dir = pickle_path+'averaged_maps/'

# _________ATLASES_________ #
atlas_HO_0 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
atlas_HO_25 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
atlas_HO_50 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')

n_labels = len(atlas_HO_0['labels'])


# _________DISTANCE_________ #
def pearson_distance(array_ref, array_obs, **kwargs):
    """Pearson distance between given arrays. 0 in ref array are ignored."""
    check_dim(array_ref, array_obs)

    sum_ref = np.sum(array_ref)
    sum_obs = np.sum(array_obs)

    if sum_ref > 0:
        array_ref /= sum_ref

    if sum_obs > 0:
        array_obs /= sum_obs

    with np.errstate(divide='ignore', invalid='ignore'):
        num = np.power(array_obs - array_ref, 2)
        return np.sum(np.nan_to_num(np.true_divide(num, array_ref), posinf=0))


def RMS(array_ref, array_obs, **kwargs):
    """Root mean square distance between given arrays."""
    check_dim(array_ref, array_obs)
    return np.sqrt(np.mean(np.power(array_ref-array_obs, 2)))


def zero(arr, arr2, **kwargs):
    """Trivial zero criterion."""
    return 0


# _________TOOLS_________ #
def check_dim(arr1, arr2):
    """Check if the given arrays have the same shape."""
    if arr1.shape != arr2.shape:
        raise ValueError(f'Dimensions missmatch: {arr1.shape} & {arr2.shape}.')


def add_uniform(maps, alpha):
    """
    Derive convex combination between given maps and uniform.

    Derive the convex combination between the probability maps and the uniform
    probability map with weigth alpha.

    Args:
        maps(Maps): Maps object containg the maps
        alpha(float): Weigth of the convex combination.
            Alpha=0 returns maps unchanged.

    Returns:
        (Maps): Maps object

    """
    if alpha == 0:
        return copy(maps)

    Ni, Nj, Nk, _ = maps.shape
    uni_array = np.ones((maps.n_v, maps.n_m))/maps.n_v
    uni_maps = Maps(uni_array, Ni=Ni, Nj=Nj, Nk=Nk, affine=maps.affine)
    return (1-alpha)*maps + alpha*uni_maps


# _________CRITERIA_________ #
def goodness_of_fit(maps_ref, atlas, **kwargs):
    """
    Compute pearson distance between ref and obs average.

    Args:
        maps_ref(Maps): Reference maps object containing all the maps on which
            the atlas is applied and variance is computed.
        atlas(dict): Atlas stored nilearn-like (dict with maps and labels key).
        kwargs: Looking for an alpha parameter. Alpha is the weight of the
            convex combination between the ref maps and an uniform maps.

    Returns:
        (float): Distance between the refernece maps and the maps on which
            the atlas has been applied.

    """
    alpha = kwargs.get('alpha', 0)

    maps_avg_ref = add_uniform(maps_ref.avg(), alpha)
    maps_avg_obs = maps_avg_ref.apply_atlas(atlas, inplace=False)

    array_ref = maps_avg_ref.to_img().get_fdata()
    array_obs = maps_avg_obs.to_img_atlas().get_fdata()

    return pearson_distance(array_ref, array_obs)


def variance_criterion(maps_ref, atlas, **kwargs):
    """
    Compute pearson distance between ref and obs variance.

    Args:
        maps_ref(Maps): Reference maps object containing all the maps on which
            the atlas is applied and variance is computed.
        atlas(dict): Atlas stored nilearn-like (dict with maps and labels key).
        kwargs: Looking for an alpha parameter. Alpha is the weight of the
            convex combination between the ref maps and an uniform maps.

    Returns:
        (float): Distance between the refernece maps and the maps on which
            the atlas has been applied.

    """
    alpha = kwargs.get('alpha', 0)

    maps_var_ref = add_uniform(maps_ref.var(), alpha)
    maps_var_obs = maps_ref.apply_atlas(atlas, inplace=False).var()
    maps_var_obs = add_uniform(maps_var_obs, alpha)

    array_ref = maps_var_ref.to_img().get_fdata()
    array_obs = maps_var_obs.to_img_atlas().get_fdata()

    return pearson_distance(array_ref, array_obs, **kwargs)


def benchmark(maps, atlases, criteria, verbose=False, **kwargs):
    """
    Benchmark given atlases on given criteria.

    Args:
        maps(Maps): Reference Maps object. Performance of atlases will be
            evaluated on this maps.
        atlases(dict): Dict containing the atlases. The keys will be used
            make a reference to the atlases in the result (e.g atlases name).
        criteria(list): List of criterion available in the benchmark.py file.
        verbose(bool): Should print log or not.
        kwargs: All kwargs are passed to each criterion.

    Returns:
        (pandas.DataFrame) Data frame containing the results of the benchmark.

    """
    df_list = []
    k, n_tot = 0, len(atlases)*len(criteria)

    for criterion in criteria:
        for name, atlas in atlases.items():
            print_percent(
                k, n_tot,
                string='Benchmarking atlas {1} out of {2} : {0:.2f}%...',
                rate=0, verbose=verbose, prefix='Benchmark')
            score = criterion(maps, atlas, **kwargs)
            df_list.append([criterion.__name__, name, score])
            k += 1

    return pd.DataFrame(df_list, columns=['Criterion', 'Atlas', 'Value'])


if __name__ == '__main__':
    keyword = 'prosopagnosia'
    sigma = 2.

    train_proportion = 0.5
    RS = 0  # Random state
    n_par = n_labels-1
    alpha = 0.1

    load = True
    tag = f'{keyword}-sigma-{sigma}-{n_par}-components-RS-{RS}'
    avg_maps_tag = f'{keyword}-sigma-{sigma}-normalized'

    params_CanICA = {
        'n_components':  n_par,
        'random_state': RS,
    }

    params_DictLearning = {
        'n_components': n_par,
        'random_state': RS,
    }

    params_Wards = {
        'n_parcels': n_par,
        'random_state': RS,
    }

    params_Kmeans = {
        'n_parcels': n_par,
        'random_state': RS,
    }

    file_path = pickle_path+get_dump_token(tag=avg_maps_tag)
    maps = pickle_load(file_path)
    if maps is None:  # No previous computation found.
        # Build keyword related maps, smooth and normalize them.
        df = build_df_from_keyword(keyword)
        maps = Maps(df, template=template, groupby_col='pmid',
                    mask=gray_mask, verbose=True)
        maps.smooth(sigma, inplace=True, verbose=True)
        maps.normalize(inplace=True)
        pickle_dump(maps, file_path)

    maps_train, maps_test = maps.split(prop=train_proportion, random_state=RS)
    del maps

    maps_train_avg = maps_train.avg()

    imgs_4D = maps_train.to_img()
    del maps_train

    CanICA_imgs = fit_Model(fit_CanICA, imgs_4D,
                            params_CanICA, tag=tag, load=load).components_img_
    DictLearning_imgs = fit_Model(fit_DictLearning, maps.to_img(),
                                  params_DictLearning, tag=tag,
                                  load=load).components_img_
    Ward_imgs = fit_Model(fit_Wards, imgs_4D, params_Wards,
                          tag=tag, load=load).labels_img_
    Kmeans_imgs = fit_Model(fit_Kmeans, imgs_4D,
                            params_Kmeans, tag=tag, load=load).labels_img_
    del imgs_4D

    atlas_CanICA = Maps(CanICA_imgs, template=template).to_atlas()
    # atlas_DictLearning = Maps(DictLearning_imgs,
    #                           template=template).to_atlas()
    atlas_Ward = Maps(Ward_imgs, template=template).to_atlas()
    atlas_Kmeans = Maps(Kmeans_imgs, template=template).to_atlas()

    array_avg = maps_train_avg.to_array()
    maps_thr = Maps(array_avg > (1-alpha)*3e-6, template=template)
    atlas_mean = maps_thr.to_atlas()
    atlas_null = Maps(np.zeros(array_avg.shape), template=template).to_atlas()

    atlas_CanICA_thr = Maps(CanICA_imgs, template=template,
                            mask=maps_thr.to_img()).to_atlas()
    atlas_Ward_thr = Maps(Ward_imgs, template=template,
                          mask=maps_thr.to_img()).to_atlas()
    atlas_Kmeans_thr = Maps(Kmeans_imgs, template=template,
                            mask=maps_thr.to_img()).to_atlas()

    del array_avg

    atlas_dict = {
        'Harvard Oxford 0': atlas_HO_0,
        'Harvard Oxford 25': atlas_HO_25,
        'Harvard Oxford 50': atlas_HO_50,
        f'CanICA {n_par} components': atlas_CanICA,
        # f'CanICA thresholded {n_par} components': atlas_CanICA_thr,
        # 'Dict Learning {} components'.format(n_par): atlas_DictLearning,
        f'Ward {n_par} components': atlas_Ward,
        # f'Ward thresholded {n_par} components': atlas_Ward_thr,
        f'Kmeans {n_par} components': atlas_Kmeans,
        # f'Kmeans thresholded {n_par} components': atlas_Kmeans_thr,
        'Mean thresholded': atlas_mean,
        'Null': atlas_null,
    }

    criteria = [
        goodness_of_fit,
        variance_criterion,
    ]

    for name, atlas in atlas_dict.items():
        plotting.plot_activity_map(atlas['maps'], title=name)

    benchmarks = benchmark(maps_test, atlas_dict, criteria, verbose=True,
                           alpha=alpha)

    print(benchmarks)
    sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks,
                height=6, kind="bar", palette="muted").set(yscale="log")
    plt.title('Atlas benchmark on \'{}\' keyword'.format(keyword))
    plt.show()
