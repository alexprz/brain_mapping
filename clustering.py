"""
Implement function used to generate brain parcellations.

Interface of Nilearn parcellations method.
"""
from nilearn.decomposition import DictLearning, CanICA
from nilearn.regions import Parcellations

from globals import pickle_path
from tools import get_dump_token, pickle_dump, pickle_load

save_dir = pickle_path+'generated_atlases/'


def fit_GroupICA(imgs, params):
    """Interface of GroupICA."""
    raise NotImplementedError('')


def fit_CanICA(imgs, params):
    """Interface of CanICA."""
    defaults = {
        'n_components':  10,
        'memory': "nilearn_cache",
        'memory_level': 2,
        'threshold': 3.,
        'n_init': 1,
        'verbose': 1,
        'mask_strategy': 'template',
        'n_jobs': -2
    }
    context = dict(defaults, **params)

    return CanICA(**context).fit(imgs)


def fit_DictLearning(imgs, params):
    """Interface of DictLearning."""
    defaults = {
        'n_components': 10,
        'memory': "nilearn_cache",
        'memory_level': 2,
        'verbose': 1,
        'random_state': 0,
        'n_epochs': 1,
        'mask_strategy': 'template',
        'n_jobs': -2
    }
    context = dict(defaults, **params)

    return DictLearning(**context).fit(imgs)


def fit_Kmeans(imgs, params):
    """Interface of Kmeans."""
    defaults = {
        'method': 'kmeans',
        'n_parcels': 10,
        'standardize': True,
        'smoothing_fwhm': 10.,
        'memory': 'nilearn_cache',
        'memory_level': 1,
        'verbose': 1,
        'n_jobs': -2
    }
    context = dict(defaults, **params)

    return Parcellations(**context).fit(imgs)


def fit_Wards(imgs, params):
    """Interface of Wards."""
    defaults = {
        'method': 'ward',
        'n_parcels': 10,
        'standardize': False,
        'smoothing_fwhm': 2.,
        'memory': 'nilearn_cache',
        'memory_level': 1,
        'verbose': 1,
        'n_jobs': -2
    }
    context = dict(defaults, **params)

    return Parcellations(**context).fit(imgs)


def fit_Model(Model, imgs, params=dict(), load=True, save=True, tag=None):
    """Fit a model to images.

    Args:
        Model(function): One of the interface funciton of clustering.py file.
        imgs(nibabel.Nifti1Image): 4D image on which to fit the model.
        params(dict): Additional parameters passed to each parcellation
            method. Default arguments are replaced by newer ones.
        load(bool): Whether to load the previous results of a fit.
        save(bool): Whether to save the results of the fit.
        tag(string): Suffix to add to the saved file.

    Returns:
        Fitted model. Same return type as given Model functions.

    """
    filepath = save_dir+get_dump_token(Model.__name__+'_', tag=tag)

    model = pickle_load(filepath, load=load)
    if model is not None:
        return model

    model = Model(imgs, params)
    return pickle_dump(model.fit(imgs), filepath, save=save)
