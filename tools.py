"""Implement some tool functions used in other files."""
import numpy as np
import nibabel as nib
import copy
from scipy.ndimage import gaussian_filter
import time
import pickle
import os
import ntpath
from colorama import Fore, Style

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from brain_mapping.globals import mem, input_path, Ni, Nj, Nk, coordinates, affine, \
    inv_affine, corpus_tfidf, gray_mask
from brain_mapping.meta_analysis.meta_analysis import Maps, print_percent


@mem.cache
def build_index(file_path):
    """
    Build decode & encode dictionnary of the given file_name.

    encode : dict
        key : line number
        value : string at the specified line number
    decode : dict (reverse of encode)
        key : string found in the file
        value : number of the line containing the string

    Used for the files pmids.txt & feature_names.txt
    """
    decode = dict(enumerate(line.strip() for line in open(file_path)))
    encode = {v: k for k, v in decode.items()}

    return encode, decode


encode_feature, decode_feature = build_index(input_path+'feature_names.txt')
encode_pmid, decode_pmid = build_index(input_path+'pmids.txt')


def build_activity_map_from_pmid(pmid, sigma=1):
    """
    Given a pmid, build its corresponding activity map.

    pmid : integer found in pmids.txt
    sigma : std used in gaussian blurr
    """
    stat_img_data = np.zeros((Ni, Nj, Nk))  # Blank img with MNI152's shape

    # For each coordinates found in pmid (in mm),
    # compute its corresponding voxels coordinates and note it as activated
    for _, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        ijk = np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int)
        i, j, k = np.minimum(ijk, [Ni-1, Nj-1, Nk-1])
        stat_img_data[i, j, k] += 1

    # Add gaussian blurr
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)


@mem.cache
def build_df_from_keyword(keyword):
    """Build DataFrame from keyword related studies."""
    nz_encoded_pmids = corpus_tfidf[:, encode_feature[keyword]].nonzero()[0]
    nz_pmids = np.array([int(decode_pmid[ind]) for ind in nz_encoded_pmids])
    df = coordinates[coordinates['pmid'].isin(nz_pmids)]
    df['weight'] = 1
    return df


def build_avg_map_corpus(sigma):
    """Build average activity map on full corpus."""
    df = copy.copy(coordinates)
    df['weight'] = 1

    all_maps = Maps(df, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col='pmid',
                    mask=gray_mask.get_data())

    return all_maps.avg().smooth(sigma=sigma, inplace=True)


# _________SAVING_TOOLS_________ #
def filename_from_path(path):
    """Extract filename from path."""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_dump_token(prefix='', tag=None, ext=None):
    """Create a dump token."""
    suffix = str(tag)

    if tag is None:
        suffix = time.strftime("%Y%m%d-%H%M%S")
    if ext is not None:
        suffix += '.'+ext

    return prefix+suffix


def pickle_dump(obj, file_path, save=True, verbose=True):
    """Dump object to a file."""
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        filename = filename_from_path(file_path)
        s = f'File {filename} {Fore.GREEN}dumped succesfully.{Style.RESET_ALL}'
        print_percent(string=s, prefix='Pickle', verbose=verbose)
    return obj


def pickle_load(file_path, verbose=True, load=True):
    """Load object from a file."""
    if not load:
        s = f'{Fore.YELLOW}No file loaded.{Style.RESET_ALL}'
        print_percent(string=s, prefix='Pickle', verbose=verbose)
        return None

    filename = filename_from_path(file_path)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        s = f'File {filename} {Fore.GREEN}loaded succesfully.{Style.RESET_ALL}'
        print_percent(string=s, prefix='Pickle', verbose=verbose)
        return obj
    except OSError:
        s = f'File {filename} does not exist. ' \
            f'{Fore.YELLOW}No file loaded.{Style.RESET_ALL}'
        print_percent(string=s, prefix='Pickle', verbose=verbose)
        return None
    except EOFError:
        s = f'File {filename} is empty. ' \
            f'{Fore.YELLOW}No file loaded.{Style.RESET_ALL}'
        print_percent(string=s, prefix='Pickle', verbose=verbose)
        return None
