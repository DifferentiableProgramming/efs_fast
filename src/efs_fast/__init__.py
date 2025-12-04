import ctypes
import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from math import comb

_curr_dir = os.path.dirname(os.path.abspath(__file__))

_lib_files = glob.glob(os.path.join(_curr_dir, "efs_lib*"))
if not _lib_files:
    raise FileNotFoundError("Could not find the compiled efs_lib extension.")
_lib_path = _lib_files[0]

_linreg_lib = ctypes.CDLL(_lib_path)

_linreg_lib.exhaustive_feature_selection.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),  # X
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # y
    ctypes.c_int,  # n
    ctypes.c_int,  # m
    ctypes.c_int,  # k_max
    ctypes.c_int,  # k_folds
    np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"
    ),  # fold_indices
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # out_mapes
    np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"
    ),  # out_combinations
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # out_lens
    ctypes.c_int,  # top_k
]
_linreg_lib.exhaustive_feature_selection.restype = ctypes.c_int


def run_efs(X, y, max_k, k_folds=5, top_k=100, random_state=42):
    if X.shape[1] > 30:
        raise ValueError(
            "Feature matrix contains more than 30 features which is currently not supported, aborting."
        )
    X = sm.add_constant(X, prepend=False)
    X_32 = np.ascontiguousarray(X, dtype=np.float32)
    y_32 = np.ascontiguousarray(y, dtype=np.float32)

    n, m = X_32.shape

    alloc_size = sum(comb(m - 1, i) for i in range(max_k))
    if top_k <= 0:
        print(f"Processing {alloc_size} combinations and returning all of them...")
    else:
        print(
            f"Processing {alloc_size} combinations and returning top {top_k} of them..."
        )
        alloc_size = top_k

    out_mapes = np.zeros(alloc_size, dtype=np.float32)
    out_combinations = np.zeros((alloc_size, m - 1), dtype=np.int32)
    out_lens = np.zeros(alloc_size, dtype=np.int32)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_indices = np.zeros(n, dtype=np.int32)
    for i, (_, test_idx) in enumerate(kf.split(X_32)):
        fold_indices[test_idx] = i

    count = _linreg_lib.exhaustive_feature_selection(
        X_32,
        y_32,
        n,
        m,
        max_k,
        k_folds,
        fold_indices,
        out_mapes,
        out_combinations,
        out_lens,
        top_k,
    )

    print(f"Computation finished. Returned {count} combinations.")

    valid_mapes = out_mapes[:count]
    valid_lens = out_lens[:count]
    valid_combos = out_combinations[:count]

    combo_list = [tuple(valid_combos[i, : valid_lens[i]]) for i in range(count)]

    df = pd.DataFrame({"combination": combo_list, "mape": valid_mapes})

    return df.sort_values("mape")
