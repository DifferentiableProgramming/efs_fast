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
    """
    Perform Fast Exhaustive Feature Selection using Linear Regression (OLS) and K-Fold Cross-Validation.

    This function searches through all feature combinations up to size `max_k` to find the
    subsets that minimize the Mean Absolute Percentage Error (MAPE). It uses an optimized
    C extension with AVX2 instructions and OpenMP parallelism to accelerate the process.

    An intercept (constant) term is automatically added to the feature matrix and is
    forced into every model combination.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature matrix. Must not contain more than 30 features.
        Data will be converted to float32 (single precision) internally.

    y : array-like of shape (n_samples,)
        The target values (dependent variable).
        Data will be converted to float32 internally.

    max_k : int
        The maximum number of features to select (excluding the intercept).
        The search will evaluate all combinations of size 0 to `max_k`.

    k_folds : int, optional (default=5)
        The number of folds to use for K-Fold Cross-Validation.

    top_k : int, optional (default=100)
        The number of best performing feature combinations to return.
        If set to <= 0, all evaluated combinations are returned.

    random_state : int, optional (default=42)
        Seed used by the random number generator for K-Fold splitting.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sorted results. The columns are:
        - `combination`: A tuple of integers representing the column indices of the features
          included in the model. The index of the added intercept term (which is `n_features`)
          will be included in these tuples.
        - `mape`: The average Mean Absolute Percentage Error across the K folds.

        The DataFrame is sorted by `mape` in ascending order.

    Raises
    ------
    ValueError
        If `X` contains more than 30 features.

    Notes
    -----
    - The metric used for evaluation is MAPE (Mean Absolute Percentage Error).
    - To handle division by zero in MAPE, the denominator is clipped to a minimum magnitude of 1e-8.
    - Since an intercept is appended as the last column, indices in the output `combination` tuple
      ranging from `0` to `n_features-1` correspond to the input columns of `X`. The index
      `n_features` corresponds to the intercept.
    """

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
