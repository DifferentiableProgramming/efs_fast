# efs-fast

Fast exhaustive feature selection for ordinary least squares regression.

`efs-fast` evaluates every subset of the input features (up to a chosen size),
scores each subset by k-fold cross-validated MAPE, and returns the best-scoring
subsets. The search is a brute force over all feature combinations; the purpose
of the package is to make that brute force fast enough to be practical. The
inner loop is written in C with AVX2 intrinsics and parallelised with OpenMP.

Up to 30 features are supported. An exhaustive search over 30 features is roughly
a billion combinations and finishes in a few minutes on a modern multi-core CPU.

## What it computes

Given a feature matrix `X` and target `y`, `run_efs` appends an intercept column
and then, for every feature combination up to size `max_k`:

- fits OLS on the training folds and computes MAPE on the held-out fold,
- averages that MAPE over the `k` folds.

It returns the combinations with the lowest average MAPE, sorted ascending. The
intercept is forced into every model. In the results, feature indices are
columns of `X`; the index equal to `n_features` is the intercept. All arithmetic
is single precision, so MAPE values are approximate, but the ranking of
combinations matches a double-precision reference.

## How it works

Most of the work is solving a small least squares problem for each combination
and fold, and evaluating MAPE on the held-out rows. The implementation leans on
a few things to stay fast.

**Gram matrix reuse.** `X^T X` and `X^T y` are accumulated once over the whole data
set and once per fold. The normal equations for any subset are a submatrix of
these, and a fold's training system is the global matrix minus that fold, so the
raw data is never touched again per combination.

**Combinations as bitmasks.** Each subset is a bitmask over the features. The
search enumerates masks and reads off the selected indices with a
count-trailing-zeros loop, visiting only the set bits.

**Batched LDL^T solve.** Each subset's normal equations are solved with an LDL^T
factorisation (no square roots) and a small ridge term on the diagonal for
stability. The `k` folds share the same feature set, so all `k` systems are
solved at once, one per AVX2 lane, in a single vectorised factorisation. The
matrix is read straight from the per-fold Gram arrays through the selected
indices instead of being copied into a dense buffer.

**Vectorised MAPE.** Predictions on the held-out fold are computed eight rows at
a time with FMA. The reciprocal of the per-fold denominator is precomputed, the
all-ones intercept column is folded into the accumulator so it costs no load or
multiply, and rows are processed in groups of four so each coefficient is
broadcast once per group rather than once per row.

**Branch and bound.** When only the top `k` results are requested, the running
cross-validation sum is checked against the worst result kept so far. Once a
combination can no longer make the cut, its remaining folds are skipped. The
result is exact, and on data with real signal this prunes most combinations
early.

**Thread-local heaps.** Each OpenMP thread keeps its own bounded max-heap of the
best combinations. The heaps are merged and sorted once at the end.

## Installation

### Step 1: Install C compiler (skip if already installed)

#### On Linux (GCC):

```bash
sudo apt install build-essential
```

#### On Mac (Clang):

```bash
xcode-select --install
```

```bash
brew install libomp
```

To support Apple Silicon, [simde](https://github.com/simd-everywhere/simde) is used to translate AVX2 instructions into native ARM NEON instructions.

#### On Windows (MSVC) using PowerShell:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

### Step 2: Install package

```bash
pip install git+https://github.com/DifferentiableProgramming/efs_fast.git
```

## Example usage

```python
import numpy as np
import pandas as pd
from efs_fast import run_efs

# Generate example data
samples = 1000
features = 20
rng = np.random.default_rng()
X = rng.standard_normal([samples, features])
y = rng.standard_normal(samples)

max_comb_len = 10
k_folds = 5

efs_results = run_efs(X, y, max_comb_len, k_folds, top_k = 100, random_state=42)
```
