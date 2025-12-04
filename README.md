# efs-fast

Fast Exhaustive Feature Selection with Linear Regression, with heavy lifting done in C using AVX2 intrinsics.

## Installation

### Step 1: Install C compiler (skip if already installed)

On Linux (GCC):

```bash
sudo apt install build-essential
```

On Mac (Clang):

```bash
xcode-select --install
```

On Windows (MSVC) using PowerShell:

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
