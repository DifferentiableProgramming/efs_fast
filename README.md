# efs-fast

Fast Exhaustive Feature Selection with Linear Regression using AVX2 optimized C extension.

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

On Windows (MSVC):

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

### Step 2: Install package

```bash
pip install git+https://github.com/DifferentiableProgramming/efs_fast.git
```
