# Building the C Extensions (Windows)

## Prerequisites

```
pip install cython numpy setuptools
```

You also need a C compiler. Options:
- **Visual Studio Build Tools** (recommended) — install "Desktop development with C++"
- **MinGW-w64** — install then set `distutils.cfg`

## Build steps

Open a terminal in this directory (`pmsm_blocks\c_src\`) and run:

```bat
build_all.bat
```

This compiles three Cython extensions and copies the `.pyd` files to `pmsm_blocks\`:

| `.pyd` produced        | Used by                  |
|------------------------|--------------------------|
| `pmsm_motor_wrapper`   | `motor_block.py`         |
| `transforms_wrapper`   | `transform_blocks.py`    |
| `pi_controller_wrapper`| `pi_controller.py`       |

## Verify

```python
from pmsm_blocks import PMSMMotorBlock
m = PMSMMotorBlock("test", use_c_backend=True)
print("C backend OK")
```

## If build fails

Build individually to isolate the problem:

```bat
python setup_pmsm_motor.py build_ext --inplace
python setup_transforms.py build_ext --inplace
python setup_pi_controller.py build_ext --inplace
```

## Then set USE_C_BACKEND = True

In `example_pmsm_foc.py`:

```python
USE_C_BACKEND = True
```
