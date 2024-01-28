```python
# Kernel for Jupyter Notebook

import sys

!echo {sys.executable}
# The output should be local to your directory.
# /Users/alextanhongpin/Documents/python/matrix-factorization/.venv/bin/python

```

    /Users/alextanhongpin/Documents/python/matrix-factorization/.venv/bin/python



```python
# Otherwise, re-run this and restart the jupyter server until you got the correct path.
!poetry run python -m pip install ipykernel
!poetry run python -m ipykernel install --user
```
