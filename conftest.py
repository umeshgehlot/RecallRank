import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # noqa: E402, F401 — load torch's OMP first to avoid faiss conflict
