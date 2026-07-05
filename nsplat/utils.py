import math

import torch


def make_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    lr_min_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR scheduler that decays from 1 to *lr_min_scale* over *num_steps*.

    Returns a ``LambdaLR`` scheduler whose lambda is::

        lr_min_scale + (1 - lr_min_scale) * 0.5 * (1 + cos(pi * step / num_steps))
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_min_scale
        + (1 - lr_min_scale) * 0.5 * (1 + math.cos(math.pi * step / num_steps)),
    )


def clear_memory():
    if torch.cuda.is_available():
        import gc

        torch.cuda.empty_cache()
        gc.collect()


def download_file_once(url: str, path: str):
    import os
    import zipfile
    import urllib.request

    if os.path.exists(path):
        print("File already exists, skipping download.")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print("Downloading", url)
    tmp_path = f"{path}.tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
