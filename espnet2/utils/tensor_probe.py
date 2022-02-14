#!/usr/bin/env python3

import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime


def imshow_tensor(t: torch.Tensor, output_path='./probe', figsize=None, **kwargs):
    output_path = Path(output_path)
    output_path = output_path / datetime.now().strftime('%Y-%m-%d')
    output_path.mkdir(parents=True, exist_ok=True)
    occupied = [int(item.stem) for item in output_path.glob('*.png')]
    if len(occupied) == 0:
        name_id = 1
    else:
        name_id = max(occupied) + 1
    name = f'{name_id}.png'

    if isinstance(t, torch.Tensor):
        t = t.data.cpu().numpy()
    elif isinstance(t, np.ndarray):
        pass
    else:
        t = np.array(t)

    plt.figure(figsize=figsize)
    plt.imshow(t)
    plt.savefig((output_path / name).as_posix())