"""Thin read-only access to FIPER's processed rollout caches for the signature study.

Nothing here mutates the repo's own modules; it only imports ProcessedRolloutDataset
and re-uses the already-built data/<task>/processed_rollouts caches.
"""
import os
import sys

ROOT = "/home/louis/fiper"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from datasets.rollout_datasets import ProcessedRolloutDataset

BASE_DATA = os.path.join(ROOT, "data")
BASE_CFG = os.path.join(ROOT, "configs")

TASKS = ["sorting", "stacking", "push_t", "pretzel", "push_chair"]


def load_task(task, required_tensors=("action_preds",)):
    ds = ProcessedRolloutDataset(
        task_data_path=os.path.join(BASE_DATA, task),
        base_config_path=BASE_CFG,
        required_tensors=list(required_tensors),
    )
    ds.load_dataset()
    assert ds.dataset_loaded, f"no processed cache for {task}"
    # load_dataset() stores tensors under "<name>.pt"; the rest of the codebase
    # expects the bare name. Alias both so ds methods keep working.
    for k in list(ds.data.keys()):
        if k.endswith(".pt"):
            ds.data[k[:-3]] = ds.data.pop(k)
    return ds


def action_slice(ds, actions=("position",)):
    """Indices of the requested named action groups inside the A-dim action vector."""
    return ds._get_action_slices(required_actions=list(actions), optional_actions=[])


def episodes(ds, subset, actions=("position",)):
    """Yield (action_preds [T,B,H,a], successful:bool) per episode of `subset`."""
    idx = action_slice(ds, actions)
    starts, ends, labels = ds._filter_start_end_episode_indices(subset=subset, subsubset="all")
    ap = ds.data["action_preds"]
    for i in range(len(starts)):
        s, e = starts[i].item(), ends[i].item()
        x = ap[s:e]
        if x.ndim == 5:  # (T, robots, B, H, A) — squeeze robot axis if singleton
            x = x[:, 0] if x.shape[1] == 1 else x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        x = x[..., idx]
        yield np.asarray(x, dtype=np.float64), bool(labels[i].item())
