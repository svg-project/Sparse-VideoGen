import os

import torch


def save_qkv(q, k, v, save_dir, layer_idx, timestep):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sparse_qkv_layer{layer_idx}_ts{timestep}.pt")
    torch.save(
        {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "timestep": timestep,
            "layer_idx": layer_idx,
        },
        filename,
    )


def save_qkvx(q, k, v, x, save_dir, layer_idx, timestep):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sparse_qkvx_layer{layer_idx}_ts{timestep}.pt")
    torch.save(
        {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "x": x.detach().cpu(),
            "timestep": timestep,
            "layer_idx": layer_idx,
        },
        filename,
    )
