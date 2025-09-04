import argparse
import json
import os
from typing import Optional

import imageio
import lpips
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torchvision import transforms
from tqdm import trange

lpips_model = lpips.LPIPS(net="vgg").cuda()


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_video(video_path):
    """
    Load a video and return a PyTorch tensor using imageio (without cv2)

    Parameters:
        video_path (str): Path to the video file

    Returns:
        video_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)
    """
    reader = imageio.get_reader(video_path)
    frames = []
    to_tensor = transforms.ToTensor()

    for frame in reader:
        # imageio gives RGB frames by default
        frame_tensor = to_tensor(frame)  # (C, H, W)
        frames.append(frame_tensor)

    reader.close()

    if not frames:
        raise ValueError(f"Fail to load frames from {video_path}.")

    video_tensor = torch.stack(frames)  # (T, C, H, W)
    return video_tensor


def compute_quantization_error(video1_tensor, video2_tensor):
    """
    Calculate MSE, PSNR, and SSIM between two videos using PyTorch.

    Parameters:
        video1_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)
        video2_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)

    Returns:
        dict: A dictionary containing 'MSE', 'PSNR', and 'SSIM' values.
    """
    # Ensure the two videos have the same shape
    assert (
        video1_tensor.shape == video2_tensor.shape
    ), f"Videos must have the same shape. {video1_tensor.shape} != {video2_tensor.shape}"
    num_frames, channels, height, width = video1_tensor.shape

    # MSE and PSNR
    mse_values = []
    psnr_values = []

    # SSIM
    ssim_values = []

    # LPIPS
    lpips_values = []

    for i in trange(num_frames):
        frame1 = video1_tensor[i].unsqueeze(0)  # Add batch dimension
        frame2 = video2_tensor[i].unsqueeze(0)

        # Calculate MSE
        mse = mse_loss(frame1, frame2, reduction="mean")
        mse_values.append(mse.item())

        # Calculate PSNR
        max_pixel_value = 1.0  # Assuming input tensors are normalized to [0, 1]
        psnr = 10 * torch.log10(max_pixel_value**2 / mse)
        psnr_values.append(psnr.item())

        # Structural Similarity Index Measure (SSIM)
        def calculate_ssim(img1, img2):
            """
            Compute SSIM for a single frame using PyTorch.
            """
            C1 = 0.01**2
            C2 = 0.03**2

            # Compute means
            mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
            mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

            # Compute variances and covariances
            sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1**2
            sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2**2
            sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2

            # SSIM calculation
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            return ssim_map.mean()

        # Compute SSIM for each frame and average across frames
        num_frames = video1_tensor.shape[0]
        ssim_values = [calculate_ssim(video1_tensor[i : i + 1], video2_tensor[i : i + 1]) for i in range(num_frames)]

        # Calculate LPIPS
        lpips_value = lpips_model(frame1, frame2)
        lpips_values.append(lpips_value.item())

    metrics = {
        "MSE": sum(mse_values) / len(mse_values),
        "PSNR": sum(psnr_values) / len(psnr_values),
        "SSIM": sum(ssim_values).item() / len(ssim_values),
        "LPIPS": sum(lpips_values) / len(lpips_values),
    }
    # Return results as a dictionary
    return metrics


def encode_video_with_vae(video_tensor, vae_model):
    """
    Use VAE to encode videos

    paras:
        video_tensor (torch.Tensor)
        vae_model (torch.nn.Module)

    return:
        encoded_video (torch.Tensor)
    """
    vae_model.eval()
    with torch.no_grad():
        # Treat video tensor as batch for encoding
        encoded_video = vae_model.encode(video_tensor)
    return encoded_video


def compute_quantization_error_after_vae(video1_tensor, video2_tensor, vae_model):
    """
    Compute MSE and PSNR with VAE encoding

    paras:
        video1_tensor (torch.Tensor)
        video2_tensor (torch.Tensor)
        vae_model (torch.nn.Module)

    return:
        average_mse (float)
        psnr (float)
    """
    # Encode both videos with VAE
    encoded_video1 = retrieve_latents(encode_video_with_vae(video1_tensor, vae_model), sample_mode="argmax")
    encoded_video2 = retrieve_latents(encode_video_with_vae(video2_tensor, vae_model), sample_mode="argmax")

    mse = torch.mean((encoded_video1 - encoded_video2) ** 2)
    # Calculate peak signal-to-noise ratio
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # Assume encoded values are in [0, 1]

    return mse.item(), psnr.item()


if __name__ == "__main__":

    # Use argparse to parse arguments
    parser = argparse.ArgumentParser(description="Compute video loss metrics.")
    parser.add_argument("--video1_path", "--v1", type=str, required=True, help="Path to the first video.")
    parser.add_argument("--video2_path", "--v2", type=str, required=True, help="Path to the second video.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the VAE model.")
    parser.add_argument("--prompt_idx", type=int, default=None, help="Start index of the video.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    video1_path = args.video1_path
    video2_path = args.video2_path
    # Load videos
    video1_tensor = load_video(video1_path)
    video2_tensor = load_video(video2_path)

    video1_tensor, video2_tensor = video1_tensor.cuda(), video2_tensor.cuda()

    print(f"video tensor shape: {video1_tensor.shape}")

    # Calculate direct comparison error
    metrics = compute_quantization_error(video1_tensor, video2_tensor)
    print("Average video tensor MSE: {mse}".format(mse=metrics["MSE"]))
    print("Average video tensor PSNR: {psnr} dB".format(psnr=metrics["PSNR"]))
    print("Average video tensor SSIM: {ssim}".format(ssim=metrics["SSIM"]))
    print("Average video tensor LPIPS: {lpips}".format(lpips=metrics["LPIPS"]))

    # Update idx and seed
    if args.prompt_idx is not None:
        metrics["idx"] = args.prompt_idx
    if args.seed is not None:
        metrics["seed"] = args.seed

    # Output to the jsonl file.
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        assert args.output_path.endswith(".jsonl"), "Output path must end with .jsonl"
        with open(args.output_path, "a") as f:
            json.dump(metrics, f)
            f.write("\n")
