#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _set_model_cache(cache_dir: Path) -> None:
    resolved = cache_dir.expanduser().resolve()
    os.environ["MODEL_CACHE_DIR"] = str(resolved)
    os.environ["TORCH_HOME"] = str(resolved)
    (resolved / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)


def _download_torchvision_weights() -> None:
    from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

    print("Downloading torchvision VGG16 weights...")
    vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

    print("Downloading torchvision AlexNet weights...")
    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).eval()


def _download_lpips_weights(include_squeeze: bool) -> None:
    import lpips

    lpips_nets = ["vgg", "alex"]
    if include_squeeze:
        lpips_nets.append("squeeze")

    for net in lpips_nets:
        print(f"Downloading LPIPS {net} weights...")
        lpips.LPIPS(net=net, verbose=False).eval()


def _download_dists_weights() -> None:
    import piq

    print("Downloading PIQ DISTS weights...")
    piq.DISTS().eval()


def _list_cached_files(cache_dir: Path) -> None:
    checkpoints_dir = cache_dir.expanduser().resolve() / "hub" / "checkpoints"
    print(f"Model cache: {cache_dir.expanduser().resolve()}")
    if not checkpoints_dir.exists():
        print("No checkpoints directory found.")
        return

    cached_files = sorted(path.name for path in checkpoints_dir.iterdir() if path.is_file())
    if not cached_files:
        print("No cached checkpoint files found.")
        return

    print("Cached checkpoint files:")
    for filename in cached_files:
        print(f"- {filename}")


def _print_dry_run(args: argparse.Namespace, cache_dir: Path) -> None:
    checkpoints_dir = cache_dir.expanduser().resolve() / "hub" / "checkpoints"
    print(f"Model cache: {cache_dir.expanduser().resolve()}")
    print(f"Checkpoint cache: {checkpoints_dir}")
    print("Would set environment variables:")
    print(f"- MODEL_CACHE_DIR={cache_dir.expanduser().resolve()}")
    print(f"- TORCH_HOME={cache_dir.expanduser().resolve()}")

    if args.skip_torchvision:
        print("Would skip torchvision VGG16/AlexNet weights.")
    else:
        print("Would download torchvision weights:")
        print("- vgg16-397923af.pth")
        print("- alexnet-owt-7be5be79.pth")

    if args.skip_lpips:
        print("Would skip LPIPS calibrated weights.")
    else:
        print("Would initialize LPIPS nets:")
        print("- vgg")
        print("- alex")
        if not args.skip_lpips_squeeze:
            print("- squeeze")

    if args.skip_dists:
        print("Would skip PIQ DISTS weights.")
    else:
        print("Would initialize piq.DISTS() to download dists_weights.pt.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights required by PMS CPU/GPU workers.")
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("MODEL_CACHE_DIR", "models"),
        help="Torch model cache directory. Defaults to MODEL_CACHE_DIR or ./models.",
    )
    parser.add_argument(
        "--skip-torchvision",
        action="store_true",
        help="Skip torchvision VGG16/AlexNet ImageNet weights.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS calibrated weights.",
    )
    parser.add_argument(
        "--skip-lpips-squeeze",
        action="store_true",
        help="Skip LPIPS SqueezeNet weights. The API accepts this net, so it is downloaded by default.",
    )
    parser.add_argument(
        "--skip-dists",
        action="store_true",
        help="Skip PIQ DISTS weights, including dists_weights.pt.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list currently cached checkpoint files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without importing model libraries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    _set_model_cache(cache_dir)

    if args.dry_run:
        _print_dry_run(args, cache_dir)
        return

    if args.list_only:
        _list_cached_files(cache_dir)
        return

    if not args.skip_torchvision:
        _download_torchvision_weights()

    if not args.skip_lpips:
        _download_lpips_weights(include_squeeze=not args.skip_lpips_squeeze)

    if not args.skip_dists:
        _download_dists_weights()

    _list_cached_files(cache_dir)


if __name__ == "__main__":
    main()
