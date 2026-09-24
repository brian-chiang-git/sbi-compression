#!/usr/bin/env python3
"""Augment map datasets by generating multiple noisy realizations per map.

Supports single or multiple input map directories. When multiple directories
are provided, maps are pooled in Sobol order into a single unified output dataset.

Usage:
    python /home/u6pf/brianycc.u6pf/sbi-compression/examples/augment_compression_data.py \
    --input-dirs \
        /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_5_tomo_bins_flm_1500_batch_0 \
        /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_5_tomo_bins_flm_1500_batch_1 \
        /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_5_tomo_bins_flm_1500_batch_2 \
        /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_5_tomo_bins_flm_1500_batch_3 \
    --output-dir /projects/u6pf/brianycc/map_compression_samples_augmented_pooled \
    --num-augmentations 3 \
    --policy resample

    python /home/u6pf/brianycc.u6pf/sbi-compression/examples/augment_compression_data.py \
    --input-dirs /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_* \
    --output-dir /projects/u6pf/kiyam/spherical_inference/sim_output/map_compression_samples_augmented_pooled \
    --num-augmentations 5 \
    --policy resample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

# Ensure differentiable_spherical package/scripts are in sys.path
default_pkg_dir = "/projects/u6pf/kiyam/spherical_inference/differentiable_spherical"
if os.path.exists(default_pkg_dir) and default_pkg_dir not in sys.path:
    sys.path.append(default_pkg_dir)

import scripts.compression_data as cd


def _atomic_write_npy(path: str, arr: np.ndarray) -> None:
    """Save numpy array via temporary file to prevent truncated writes."""
    base, ext = os.path.splitext(os.path.basename(path))
    tmp = os.path.join(os.path.dirname(path), f".{base}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description="Augment map dataset(s) with multiple noisy realizations per map."
    )
    parser.add_argument(
        "--input-dirs",
        "--input-dir",
        dest="input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more input map block directories (e.g. .../batch_0 .../batch_1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save augmented noisy maps and parameters",
    )
    parser.add_argument(
        "--num-augmentations",
        "-n",
        type=int,
        default=5,
        help="Number of noisy realizations to generate per map (default: 5)",
    )
    parser.add_argument(
        "--policy",
        choices=cd.NOISE_POLICIES,
        default="resample",
        help="Noise policy: 'resample' (fresh noise), 'fixed' (reproducible per seed/index), 'none'",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Base noise seed for 'fixed' noise policy or RNG initialization (default: 0)",
    )
    parser.add_argument(
        "--roll",
        action="store_true",
        help="Apply random z-axis rotation (phi roll) as additional augmentation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing maps in the output directory if present",
    )

    args = parser.parse_args()

    input_paths = [os.path.abspath(p) for p in args.input_dirs]
    output_dir = os.path.abspath(args.output_dir)

    print(f"Loading {len(input_paths)} input block(s)...")
    blocks = [cd.load_block(p, derive_sigma=True) for p in input_paths]

    # Sort blocks by Sobol offset to ensure contiguous ordering
    blocks = sorted(blocks, key=lambda b: b.sobol_offset)
    
    # Pool parameters and create mapping: global index -> (block, local_map_index)
    pooled_params, lookup = cd.pool(blocks)
    total_input_maps = len(lookup)

    n_aug = args.num_augmentations
    total_output_maps = total_input_maps * n_aug

    print(f"\nSummary of Input Blocks:")
    for b in blocks:
        print(f"  - {b.path} (offset: {b.sobol_offset}, maps: {len(b)})")
    print(f"Total pooled input maps: {total_input_maps}")
    print(f"Augmentations per map: {n_aug}")
    print(f"Total output maps to write: {total_output_maps}")
    print(f"Noise policy: {args.policy} | Roll augmentation: {args.roll}")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare augmented parameters array (repeat each pooled parameter set n_aug times)
    augmented_params = np.repeat(pooled_params, n_aug, axis=0)
    params_out_path = os.path.join(output_dir, "params.npy")
    _atomic_write_npy(params_out_path, augmented_params)
    print(f"\nSaved augmented params.npy shape: {augmented_params.shape}")

    # Save sigma_pix.npy from the first block (verify consistency across blocks)
    ref_sigma = blocks[0].sigma_pix
    for b in blocks[1:]:
        if not np.allclose(b.sigma_pix, ref_sigma):
            raise ValueError(f"sigma_pix in {b.path} does not match reference block {blocks[0].path}")

    sigma_out_path = os.path.join(output_dir, "sigma_pix.npy")
    _atomic_write_npy(sigma_out_path, ref_sigma)
    print(f"Saved sigma_pix.npy shape: {ref_sigma.shape}")

    # Write meta.json
    first_meta = dict(blocks[0].meta) if blocks[0].meta else {}
    out_meta = {
        **first_meta,
        "noise": "baked" if args.policy != "none" else "none",
        "save_dtype": "float32",
        "sobol_offset": blocks[0].sobol_offset * n_aug,
        "num_maps": total_output_maps,
        "map_shape": list(blocks[0].map_shape),
        "n_pix_flat": int(np.prod(blocks[0].map_shape)),
        "param_names": list(blocks[0].param_names),
        "augmentation": {
            "source_dirs": input_paths,
            "num_augmentations": n_aug,
            "policy": args.policy,
            "noise_seed": args.noise_seed,
            "roll": args.roll,
        }
    }

    meta_out_path = os.path.join(output_dir, "meta.json")
    with open(meta_out_path, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Saved meta.json")

    # RNG stream for resample policy
    rng = np.random.default_rng(args.noise_seed)

    start_time = time.time()
    written_count = 0
    skipped_count = 0

    print("\nProcessing map augmentations...")
    for pooled_idx, (blk, local_idx) in enumerate(lookup):
        for aug_j in range(n_aug):
            out_idx = pooled_idx * n_aug + aug_j
            out_map_path = os.path.join(output_dir, f"map_{out_idx}.npy")

            if os.path.exists(out_map_path) and not args.overwrite:
                skipped_count += 1
                continue

            noisy_map = cd.load_map(
                blk,
                local_idx,
                policy=args.policy,
                rng=rng if args.policy == "resample" else None,
                noise_seed=args.noise_seed + aug_j,
                roll=args.roll,
                dtype=np.float32,
            )

            _atomic_write_npy(out_map_path, noisy_map)
            written_count += 1

        if (pooled_idx + 1) % 50 == 0 or (pooled_idx + 1) == total_input_maps:
            elapsed = time.time() - start_time
            processed_outputs = (pooled_idx + 1) * n_aug
            rate = processed_outputs / max(elapsed, 1e-5)
            remaining_outputs = total_output_maps - processed_outputs
            eta_str = f"{remaining_outputs / rate / 60:.1f} min" if rate > 0 else "N/A"
            print(
                f"  Processed pooled map {pooled_idx + 1}/{total_input_maps} "
                f"({processed_outputs}/{total_output_maps} outputs) "
                f"[{rate:.1f} maps/s, ETA: {eta_str}]",
                flush=True,
            )

    print(
        f"\nDone! Written: {written_count}, Skipped: {skipped_count}. "
        f"Total time: {(time.time() - start_time) / 60:.2f} min"
    )
    print(f"Augmented dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
