#!/bin/bash

#SBATCH --job-name=s2VMIM_L1500_comp
#SBATCH --output=slurm_logs/s2VMIM_L1500_comp.out
#SBATCH --error=slurm_logs/s2VMIM_L1500_comp.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

hostname
nvidia-smi --list-gpus

apptainer exec --nv --overlay sandbox_overlay.img:ro \
    -B /home/u6pf/brianycc.u6pf/sbi-compression:/opt/sbi-compression \
    -B /scratch:/scratch \
    -B /projects:/projects \
    sbi_container.sif /opt/venv/bin/python \
    examples/compression/s2VMIM_L1500_comp.py \
    --data_directory '/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_holdout' \
    --output_directory '/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_holdout_compressed_final' \
    -lr 1e-3 \
    -tts 0.9 \
    -bs 32 \
    -s 12000 \
    --print_every 100 \
    -L 1500 \
    -N 98304 \
    --n_samples 16384 \
    --n_transforms 4 \
    --n_bins 8 \
    --conditioner_dimensions '((32,32),(32,32))' \
    --normalise_maps \
    -v \
    --dropout \
    --save_compressed
