#!/bin/bash

#SBATCH --job-name=s2MSE_L1500_comp
#SBATCH --output=slurm_logs/s2MSE_L1500_comp.out
#SBATCH --error=slurm_logs/s2MSE_L1500_comp.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

cd $HOME/sbi-compression
hostname
nvidia-smi --list-gpus

apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
    -B $HOME/sbi-compression:/opt/sbi-compression \
    -B /share/lustre/brianycc \
    sbi_container_hypatia.sif /opt/venv/bin/python \
    examples/compression/s2MSE_L1500_comp.py \
    --data_directory '/share/lustre/brianycc/spherical_maps/map_compression_L1500_mwss_samples_holdout' \
    --output_directory '/share/lustre/brianycc/spherical_maps/map_compression_L1500_mwss_samples_holdout_compressed_final' \
    -lr 1e-3 \
    -tts 0.9 \
    -bs 32 \
    -s 12000 \
    --print_every 100 \
    -L 1500 \
    -N 98304 \
    --n_samples 16384 \
    --normalise_maps \
    -v \
    --save_compressed
