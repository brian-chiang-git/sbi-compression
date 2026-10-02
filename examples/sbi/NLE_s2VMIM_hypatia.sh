#!/bin/bash

#SBATCH --job-name=NLE_s2VMIM
#SBATCH --output=slurm_logs/NLE_s2VMIM.out
#SBATCH --error=slurm_logs/NLE_s2VMIM.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

cd $HOME/sbi-compression
hostname
nvidia-smi --list-gpus

apptainer exec --userns --nv --overlay sandbox_overlay_hypatia.img:ro \
    -B $HOME/sbi-compression:/opt/sbi-compression \
    -B /share/lustre/brianycc \
    sbi_container_hypatia.sif /opt/venv/bin/python \
    examples/sbi/NLE_s2.py \
    --compression_types "'s2VMIM_L1500_augpool'" \
    --compression_directories "'/share/lustre/brianycc/spherical_maps/map_compression_L1500_mwss_samples_augmented_pooled_compressed/s2VMIM_L1500_98304N_9000steps_32batch_0.001lr_4transforms_8bins.pkl'" \
    --n_samples 98304


# --compression_directories "'examples/compressed_data/s2MSE_L1500_98304N_6500steps_32batch_0.001lr.pkl'" \