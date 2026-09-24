#!/bin/bash

#SBATCH --job-name=NLE_s2VMIM
#SBATCH --output=slurm_logs/NLE_s2VMIM.out
#SBATCH --error=slurm_logs/NLE_s2VMIM.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

hostname
nvidia-smi --list-gpus

apptainer exec --nv --overlay sandbox_overlay.img:ro \
    -B /home/u6pf/brianycc.u6pf/sbi-compression:/opt/sbi-compression \
    -B /scratch:/scratch \
    -B /projects:/projects \
    sbi_container.sif /opt/venv/bin/python \
    examples/NLE_s2.py \
    --compression_types "'s2VMIM_L1500_augpool'" \
    --compression_directories "'/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_augmented_pooled_compressed/s2VMIM_L1500_98304N_9000steps_32batch_0.001lr_4transforms_8bins.pkl'" \
    --n_samples 98304


# --compression_directories "'examples/compressed_data/s2MSE_L1500_98304N_6500steps_32batch_0.001lr.pkl'" \