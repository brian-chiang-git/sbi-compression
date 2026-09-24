#!/bin/bash

#SBATCH --job-name=s2VMIM_L1500_bs32
#SBATCH --output=slurm_logs/s2VMIM_L1500_bs32.out
#SBATCH --error=slurm_logs/s2VMIM_L1500_bs32.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00

# Clear previous output
# rm -rf slurm_logs/s2VMIM_L1500_bs64.*

hostname
nvidia-smi --list-gpus

apptainer exec --nv --overlay sandbox_overlay.img:ro \
    -B /home/u6pf/brianycc.u6pf/sbi-compression:/opt/sbi-compression \
    -B /scratch:/scratch \
    -B /projects:/projects \
    sbi_container.sif /opt/venv/bin/python \
    examples/s2VMIM_L1500.py \
    -lr 1e-3 \
    -tts 0.9 \
    -bs 32 \
    -s 12000 \
    --print_every 100 \
    -L 1500 \
    -N 98304 \
    --n_transforms 4 \
    --n_bins 8 \
    --conditioner_dimensions '((32,32),(32,32))' \
    --normalise_params \
    --normalise_maps \
    -v \
    --dropout \
    --save_model