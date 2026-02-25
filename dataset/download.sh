#!/bin/bash
#SBATCH --job-name=mimic_download
#SBATCH --output=download_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH -p priority
#SBATCH -c 1

# get current directory
SCRIPT_DIR="/n/groups/training/bmif203/AIM2/dataset"
source "${SCRIPT_DIR}/.env_roshan_private"
cd "${SCRIPT_DIR}"

wget -r -N -c -np --user "${PHYSIONET_USERNAME}" --password "${PHYSIONET_PASSWORD}" https://physionet.org/files/mimic-cxr-jpg/2.1.0/