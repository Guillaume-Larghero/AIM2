#!/bin/bash
#SBATCH --job-name=mimic_download
#SBATCH --output=download_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"
cd "${SCRIPT_DIR}"

wget -r -N -c -np -nH --cut-dirs=1 \
  --user "${PHYSIONET_USERNAME}" \
  --password "${PHYSIONET_PASSWORD}" \
  -i FILES \
  --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
