#!/bin/bash
cd /mlep
conda env list
conda env create --file env.yml
conda env list
echo "new env created"
source /etc/profile.d/conda.sh
conda activate mlep-dev
activate mlep-dev
conda env list
pip install dist/demo_mlep_package-3.0-py3-none-any.whl
python -m demo_mlep_package.main
