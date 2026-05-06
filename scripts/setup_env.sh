#!/bin/bash
# Environment setup script for SoftDroPE project

# Activate the softdrope environment
echo "Activating softdrope environment..."
micromamba activate softdrope

# Check Python version
python --version

# Verify key packages
echo "Verifying packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

echo "Environment ready!"