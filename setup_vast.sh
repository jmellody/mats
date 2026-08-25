#!/usr/bin/env bash
set -euo pipefail

# vast.ai: PyTorch template, >=24GB VRAM for 8B in bf16
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

pip install -q -r requirements.txt
mkdir -p data results

python - <<'EOF'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF

echo "hf auth login   # for gated repos (Llama)"
