#!/usr/bin/env bash
# Vast.ai pod -> ready to run.
#
#   cd /workspace && git clone https://github.com/jmellody/mats.git
#   cd mats && bash vast_setup.sh
#
# Assumes a PyTorch template. Does NOT reinstall torch: the template's build is
# matched to the host driver, and reinstalling costs a 2.5GB download and often
# breaks CUDA. If CUDA fails below, the machine's driver is older than the
# template expects -- destroy the instance and rent one with a higher CUDA
# version rather than trying to fix it. Vast machines are individually owned
# and drivers vary, so filter on CUDA version when searching.

set -e

echo "=== hardware ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== cuda check (destroy the pod now if this fails) ==="
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE")
    print("torch built for CUDA:", torch.version.cuda)
    print("The host driver is too old for this template. Destroy this")
    print("instance and rent one with a higher CUDA version.")
    sys.exit(1)
print("gpu     ", torch.cuda.get_device_name(0))
print("vram GB ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
PY

echo
echo "=== deps ==="
pip install -q --root-user-action=ignore \
    transformers accelerate scikit-learn matplotlib python-dotenv

echo
echo "=== hf cache on /workspace ==="
# Default cache sits on the container layer and dies with the pod. /workspace
# persists across restarts, so weights survive and you do not pay twice to
# download them.
export HF_HOME=/workspace/.hf
mkdir -p "$HF_HOME"
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/.hf' >> ~/.bashrc

echo
echo "=== versions ==="
python -c "
import transformers, sklearn, numpy
print('transformers', transformers.__version__)
print('sklearn     ', sklearn.__version__)
print('numpy       ', numpy.__version__)
"

df -h /workspace | tail -1 | awk '{print "\ndisk free on /workspace:", $4}'

cat <<'EOF'

=== ready ===

Run inside tmux so an SSH drop does not kill the job:

  tmux new -s run

  export PROBE_MODEL=Qwen/Qwen3.6-4B-Instruct
  export PROBE_TAG=qwen36-4b

  python smoke.py                        # ~3 min, all checks must pass
  python run_all.py                      # stops at the layer sweep
  python run_all.py --from dose --layer <N>

  Ctrl-B then D detaches.  tmux attach -t run  returns.

Pull results back BEFORE destroying the pod -- activations and figures are
gitignored and the disk goes with the instance:

  git add -f data/*/chosen_layer.json figures/
  git commit -m "results" && git push

Then DESTROY the instance. Vast bills for the instance existing, not for GPU
use, so a forgotten pod costs more than the experiment.
EOF
