"""Configuration. One place, so scaling does not silently desync.

Right now LAYER, C, and MODEL are duplicated across six scripts. That is a
real hazard when you move to GPU and start varying models -- change it in
five files and forget the sixth, and you get results that look fine and are
wrong. Everything imports from here instead.
"""

import os

import torch

# --- model ------------------------------------------------------------------
MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# Layers where expertise was decodable for Qwen2.5-3B. Re-sweep per model --
# this is not transferable across architectures or sizes.
LAYER = int(os.environ.get("PROBE_LAYER", 27))

# --- probe ------------------------------------------------------------------
# C=0.01 chosen empirically: Cohen's d 0.83 with no margin saturation.
# Higher C saturates decision_function and destroys graded signal.
C = 0.01

# --- hardware ---------------------------------------------------------------
def device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype():
    # bf16 on GPU is fine for forward-only activation extraction and halves
    # memory. On CPU it is slow or unsupported depending on the torch build.
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def batch_size():
    if not torch.cuda.is_available():
        return 1  # CPU gains nothing from batching at this size
    free = torch.cuda.get_device_properties(0).total_memory / 1e9
    if free > 70:
        return 32
    if free > 35:
        return 16
    return 8


def setup():
    """Call once at the start of any script."""
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)
    torch.set_grad_enabled(False)
    return device()


# --- conversation scaffolding -----------------------------------------------
# Shared across all experiments so that probe training and every experiment
# read at an identical final position. Mismatched scaffolding was the bug that
# put the first experiment out of distribution.
ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"
