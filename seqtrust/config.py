"""Central configuration for SeqTrust.

All absolute paths, pretrained weight locations and default model /
inference hyperparameters used internally by SeqTrust are defined here.

Usage
-----
1. Edit this file directly to adapt SeqTrust to your environment.

2. Or override values at runtime before calling analysis methods::

    from seqtrust import config

    config.MODEL_PRETRAINED_MAP["alphagenome"] = "/path/to/weights"
    config.DEFAULT_LENGTH = 500

Values are read at call time, so both ways take effect without
restarting anything (external model directories take effect before the
first ``add_seq2func_*`` call).
"""

from pathlib import Path

# ============================================================
# Bundled data files (shipped inside the package)
# ============================================================

_TOOLS_DIR = Path(__file__).resolve().parent / "tools"

# Default MEME motif file used by motif scanning.
DEFAULT_MEME_PATH = _TOOLS_DIR / "H14CORE_meme_format.meme"

# Default motif cluster annotation file.
DEFAULT_CLUSTER_PATH = _TOOLS_DIR / "motif_clusters.tsv"

# ============================================================
# External model source directories
# ============================================================
# Third-party model definitions that are imported lazily (only when
# model_type="sei" is actually used).

# Directory containing sei.py (Sei model definition).
SEI_MODEL_DIR = "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Sei/zenodo/sei_model"

# ============================================================
# Pretrained model weights
# ============================================================
# Used when pretrained_path="default" in add_seq2func_* methods.

MODEL_PRETRAINED_MAP = {
    "enformer": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/EleutherAI/enformer-official-rough",
    "borzoi": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/johahi/borzoi-replicate-0",
    "alphagenome": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/gtca/alphagenome_pytorch/model_all_folds.safetensors",
    "borzoi_transformer": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/johahi/borzoi-replicate-0",
    "sei": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Sei/zenodo/sei_model/model/sei.pth",
}

# ============================================================
# Seq2Function model / inference defaults
# ============================================================
# Function-level arguments left as None fall back to these values.

# Sequence length fed into the model.
DEFAULT_LENGTH = 200

# Inference batch size.
DEFAULT_BATCH_SIZE = 128

# Number of dinucleotide-shuffled background sequences for attribution.
DEFAULT_BACKGROUND_NUM = 50

# Hidden width of the prediction header MLP.
DEFAULT_HEADER_HIDDEN = 512

# Dropout rate of the prediction header.
DEFAULT_DROPOUT = 0.1
