"""Seq2Function model wrapper and inference utilities.

This module contains:

- ``Predictor``: unified wrapper around Seq2Function backbones
  (enformer, borzoi, borzoi_transformer, alphagenome, sei).
- ``calculate_seq2func_embedding``: trunk embedding extraction.
- ``calculate_seq2func_prediction``: scalar prediction.
- ``calculate_seq2func_attribution``: gradient-based attribution maps.

Paths and default hyperparameters are read from ``seqtrust.config``.
"""

from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from tqdm import tqdm

from enformer_pytorch import Enformer, from_pretrained, seq_indices_to_one_hot, str_to_one_hot
from borzoi_pytorch import Borzoi
from alphagenome_pytorch import AlphaGenome

from shap.explainers._gradient import GradientExplainer
from deeplift.dinuc_shuffle import dinuc_shuffle

from seqtrust import config


def _import_external_module(dir_path: str, module_name: str):
    """Import a third-party model module from an external directory.

    The directory is added to ``sys.path`` on first use; see
    ``SEI_MODEL_DIR`` in ``seqtrust.config``.
    """
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
    return importlib.import_module(module_name)


class Predictor(nn.Module):

    def __init__(
        self,
        model_type,
        pretrained_path=None,
        config_path=None,
        length=200,
        device="cuda",
        header_hidden=512,
        dropout=0.1,
        whether_softplus=False,
    ):

        super().__init__()

        self.model_type = model_type.lower()
        self.device = device
        self.whether_softplus = whether_softplus
        self.length = length
        self.header_hidden = header_hidden
        self.dropout = dropout

        # build trunk
        self.trunk = self._build_trunk(
            self.model_type,
            pretrained_path,
            config_path,
        )

        self.trunk.to(device)

        trunk_dim = self._infer_trunk_dim(length=length)

        # header (2-layer MLP)
        self.header = nn.Sequential(
            nn.Linear(trunk_dim, header_hidden),
            nn.LayerNorm(header_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(header_hidden, 1)
        )

        self.header.to(device)


    # ------------------------------------------------
    # build trunk
    # ------------------------------------------------
    def _build_trunk(
        self,
        model_type,
        pretrained_path,
        config_path,
    ):

        # pretrained mode
        if pretrained_path is not None:
            if model_type == "enformer":
                model = from_pretrained(
                    pretrained_path,
                    target_length=-1
                )
                trunk = model._trunk

            elif model_type == "borzoi":
                model = Borzoi.from_pretrained(
                    pretrained_path,
                    return_center_bins_only=True,
                    bins_to_return=-1
                )
                trunk = nn.Sequential(
                    model.conv_dna.conv_layer,
                    model.conv_dna.max_pool,
                    model._max_pool,
                    model.res_tower,
                )

            elif model_type == "borzoi_transformer":
                model = Borzoi.from_pretrained(
                    pretrained_path,
                    return_center_bins_only=True,
                    bins_to_return=-1
                )
                model.human_head = nn.Identity()
                model.final_softplus = nn.Identity()
                trunk = model

            elif model_type == "alphagenome":
                model = AlphaGenome.from_pretrained(pretrained_path)
                trunk = model.encoder

            elif model_type == "sei":
                sei_module = _import_external_module(
                    config.SEI_MODEL_DIR,
                    "sei",
                )
                model = sei_module.Sei(sequence_length=4096, n_genomic_features=21907)
                state_dict = torch.load(pretrained_path)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if "classifier" in k:
                        continue
                    new_key = k.replace("module.model.", "")
                    new_state_dict[new_key] = v

                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                print("missing:", missing)
                print("unexpected:", unexpected)
                trunk = model

            else:
                raise ValueError(f"Unknown model_type {model_type}")

            return trunk

        # ckpt mode
        if config_path is None:
            raise ValueError(
                "If pretrained_path is None, config_path must exist"
            )


        if model_type == "enformer":
            with open(config_path, "r") as f:
                config = json.load(f)
            config["target_length"] = -1
            model = Enformer.from_hparams(**config)
            trunk = model._trunk

        elif model_type == "borzoi":
            with open(config_path, "r") as f:
                config = json.load(f)
            config["return_center_bins_only"] = True
            config["bins_to_return"] = -1
            model = Borzoi.from_hparams(**config)
            trunk = nn.Sequential(
                model.conv_dna.conv_layer,
                model.conv_dna.max_pool,
                model._max_pool,
                model.res_tower,
            )

        elif model_type == "borzoi_transformer":
            with open(config_path, "r") as f:
                config = json.load(f)
            config["return_center_bins_only"] = True
            config["bins_to_return"] = -1
            model = Borzoi.from_hparams(**config)
            model.human_head = nn.Identity()
            model.final_softplus = nn.Identity()
            trunk = model

        elif model_type == "alphagenome":
            model = AlphaGenome.from_pretrained(config_path)
            trunk = model.encoder

        elif model_type == "sei":
            sei_module = _import_external_module(
                config.SEI_MODEL_DIR,
                "sei",
            )
            model = sei_module.Sei(sequence_length=4096, n_genomic_features=21907)
            state_dict = torch.load(config_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                if "classifier" in k:
                    continue
                new_key = k.replace("module.model.", "")
                new_state_dict[new_key] = v

            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print("missing:", missing)
            print("unexpected:", unexpected)

            trunk = model

        else:
            raise ValueError(f"Unknown model_type {model_type}")

        return trunk


    # ------------------------------------------------
    # infer trunk feature size
    # ------------------------------------------------
    def _infer_trunk_dim(self, length):
        with torch.no_grad():
            bases = "ACGT"
            dummy_seq = "".join(np.random.choice(list(bases), size=length))
            dummy = str_to_one_hot([dummy_seq])  # [1, length, 4]
            if self.model_type in ["borzoi","borzoi_transformer","sei"]:
                dummy = dummy.transpose(-1,-2)
            dummy = dummy.to(self.device)
            if self.model_type == "alphagenome":
                out,_ = self.trunk(dummy)
            else:
                out = self.trunk(dummy)
            trunk_dim = out.flatten(1).shape[1]
        print("Trunk output dim:", trunk_dim)
        return trunk_dim


    def forward(self, seq):
        if self.model_type in ["borzoi","borzoi_transformer","sei"]:
            seq = seq.transpose(-1,-2)
        seq = seq.to(self.device)
        if self.model_type == "alphagenome":
            features,_ = self.trunk(seq)
        else:
            features = self.trunk(seq)
        features = features.flatten(1)
        pred = self.header(features)

        if self.whether_softplus:
            pred = F.softplus(pred)

        return pred.squeeze(-1)


    @torch.no_grad()
    def predict(self, seq):
        self.eval()
        return self.forward(seq)


    def save_ckpt(self,path):
        torch.save({
            "model_type": self.model_type,
            "trunk_state": self.trunk.state_dict(),
            "header_state": self.header.state_dict(),

            "model_type": self.model_type,
            "whether_softplus": self.whether_softplus,
            "length": self.length,
            "header_hidden": self.header_hidden,
            "dropout": self.dropout,
            "device": self.device,
        },path)
        print("Saved checkpoint:",path)


    def load_ckpt(self,path):
        ckpt = torch.load(path,map_location=self.device)
        self.trunk.load_state_dict(ckpt["trunk_state"])
        self.header.load_state_dict(ckpt["header_state"])
        print("Loaded checkpoint:",path)


def _resolve(model_param, config_value):
    """Return the per-call argument, falling back to the config default."""
    return config_value if model_param is None else model_param


def calculate_seq2func_embedding(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    output_col: str = "xxx_embedding",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int | None = None,
    batch_size: int | None = None,
    whether_softplus: bool = False,

    device: str = "cuda",
    header_hidden: int | None = None,
    dropout: float | None = None,
):
    """
    Extract trunk embeddings from Seq2Function models and store them in a DataFrame column.

    Important
    ---------
    Even when `load_ckpt=True`, this function returns trunk embeddings rather than
    final scalar predictions.

    Supported model_type:
        enformer, borzoi, borzoi_transformer, alphagenome, sei.

    Arguments left as None fall back to the defaults in ``seqtrust.config``.
    """

    length = _resolve(length, config.DEFAULT_LENGTH)
    batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)
    header_hidden = _resolve(header_hidden, config.DEFAULT_HEADER_HIDDEN)
    dropout = _resolve(dropout, config.DEFAULT_DROPOUT)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_path == "default":
        if model_type not in config.MODEL_PRETRAINED_MAP:
            raise ValueError(f"Unknown model_type: {model_type}")
        pretrained_path = config.MODEL_PRETRAINED_MAP[model_type]
        print(f"[INFO] Using pretrained path:\n{pretrained_path}")

    model = Predictor(
        model_type=model_type,
        pretrained_path=pretrained_path,
        length=length,
        device=str(device),
        header_hidden=header_hidden,
        dropout=dropout,
        whether_softplus=whether_softplus,
    )
    model.to(device)
    model.eval()

    if load_ckpt:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when load_ckpt=True.")
        model.load_ckpt(ckpt_path)
        model.eval()

    def _pad_or_trim_sequence(seq: str, target_length: int) -> str:
        seq = str(seq).upper()

        if len(seq) > target_length:
            # center crop
            start = (len(seq) - target_length) // 2
            seq = seq[start:start + target_length]

        elif len(seq) < target_length:
            # pad N at the end
            seq = seq + "N" * (target_length - len(seq))

        return seq

    df = df.copy()
    sequences = df[sequence_col].astype(str).str.upper().values
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seq = [_pad_or_trim_sequence(seq, length) for seq in sequences[i:i + batch_size]]
            seq_tensor = str_to_one_hot(batch_seq).float()

            # if model_type in {"borzoi", "borzoi_transformer", "sei"}:
            #     seq_tensor = seq_tensor.transpose(-1, -2)

            seq_tensor = seq_tensor.to(device)

            if model_type == "alphagenome":
                trunk_out, _ = model.trunk(seq_tensor)
            else:
                trunk_out = model.trunk(seq_tensor)

            # if trunk_out.ndim != 2:
            #     # expected: (batch, embedding_dim)
            #     but for alphagenome, trunk output is (batch, embedding_dim, 2), two strand
            #     raise ValueError(
            #         f"Unexpected trunk_out dimension: {trunk_out.shape}"
            #     )

            emb = trunk_out.flatten(1)
            embeddings.append(emb.detach().cpu().numpy())

    features = np.concatenate(embeddings, axis=0).astype(np.float32)


    df[output_col] = list(features)

    return df


def calculate_seq2func_prediction(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    output_col: str = "xxx_prediction",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int | None = None,
    batch_size: int | None = None,
    whether_softplus: bool = False,

    device: str = "cuda",
    header_hidden: int | None = None,
    dropout: float | None = None,
):
    """
    Calculate Seq2Function scalar predictions and store them in a DataFrame column.

    Arguments left as None fall back to the defaults in ``seqtrust.config``.
    """

    length = _resolve(length, config.DEFAULT_LENGTH)
    batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)
    header_hidden = _resolve(header_hidden, config.DEFAULT_HEADER_HIDDEN)
    dropout = _resolve(dropout, config.DEFAULT_DROPOUT)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_path == "default":
        if model_type not in config.MODEL_PRETRAINED_MAP:
            raise ValueError(f"Unknown model_type: {model_type}")
        pretrained_path = config.MODEL_PRETRAINED_MAP[model_type]
        print(f"[INFO] Using pretrained path:\n{pretrained_path}")

    model = Predictor(
        model_type=model_type,
        pretrained_path=pretrained_path,
        length=length,
        device=str(device),
        header_hidden=header_hidden,
        dropout=dropout,
        whether_softplus=whether_softplus,
    )
    model.to(device)
    model.eval()

    if load_ckpt:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when load_ckpt=True.")
        model.load_ckpt(ckpt_path)
        model.eval()

    def _pad_or_trim_sequence(seq: str, target_length: int) -> str:
        seq = str(seq).upper()
        if len(seq) > target_length:
            start = (len(seq) - target_length) // 2
            seq = seq[start:start + target_length]
        elif len(seq) < target_length:
            left = (target_length - len(seq)) // 2
            right = target_length - len(seq) - left
            seq = "N" * left + seq + "N" * right
        return seq.replace("N", "A")

    df = df.copy()
    sequences = df[sequence_col].astype(str).str.upper().values
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seq = [_pad_or_trim_sequence(seq, length) for seq in sequences[i:i + batch_size]]
            seq_tensor = str_to_one_hot(batch_seq).float()

            # if model_type in {"borzoi", "borzoi_transformer", "sei"}:
            #     seq_tensor = seq_tensor.transpose(-1, -2)

            seq_tensor = seq_tensor.to(device)
            pred = model.predict(seq_tensor)
            pred = pred.detach().cpu().numpy()
            if pred.ndim > 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    df[output_col] = list(preds)

    return df


def calculate_seq2func_attribution(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int | None = None,
    batch_size: int | None = None,
    whether_softplus: bool = False,
    background_num: int | None = None,
):
    """
    Calculate attribution map for Seq2Function models.

    Arguments left as None fall back to the defaults in ``seqtrust.config``.

    Returns
    -------
    attr:
        np.ndarray
        shape:
        (N,4,L)

        masked attribution:
        shap_values * one_hot
    """

    length = _resolve(length, config.DEFAULT_LENGTH)
    batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)
    background_num = _resolve(background_num, config.DEFAULT_BACKGROUND_NUM)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # ==========================
    # load model
    # ==========================

    if pretrained_path == "default":
        if model_type not in config.MODEL_PRETRAINED_MAP:
            raise ValueError(
                f"Unknown model_type {model_type}"
            )

        pretrained_path = config.MODEL_PRETRAINED_MAP[model_type]


    model = Predictor(
        model_type=model_type,
        pretrained_path=pretrained_path,
        length=length,
        device=device,
        whether_softplus=whether_softplus,
    )

    model.to(device)
    model.eval()


    if load_ckpt:

        if ckpt_path is None:
            raise ValueError(
                "ckpt_path is required when load_ckpt=True"
            )

        model.load_ckpt(ckpt_path)
        model.eval()


    # ==========================
    # sequence preprocessing
    # ==========================

    def _pad_or_trim_sequence(seq, target_length):

        seq = str(seq).upper()

        if len(seq) > target_length:
            start = (len(seq)-target_length)//2
            seq = seq[start:start+target_length]

        elif len(seq)<target_length:

            left = (target_length-len(seq))//2
            right = target_length-len(seq)-left

            seq = (
                "N"*left +
                seq +
                "N"*right
            )

        return seq.replace("N","A")



    sequences = (
        df[sequence_col]
        .astype(str)
        .tolist()
    )


    # ==========================
    # one hot
    # ==========================

    seqs = [
        _pad_or_trim_sequence(
            x,
            length
        )
        for x in sequences
    ]

    ohe = str_to_one_hot(seqs).float()

    ohe = ohe.contiguous()
    explained = ohe.to(device)



    # ==========================
    # background
    # ==========================

    n_bg = min(
        background_num,
        len(seqs)
    )

    bg_idx = np.random.choice(
        len(seqs),
        n_bg,
        replace=False,
    )

    background = []

    rng = np.random.RandomState(42)


    for idx in bg_idx:
        x = str_to_one_hot(
            seqs[idx]
        )

        x = dinuc_shuffle(
            x.detach().cpu().numpy(),
            rng=rng,
        )

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        background.append(x)



    background = torch.stack(
        background
    )


    background = (
        background
        .contiguous()
        .to(device)
    )


    class AttributionWrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):

            pred = self.model(x)

            # (batch,) -> (batch,1)

            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)

            return pred

    wrapper = AttributionWrapper(model)

    explainer = GradientExplainer(
        wrapper,
        background,
    )

    # ==========================
    # shap calculation
    # ==========================

    attrs=[]

    for i in tqdm(
        range(
            0,
            len(explained),
            batch_size,
        ),
        desc="Attribution"
    ):
        batch = explained[
            i:i+batch_size
        ]

        shap_values = (
            explainer
            .shap_values(batch)
        )

        if isinstance(
            shap_values,
            list
        ):
            shap_values = shap_values[0]

        if torch.is_tensor(
            shap_values
        ):
            shap_values = (
                shap_values
                .detach()
                .cpu()
                .numpy()
            )

        else:

            shap_values = np.asarray(
                shap_values
            )


        attrs.append(
            shap_values
        )


    shap_values = np.concatenate(
        attrs,
        axis=0
    )
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    # ==========================
    # observed base attribution
    # ==========================

    ohe_np = (
        explained
        .detach()
        .cpu()
        .numpy()
    )


    attr = (
        shap_values *
        ohe_np
    )


    del model
    del explainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return attr.astype(
        np.float32
    )
