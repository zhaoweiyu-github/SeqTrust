import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import json



def _str_to_one_hot(sequences):
    """Encode DNA strings without importing a model-specific package."""
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = torch.zeros((len(sequences), len(sequences[0]), 4), dtype=torch.float32)
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence.upper()):
            index = lookup.get(base)
            if index is not None:
                result[row, position, index] = 1.0
    return result


class Predictor(nn.Module):
    def __init__(
        self,
        model_type,
        pretrained_path=None,
        config_path=None,
        length=200, 
        device="cuda",
        conv_dim=1536,
        header_hidden=512, 
        dropout=0.1,
        whether_softplus=False,
    ):

        super().__init__()

        self.model_type = model_type.lower()
        self.device = device
        self.whether_softplus = whether_softplus
        self.length = length
        self.conv_dim = conv_dim
        self.header_hidden = header_hidden
        self.dropout = dropout

        # build trunk
        self.trunk = self._build_trunk(
            self.model_type,
            pretrained_path,
            config_path,
            conv_dim
        )

        self.trunk.to(device)

        trunk_dim = self._infer_trunk_dim(length=length)

        # # header (2-layer MLP)
        # self.header = nn.Sequential(
        #     nn.Linear(trunk_dim, header_hidden),
        #     nn.LayerNorm(header_hidden),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(header_hidden, 1)
        # )
        # header (2-layer MLP)
        if self.model_type != "alphagenome_chromdna":
            self.header = nn.Sequential(
                nn.Linear(trunk_dim, header_hidden),
                nn.LayerNorm(header_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(header_hidden, 1)
            )
        else:
            self.header = nn.Sequential(
                nn.Linear(trunk_dim, header_hidden),
                nn.GELU(),
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
        conv_dim
    ):

        # pretrained mode
        if pretrained_path is not None:
            if model_type == "enformer":
                from enformer_pytorch import from_pretrained

                model = from_pretrained(
                    pretrained_path,
                    target_length=-1
                )
                trunk = model._trunk

            elif model_type == "borzoi":
                from borzoi_pytorch import Borzoi

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
                from borzoi_pytorch import Borzoi

                model = Borzoi.from_pretrained(
                    pretrained_path,
                    return_center_bins_only=True,
                    bins_to_return=-1
                )
                model.human_head = nn.Identity()
                model.final_softplus = nn.Identity()
                trunk = model

            elif model_type in ["alphagenome", "alphagenome_chromdna"]:
                from alphagenome_pytorch import AlphaGenome

                model = AlphaGenome.from_pretrained(pretrained_path)
                trunk = model.encoder

            else:
                raise ValueError(f"Unknown model_type {model_type}")

            return trunk


        # ckpt mode
        if config_path is None:
            raise ValueError(
                "If pretrained_path is None, config_path must exist"
            )

        
        if model_type == "enformer":
            from enformer_pytorch import Enformer

            with open(config_path, "r") as f:
                config = json.load(f)
            config["target_length"] = -1
            model = Enformer.from_hparams(**config)
            trunk = model._trunk

        elif model_type == "borzoi":
            from borzoi_pytorch import Borzoi

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
            from borzoi_pytorch import Borzoi

            with open(config_path, "r") as f:
                config = json.load(f)
            config["return_center_bins_only"] = True
            config["bins_to_return"] = -1
            model = Borzoi.from_hparams(**config)
            model.human_head = nn.Identity()
            model.final_softplus = nn.Identity()
            trunk = model

        elif model_type in ["alphagenome", "alphagenome_chromdna"]:
            from alphagenome_pytorch import AlphaGenome

            model = AlphaGenome.from_pretrained(config_path)
            trunk = model.encoder

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
            dummy = _str_to_one_hot([dummy_seq])  # [1, length, 4]
            if self.model_type in ["borzoi", "borzoi_transformer"]:
                dummy = dummy.transpose(-1,-2)
            dummy = dummy.to(self.device)
            if self.model_type in ["alphagenome", "alphagenome_chromdna"]:
                out,_ = self.trunk(dummy)
            else:
                out = self.trunk(dummy)
            trunk_dim = out.flatten(1).shape[1]
        print("Trunk output dim:", trunk_dim)
        return trunk_dim



    def forward(self, seq):
        if self.model_type in ["borzoi", "borzoi_transformer"]:
            seq = seq.transpose(-1,-2)
        seq = seq.to(self.device)
        if self.model_type in ["alphagenome", "alphagenome_chromdna"]:
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
            "conv_dim": self.conv_dim,
            "device": self.device,
        },path)
        print("Saved checkpoint:",path)


    def load_ckpt(self,path):
        ckpt = torch.load(path,map_location=self.device)
        self.trunk.load_state_dict(ckpt["trunk_state"])
        self.header.load_state_dict(ckpt["header_state"])
        print("Loaded checkpoint:",path)
