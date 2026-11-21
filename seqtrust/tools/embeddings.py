import itertools
import pandas as pd
import numpy as np
from typing import Any, Optional

model_pretrained_map = {
    "enformer": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/EleutherAI/enformer-official-rough",
    "borzoi": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/johahi/borzoi-replicate-0",
    "alphagenome": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/gtca/alphagenome_pytorch/model_all_folds.safetensors",
    "alphagenome_chromdna": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/gtca/alphagenome_pytorch/model_all_folds.safetensors",
    "borzoi_transformer": "/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/huggingface/johahi/borzoi-replicate-0",
}


def calculate_seq2function_embedding(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    output_col: str = "xxx_embedding",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int = 200,
    batch_size: int = 64,
    whether_softplus: bool = False,
    
    device: str = "cuda",
    conv_dim: int = 1536,
    header_hidden: int = 512,
    dropout: float = 0.1,
):
    """
    Extract trunk embeddings from Seq2Function models and store them in a DataFrame column.

    Important
    ---------
    Even when `load_ckpt=True`, this function returns trunk embeddings rather than
    final scalar predictions.

    Supported model_type:
        enformer, borzoi, borzoi_transformer, alphagenome,
        alphagenome_chromdna.
    """
    
    import torch
    from enformer_pytorch import str_to_one_hot
    from tqdm import tqdm
    from seq2func.Seq2Func import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_path == "default":
        if model_type not in model_pretrained_map:
            raise ValueError(f"Unknown model_type: {model_type}")
        pretrained_path = model_pretrained_map[model_type]
        print(f"[INFO] Using pretrained path:\n{pretrained_path}")

    model = Predictor(
        model_type=model_type,
        pretrained_path=pretrained_path,
        length=length,
        device=str(device),
        conv_dim=conv_dim,
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

            # if model_type in {"borzoi", "baseline", "borzoi_transformer", "sei"}:
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


def calculate_seq2function_prediction(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    output_col: str = "xxx_prediction",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int = 200,
    batch_size: int = 64,
    whether_softplus: bool = False,
    device: str = "cuda",
    conv_dim: int = 1536,
    header_hidden: int = 512,
    dropout: float = 0.1,
):

    import torch
    from enformer_pytorch import str_to_one_hot
    from tqdm import tqdm
    from seq2func.Seq2Func import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_path == "default":
        if model_type not in model_pretrained_map:
            raise ValueError(f"Unknown model_type: {model_type}")
        pretrained_path = model_pretrained_map[model_type]
        print(f"[INFO] Using pretrained path:\n{pretrained_path}")

    model = Predictor(
        model_type=model_type,
        pretrained_path=pretrained_path,
        length=length,
        device=str(device),
        conv_dim=conv_dim,
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
        return seq.replace("N", "N")

    df = df.copy()
    sequences = df[sequence_col].astype(str).str.upper().values
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seq = [_pad_or_trim_sequence(seq, length) for seq in sequences[i:i + batch_size]]
            seq_tensor = str_to_one_hot(batch_seq).float()

            # if model_type in {"borzoi", "baseline", "borzoi_transformer", "sei"}:
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


def calculate_seq2function_attribution(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    model_type: str = "alphagenome",
    pretrained_path: str = "default",
    load_ckpt: bool = False,
    ckpt_path: Optional[str] = None,
    length: int = 200,
    batch_size: int = 128,
    whether_softplus: bool = False,
    background_num: int = 50,
):
    """
    Calculate attribution map for Seq2Function models.

    Returns
    -------
    attr:
        np.ndarray
        shape:
        (N,4,L)

        masked attribution:
        shap_values * one_hot
    """
    import torch
    from deeplift.dinuc_shuffle import dinuc_shuffle
    from enformer_pytorch import str_to_one_hot
    from shap.explainers._gradient import GradientExplainer
    from tqdm import tqdm
    from seq2func.Seq2Func import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # ==========================
    # load model
    # ==========================

    if pretrained_path == "default":
        if model_type not in model_pretrained_map:
            raise ValueError(
                f"Unknown model_type {model_type}"
            )

        pretrained_path = model_pretrained_map[model_type]


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

        return seq.replace("N","N")



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


def calculate_dnabert2_embedding(
    df,
    sequence_col="Sequence",
    output_col="dnabert2_embedding",
    model_path="/home/wangyucheng/projects/Benchmark/Evaluation/scripts/predictors/zhihan1996/DNABERT-2-117M",
    conda_env="dnabert2",
    batch_size=32,
    max_length=200,
):

    import tempfile
    import subprocess
    import pickle
    import numpy as np
    from pathlib import Path


    df=df.copy()


    sequences=(
        df[sequence_col]
        .astype(str)
        .str.upper()
        .tolist()
    )


    with tempfile.TemporaryDirectory() as tmpdir:


        input_file=Path(tmpdir)/"seq.pkl"
        output_file=Path(tmpdir)/"embedding.npy"


        with open(input_file,"wb") as f:
            pickle.dump(
                sequences,
                f
            )


        runner=Path(
            __file__
        ).parent/"dnabert2_runner.py"


        cmd=[
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(runner),
            str(input_file),
            str(output_file),
            model_path,
            str(batch_size),
            str(max_length),
        ]


        subprocess.run(
            cmd,
            check=True,
        )


        embedding=np.load(
            output_file
        )


    df[output_col]=list(
        embedding
    )


    return df


def calculate_kmer_embedding(
    df: pd.DataFrame,
    sequence_col: str,
    output_col: str = "kmer_embedding",
    kmer_size: int = 6,
):
    """
    Calculate normalized k-mer frequency embedding.

    Parameters
    ----------
    df:
        Input dataframe

    sequence_col:
        DNA sequence column

    output_col:
        Output embedding column

    kmer_size:
        k-mer length

    Returns
    -------
    df
    """

    # all possible kmers
    kmers = [
        "".join(x)
        for x in itertools.product(
            "ACGT",
            repeat=kmer_size
        )
    ]

    kmer_to_idx = {
        kmer: i
        for i, kmer in enumerate(kmers)
    }


    embeddings = []


    for seq in df[sequence_col].astype(str):

        seq = seq.upper()

        vector = np.zeros(
            len(kmers),
            dtype=np.float32
        )


        for i in range(
            len(seq)-kmer_size+1
        ):

            kmer = seq[
                i:i+kmer_size
            ]

            if kmer in kmer_to_idx:
                vector[
                    kmer_to_idx[kmer]
                ] += 1


        # normalize frequency
        total = vector.sum()

        if total > 0:
            vector /= total


        embeddings.append(vector)


    df[output_col] = embeddings

    return df



