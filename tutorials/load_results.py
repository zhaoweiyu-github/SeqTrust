### io module

"""Loader for generated sequences from the SeqEval benchmark results."""

import os
import glob
import pickle

import pandas as pd

# TACO and CtrlDNA: read the csv WITHOUT "summary" in its name,
# and keep only sequences from the last round
ROUND_BASED_TOOLS = {"TACO", "CtrlDNA"}


def load_method_seqs(method_dir: str, method_name: str) -> pd.DataFrame:
    """Load generated sequences from one method directory.

    Returns a DataFrame with columns: sequence, Tools.
    """
    if method_name in ROUND_BASED_TOOLS:
        # Find the csv file without "summary" in its name
        csv_files = [
            f for f in glob.glob(os.path.join(method_dir, "*.csv"))
            if "summary" not in os.path.basename(f)
        ]
        if not csv_files:
            print(f"[Warning] No non-summary csv found in {method_dir}, skipped")
            return pd.DataFrame(columns=["sequence", "Tools"])
        df = pd.read_csv(csv_files[0])
        # Keep only the last round
        last_round = df["round"].max()
        df = df[df["round"] == last_round]
    else:
        csv_path = os.path.join(method_dir, "generate_seqs.csv")
        if not os.path.exists(csv_path):
            print(f"[Warning] {csv_path} not found, skipped")
            return pd.DataFrame(columns=["sequence", "Tools"])
        df = pd.read_csv(csv_path)

    return pd.DataFrame({
        "sequence": df["sequence"].values,
        "Tools": method_name,
    })


def load_task_seqs(results_dir: str, verbose: bool = True) -> dict:
    """Load generated sequences for all tasks under `results_dir`.

    Only tasks whose name contains "Complete" are kept. Tasks whose name
    contains "DNase" are dropped. The returned keys drop the trailing
    setting, keeping `<modality>_<cell>` (for example
    `LentiMPRA_HepG2_Complete` becomes `LentiMPRA_HepG2`). The sequence
    column is renamed to `Sequence`.

    Args:
        results_dir: Root directory structured as <task>/<method>/*.csv.
            Task names are expected as `<modality>_<cell>_Complete`.
            Names containing "DNase" are excluded.
        verbose: If True, print an overview of the loaded data.

    Returns:
        Dict mapping `<modality>_<cell>` -> DataFrame with `Sequence` and
        `Tools` columns.
    """
    task_seqs_dict = {}

    for task_name in sorted(os.listdir(results_dir)):
        task_dir = os.path.join(results_dir, task_name)
        if not os.path.isdir(task_dir):
            continue

        df_list = []
        for method_name in sorted(os.listdir(task_dir)):
            method_dir = os.path.join(task_dir, method_name)
            if not os.path.isdir(method_dir):
                continue
            df_method = load_method_seqs(method_dir, method_name)
            if not df_method.empty:
                df_list.append(df_method)

        if df_list:
            task_seqs_dict[task_name] = pd.concat(df_list, ignore_index=True)

    # Keep only Complete tasks, drop DNase, remove the setting suffix from
    # the key, and rename the sequence column.
    filtered = {}
    for task_name, df in task_seqs_dict.items():
        if "Complete" not in task_name or "DNase" in task_name:
            continue
        key = task_name.replace("_Complete", "")
        filtered[key] = df.rename(columns={"sequence": "Sequence"})
    task_seqs_dict = filtered

    if verbose:
        for task_name, df in task_seqs_dict.items():
            print(f"{task_name}: {len(df)} seqs, tools: {df['Tools'].value_counts().to_dict()}")

    return task_seqs_dict


COMPLETE_CKPTS_PATH = {
    "LentiMPRA_HepG2": "/home/wangyucheng/projects/Benchmark/Data_v2/LentiMPRA/HepG2/HepG2_complete_alphagenome/best_loss_epoch19_loss0.1518.pt",
    "LentiMPRA_K562": "/home/wangyucheng/projects/Benchmark/Data_v2/LentiMPRA/K562/K562_complete_alphagenome/best_loss_epoch2_loss0.0742.pt",
    "LentiMPRA_WTC11": "/home/wangyucheng/projects/Benchmark/Data_v2/LentiMPRA/WTC11/WTC11_complete_alphagenome/best_loss_epoch1_loss0.5075.pt",
    "EpisomalMPRA_A549": "/home/wangyucheng/projects/Benchmark/Data_v2/EpisomalMPRA/A549/A549_complete_alphagenome/best_loss_epoch4_loss0.1383.pt",
    "EpisomalMPRA_HCT116": "/home/wangyucheng/projects/Benchmark/Data_v2/EpisomalMPRA/HCT116/HCT116_complete_alphagenome/best_loss_epoch2_loss0.1823.pt",
    "EpisomalMPRA_HepG2": "/home/wangyucheng/projects/Benchmark/Data_v2/EpisomalMPRA/HepG2/HepG2_complete_alphagenome/best_loss_epoch1_loss0.1986.pt",
    "EpisomalMPRA_K562": "/home/wangyucheng/projects/Benchmark/Data_v2/EpisomalMPRA/K562/K562_complete_alphagenome/best_loss_epoch1_loss0.2687.pt",
    "EpisomalMPRA_SKNSH": "/home/wangyucheng/projects/Benchmark/Data_v2/EpisomalMPRA/SKNSH/SKNSH_complete_alphagenome/best_loss_epoch4_loss0.2790.pt",
}

OFFLINE_CKPTS_PATH = {
    "LentiMPRA_HepG2_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/HepG2/HepG2_offline_alphagenome/best_loss_epoch18_loss0.1153.pt",
    "LentiMPRA_K562_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/K562/K562_offline_alphagenome/best_loss_epoch1_loss0.0593.pt",
    "LentiMPRA_WTC11_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/WTC11/WTC11_offline_alphagenome/best_loss_epoch0_loss0.4039.pt",
    "EpisomalMPRA_A549_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/A549/A549_offline_alphagenome/best_loss_epoch2_loss0.0689.pt",
    "EpisomalMPRA_HCT116_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/HCT116/HCT116_offline_alphagenome/best_loss_epoch2_loss0.1205.pt",
    "EpisomalMPRA_HepG2_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/HepG2/HepG2_offline_alphagenome/best_loss_epoch3_loss0.1349.pt",
    "EpisomalMPRA_K562_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/K562/K562_offline_alphagenome/best_loss_epoch3_loss0.1675.pt",
    "EpisomalMPRA_SKNSH_Offline": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/SKNSH/SKNSH_offline_alphagenome/best_loss_epoch2_loss0.1701.pt",
}


def load_ckpts_path() -> tuple:
    """Return (complete_ckpts_path, offline_ckpts_path) dicts.

    Checks that each checkpoint file exists; prints a warning only for
    missing files (existing files produce no message).
    """
    for ckpts_path in (COMPLETE_CKPTS_PATH, OFFLINE_CKPTS_PATH):
        for name, path in ckpts_path.items():
            if not os.path.exists(path):
                print(f"[Warning] Checkpoint not found for {name}: {path}")
    return dict(COMPLETE_CKPTS_PATH), dict(OFFLINE_CKPTS_PATH)


# Min/max of the training labels for each dataset: dataset_tag -> (min, max)
DATASET_MIN_MAX = {
    "DNase_GM12878": (0.0, 4.656977),
    "DNase_HepG2": (0.0, 5.285814),
    "DNase_K562": (0.0, 6.625859),
    "EpisomalMPRA_A549": (-4.502754, 7.907343),
    "EpisomalMPRA_HCT116": (-4.273855, 8.208215),
    "EpisomalMPRA_HepG2": (-6.227728, 8.589416),
    "EpisomalMPRA_K562": (-5.609770, 9.838334),
    "EpisomalMPRA_SKNSH": (-8.162227, 9.963708),
    "LentiMPRA_HepG2": (-3.338, 5.036),
    "LentiMPRA_K562": (-3.527, 3.273),
    "LentiMPRA_WTC11": (-5.263, 6.657),
}

# Background nucleotide frequencies per task: dataset_tag -> {base: freq}
TASK_TO_BG_FREQ = {
    "LentiMPRA_HepG2": {"A": 0.2485187837643396, "C": 0.25093948253083753, "G": 0.25139439364805294, "T": 0.24914734005676986},
    "LentiMPRA_K562": {"A": 0.24361650632823548, "C": 0.2563461705283795, "G": 0.2562297552850959, "T": 0.24380756785828916},
    "LentiMPRA_WTC11": {"A": 0.2184018326852699, "C": 0.2813201195105215, "G": 0.28106858989021044, "T": 0.21920945791399818},
    "EpisomalMPRA_A549": {"A": 0.27808155612144225, "C": 0.22016753417397092, "G": 0.22241047029891328, "T": 0.27934043940567355},
    "EpisomalMPRA_HCT116": {"A": 0.2668014838261277, "C": 0.23187193486349414, "G": 0.23354588668101367, "T": 0.26778069462936444},
    "EpisomalMPRA_HepG2": {"A": 0.27163101954472, "C": 0.22684905488002546, "G": 0.22876446814667223, "T": 0.27275545742858226},
    "EpisomalMPRA_K562": {"A": 0.27137775616174226, "C": 0.2270740421793547, "G": 0.22897209976197583, "T": 0.2725761018969272},
    "EpisomalMPRA_SKNSH": {"A": 0.27165314865570417, "C": 0.2267790815686754, "G": 0.2287723799077559, "T": 0.27279538986786456},
    "DNase_GM12878": {"A": 0.2389284497183424, "C": 0.2602876704582082, "G": 0.261710673246344, "T": 0.23907320657710537},
    "DNase_HepG2": {"A": 0.24913067361162813, "C": 0.25057147345171005, "G": 0.25109082011153183, "T": 0.24920703282513},
    "DNase_K562": {"A": 0.23849123402500144, "C": 0.2607984467034979, "G": 0.2620726261683269, "T": 0.2386376931031738},
}


# Reference motif enrichment results (dataset_tag -> DataFrame), stored as a
# pickle because of its size (~2.4 MB, 11 DataFrames of ~1600 x 17)
MOTIF_ENRICHMENT_PKL = "/home/wangyucheng/projects/Benchmark/Generate_Seqs_v4/Summary/Calculate_MotifEnrichment_Natural/SingleCell/task_to_motif_enrichment.pkl"

def load_motif_enrichment(
    pkl_path: str = MOTIF_ENRICHMENT_PKL,
    qval_cutoff: float = 0.1,
    pval_cutoff: float = 1e-3,
    min_log2_enrichment: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Load reference motif enrichment results and keep the motifs
    significantly enriched in natural_high.

    Args:
        pkl_path: Path to task_to_motif_enrichment.pkl.
        qval_cutoff: Keep motifs with q_value < qval_cutoff.
        pval_cutoff: Keep motifs with p_value < pval_cutoff.
        min_log2_enrichment: Keep motifs with log2_enrichment above this
            threshold (1.0 = at least 2-fold enrichment; set to 0 to disable).
        verbose: If True, print the number of selected motifs per task.

    Returns:
        Dict mapping dataset_tag -> list of significant motif names.
    """
    with open(pkl_path, "rb") as f:
        task_to_motif_enrichment = pickle.load(f)

    sig_motifs_high = {}
    for task, enrichment in task_to_motif_enrichment.items():
        sig = enrichment[
            (enrichment["enriched_in"] == "natural_high")
            & (enrichment["q_value"] < qval_cutoff)
            & (enrichment["p_value"] < pval_cutoff)
            & (enrichment["log2_enrichment"] > min_log2_enrichment)
        ]
        if verbose:
            print(f"{task}: {len(sig)}/{len(enrichment)} motifs significantly enriched in natural_high")
        sig_motifs_high[task] = list(sig["motif"])
    return sig_motifs_high




# Natural data path: dataset_tag -> tsv file path
NATURALDATA_PATH = {
    "LentiMPRA_HepG2": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/HepG2/HepG2_complete.tsv",
    "LentiMPRA_K562": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/K562/K562_complete.tsv",
    "LentiMPRA_WTC11": "/home/wangyucheng/projects/Benchmark/Data_v3/LentiMPRA/WTC11/WTC11_complete.tsv",
    "EpisomalMPRA_A549": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/A549/A549_complete.tsv",
    "EpisomalMPRA_HCT116": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/HCT116/HCT116_complete.tsv",
    "EpisomalMPRA_HepG2": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/HepG2/HepG2_complete.tsv",
    "EpisomalMPRA_K562": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/K562/K562_complete.tsv",
    "EpisomalMPRA_SKNSH": "/home/wangyucheng/projects/Benchmark/Data_v3/EpisomalMPRA/SKNSH/SKNSH_complete.tsv",
    "DNase_GM12878": "/home/wangyucheng/projects/Benchmark/Data_v3/DNase/GM12878/GM12878_complete.tsv",
    "DNase_HepG2": "/home/wangyucheng/projects/Benchmark/Data_v3/DNase/HepG2/HepG2_complete.tsv",
    "DNase_K562": "/home/wangyucheng/projects/Benchmark/Data_v3/DNase/K562/K562_complete.tsv",
}


def load_natural_data(naturaldata_path: dict = NATURALDATA_PATH, verbose: bool = True) -> dict:
    """Load natural sequences and their functional scores.

    Args:
        naturaldata_path: Dict mapping dataset_tag -> tsv file path.
        verbose: If True, print the number of sequences per task.

    Returns:
        Dict mapping dataset_tag -> DataFrame with columns
        `sequence` and `functional_score`.
    """
    natural_data_dict = {}
    for task, file_path in naturaldata_path.items():
        df = pd.read_csv(file_path, sep="\t", low_memory=False)
        # keep required columns
        df = df[["Sequence", "functional_score"]].copy()
        df = df.rename(columns={"Sequence": "sequence"})
        natural_data_dict[task] = df
        if verbose:
            print(f"{task}: {len(df)} natural sequences")
    return natural_data_dict
