SeqTrust
========

Regulatory plausibility analysis based on predictive sequence features enables reliable prioritization of AI-designed regulatory sequences.

Installation
------------

**1. Create the conda environment** (Python 3.12.13)::

    conda create -n SeqTrust python=3.12.13 -y
    conda activate SeqTrust

**2. Install the Python dependencies**::

    pip install -r requirements.txt

**3. Install flash-attn** (2.8.1, built for torch 2.10 + cu12) from the
wheel file. The wheel is not bundled with this repository; obtain it
separately, e.g. from the `flash-attention releases
<https://github.com/Dao-AILab/flash-attention/releases>`_ page::

    pip install flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

Usage
-----

A complete workflow example (prediction, attribution, motif scan,
seqlets and metrics) is available in ``seqtrust/example/tutorial.ipynb``.
