from __future__ import annotations

from typing import Optional


class EmbeddingMixin:
    """Whole-dataset embeddings and Seq2Function predictions."""

    def add_seq2function_embedding(
        self,
        model_type: str,
        output_col: str = "seq2function_embedding",
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: Optional[str] = None,
        length: int = 200,
        batch_size: int = 128,
        whether_softplus: bool = False,
    ):
        """Add one Seq2Function trunk-embedding vector per sequence."""
        from seqtrust.tools.embeddings import calculate_seq2function_embedding

        result = calculate_seq2function_embedding(
            self.trust.df,
            sequence_col=self.sequence_col,
            model_type=model_type,
            output_col=output_col,
            pretrained_path=pretrained_path,
            load_ckpt=load_ckpt,
            ckpt_path=ckpt_path,
            length=length,
            batch_size=batch_size,
            whether_softplus=whether_softplus,
        )
        self.trust.df[output_col] = result[output_col].to_numpy()
        self._record(
            "add_seq2function_embedding",
            model_type=model_type,
            output_col=output_col,
            length=length,
            batch_size=batch_size,
        )
        return self.trust.df

    def add_DNABERT2_embedding(
        self,
        model_path: str,
        output_col: str = "dnabert2_embedding",
        conda_env: str = "dnabert2",
        batch_size: int = 32,
        max_length: int = 200,
    ):
        """Add DNABERT-2 embeddings by running the configured conda environment."""
        from seqtrust.tools.embeddings import calculate_dnabert2_embedding

        result = calculate_dnabert2_embedding(
            self.trust.df,
            sequence_col=self.sequence_col,
            output_col=output_col,
            model_path=model_path,
            conda_env=conda_env,
            batch_size=batch_size,
            max_length=max_length,
        )
        self.trust.df[output_col] = result[output_col].to_numpy()
        self._record(
            "add_DNABERT2_embedding",
            output_col=output_col,
            model_path=model_path,
            conda_env=conda_env,
            batch_size=batch_size,
            max_length=max_length,
        )
        return self.trust.df

    def add_kmer_embedding(
        self,
        output_col: str = "kmer_embedding",
        kmer_size: int = 6,
    ):
        """Add normalized frequencies of all possible DNA k-mers."""
        from seqtrust.tools.embeddings import calculate_kmer_embedding

        result = calculate_kmer_embedding(
            self.trust.df,
            sequence_col=self.sequence_col,
            output_col=output_col,
            kmer_size=kmer_size,
        )
        self.trust.df[output_col] = result[output_col].to_numpy()
        self._record(
            "add_kmer_embedding",
            output_col=output_col,
            kmer_size=kmer_size,
        )
        return self.trust.df

    def add_seq2function_prediction(
        self,
        model_type: str,
        output_col: str = "seq2function_prediction",
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: Optional[str] = None,
        length: int = 200,
        batch_size: int = 128,
        whether_softplus: bool = False,
    ):
        """Add one Seq2Function model prediction per sequence."""
        from seqtrust.tools.embeddings import calculate_seq2function_prediction

        result = calculate_seq2function_prediction(
            self.trust.df,
            sequence_col=self.sequence_col,
            model_type=model_type,
            output_col=output_col,
            pretrained_path=pretrained_path,
            load_ckpt=load_ckpt,
            ckpt_path=ckpt_path,
            length=length,
            batch_size=batch_size,
            whether_softplus=whether_softplus,
        )
        self.trust.df[output_col] = result[output_col].to_numpy()
        self._record(
            "add_seq2function_prediction",
            model_type=model_type,
            output_col=output_col,
            length=length,
            batch_size=batch_size,
        )
        return self.trust.df
