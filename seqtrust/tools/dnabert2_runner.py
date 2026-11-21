from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd


def load_dnabert2(
    model_path,
    device="cuda",
):
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device



def get_embedding(
    sequences,
    model_path,
    batch_size=32,
    max_length=200,
    pooling="mean",
    device="cuda",
):

    import torch

    tokenizer, model, device = load_dnabert2(
        model_path,
        device,
    )

    embeddings=[]

    with torch.no_grad():

        for i in range(
            0,
            len(sequences),
            batch_size
        ):

            batch = sequences[
                i:i+batch_size
            ]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            inputs = {
                k:v.to(device)
                for k,v in inputs.items()
            }


            outputs=model(**inputs)

            hidden=outputs[0]


            if pooling=="mean":

                mask=inputs["attention_mask"].unsqueeze(-1)

                emb=(
                    hidden*mask
                ).sum(1)/mask.sum(1).clamp(min=1)

            elif pooling=="cls":

                emb=hidden[:,0,:]

            embeddings.append(
                emb.cpu().numpy()
            )


    return np.concatenate(
        embeddings,
        axis=0
    ).astype(np.float32)



if __name__=="__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = sys.argv[3]

    batch_size = int(sys.argv[4])
    max_length = int(sys.argv[5])


    with open(input_file,"rb") as f:
        sequences = pickle.load(f)


    embedding = get_embedding(
        sequences,
        model_path=model_path,
        batch_size=batch_size,
        max_length=max_length,
    )

    np.save(
        output_file,
        embedding,
    )