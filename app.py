"""
Pokemon search application in streamlit, using the 
"""

import os
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple
from tqdm.auto import tqdm

import pandas as pd
import streamlit as st
import torch
from transformers.models.auto import AutoModel, AutoTokenizer

from streamlit_search_ui import Document
from streamlit_search_ui import Result
from streamlit_search_ui import search_app

# Constants.
DEVICE = "mps"  # Mac hardware acceleration.
MODEL_NAME = "Snowflake/snowflake-arctic-embed-xs"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5


class CacheableResourceState(NamedTuple):
    model: AutoModel
    tokenizer: AutoTokenizer
    df_pokemon: pd.DataFrame
    document_embeddings: torch.Tensor


def main() -> None:
    search_app(
        search_fn=search,
        init_fn=get_resources,  # Warm up the cache at initialization.
        title="Pokemon Search",
        show_time=True,  # Shows runtime per search.
    )


@st.cache_resource
def get_resources() -> CacheableResourceState:
    df_pokemon = _load_data_from_disk()
    model, tokenizer = _load_model_from_huggingface(device=DEVICE)
    emb = _embed_docs(model, tokenizer, df_pokemon["description"].tolist())
    return CacheableResourceState(model, tokenizer, df_pokemon, emb)


def search(query: str) -> List[Result]:
    model, tokenizer, df_pokemon, document_embeddings = get_resources()
    query_vector = _embed(model, tokenizer, [query], prefix=QUERY_PREFIX)
    scores = (query_vector @ document_embeddings.T).squeeze()
    topk = torch.topk(scores, TOP_K)
    topk_scores = topk.values.cpu().tolist()
    topk_ind = topk.indices.cpu().tolist()
    return [
        Result(Document(title=row.name, text=row.description), score)
        for row, score in zip(df_pokemon.iloc[topk_ind].itertuples(), topk_scores)
    ]


def _load_data_from_disk() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "all_the_pokemon.csv")


def _load_model_from_huggingface(device: str) -> Tuple[AutoModel, AutoTokenizer]:
    # Disable parallelized tokenization to avoid warnings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model and configure for embedding.
    model = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
    model.eval()
    model.to(device)

    return model, tokenizer


def _embed_docs(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    docs: Sequence[str],
    batch_size: int = 16,
) -> torch.Tensor:
    embeddings_list = []
    with tqdm(
        total=len(docs), desc="Embedding The Pokemon!", unit="doc", smoothing=0
    ) as pbar:
        for start in range(0, len(docs), batch_size):
            end = start + batch_size
            batch = docs[start:end]
            batch_embeddings = _embed(model, tokenizer, batch)
            embeddings_list.append(batch_embeddings)
            pbar.update(len(batch))
    document_embeddings = torch.cat(embeddings_list)
    return document_embeddings


@torch.no_grad()
def _embed(
    model: AutoModel, tokenizer: AutoTokenizer, texts: Sequence[str], prefix: str = ""
) -> torch.Tensor:
    # Add prefix.
    if prefix != "":
        texts = [f"{prefix}{text}" for text in texts]

    # Tokenize.
    input_dict = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )

    # Move tokens to model device.
    input_dict = {name: tensor.to(model.device) for name, tensor in input_dict.items()}

    # Model forward pass.
    last_hidden_state = model.forward(**input_dict, return_dict=True).last_hidden_state

    # The normalized output vector for [cls] token is the embedding.
    batch_size = len(texts)
    sequence_length = input_dict["input_ids"].size(1)
    assert last_hidden_state.ndim == 3  # batch_size x seq_len x hidden_dim
    assert last_hidden_state.size()[:2] == (batch_size, sequence_length)
    cls_vec = last_hidden_state[:, 0, :]
    embedding_vec = torch.nn.functional.normalize(cls_vec, dim=1)

    return embedding_vec


if __name__ == "__main__":
    main()
