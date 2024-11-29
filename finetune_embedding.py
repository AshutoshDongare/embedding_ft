import json
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")


from llama_index.finetuning import SentenceTransformersFinetuneEngine

from datasets import load_dataset


finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-m3",
    model_output_path="FineTuned_Model",
    val_dataset=val_dataset,
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()

# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core import VectorStoreIndex
# from llama_index.core.schema import TextNode
# from tqdm.notebook import tqdm
# import pandas as pd



