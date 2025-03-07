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

# def evaluate(
#     dataset,
#     embed_model,
#     top_k=5,
#     verbose=False,
# ):
#     corpus = dataset.corpus
#     queries = dataset.queries
#     relevant_docs = dataset.relevant_docs

#     nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
#     index = VectorStoreIndex(
#         nodes, embed_model=embed_model, show_progress=True
#     )
#     retriever = index.as_retriever(similarity_top_k=top_k)

#     eval_results = []
#     for query_id, query in tqdm(queries.items()):
#         retrieved_nodes = retriever.retrieve(query)
#         retrieved_ids = [node.node.node_id for node in retrieved_nodes]
#         expected_id = relevant_docs[query_id][0]
#         is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

#         eval_result = {
#             "is_hit": is_hit,
#             "retrieved": retrieved_ids,
#             "expected": expected_id,
#             "query": query_id,
#         }
#         eval_results.append(eval_result)
#     return eval_results




