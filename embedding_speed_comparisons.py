"""Python script to compare the speed of different embedding models."""

import argparse
import json
import logging
import os
import timeit
import uuid

import google.generativeai as gemini_client
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()

parser.add_argument("-q", type=str)
parser.add_argument("--gemini", type=bool, default=False)
parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--vector_size", type=int, default=384)
parser.add_argument("--collection", type=str, default="shooting")

parser.add_argument("--data_path", type=str, default="./data/shooting.json")
parser.add_argument("--qdrant_host", type=str, default="localhost")
parser.add_argument("--qdrant_port", type=int, default=6333)
parser.add_argument("-n", type=int, default=10)
parser.add_argument("-r", type=int, default=3)
parser.add_argument(
    "--log",
    type=str,
    default=None,
    help="Path to log file. If not specified, logs will be printed to console.",
)

args = parser.parse_args()

# Configure logging based on whether log_file is specified.
if args.log:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=args.log,
        filemode="a",
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Doc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    title: str
    source: str
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document since SHA256 hash not supported. More info: https://qdrant.tech/documentation/concepts/points/",
    )
    doc_length: int | None = None
    url: str
    embedding: list[float] | None = None
    tags: list[str]  # Fixed: list of strings, not list[str | None]
    # Needed for de-duping. We'll match the hash for incoming documents.
    content_hash: str | None = None


def load_data(path: str = "./data/shooting.json"):
    docs = []

    with open(path, "r") as f:
        j = json.load(f)
        for e in j:
            new_doc = Doc(**e)
            new_doc.doc_length = len(new_doc.text)
            docs.append(new_doc)

    return docs


def init_gemini_client() -> gemini_client.Client:
    gemini_client.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return gemini_client.Client()


def qdrant_client(host: str, port: int):
    client = QdrantClient(host=host, port=port)
    return client


def qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):

    ## Check if collection exists.
    collections = client.get_collections()
    if collection_name not in [c.name for c in collections.collections]:

        print(f"{collection_name} not found; creating")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def embed_qdrant(model, text: str) -> list[float]:
    return model.encode(text)


def embed_gemini(client: gemini_client.Client, model: str, text: str) -> list[float]:
    return client.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document",
        title="Qdrant x Gemini",
    )["embedding"]


def search(
    embed_query: list[float], collection: str, qdrant_client
) -> list[models.PointStruct]:
    qdrant_results: list[models.PointStruct] = qdrant_client.search(
        collection_name=collection,
        query_vector=embed_query,
        limit=100,
    )

    return qdrant_results


def main():
    logging.info("Starting embedding speed comparisons")

    logging.info("Loading models")

    logging.info("Initializing Qdrant client")
    qdrant_client = qdrant_client(args.qdrant_host, args.qdrant_port)
    qdrant_collection(qdrant_client, args.collection, args.vector_size)

    logging.info("Loading data")
    docs = load_data(args.data_path)

    model = None
    gclient = None
    points = None

    if args.gemini:
        gclient = init_gemini_client()

        logging.info("embedding gemini")
        for i in range(len(docs)):
            docs[i].embedding = embed_gemini(gclient, args.model, docs[i].text)

    else:
        model = SentenceTransformer(args.model)

        logging.info("embedding")
        for i in range(len(docs)):
            docs[i].embedding = embed_qdrant(model, docs[i].text)

    logging.info(docs)

    points = [
        models.PointStruct(
            id=d.doc_id,
            vector=d.embedding,
            payload=d.model_dump(exclude={"embedding"}),
        )
        for d in docs
    ]

    logging.info("Uploading to Qdrant")
    qdrant_client.upsert(collection_name=args.collection, points=points)
    logging.info("Done")

    logging.info("Performing evaluations")

    if args.gemini:
        q_embed = embed_gemini(gclient, args.model, args.q)

        # Measure search performance
        search_time = timeit.timeit(
            lambda: search(q_embed, args.collection, qdrant_client),
            number=args.n,
            repeat=args.r,
        )

        result = search(q_embed, args.collection, qdrant_client)
        logging.info(f"Search time: {search_time:.4f} seconds for {args.n} iterations")
        logging.info(result)

    else:
        q_embed = embed_qdrant(model, args.q)

        # Measure search performance
        search_time = timeit.timeit(
            lambda: search(q_embed, args.collection, qdrant_client),
            number=args.n,
            repeat=args.r,
        )

        result = search(q_embed, args.collection, qdrant_client)
        logging.info(f"Search time: {search_time:.4f} seconds for {args.n} iterations")
        logging.info(result)


if __name__ == "__main__":
    main()
