from pymilvus import model
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")


embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))


query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)

