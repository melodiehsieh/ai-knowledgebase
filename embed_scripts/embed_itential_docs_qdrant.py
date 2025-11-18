from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# Define path to root folder
ROOT_DIR = "[redacted]"

def get_url(filepath):
    base_url = "[redacted]"
    filepath = filepath.replace("[redacted]/", "")
    if "articles" in filepath:
        filepath = filepath.replace("articles", "docs")
    if "categories" in filepath:
        filepath = filepath.replace("categories", "docs")
    filepath = filepath.replace(".md", "")
    if filepath.endswith("-1"):
        filepath = filepath[:-2]
    print(base_url + filepath)
    return base_url + filepath

def collect_markdown_documents(root_path):
    documents = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        page_url = get_url(file_path)
                        title = os.path.splitext(filename)[0]
                        template = f"Page Title: {title}, Page URL: {page_url}, Page Content: {content}"
                        documents.append(Document(text=template))
                        print(documents[len(documents) - 1])
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
    return documents

# Load documents from only .md files
documents = collect_markdown_documents(ROOT_DIR)

# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "[redacted]"

# Delete and create fresh collection
if collection_name in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.delete_collection(collection_name=collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config={"custom_vector": VectorParams(distance=Distance.COSINE, size=768)}
)
print("embedding now")
# Store in vector DB
vector_store = QdrantVectorStore(
    collection_name=collection_name,
    client=qdrant_client,
    dense_vector_name="custom_vector",
    sparse_vector_name="custom_vector"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)
