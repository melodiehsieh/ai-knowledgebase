from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from qdrant_client import models
from qdrant_client import QdrantClient
import snowflake.connector
import os
from dotenv import load_dotenv
load_dotenv()

def template_map(x):
    template = "Confluence Space Name: " + (x[0] if x[0] else "") + ", Page Title: " + (x[1] if x[1] else "") + ", Page URL: " + (("https://[redacted].atlassian.net/wiki/spaces/LLMKB/pages/" + str(x[2])) if x[2] else "") + ", Page Content: " + (x[3] if x[3] else "")
    return template

conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASS'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
)

cursor = conn.cursor()

#UPDATE THE FOLLOWING ONCE CONFLUENCE DATA MODELLING IS DONE
cursor.execute("USE WAREHOUSE DEV_WH")
cursor.execute("USE DATABASE DEV_DB")
cursor.execute("USE SCHEMA DEV_DB.DEV_JSHAH")
cursor.execute("[redacted]")

dataframe = cursor.fetch_pandas_all()
dataframe = dataframe.drop_duplicates()
cols = dataframe.values.tolist()

docs = list(map(lambda x: template_map(x), cols))
documents = [Document(text=d) for d in docs]
for doc in documents:
    print(doc)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "qdrant_confluence_sparse"

# create new collection (w sparse)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "text-dense": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": models.SparseVectorParams(
            index=models.SparseIndexParams()
        )
    },
)

vector_store = QdrantVectorStore(collection_name=collection_name, client=qdrant_client, dense_vector_name="text-dense",
                sparse_vector_name="text-sparse", enable_hybrid=True, fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions", batch_size=1)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)