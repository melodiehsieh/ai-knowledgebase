import snowflake.connector
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
import json, os

from dotenv import load_dotenv
load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASS'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
)

cursor = conn.cursor()

ret_schema = []
cursor.execute("USE DATABASE [redacted]")
cursor.execute("USE SCHEMA [redacted]")

try:

    tables = cursor.execute("SHOW TABLES").fetchall()
    table_names_descriptions = list(map(lambda x: (x[1], x[5]), tables))

    views = cursor.execute("SHOW VIEWS").fetchall()

    view_names_descriptions = list(map(lambda x: (x[1], x[6]), views))

    tables_and_views = table_names_descriptions + view_names_descriptions

    for table_view in tables_and_views:
        table_view_name = table_view[0]
        description = table_view[1]

        if description == "" or not description:
            continue
        
        try:
            ret_schema.append(f"Table: [redacted].[redacted].{table_view_name}, Description: {description}")
            print(f"Table: [redacted].[redacted].{table_view_name}, Description: {description}")
        except Exception as e:
            print(e)
            continue

except Exception as e:
        print(e)

with open('snowflake_schema.txt', 'w') as file:
    file.write(str(ret_schema))


documents = [Document(text=str(d)) for d in ret_schema]
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant_collection = qdrant_client.create_collection(collection_name="snowflake_table_desc", vectors_config= {"custom_vector": VectorParams(distance=Distance.COSINE, size=768)})
vector_store = QdrantVectorStore(collection_name="snowflake_table_desc", client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)