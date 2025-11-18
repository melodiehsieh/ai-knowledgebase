from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
import snowflake.connector
import os
import re
from dotenv import load_dotenv
load_dotenv()
QUERY = ""

conn = snowflake.connector.connect(
    query=QUERY,
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASS'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
)

cursor = conn.cursor()

cursor.execute("USE WAREHOUSE [redacted]")
cursor.execute("USE DATABASE [redacted]")
cursor.execute("USE SCHEMA [redacted].[redacted]")

cursor.execute("[redacted]")
ticket_dataframe = cursor.fetch_pandas_all()
ticket_cols = ticket_dataframe.values.tolist()

cursor.execute("[redacted]")
comment_dataframe = cursor.fetch_pandas_all()
comment_cols = comment_dataframe.values.tolist()

docs_dict = {}
for x in ticket_cols:
    ticket_str = "Issue URL: " + (("https://[redacted]/jira/servicedesk/projects/ISD/issues/" + x[16]) if x[16] else "") + ", Issue Summary: " + str(x[0] if x[0] else "") + ", Customer: " + str(x[1] if x[1] else "") + ", Platform Component: " + str(x[2] if x[2] else "") + ", Version: " + str(x[3] if x[3] else "") + ", Environment: " + str(x[4] if x[4] else "") + ", Implementation Type: " + str(x[12] if x[12] else "") + ", Issue Priority: " + str(x[5] if x[5] else "") + ", Business Impact: " + str(x[11] if x[11] else "") + ", Root Cause Category: " + str(x[6] if x[6] else "") + ".\n\nDescription of Issue: " + str(x[7] if x[7] else "") + "\n\nSteps to Reproduce Issue: " + str(x[13] if x[13] else "") + "\nExpected Results: " + str(x[14] if x[14] else "") + "\nActual Results: " + str(x[15] if x[15] else "") + ".\n\nThe issue was reported on " + str(x[8] if x[8] else "") + ((" and resolved on " + str(x[9])) if x[9] else (" and is currently unresolved")) + ((". The category of the root cause was determined to be: " + x[6]) if x[6] else (". No root cause has been determined yet."))
    
    id = x[10]
    #There are a lot of duplicate tickets
    docs_dict[id] = ticket_str

for c in comment_cols:
    id = c[0]
    if c[1]:
        #Skipping any autogen comments
        if "Hi there, we haven't heard from you in a while. If you would like to keep this request open or if you have additional information, just reply." in c[1]:
            continue
        if "We have received your ticket entry and apologize for the issues you are experiencing. Your ticket has been assigned" in c[1]:
            continue
        if id in docs_dict.keys():
            comment = c[1]
            #removing extra stuff (account ids, text formatting, embedded images)
            comment = re.sub(r'\[.*?\]', '', comment)
            comment = re.sub(r'\!.*?\!', '', comment)
            comment = re.sub(r'\{.*?\}', '', comment)

            docs_dict[id] += ("\n\nComment: " + comment)
                
docs = list(docs_dict.values())
documents = []
for i, d in enumerate(docs):
    print(f"\n--- Document {i + 1} ---\n")
    print(d)
    documents.append(Document(text=d))
# documents = [Document(text=d) for d in docs]#[:20]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "qdrant_JIRA_with_comments"

if collection_name in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.delete_collection(collection_name=collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config={"custom_vector": VectorParams(distance=Distance.COSINE, size=768)}
)

vector_store = QdrantVectorStore(collection_name=collection_name, client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)