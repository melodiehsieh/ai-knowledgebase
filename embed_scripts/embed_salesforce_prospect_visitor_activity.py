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
prospect_rows = ticket_dataframe.values.tolist()

# cursor.execute("SELECT ISSUE_ID, COMMENT FROM ISD_ISSUE_COMMENT_HISTORY")
# comment_dataframe = cursor.fetch_pandas_all()
# comment_cols = comment_dataframe.values.tolist()


docs_dict = {}
for x in prospect_rows:
    prospect_str = "\nSalesforce Contact ID: " + (str(x[0]) if x[0] else "") + "\nSalesforce Account ID: " + (str(x[1]) if x[1] else "") + "\nFirst Name: " + (str(x[2]) if x[2] else "") + "\nLast Name: " + (str(x[3]) if x[3] else "") + "\nAccount: " + (str(x[4]) if x[4] else "") + "\nHow did they hear about [redacted]?: " + (str(x[5]) if x[5] else "") + "\n6Sense Buying Stage: " + (str(x[6]) if x[6] else "") + "\nPerson Job Level: " + (str(x[7]) if x[7] else "") + "\nPerson Job Function: " + (str(x[8]) if x[8] else "") + "\nAccount Strategy: " + (str(x[9]) if x[9] else "") + "\nMarket Segment: " + (str(x[10]) if x[10] else "") + "\nAccount Region: " + (str(x[11]) if x[11] else "") + "\nEmail: " + (str(x[12]) if x[12] else "") + "\nVisitor Activity ID: " + (str(x[13]) if x[13] else "") + "\nVisit ID: " + (str(x[14]) if x[14] else "") + "\nVisitor Activity Sub-Type: " + (str(x[15]) if x[15] else "") + "\nVisitor Activity Type: " + (str(x[16]) if x[16] else "") + "\nEmail Sent Date: " + (str(x[17]) if x[17] else "") + "\nEmail ID: " + (str(x[18]) if x[18] else "") + "\nList Email ID: " + (str(x[19]) if x[19] else "") + "\nList Email Tag: " + (str(x[20]) if x[20] else "") + "\nActivity Detail: " + (str(x[21]) if x[21] else "") + "\nEmail Subject: " + (str(x[22]) if x[22] else "") + "\nVisitor Activity Date: " + (str(x[23]) if x[23] else "") + "\nWebsite URL: " + (str(x[24]) if x[24] else "") + "\nWebsite URL Host: " + (str(x[25]) if x[25] else "") + "\nWebsite URL Path: " + (str(x[26]) if x[26] else "") + "\nWebsite Title: " + (str(x[27]) if x[27] else "") + "\nVisitor Visit Page View Count: " + (str(x[28]) if x[28] else "") + "\nVisitor Website Duration (Sec): " + (str(x[29]) if x[29] else "") + "\nFirst Page Viewed Date by Visitor: " + (str(x[30]) if x[30] else "") + "\nLast Page Viewed Date by Visitor: " + (str(x[31]) if x[31] else "") + "\nWebsite Visit Date: " + (str(x[32]) if x[32] else "")

    
    prospect_id = x[0]

    docs_dict[prospect_id] = prospect_str


                
docs = list(docs_dict.values())
documents = []
for i, d in enumerate(docs):
    print(f"\n--- Document {i + 1} ---\n")
    print(d)
    documents.append(Document(text=d))


documents = [Document(text=d) for d in docs][:20]
print(documents)

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# qdrant_client = QdrantClient(
#     url=os.getenv('QDRANT_URL'), 
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

# collection_name = "salesforce"

# if collection_name in [c.name for c in qdrant_client.get_collections().collections]:
#     qdrant_client.delete_collection(collection_name=collection_name)

# qdrant_client.create_collection(
#     collection_name=collection_name,
#     vectors_config={"custom_vector": VectorParams(distance=Distance.COSINE, size=768)}
# )

# vector_store = QdrantVectorStore(collection_name=collection_name, client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )