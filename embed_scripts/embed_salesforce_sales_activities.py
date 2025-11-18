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
    prospect_str = "\nUnique ID: " + (str(x[0]) if x[0] else "") + "\nTask or Event: " + (str(x[1]) if x[1] else "") + "\nActivity Status: " + (str(x[2]) if x[2] else "") + "\nActivity Type: " + (str(x[3]) if x[3] else "") + "\nActivity Subtype: " + (str(x[4]) if x[4] else "") + "\nMeeting Obtained Through: " + (str(x[5]) if x[5] else "") + "\nCall Result: " + (str(x[6]) if x[6] else "") + "\nCall Duration: " + (str(x[7]) if x[7] else "") + "\nCall Sentiment: " + (str(x[8]) if x[8] else "") + "\nActivity Date: " + (str(x[9]) if x[9] else "") + "\nActivity Subject: " + (str(x[10]) if x[10] else "") + "\nActivity Description: " + (str(x[11]) if x[11] else "") + "\nCustomer Sentiment: " + (str(x[12]) if x[12] else "") + "\nMeeting Minutes: " + (str(x[13]) if x[13] else "") + "\nPOC Use Cases: " + (str(x[14]) if x[14] else "") + "\nCustomer Persona: " + (str(x[15]) if x[15] else "") + "\nLead Source Category: " + (str(x[16]) if x[16] else "") + "\nLead Source Type: " + (str(x[17]) if x[17] else "") + "\nLead Source Detail: " + (str(x[18]) if x[18] else "") + "\nQualified Lead (Yes/No): " + (str(x[19]) if x[19] else "") + "\nReason Disqualified: " + (str(x[20]) if x[20] else "") + "\nAccount Name: " + (str(x[21]) if x[21] else "") + "\nAccount Country: " + (str(x[22]) if x[22] else "") + "\nAccount Primary Industry: " + (str(x[23]) if x[23] else "") + "\nAccount Sub Industry: " + (str(x[24]) if x[24] else "") + "\nMarket Segment: " + (str(x[25]) if x[25] else "") + "\nAccount Type: " + (str(x[26]) if x[26] else "") + "\nOpportunity Name: " + (str(x[27]) if x[27] else "") + "\nOpportunity Type: " + (str(x[28]) if x[28] else "") + "\nOpportunity Stage: " + (str(x[29]) if x[29] else "") + "\nTotal Opportunity $ Value: " + (str(x[30]) if x[30] else "") + "\nOpportunity Annual Contract Value: " + (str(x[31]) if x[31] else "") + "\nOpportunity Close Date: " + (str(x[32]) if x[32] else "") + "\nOpportunity Status: " + (str(x[33]) if x[33] else "") + "\nQualified start date: " + (str(x[34]) if x[34] else "") + "\nEducate [redacted] Vision start date: " + (str(x[35]) if x[35] else "") + "\nConfirm Need & Impact start date: " + (str(x[36]) if x[36] else "") + "\nValidate Solution start date: " + (str(x[37]) if x[37] else "") + "\nNegotiation start date: " + (str(x[38]) if x[38] else "") + "\nOpportunity close date: " + (str(x[39]) if x[39] else "") + "\nDiscovery duration: " + (str(x[40]) if x[40] else "") + "\nQualified duration: " + (str(x[41]) if x[41] else "") + "\nEducate [redacted] Vision duration: " + (str(x[42]) if x[42] else "") + "\nConfirm Need & Impact duration: " + (str(x[43]) if x[43] else "") + "\nValidate Solution duration: " + (str(x[44]) if x[44] else "") + "\nNegotiation duration: " + (str(x[45]) if x[45] else "") + "\nEmail Click count: " + (str(x[46]) if x[46] else "") + "\nSales Attribution: " + (str(x[47]) if x[47] else "")
    
    unique_id = x[0]

    docs_dict[unique_id] = prospect_str

                
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