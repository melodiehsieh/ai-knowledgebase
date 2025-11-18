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
    prospect_str = "Opportunity ID: " + (str(x[0]) if x[0] else "") + "\nOpportunity name: " + (str(x[1]) if x[1] else "")+ "\nAccount ID: " + (x[2] if x[2] else "") + "\nAccount Name: " + (x[3] if x[3] else "") + "\nIndustry: " + (x[4] if x[4] else "") + "\nSub Industry: " + (x[5] if x[5] else "") + "\nAccount revenue: " + (x[6] if x[6] else "") + "\nAccount employee count: " + (x[7] if x[7] else "") + "\nMarket segment: " + (str(x[8]) if x[8] else "") + "\nAccount health: " + (str(x[9]) if x[9] else "") + "\nExpansion type: " + (x[10] if x[10] else "") + "\nTotal cost: " + (x[11] if x[11] else "") + "\nSoftware Cost: " + (str(x[12]) if x[12] else "") + "\nMaintenance Cost: " + (str(x[13]) if x[13] else "") + "\nTraining cost: " + (str(x[14]) if x[14] else "") + "\nServices cost: " + (str(x[15]) if x[15] else "") + "\nTotal deal size: " + (str(x[16]) if x[16] else "") + "\nAnnual Contract Value: " + (str(x[17]) if x[17] else "") + "\nAnnual Contract Value Deal Size: " + (str(x[18]) if x[18] else "") + "\nOn prem or SaaS: " + (str(x[19]) if x[19] else "") + "\nCreated Date: " + (str(x[20]) if x[20] else "") + "\nCurrent Stage Number: " + (str(x[21]) if x[21] else "") + "\nDiscovery Date: " + (str(x[22]) if x[22] else "")+ "\nQualified Date: " + (str(x[23]) if x[23] else "")+ "\nEducate [redacted] Vision Date: " + (str(x[24]) if x[24] else "")+ "\nConfirm Need & Impact Date: " + (str(x[25]) if x[25] else "")+ "\nValidate Solution Date: " + (str(x[26]) if x[26] else "")+ "\nNegotiation Date: " + (str(x[27]) if x[27] else "")+ "\nClose Date: " + (str(x[28]) if x[28] else "")+ "\nOpportunity Age at Close: " + (str(x[29]) if x[29] else "")+ "\nCurrent Opportunity Age: " + (str(x[30]) if x[30] else "")+ "\nCurrent Opportunity Stage Duration: " + (str(x[31]) if x[31] else "")+ "\nOpportunity Stage Name: " + (str(x[32]) if x[32] else "")+ "\nOpportunity Source: " + (str(x[33]) if x[33] else "") + "\nDeal Length: " + (str(x[34]) if x[34] else "") + "\nLead Source: " + (str(x[35]) if x[35] else "")+ "\nOpportunity Source Bucket: " + (str(x[36]) if x[36] else "")+ "\nAssigned Sales Engineer: " + (str(x[37]) if x[37] else "")+ "\nDiscovery Duration: " + (str(x[38]) if x[38] else "")+ "\nQualified Duration: " + (str(x[39]) if x[39] else "")+ "\nEducate [redacted] Vision Duration: " + (str(x[40]) if x[40] else "")+ "\nConfirm Need & Impact: " + (str(x[41]) if x[41] else "")+ "\nValidate Solution Duration: " + (str(x[42]) if x[42] else "")+ "\nNegotiation Duration: " + (str(x[43]) if x[43] else "")+ "\nTotal Contract Value: " + (str(x[44]) if x[44] else "")+ "\nDeal length in months: " + (str(x[45]) if x[45] else "")+ "\nSalesforce Contact ID: " + (str(x[46]) if x[46] else "")+ "\nNumber of Proof-of-concepts: " + (str(x[47]) if x[47] else "")+ "\nNext Step: " + (str(x[48]) if x[48] else "")+ "\nLast Activity Date: " + (str(x[49]) if x[49] else "")+ "\nCustomer Contract Expration Date: " + (str(x[50]) if x[50] else "")+ "\nRenewal Amount: " + (str(x[51]) if x[51] else "")+ "\nOpportunity Loss Reason: " + (str(x[52]) if x[52] else "")+ "\nNext Step Due Date: " + (str(x[53]) if x[53] else "")+ "\n6Sense Buying Stage: " + (str(x[54]) if x[54] else "")+ "\nOpportunity Source Detail: " + (str(x[55]) if x[55] else "")+ "\nManager Notes: " + (str(x[56]) if x[56] else "")+ "\nMonthly Recurring Revenue: " + (str(x[57]) if x[57] else "")+ "\nAnnual Recurring Revenue: " + (str(x[58]) if x[58] else "")+ "\nFirst meeting saleforce ID: " + (str(x[59]) if x[59] else "")+ "\nProspect First meeting date: " + (str(x[60]) if x[60] else "")+ "\nFirst Meeting obtained through: " + (str(x[61]) if x[61] else "")+ "\nBusiness Value Driver: " + (str(x[62]) if x[62] else "")+ "\nBusiness Value Driver Explanation: " + (str(x[63]) if x[63] else "")+ "\nLast Communication to Prospect: " + (str(x[64]) if x[64] else "")+ "\nOpportunity Sourcing: " + (str(x[65]) if x[65] else "")+ "\nCustomer Procurement Channel: " + (str(x[66]) if x[66] else "")+ "\nCustomer Renewal Date: " + (str(x[67]) if x[67] else "")+ "\nDays until next renewal for customer: " + (str(x[68]) if x[68] else "")
    
    account_id = x[2]

    docs_dict[account_id] = prospect_str


                
docs = list(docs_dict.values())
documents = []
for i, d in enumerate(docs):
    print(f"\n--- Document {i + 1} ---\n")
    print(d)
    documents.append(Document(text=d))


documents = [Document(text=d) for d in docs]#[:20]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "salesforce"

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