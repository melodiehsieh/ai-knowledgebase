from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import anthropic

from qdrant_client import QdrantClient

from dotenv import load_dotenv

import os

load_dotenv()

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

''' 
Cost tracker pulled from Snehith's AIdoc tool
'''

class text_format:
    GREEN = '\033[92m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    ENDC = '\033[0m'

class CostTracker:
    """Track Claude API usage and costs"""
    
    # Claude Sonnet 4.0 pricing (as of 2024)
    INPUT_COST_PER_1K = 0.003  # $0.003 per 1K input tokens
    OUTPUT_COST_PER_1K = 0.015  # $0.015 per 1K output tokens
    
    def print_summary(self, usage):
        """Print cost summary"""

        input_cost = (usage.input_tokens / 1000) * self.INPUT_COST_PER_1K
        output_cost = (usage.output_tokens / 1000) * self.OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost

        print(f"\n{text_format.BOLD}=== Claude API Usage Summary ==={text_format.ENDC}")
        print(f"Input Tokens: {text_format.GREEN}{usage.input_tokens:,}{text_format.ENDC}")
        print(f"Output Tokens: {text_format.GREEN}{usage.output_tokens:,}{text_format.ENDC}")
        print(f"Total Cost: {text_format.GREEN}${total_cost:.4f}{text_format.ENDC}")

cost_tracker = CostTracker()

def answer_query(retrievers, question):
    print("\n\nGenerating Answer...")

    for retriever in retrievers:
        retrieved = retriever.retrieve(question)
        context = ''
        docs = []
        for doc in retrieved:
            if doc in docs:
                continue
            print('\n\nDocument Retrieved:\n\n' + doc.node.text[:50] + "...")
            print(str(doc.score))
            context += doc.node.text
            docs.append(doc.node.text)

    query_engine = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    #Swap the commented lines depending on if you are using Confluence or Jira
    instructions = "Instructions: You are assisting [redacted] with a broad range of questions their employees may have. " \
    "You are provided data that can be sourced from Jira issue tickets created by customers using [redacted] services, Confluence documentation articles created by employees, and official Itential product documentation. " \
    "The information provided consists of a 5 documents that may be relevant to the question." \
    "If the document is from Jira, pay extra attention to the Description of Issue, Root Cause Category fields, and Comments, if provided. The comments may include the steps to the solution of the issue." \
    "If the document is from Confluence, pay extra attention to the Confluence Space Name and Page Title, if provided." \
    "If the document is from [redacted] Docs, pay extra attention to the Page Title."\
    "For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially."\
    "Your response should begin by directly answering the question. Avoid restating these instructions. Avoid beginning your response with 'Based on the documentation provided' or similar introductions.\n\n" 
    #Include one or more example Q&As here

    response = query_engine.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=10000,
        system=instructions + "\n Data: " + context,
        messages=[
            {
                "role": "user",
                "content": "Question: " + question,
            }
        ],
    )

    print("\n\nAnswer: " + str(response.content[0].text))

    usage = response.usage
    cost_tracker.print_summary(usage=usage)

    return str(response.content[0].text)

def get_index_from_qdrant(db_name, embed_model):
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vector_store = QdrantVectorStore(collection_name=db_name, client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

def main(question):

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    index_combined = get_index_from_qdrant("qdrant_comprehensive_db", embed_model)
    retriever = index_combined.as_retriever(similarity_top_k=2)

    #BM25 retriever doesn't work with remote vector db, so need to fetch all nodes from Qdrant
    source_nodes = index_combined.as_retriever(similarity_top_k=42000).retrieve("blank")
    nodes = [x.node for x in source_nodes]
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

    response = answer_query([bm25_retriever, retriever], question)

    return response

if __name__ == '__main__':
    exit(main())
