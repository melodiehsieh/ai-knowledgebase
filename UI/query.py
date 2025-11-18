from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import snowflake.connector

from datetime import datetime
from logger import log_chat_to_snowflake, generate_request_id
from retrieve_sparse_docs import retrieve_sparse_docs

import re
import os
import time
from contextlib import contextmanager
import asyncio
from anthropic import AsyncAnthropic

from dotenv import load_dotenv
load_dotenv()

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

@contextmanager
def time_block(name):
    start = time.time()
    yield
    end = time.time()
    elapsed = round(end - start, 3)
    print(f"{text_format.PURPLE}[Timer] {name}: {elapsed} seconds{text_format.ENDC}")

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
    
    def print_summary(self, usages):
        """Print cost summary"""
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        for usage in usages:
            input_cost = (usage.input_tokens / 1000) * self.INPUT_COST_PER_1K
            output_cost = (usage.output_tokens / 1000) * self.OUTPUT_COST_PER_1K
            total_cost += input_cost + output_cost

            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens

        print(f"\n{text_format.BOLD}=== Claude API Usage Summary ==={text_format.ENDC}")
        print(f"Input Tokens: {text_format.GREEN}{total_input_tokens:,}{text_format.ENDC}")
        print(f"Output Tokens: {text_format.GREEN}{total_output_tokens:,}{text_format.ENDC}")
        print(f"Total Cost: {text_format.GREEN}${total_cost:.4f}{text_format.ENDC}")

cost_tracker = CostTracker()

def get_index_from_qdrant(db_name, embed_model):
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vector_store = QdrantVectorStore(collection_name=db_name, client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

def get_index_from_qdrant_sparse(db_name, embed_model):
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vector_store = QdrantVectorStore(collection_name=db_name, client=qdrant_client, enable_hybrid=True, fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions", batch_size=1)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

def run_query(query):
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASS"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),

    )

    cursor = conn.cursor()
    dataframe = cursor.execute(query).fetchall()
    print(dataframe)
    return dataframe

def get_table_schema(table_name):
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASS"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),

    )

    cursor = conn.cursor()
    schema_info = cursor.execute(f"SELECT COLUMNS FROM DEV_DB.DEV_JSHAH.AI_SNOWFLAKE_METADATA WHERE TABLE_VIEW_NAME = '{table_name}'").fetchall()
    return schema_info

async def queries(qual_instructions, qual_context, question, quant_sql_instructions, combined_instructions, quant_snowflake_context, table_descriptions, embed_model, status_callback=None):
    request_date = datetime.utcnow()
    request_id = generate_request_id(question, request_date)
    start_time = time.time()


    query_engine = AsyncAnthropic(api_key=CLAUDE_API_KEY)

    with time_block("Qual + SQL Claude calls"):
        qual_task = query_engine.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=5000,
            system=qual_instructions + "\n Context: " + qual_context,
            messages=[
                {
                    "role": "user",
                    "content": "Question: " + question,
                }
            ],
        )
        if status_callback:
            status_callback("Retrieving context documents...")
        snowflake_query_task = query_engine.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=300,
            system=quant_sql_instructions + "\n Snowflake Tables Info: " + str(quant_snowflake_context),
            messages=[
                {
                    "role": "user",
                    "content": "Question: " + question,
                }
            ],
        )

        qual_response, snowflake_query = await asyncio.gather(qual_task, snowflake_query_task)


    print("\n\nQual: " + qual_response.content[0].text)

    snowflake_query_usage = snowflake_query.usage

    # preprocessing the query before sending it to Snowflake.
    # sometimes there is 'chain of thought', where Claude generates multiple queries after realizing the previous one is incorrect
    snowflake_query = snowflake_query.content[0].text
    snowflake_query = re.sub(r"\n\n.*\n\n", "split", snowflake_query) # for when multiple queries are generated
    snowflake_query = snowflake_query.split('split')[-1] # this assumes the last query is the correct one. Not sure how else to evaluate
    snowflake_query = snowflake_query.replace('```sql', "")
    snowflake_query = snowflake_query.replace('```', "")
    snowflake_query = snowflake_query.replace('\n', " ")

    print("\n\nSnowflake Query:" + snowflake_query)
    
    with time_block("Execute Snowflake Query + Source Retrieval"):
        dataframe_task = asyncio.to_thread(run_query, snowflake_query)
    
        async def get_source():
            documents = [Document(text=str(d)) for d in table_descriptions]
            quant_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            
            source_table_columns = snowflake_query[6:].split("WHERE")[0] #just getting which tables and columns were used in the query
            source_retriever = quant_index.as_retriever(similarity_top_k=1)
            retrieved = source_retriever.retrieve(source_table_columns)
            source = retrieved[0].node.text
            return source
        
        source_task = get_source()
        
        try:
            dataframe, source = await asyncio.gather(dataframe_task, source_task)

            if status_callback:
                status_callback("Searching data warehouse...")
        except:
            print("Error: Snowflake query failed.")
            print(f"\n\n{text_format.BOLD}Final Synthesized Answer:{text_format.ENDC}\n{qual_response.content[0].text}")

            qual_usage = qual_response.usage
            cost_tracker.print_summary([qual_usage, snowflake_query_usage])

            end_time = time.time()
            runtime_seconds = round(end_time - start_time, 2)

            log_chat_to_snowflake(
                request_date=datetime.utcnow(),
                user_question=question,
                retrieved_docs=str(quant_snowflake_context) + str(qual_context),
                generated_sql=snowflake_query,
                sql_result_data='ERROR',
                response=qual_response.content[0].text,
                runtime_seconds=runtime_seconds,
                request_id=request_id,
                user_rating=None,          
                user_feedback=None         
            )

            return qual_response.content[0].text, request_id
    
    if dataframe == []:
        # the query returned nothing - return just the qual result
        print("Error: Snowflake query did not return anything.")
        print(f"\n\n{text_format.BOLD}Final Synthesized Answer:{text_format.ENDC}\n{qual_response.content[0].text}")

        qual_usage = qual_response.usage
        cost_tracker.print_summary([qual_usage, snowflake_query_usage])

        end_time = time.time()
        runtime_seconds = round(end_time - start_time, 2)

        log_chat_to_snowflake(
            request_date=datetime.utcnow(),
            user_question=question,
            retrieved_docs=str(quant_snowflake_context) + str(qual_context),
            generated_sql=snowflake_query,
            sql_result_data=dataframe,
            response=qual_response.content[0].text,
            runtime_seconds=runtime_seconds,
            request_id=request_id,
            user_rating=None,          
            user_feedback=None         
        )

        return qual_response.content[0].text, request_id
    
    print('\n\nQuery Source:\n\n' + source)

    '''
    with time_block("Quant Claude Response"):
        quant_response = await query_engine.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=200,
            system=data_instructions + "\n Data: " + str(dataframe) + "\n Query: " + str(snowflake_query) + "\n Relevant Table Info: " + str(source),
            messages=[
                {
                    "role": "user",
                    "content": "Question: " + question,
                }
            ],
        )
    '''

    #print("\n\nQuant: " + quant_response.content[0].text) # replaced with approach 1 impl

    final_response = await query_engine.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=600,
        system=combined_instructions + "\nDocumentation Summary: " + qual_response.content[0].text + "\nData: " + str(dataframe) + "\nQuery: " + str(snowflake_query) + "\nRelevant Table Info: " + str(source),
        messages=[
            {
                "role": "user",
                "content": "Question: " + question,
            }
        ],
    )

    print(f"\n\n{text_format.BOLD}Final Synthesized Answer:{text_format.ENDC}\n{final_response.content[0].text}")
    
    qual_usage = qual_response.usage
    #quant_usage = quant_response.usage
    final_usage = final_response.usage
    cost_tracker.print_summary([qual_usage, snowflake_query_usage, final_usage])

    end_time = time.time()
    runtime_seconds = round(end_time - start_time, 2)

    log_chat_to_snowflake(
        request_date=datetime.utcnow(),
        user_question=question,
        retrieved_docs=str(quant_snowflake_context) + str(qual_context),
        generated_sql=snowflake_query,
        sql_result_data=dataframe,
        response=final_response.content[0].text,
        runtime_seconds=runtime_seconds,
        request_id=request_id,
        user_rating=None,          
        user_feedback=None         
    )

    return final_response.content[0].text, request_id

async def main(question, status_callback=None):

    if status_callback:
        status_callback("Setting up embedding model...")
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # prompts
    qual_instructions = "Instructions: You are assisting [redacted], with a broad range of questions their employees may have. " \
    "You are provided data that can be sourced from Jira issue tickets created by customers using [redacted] services, Confluence documentation articles created by employees, and official [redacted] product documentation. " \
    "The information provided consists of a 5 documents that may be relevant to the question." \
    "If the document is from Jira, pay extra attention to the Description of Issue, Root Cause Category fields, and Comments, if provided. The comments may include the steps to the solution of the issue." \
    "If the document is from Confluence, pay extra attention to the Confluence Space Name and Page Title, if provided." \
    "If the document is from Ite[redacted]ntial Docs, pay extra attention to the Page Title."\
    "For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially."\
    "Your response should begin by directly answering the question. Avoid restating these instructions. Avoid beginning your response with 'Based on the documentation provided' or similar introductions.\n\n" 
    # include one or more example Q&As here

    quant_sql_instructions = "You are assisting [redacted], with their automated Snowflake query system. " \
    "Your task is to create a SQL query from a user's question that can successfully query Snowflake and yield an informative result. " \
    "You are provided a few paragraphs containing information about Snowflake tables and columns, along with the schemas for each table. Assume these are the only tables of the data warehouse you can query. " \
    "It is essential that this query runs without errors. You must ensure that the columns correspond with the correct tables, databases, and schemas in your generated query. " \
    "Do not assume the contents of cells unless the column data type specifies it. " \
    "DO NOT make up names of tables, columns, schemas, or databases beyond what is provided in the info. This will cause the query to fail, thus causing you to fail this task." \
    "If there is not enough information in the schema info to make a specific query, broaden the query columns in order to guarantee results to the user. You should still aim to output a small number of rows or 1 row unless the question explicitly asks for all relevant results. " \
    "Keep in mind case sensitivities and other relevant considerations as you form the query. " \
    "For example, if you are asked to find the number of requests made by Company X, you should include both 'Company X' and 'company x' in your query. " \
    "Be mindful of how you deal with null values. Almost every column can contain nulls. " \
    "Prioritize simple queries that will run without errors, even if the result is less specific to the question. " \
    "Only output the final code. Do not include any explanations or chain of thought in your response.\n\n"

    combined_instructions = "You are assisting [redacted] with questions they have about company data and documentation. " \
    "Your task is to combine a documentation summary and relevant data to the user's question into one concise and comprehensive answer. " \
    "You are provided the documentation summary and data along with the user's question, the SQL query that was used to fetch the data, and information about the source database for context. " \
    "The data could range from a dataframe to a single number. Use the data, query, and table context directly to form your response to the question. " \
    "Assume the columns in the table context are in order as they appear in the actual table. " \
    "Do not attempt to analyze the data, including counting or numerical operations. " \
    "Do not assume units of data, or any information about the source table beyond what is provided to you.\n" \
    "Prioritize factual accuracy and avoid redundancy. Choose the most informative parts of the documentation summary and data and integrate them into your response. " \
    "If the documentation and data contradict each other, briefly acknowledge and explain the discrepancy if possible. " \
    "If either of the documentation or data is not relevant to the question, do not include it in your response." \
    "Do not comment on the source of the data being qualitative or quantitative. " \
    "Your response should begin by directly answering the question. Avoid restating these instructions. Avoid beginning your response with 'Based on the documentation provided' or similar introductions.\n\n" 

    # fetch contexts
    print("\n\nGenerating Answer...")
    
    # Set up retrievers
    
    #ONLY DENSE VECTORS
    '''
    qual_index = get_index_from_qdrant("qdrant_comprehensive_db", embed_model)
    quant_index = get_index_from_qdrant("snowflake_table_desc", embed_model)
    qual_retriever = qual_index.as_retriever(similarity_top_k=4)
    quant_retriever = quant_index.as_retriever(similarity_top_k=3)
    '''

    #SPARSE AND DENSE VECTORS
    qual_index = get_index_from_qdrant_sparse("qdrant_sparse", embed_model)
    quant_index = get_index_from_qdrant_sparse("snowflake_table_desc_sparse", embed_model)
    qual_retriever = qual_index.as_retriever(sparse_top_k=4200, similarity_top_k=4200, vector_store_query_mode="hybrid")
    quant_retriever = quant_index.as_retriever(sparse_top_k=5, similarity_top_k=3, vector_store_query_mode="hybrid")

    # Retrieve qual context documents
    qual_retrieved = qual_retriever.retrieve(question)

    qual_context = ""
    csv = ""
    for doc in qual_retrieved:
        csv += str(doc.score) + '\n'
    
    with open('scores_relevant_question.csv', 'w') as file:
        file.write(csv)
    
    top_docs = retrieve_sparse_docs()
    qual_retrieved = qual_retrieved[:top_docs]

    #COMMENT OUT ABOVE SECTION AND UNCOMMENT THE ONLY DENSE VECTORS BLOCK AND THE LINE BELOW FOR "NORMAL" VECTOR SEARCH
    
    # qual_retrieved = qual_retriever.retrieve(question)
    qual_context = ''.join([doc.node.text for doc in qual_retrieved])

    for doc in qual_retrieved:
        print('\n\nDocument Retrieved:\n\n' + doc.node.text[:50] + "...")
        print(str(doc.score))

    quant_retrieved = quant_retriever.retrieve(question)

    quant_snowflake_context = []
    table_descriptions = []
    for doc in quant_retrieved:
        table_name = doc.node.text.split(", Description")[0].split("Table: ")[1]
        schema = get_table_schema(table_name)
        print('\n\nTable Retrieved:\n\n' + doc.node.text)
        print(schema)
        print(str(doc.score))
        table_descriptions.append(doc.node.text)
        quant_snowflake_context.append(doc.node.text + "\n" + str(schema))

    # Run the core async logic (Claude + Snowflake)
    final_answer = await queries(
        qual_instructions,
        qual_context,
        question,
        quant_sql_instructions,
        combined_instructions,
        quant_snowflake_context,
        table_descriptions,
        embed_model,
        status_callback
    )

    # After LLM + data queries complete
    if status_callback:
        status_callback("Sending data to LLM...")  # More accurate after LLM query
        status_callback("Generating final response...")  # Can keep if last phase

    return final_answer