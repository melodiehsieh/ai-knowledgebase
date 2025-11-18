from query_tool import get_index_from_qdrant 
from quantitative.test_quant_LLM import run_query
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
import os
import time
from contextlib import contextmanager
import asyncio
from anthropic import AsyncAnthropic

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

async def queries(qual_instructions, qual_context, question, quant_sql_instructions, data_instructions, quant_snowflake_context, embed_model):
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

        snowflake_query_task = query_engine.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=300,
            system=quant_sql_instructions + "\n Snowflake Schema: " + str(quant_snowflake_context),
            messages=[
                {
                    "role": "user",
                    "content": "Question: " + question,
                }
            ],
        )

        qual_response, snowflake_query = await asyncio.gather(qual_task, snowflake_query_task)

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
            quant_index = get_index_from_qdrant("[redacted]", embed_model)
            source_retriever = quant_index.as_retriever(similarity_top_k=1)
            retrieved = source_retriever.retrieve(snowflake_query)
            source = ''
            for doc in retrieved:
                print('\n\nQuery Source:\n\n' + doc.node.text)
                source += doc.node.text
            return source
        
        source_task = get_source()
        
        dataframe, source = await asyncio.gather(dataframe_task, source_task)
    
    if dataframe == []:
        print("Error: unable to execute quant query.")
        print("\n\nQual: " + qual_response.content[0].text)
        exit()

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

    print("\n\nQual: " + qual_response.content[0].text)
    print("\n\nQuant: " + quant_response.content[0].text)
    final_instructions = (
        "You are assisting employees at [redacted] by synthesizing both qualitative documentation and structured data analysis into a final response.\n"
        "You are provided with two responses:\n"
        "1. A qualitative response generated from Jira, Confluence, or internal documentation.\n"
        "2. A quantitative response based on structured data from Snowflake.\n\n"
        "Your job is to combine the insights from both to produce a clear, concise, and helpful answer to the user's question.\n"
        "Prioritize factual accuracy and avoid redundancy. If one response provides additional context that enhances the other, integrate that into your summary.\n"
        "If the two responses contradict each other, briefly acknowledge and explain the discrepancy if possible.\n"
    )

    final_response = await query_engine.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=600,
        system=final_instructions,
        messages=[
            {
                "role": "user",
                "content": f"""Question: {question}
                            Qualitative Response: {qual_response.content[0].text}
                            Quantitative Response: {quant_response.content[0].text}
                            """ 
            }
        ],
    )

    print(f"\n\n{text_format.BOLD}Final Synthesized Answer:{text_format.ENDC}\n{final_response.content[0].text}")


    
    qual_usage = qual_response.usage
    quant_usage = quant_response.usage
    final_usage = final_response.usage
    cost_tracker.print_summary([qual_usage, quant_usage, snowflake_query_usage, final_usage])

async def main():
    question = input("Please enter your question: ")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # prompts
    qual_instructions = "Instructions: You are assisting [redacted] with a broad range of questions their employees may have. " \
    "You are provided data that can be sourced from Jira issue tickets created by customers using [redacted] services, Confluence documentation articles created by employees, and official [redacted] product documentation. " \
    "The information provided consists of a 5 documents that may be relevant to the question." \
    "If the document is from Jira, pay extra attention to the Description of Issue, Root Cause Category fields, and Comments, if provided. The comments may include the steps to the solution of the issue." \
    "If the document is from Confluence, pay extra attention to the Confluence Space Name and Page Title, if provided." \
    "If the document is from [redacted] Docs, pay extra attention to the Page Title."\
    "For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially."\
    "Your response should begin by directly answering the question. Avoid restating these instructions. Avoid beginning your response with 'Based on the documentation provided' or similar introductions.\n\n" 
    # include one or more example Q&As here

    quant_sql_instructions = "You are assisting [redacted], a network automation and orchestration company, with their automated Snowflake query system. " \
    "Your goal is to create a SQL query from a user's question that can successfully query Snowflake and yield a relevant result. " \
    "You are provided a few JSON objects containing information about Snowflake tables and columns. Assume these are the only parts of the data warehouse you can query. " \
    "It is essential that this query runs with no modifications. You must ensure that the columns correspond with the correct tables, databases, and schemas in your generated query. " \
    "Do not infer the contents of cells unless the column data type specifies it. Do not make up names of tables, columns, schemas, or databases beyond what is provided in the info. " \
    "For example, if the question includes a term not mentioned within the Snowflake information, do not make up a table or column name that matches the term. This will cause the query to fail, thus causing you to fail this task." \
    "If there is not enough information in the schema info to make a specific query, broaden the query in order to guarantee results to the user. " \
    "Keep in mind case sensitivities and other relevant considerations as you form the query. " \
    "For example, if you are asked to find the number of requests made by Company X, you should include both 'Company X' and 'company x' in your query. " \
    "Be mindful of how you deal with null values. Almost every column can contain nulls. " \
    "There could be multiple ways to arrive at the same answer. Choose the option with the least risk of failing to execute and the highest chance of being accurate. " \
    "Only output the final code. Do not include any explanations or chain of thought in your response."

    data_instructions = "You are assisting [redacted] with data-oriented questions they have about company operations. " \
    "You are provided relevant data to the user's question, the SQL query that was used to fetch the data, and information about the database for context. " \
    "The data could range from a dataframe to a single number. Use the data, query, and table context directly to form your response to the question. " \
    "Assume the columns in the table context are in order as they appear in the actual table. " \
    "If you are unsure what the answer should be, provide as much information as you can based on the provided information. " \
    "Do not attempt to analyze the data, including counting or numerical operations. " \
    "Do not assume units of data, or any information about the source table beyond what is provided to you. Answer the question as accurately as possible. "

    # fetch contexts
    print("\n\nGenerating Answer...")

    async def get_qual_retrievers():
        qual_index = get_index_from_qdrant("qdrant_comprehensive_db", embed_model)
        qual_retriever = qual_index.as_retriever(similarity_top_k=4)
        #qual_source_nodes = qual_index.as_retriever(similarity_top_k=42000).retrieve("blank")  # BM25 retriever doesn't work with remote vector db, so need to fetch all nodes from Qdrant. Takes a couple of seconds
        #qual_nodes = [x.node for x in qual_source_nodes]
        #qual_bm25_retriever = BM25Retriever.from_defaults(nodes=qual_nodes, similarity_top_k=3) # this was taking too long
        qual_retrievers = [qual_retriever]
        return qual_retrievers

    async def get_quant_retriever():
        quant_index = get_index_from_qdrant("snowflake_dwh_schema", embed_model)
        quant_retriever = quant_index.as_retriever(similarity_top_k=3)
        return quant_retriever
    
    qual_retrievers, quant_retriever = await asyncio.gather(get_qual_retrievers(), get_quant_retriever())

    async def retrieve_qual_docs():
        for retriever in qual_retrievers:
            qual_retrieved = retriever.retrieve(question)
            qual_context = ''
            for doc in qual_retrieved:
                print('\n\nDocument Retrieved:\n\n' + doc.node.text[:50] + "...")
                print(str(doc.score))
                qual_context += doc.node.text
        return qual_context
    
    async def retrieve_quant_docs():
        quant_retrieved = quant_retriever.retrieve(question)
        quant_snowflake_context = ''
        for doc in quant_retrieved:
            print('\n\nTable Retrieved:\n\n' + doc.node.text)
            print(str(doc.score))
            quant_snowflake_context += doc.node.text
        return quant_snowflake_context

    qual_context, quant_snowflake_context = await asyncio.gather(retrieve_qual_docs(), retrieve_quant_docs())

    # Claude queries (async) 
    await asyncio.gather(queries(qual_instructions, qual_context, question, quant_sql_instructions, data_instructions, quant_snowflake_context, embed_model))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
    
