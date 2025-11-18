import anthropic
import snowflake.connector
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
import os

# Approach 1: Broad schema of prod dbs to give to Claude in format {db_1: [{schema_1: [{table_1: [table_1_columns]}, ...]}, ...}
# This was too large of an object to feed to Claude (> $10 per query). I considered embedding + vector search approach but depending on chunking, would lose context (always need the db name, schema name, and table name along with all table columns ideally)

# Approach 2: A list of schema objects to give to Claude in format [{'db': db_1, 'schema': schema_1, 'table': table_1, 'columns': [20 columns]}, ...] and each list element is a chunk, then embed

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

''' 
Cost tracker partially pulled from Snehith's AIdoc tool
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

def get_index():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    qdrant_client = QdrantClient(
        url="http://localhost:6333"
    )
    vector_store = QdrantVectorStore(collection_name="snowflake_dwh_schema", client=qdrant_client, dense_vector_name="custom_vector", sparse_vector_name="custom_vector")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index
    
def answer_question(question):

    print("\n\nGenerating Answer...")

    index = get_index()
    # Consider switching to BM25
    retriever = index.as_retriever(similarity_top_k=3)
    retrieved = retriever.retrieve(question)

    context = ''
    for doc in retrieved:
        print('\n\nDocument Retrieved:\n\n' + doc.node.text)
        print(str(doc.score))
        context += doc.node.text

    sql_instructions = "You are assisting Itential, a network automation and orchestration company, with their automated Snowflake query system. " \
    "Your goal is to create a SQL query from a user's question that can successfully query Snowflake and yield a relevant result. " \
    "You are provided information about the Snowflake databases, schemas, tables, columns, and info about the columns. Assume these are the only parts of the data warehouse you can query. " \
    "It is essential that this query runs with no modifications. Do not infer the contents of cells unless the column data type specifies it. Do not make up names of tables, columns, schemas, or databases beyond what is provided in the info. " \
    "For example, if the question includes a term not mentioned within the Snowflake information, do not make up a table or column name that matches the term. This will cause the query to fail, thus causing you to fail this task." \
    "If there is not enough information in the schema info to make a specific query, broaden the query in order to guarantee results to the user. " \
    "Keep in mind case sensitivities and other relevant considerations as you form the query. " \
    "For example, if you are asked to find the number of requests made by Company X, you should include both 'Company X' and 'company x' in your query. " \
    "Be mindful of how you deal with null values. Almost every column could contain nulls. " \
    "There could be multiple ways to arrive at the same answer. Choose the option with the least risk of failing to execute and the highest chance of being accurate. " \
    "Only output the code. Do not include any explanations in your response."

    # get cost
    query_engine = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    OUTPUT_COST_PER_1K = 0.015
    INPUT_COST_PER_1K = 0.003
    response = query_engine.messages.count_tokens(
        model="claude-sonnet-4-0",
        system=sql_instructions + "\n Snowflake Schema: " + str(context),
        messages=[
            {
                "role": "user",
                "content": "Question: " + question,
            }
        ],
    )
    input_tokens = int(response.input_tokens)
    cost = (input_tokens / 1000) * INPUT_COST_PER_1K + (300 / 1000) * OUTPUT_COST_PER_1K
    print(str(input_tokens) + " input tokens, Estimated Cost: $" + str(cost))



    query_engine = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    snowflake_query = query_engine.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=300,
        system=sql_instructions + "\n Snowflake Schema: " + str(context),
        messages=[
            {
                "role": "user",
                "content": "Question: " + question,
            }
        ],
    )

    snowflake_query = snowflake_query.content[0].text
    snowflake_query = snowflake_query.replace('```sql', "")
    snowflake_query = snowflake_query.replace('```', "")
    snowflake_query = snowflake_query.replace('\n', " ")
    print(snowflake_query)

    dataframe = run_query(snowflake_query)
    if dataframe == []:
        return "Error: unable to execute query."
    
    # get source from generated query
    source = ''
    retriever = index.as_retriever(similarity_top_k=1)
    retrieved = retriever.retrieve(snowflake_query)
    for doc in retrieved:
        print('\n\nDocument Retrieved:\n\n' + doc.node.text)
        source += doc.node.text

    data_instructions = "You are assisting Itential, a network automation and orchestration company, with data-oriented questions they have about company operations. " \
    "You are provided relevant data to the user's question, the SQL query that was used to fetch the data, and information about the database for context. " \
    "The data could range from a dataframe to a single number. Use the data, query, and table context directly to form your response to the question. " \
    "Assume the columns in the table context are in order as they appear in the actual table. " \
    "If you are unsure what the answer should be, provide as much information as you can based on the provided information. " \
    "Do not attempt to analyze the data, including counting or numerical operations. " \
    "Do not assume units of data, or any information about the source table beyond what is provided to you. Answer the question as accurately as possible. "

    # get cost
    response = query_engine.messages.count_tokens(
        model="claude-sonnet-4-0",
        system=data_instructions + "\n Data: " + str(dataframe) + "\n Query: " + str(snowflake_query) + "\n Relevant Table Info: " + str(source),
        messages=[
            {
                "role": "user",
                "content": "Question: " + question,
            }
        ],
    )
    input_tokens = int(response.input_tokens)
    cost = (input_tokens / 1000) * INPUT_COST_PER_1K + (200 / 1000) * OUTPUT_COST_PER_1K
    print(str(input_tokens) + " input tokens, Estimated Cost: $" + str(cost))



    response = query_engine.messages.create(
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

    print(response.content[0].text)
    return response.content[0].text

def run_query(query):

    user=os.getenv("SNOWFLAKE_USER")
    password=os.getenv("SNOWFLAKE_PASS")
    account=os.getenv("SNOWFLAKE_ACCOUNT")
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
    database=os.getenv("SNOWFLAKE_DATABASE")
    schema=os.getenv("SNOWFLAKE_SCHEMA")


    conn = snowflake.connector.connect(
        user=user,
        password=password,
        authenticator='USERNAME_PASSWORD_MFA',
        clientRequestMFAToken=True,
        account=account,
        warehouse=warehouse,
    )

    cursor = conn.cursor()
    dataframe = cursor.execute(query).fetchall()
    print(dataframe)
    return dataframe

def main():
    question = input("Please enter your question: ")
    answer_question(question)

if __name__ == '__main__':
    exit(main())


