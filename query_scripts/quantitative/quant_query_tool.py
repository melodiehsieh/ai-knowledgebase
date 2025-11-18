import snowflake.connector
import os
import subprocess
import anthropic  # You can use this for Claude if routing via OpenRouter or similar
from dotenv import load_dotenv
load_dotenv()
# Step 0: Configure Claude (replace with actual key and model)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.txt")

claude_model = "claude-3-opus"  # or your preferred route/model

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

# Step 1: Snowflake connection details
conn = snowflake.connector.connect(
    
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASS"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

cursor = conn.cursor()

def load_schema_string():
    if not os.path.exists("schema.txt"):
        print("schema.txt not found. Running get_schema.py to generate schema...")
        result = subprocess.run(["python3", "get_snowflake_schema.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to generate schema.txt:\n", result.stderr)
            exit(1)
        else:
            print("Schema file created.")
    
    with open("schema.txt") as f:
        return f.read()

def ask_claude(prompt):
    """Send prompt to Claude."""
    query_engine = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    response = query_engine.messages.create(
        model="claude-sonnet-4-0",
        max_tokens = 10000,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    usage = response.usage
    cost_tracker.print_summary(usage=usage)
    return str(response.content[0].text)

def generate_sql_query(question, schema_str):
    """Ask Claude to generate a SQL query for the user's question."""
    prompt = f"""
    You are a helpful assistant. The user will ask a question that can be answered using SQL.

    Here is the database schema:\n{schema_str}

    User Question: {question}

    Please write a SQL query (Snowflake SQL) that answers the user's question.
    Respond with ONLY the SQL query, nothing else.
    """
    raw_sql = ask_claude(prompt)
    # Strip code block markers if they exist
    if raw_sql.startswith("```sql"):
        raw_sql = raw_sql.strip("`").replace("sql", "").strip()
    elif raw_sql.startswith("```"):
        raw_sql = raw_sql.strip("`").strip()
    return raw_sql.strip()

def execute_query(sql):
    """Run the SQL query on Snowflake and return results."""
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except Exception as e:
        return [], [["SQL Error", str(e)]]

def answer_with_context(question, sql_results):
    """Send the SQL results back to Claude with the original question to get a natural language answer."""
    columns, rows = sql_results
    formatted_results = f"Columns: {columns}\nRows:\n" + "\n".join(str(row) for row in rows)

    prompt = f"""
The user asked: "{question}"

Here is the raw SQL query result:\n{formatted_results}

Please answer the user's question in plain English, using the SQL result to guide your answer.
"""
    return ask_claude(prompt)

def main():
    user_question = input("Enter your question: ")

    print("\n Getting schema...")
    schema_str  = load_schema_string()
    #print(schema_str)

    print("\n generating SQL query")
    sql_query = generate_sql_query(user_question, schema_str)
    print(f"\n Generated SQL:\n{sql_query}")

    print("\n running SQL query")
    sql_results = execute_query(sql_query)
    print("\nSQL results:\n")
    print(sql_results)

    print("\n feeding SQL results and original question to claude")
    answer = answer_with_context(user_question, sql_results)

    print("\n Final Answer:\n")
    print(answer)

if __name__ == "__main__":
    main()
