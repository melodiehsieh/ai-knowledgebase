# logger.py

import os
import snowflake.connector
from datetime import datetime
import hashlib


def generate_request_id(question: str, timestamp: datetime) -> int:
    # Combine question and timestamp string
    base_str = question + timestamp.isoformat()
    # SHA256 hash, take first 16 hex digits (64 bits)
    hash_hex = hashlib.sha256(base_str.encode()).hexdigest()[:16]
    # Convert hex to int
    return int(hash_hex, 16)

def log_chat_to_snowflake(
    request_date: datetime,
    user_question: str,
    retrieved_docs: str,
    generated_sql: str,
    sql_result_data: str,
    response: str,
    runtime_seconds: float,
    request_id: int,
    user_rating: int = None,
    user_feedback: str = None
):
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASS"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse="[redacted]",
            database="[redacted]",
            schema="[redacted]",
            authenticator='[redacted]',
            clientRequestMFAToken=[redacted],
        )
        cursor = conn.cursor()

        insert_stmt = """
            INSERT INTO [redacted]
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_stmt, (
            request_date,
            request_id,
            user_question,
            retrieved_docs,
            generated_sql,
            str(sql_result_data),  # Cast to string if it's a list or tuple
            response,
            runtime_seconds,
            user_rating,
            user_feedback
        ))
        conn.commit()

    except Exception as e:
        print(f"[LOGGING ERROR] Failed to log interaction to Snowflake:\n{e}")

    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def log_feedback(request_id: int, user_rating: int, user_feedback: str):
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASS"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse="[redacted]",
            database="[redacted]",
            schema="[redacted]",
            authenticator='USERNAME_PASSWORD_MFA',
            clientRequestMFAToken=True,
        )
        cursor = conn.cursor()

        update_sql = """
            [redacted]
        """

        cursor.execute(update_sql, (user_rating, user_feedback, request_id))
        conn.commit()

        print("Feedback logged.")

    except Exception as e:
        print(f"Failed to log feedback: {e}")

    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
