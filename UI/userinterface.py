import streamlit as st
import random
import re
import markdown
import asyncio

from query import main  # Moved query.py into UI folder
from logger import log_feedback

st.set_page_config(
    page_title="Assistant",
    page_icon="👽",
    layout="wide"
)

# Define progress steps
STEPS = [
    "Setting up embedding model...",
    "Retrieving context documents...",
    "Searching data warehouse...",
    "Sending data to LLM...",
    "Generating final response..."
]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'progress' not in st.session_state:
    st.session_state.progress = {step: False for step in STEPS}

# ---------- STYLES ----------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .chat-message-wrapper {
        display: flex;
        margin: 0.5rem 0;
    }
    .chat-message {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        max-width: 75%;
        word-wrap: break-word;
        display: inline-block;
        white-space: pre-wrap;
    }
    .user-message-wrapper {
        justify-content: flex-end;
    }
    .assistant-message-wrapper {
        justify-content: flex-start;
    }
    .user-message {
        background-color: #DCF8C6;
        text-align: right;
    }
    .assistant-message {
        background-color: #F1F0F0;
        text-align: left;
        line-height: 120%;
    }
    .assistant-message ul {
        margin: 0;
        padding-left: 1.2em;
    }
    .assistant-message h1,
    .assistant-message h2,
    .assistant-message h3 {
        margin: 0;
        line-height: 1.3;
    }
    .feedback-wrapper {
        display: flex;
        justify-content: flex-start;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .feedback-container {
        max-width: 75%;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def clear_input():
    st.session_state.user_input = ""

def submit_feedback(req_id, user_rating, user_feedback):
    print(req_id, user_rating, user_feedback)
    log_feedback(str(req_id), user_rating, user_feedback)
    st.success("Thank you for your feedback!")



def render_progress_bar():
    step_html_blocks = []

    steps = list(st.session_state.progress.keys())
    completed_steps = [step for step, complete in st.session_state.progress.items() if complete]
    current_step_index = len(completed_steps)  # The next step to be completed

    for i, step in enumerate(steps):
        if i < current_step_index:
            icon = "✅"
            color = "#2ecc71"  # green
            text = step
        elif i == current_step_index:
            icon = "⏳"
            color = "#f1c40f"  # yellow
            text = f"{step}..."  # Add ellipsis for current
        else:
            icon = "⏳"
            color = "#bdc3c7"  # gray
            text = step

        step_html_blocks.append(f"""<div style="
            flex: 1;
            text-align: center;
            padding: 10px 8px;
            border-radius: 8px;
            background-color: {color};
            color: #2c3e50;
            font-weight: 600;
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
        {icon} {text}
        </div>""")

    full_html = f"""
    <div style="display: flex; gap: 8px; margin: 1rem 0;">
        {''.join(step_html_blocks)}
    </div>
    """
    progress_placeholder.markdown(full_html, unsafe_allow_html=True)




# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# 👽 [redacted]")
    st.markdown("## Sample Queries")
    for query in [
        "Why is my IAP instance missing app-jst?",
        "Which customer had issues with JSTs being removed from workflows?",
        "I can't find the tasks panel in IAP when I use the legacy format workflow. Why?"
    ]:
        st.write(f"• {query}")

# ---------- TITLE ----------
st.markdown('<h1 class="main-title">[redacted] Assistant</h1>', unsafe_allow_html=True)

# ---------- MESSAGE HISTORY ----------
if st.session_state.messages:
    st.markdown("---")
    st.markdown("## Conversation")
    last_assistant_index = max(i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant")
    
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message-wrapper user-message-wrapper">'
                f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message-wrapper assistant-message-wrapper">'
                f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br><br>{message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            if i == last_assistant_index:
                st.markdown('<div class="feedback-wrapper"><div class="feedback-container">', unsafe_allow_html=True)
                with st.form(key=f"feedback_form_{st.session_state.last_request_id}"):
                    user_feedback = st.text_area(
                        "Provide your feedback on this response:",
                        height=68,
                        placeholder="What was helpful or unclear?",
                        key=f"{st.session_state.last_request_id}_feedback"
                    )
                    user_rating = st.text_area(
                        "Provide your rating from 1-10 on this response:",
                        height=68,
                        key=f"{st.session_state.last_request_id}_rating"
                    )
                    if st.form_submit_button("Submit Feedback"):
                        submit_feedback(st.session_state.last_request_id, user_rating, user_feedback)
                st.markdown('</div></div>', unsafe_allow_html=True)

# ---------- INPUT ----------
col1, col2 = st.columns([1, 4.5])

with col1:
    st.markdown("""
    <div style="height: 160px; display: flex; align-items: flex-end;">
        <h2 style="font-size: 30px; font-weight: 600; color: #34495e; margin-bottom: 0.25rem;">
            Ask Question
        </h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    gif_placeholder = st.empty()
    status_placeholder = st.empty()

st.markdown('<div style="margin-top: -2rem;"></div>', unsafe_allow_html=True)

progress_placeholder = st.empty()

user_input = st.text_input(
    "Enter below:",
    placeholder="Type your question here...",
    key="user_input"
)

# ---------- ON SEND ----------
if st.button("Send") and user_input:
    gif_placeholder.image(f"assets/{random.randint(1, 6)}.gif", width=100)

    # Reset and show initial progress
    for step in STEPS:
        st.session_state.progress[step] = False
    render_progress_bar()

    def update_status(msg):
        for step in STEPS:
            if step == msg:
                st.session_state.progress[step] = True
        render_progress_bar()

    try:
        assistant_response, request_id = asyncio.run(main(user_input, status_callback=update_status))
    except RuntimeError:
        assistant_response, request_id = asyncio.get_event_loop().run_until_complete(main(user_input, status_callback=update_status))
    except Exception as e:
        print(e)
        request_id = None
        assistant_response = "Sorry, I can't answer your question right now."

    # Track conversation
    st.session_state.last_request_id = request_id
    assistant_response = re.sub(r'\*\*([^*]+):\*\*', r'### \1:', assistant_response)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    gif_placeholder.empty()
    status_placeholder.empty()
    st.rerun()
