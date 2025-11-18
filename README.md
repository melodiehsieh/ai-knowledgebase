# AI Knowledgebase Project

## About
This AI knowledgebase chatbot embeds a variety of different types of data into a vector database for for RAG retrieval. This allows users to query the data using natural language and receive a natural language response. This project uses Qdrant as the vector database, the LlamaIndex BM25 retriever, a HuggingFace embedding model, and Anthropic API.

## Getting Started

NOTE: I had to remove all identifying/private information so it will not run, but you can see the architecture of the system I built.

This will not run.
After cloning this repo, run the following commands:

```
brew install python@3.9
cd ai-knowledgebase
python3.9 -m venv --without-pip venv
source venv/bin/activate
path/to/python3.9 -m pip install -r requirements.txt
cd UI
streamlit run userinterface.py
```
**NOTE:** find your path to python 3.9 by running (`which python3.9`) and copy this to the beginning of the `pip install` command.

Access the link returned in the terminal to use the chatbot.
