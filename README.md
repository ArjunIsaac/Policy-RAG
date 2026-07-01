# Policy-RAG

A Retrieval-Augmented Generation (RAG) application for querying insurance policy documents using Qwen3-4B-AWQ served with vLLM.





Create a virtual environment:
# Setup

## Creating venv 

```bash
source .venv/bin/activate
```

## Install dependencies:

```bash
pip install -r requirements.txt
```


# Running the Project

## Terminal 1 - Start the vLLM Server

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-AWQ \
    --quantization awq_marlin \
    --max-model-len 4096
```

Wait until the server finishes loading before starting the RAG application.

---

## Terminal 2 - Start the RAG Application

Open a new terminal.

Navigate to the project directory:

```bash
cd Policy-RAG
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run Streamlit:

```bash
streamlit run main.py
```

The application will open in your browser.

---

# Notes

- Make sure the vLLM server is running before launching the Streamlit application.
- If using Hugging Face gated models, log in first:

```bash
huggingface-cli login
```