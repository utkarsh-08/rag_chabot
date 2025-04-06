# Conversational RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to assist users by answering queries based on pre-trained insurance content. The chatbot uses PDF documents as its knowledge base and can handle both text and tabular data.

## Features

- Conversational interface using Streamlit
- Retrieval-Augmented Generation for accurate responses
- Handles both text and tables from PDF documents
- Maintains chat history for context-aware responses

## Prerequisites

- Python 3.9
- pip (Python package manager)

## Installation


1. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install Dependencies**

   Install the required packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file in the root directory and add the following environment variables:

```plaintext
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_PROJECT=your_project_name
```

## Usage

1. **Prepare PDF Documents**

   Place your PDF documents in the `./Insurance PDFs` directory. Ensure the directory exists and contains the PDFs you want to use as the knowledge base.

2. **Run the Application**

   Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

3. **Interact with the Chatbot**

   - Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
   - Enter your session ID and start asking questions.
