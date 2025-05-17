# Document Assistant

A Streamlit-based web application that allows you to upload documents (PDF, DOCX, TXT, HTML), process and index them using TF-IDF embeddings with FAISS, and then chat with the content using an integrated language model from Hugging Face or a fallback simple retrieval system.

---

## Features

* **Multi-format document upload:** Supports PDF, DOCX, TXT, and HTML files.
* **Document chunking:** Splits large documents into manageable chunks for better retrieval.
* **TF-IDF Embeddings + FAISS:** Vectorizes document chunks and enables fast similarity search.
* **Hugging Face model integration:** Supports conversational and question-answering models using your Hugging Face API key.
* **Fallback simple retrieval mode:** Provides basic document search when no API key is available or if the LLM fails.
* **Dark theme UI:** Stylish, easy-on-the-eyes dark mode interface.
* **Chat interface:** Ask questions and receive answers based on the uploaded documents.
* **Session state management:** Keeps conversation history and uploaded documents within the session.

---

## Installation

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install streamlit langchain langchain_community langchain_core faiss-cpu
pip install scikit-learn numpy
pip install langchain_huggingface huggingface_hub
pip install pypdf python-docx unstructured
```

---

## Usage

1. Clone the repository or copy the script into a file `app.py`.

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload your documents (PDF, DOCX, TXT, or HTML).

4. Provide your Hugging Face API token in the sidebar to enable powerful LLM question answering (optional).

5. Choose the model from the sidebar:

   * Mistral Nemo Instruct (Conversational)
   * RoBERTa for Question Answering
   * BART for Summarization

6. Click the "Process Documents" button to index your uploaded files.

7. Enter your queries/questions in the chat input box and get answers based on your documents.

---

## How It Works

* Uploaded documents are loaded and split into smaller chunks.
* A TF-IDF vectorizer converts chunks into embeddings.
* FAISS index is built for similarity search on embeddings.
* If a Hugging Face API key and model are provided, the app uses a language model endpoint to answer queries.
* If not, the app uses a fallback simple retriever to provide basic answers from the most relevant chunks.
* The app displays conversation history styled in a dark mode theme.

---

## Supported File Formats

* PDF (.pdf)
* Microsoft Word (.docx)
* Plain text (.txt)
* HTML (.html, .htm)

Unsupported formats will show a warning message.

---

## Configuration

* **Hugging Face API Token:** Required for using LLM models. Enter it in the sidebar.
* **Model Selection:** Choose from predefined Hugging Face models in the sidebar.
* **TF-IDF vectorizer:** Configured with a max feature size of 5000.

---

## Limitations

* Without a Hugging Face API token, the app uses a simple retrieval fallback that may provide less detailed answers.
* Processing large documents might take some time depending on your machine.
* Currently, the app supports a limited set of file formats.

---

## License

MIT License

---

## Acknowledgements

* [Streamlit](https://streamlit.io/)
* [LangChain](https://github.com/hwchase17/langchain)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Hugging Face](https://huggingface.co/)
* Various open-source libraries and community contributions


