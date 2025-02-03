# README.md

# Retrieval-Augmented Generation (RAG) Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for processing PDF documents. The pipeline extracts text from PDFs, processes the text, generates embeddings, and stores them in a vector database. It also allows querying a language model to retrieve information from the processed PDFs.

## Project Structure

```
rag-pipeline
├── src
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── pdf_processor.py
│   │   ├── text_processor.py
│   │   ├── embeddings_generator.py
│   │   └── vector_store.py
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── main.py
├── data
│   ├── pdfs
│   ├── processed
│   └── embeddings
├── tests
│   ├── __init__.py
│   ├── test_pdf_processor.py
│   ├── test_text_processor.py
│   └── test_embeddings.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-pipeline
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your PDF files in the `data/pdfs` directory.

## Usage

To run the pipeline, execute the following command:
```
python src/main.py
```

This will process the PDFs, generate embeddings, and store them in the Milvus vector database. You can also query the language model to test the retrieval capabilities.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.