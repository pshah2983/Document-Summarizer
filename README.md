# Document Summarization and Q&A System

An intelligent document processing system that provides summarization and question-answering capabilities for PDF and Word documents.

## Features

- Document processing (PDF and Word)
- Text summarization using advanced NLP models
- Question answering system
- Document similarity search
- Interactive web interface

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
5. Download SpaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
6. Run the application:
   ```bash
   python app.py
   ```

## Project Structure

```
document_summarizer/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
├── utils/              # Utility functions
│   ├── document_processor.py
│   ├── summarizer.py
│   └── qa_system.py
└── models/             # Model storage
```

## Technologies Used

- Flask (Web Framework)
- Hugging Face Transformers
- FAISS (Vector Search)
- NLTK & SpaCy (NLP)
- PyPDF2 & python-docx (Document Processing)

## License

MIT License 