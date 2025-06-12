# Document Summarizer & Q&A System

A modern web application that allows users to upload documents (PDF, DOCX, DOC), get AI-generated summaries, and ask questions about the content.

## Features

- User authentication (register, login, logout)
- Document upload with drag-and-drop support
- AI-powered document summarization
- Question answering about document content
- Modern, responsive UI with glass morphism design
- Secure document storage and processing

## Tech Stack

- Backend: Flask (Python)
- Frontend: HTML, Tailwind CSS, JavaScript
- Database: SQLite (SQLAlchemy)
- AI Models:
  - BART for text summarization
  - RoBERTa for question answering
- Document Processing: PyPDF2, python-docx

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-summarizer.git
cd document-summarizer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. Set up environment variables:
```bash
# Create a .env file with:
SECRET_KEY=your-secret-key-here
```

6. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

7. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Register a new account or log in with existing credentials
2. Upload a document (PDF, DOCX, or DOC) using the drag-and-drop interface
3. View the AI-generated summary of the document
4. Ask questions about the document content using the Q&A interface
5. Manage your uploaded documents (view, delete)

## Security Features

- Password hashing using Werkzeug's security functions
- User authentication with Flask-Login
- Secure file upload handling
- Document access control (users can only access their own documents)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 