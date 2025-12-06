# DocSum AI - Document Summarizer & Q&A System

A modern web application that allows users to upload documents (PDF, DOCX, DOC), get AI-generated summaries, and ask questions about the content. Powered by Google Gemini for intelligent document understanding.

## ✨ Features

- **User Authentication** - Secure register, login, and logout functionality
- **Document Upload** - Drag-and-drop support for PDF, DOCX, and DOC files
- **AI-Powered Summarization** - Intelligent summaries that understand document context
- **Smart Q&A** - Ask natural language questions about your documents
- **Modern UI** - Responsive design with glass morphism aesthetics
- **Secure Storage** - Protected document storage with user-level access control

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite with SQLAlchemy ORM
- **AI Engine:** Google Gemini 2.0 Flash
- **Document Processing:** PyPDF2, python-docx

## 📋 Prerequisites

- Python 3.9+
- A free Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/document-summarizer.git
cd document-summarizer
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```bash
# Required: Get your free API key from https://aistudio.google.com/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: Flask secret key for session security
SECRET_KEY=your-secret-key-here
```

### 5. Run the application
```bash
python app.py
```

The application will be available at `http://localhost:5001`

> **Note:** The app runs on port 5001 by default to avoid conflicts with macOS AirPlay Receiver (which uses port 5000).

## 📖 Usage

1. **Register** a new account or **log in** with existing credentials
2. **Upload** a document (PDF, DOCX, or DOC) using the drag-and-drop interface
3. **View** the AI-generated summary of your document
4. **Ask questions** about the document content using natural language
5. **Manage** your uploaded documents (view details, delete)

## 🔒 Security Features

- Password hashing using Werkzeug's security functions
- User authentication with Flask-Login
- Secure file upload handling with filename sanitization
- Document access control (users can only access their own documents)
- Environment-based API key management

## 🤖 AI Capabilities

### Document Summarization
The system uses Google Gemini to:
- Automatically identify document types (invoices, reports, articles, etc.)
- Extract key information based on document type
- Generate concise, readable summaries

### Question Answering
Ask natural language questions and get:
- Direct, accurate answers based on document content
- Context-aware responses for complex queries
- Clear indication when information is not available

## 📁 Project Structure

```
Document-Summarizer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── templates/             # HTML templates
├── static/                # CSS, JavaScript, images
├── uploads/               # Uploaded documents (auto-created)
├── instance/              # SQLite database
└── utils/
    └── document_processor.py  # Document processing & AI logic
```

## 🔧 Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | - |
| `SECRET_KEY` | Flask session secret key | Auto-generated |
| `PORT` | Server port number | 5001 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for powerful AI capabilities
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing