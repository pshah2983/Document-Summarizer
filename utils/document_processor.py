import PyPDF2
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading SpaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def process_document(file_path):
    """
    Process a document (PDF or Word) and return its text content.
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return process_pdf(file_path)
    elif file_extension in ['docx', 'doc']:
        return process_word(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_pdf(file_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return clean_text(text)

def process_word(file_path):
    """
    Extract text from a Word document.
    """
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return clean_text(text)

def clean_text(text):
    """
    Clean and preprocess the extracted text.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic text cleaning
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text

def chunk_text(text, chunk_size=1000):
    """
    Split text into smaller chunks for processing.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 