import PyPDF2
import docx
import os
import re
import google.generativeai as genai

# Configure Gemini API - supports both GEMINI_API_KEY and GOOGLE_API_KEY
API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
if API_KEY:
    genai.configure(api_key=API_KEY)

# Initialize the Gemini model
def get_gemini_model():
    """Get the Gemini model instance."""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use the summarization and Q&A features.")
    genai.configure(api_key=api_key)
    # Use gemini-2.0-flash which is the current available model
    return genai.GenerativeModel('gemini-2.0-flash')

def clean_text(text):
    """Cleans extracted text by fixing common OCR errors and formatting issues."""
    # Remove common problematic ligatures/broken words from PDF extraction
    text = re.sub(r'oﬀ', 'off', text, flags=re.IGNORECASE)
    text = re.sub(r'e-tick eting', 'e-ticketing', text, flags=re.IGNORECASE)
    text = re.sub(r'c are', 'care', text, flags=re.IGNORECASE)
    text = re.sub(r'u/ s', 'u/s', text, flags=re.IGNORECASE)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Attempt to fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    return text

def process_document(filepath):
    """Extract text from a document file and clean it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document file not found: {filepath}")

    file_ext = os.path.splitext(filepath)[1].lower()
    
    raw_text = ""
    if file_ext == '.pdf':
        raw_text = extract_text_from_pdf(filepath)
    elif file_ext in ['.docx', '.doc']:
        raw_text = extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    return clean_text(raw_text)

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file."""
    text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(filepath):
    """Extract text from a DOCX file."""
    doc = docx.Document(filepath)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_summary(text, max_length=300):
    """Generate an intelligent summary using Google Gemini."""
    try:
        model = get_gemini_model()
        
        # Truncate text if too long (Gemini can handle large context, but we limit for speed)
        if len(text) > 15000:
            text = text[:15000] + "..."
        
        prompt = f"""Analyze the following document and provide a clear, well-structured summary.

Instructions:
1. Identify the type of document (invoice, receipt, report, article, letter, etc.)
2. Extract and highlight the most important information:
   - For invoices/receipts: amounts, dates, parties involved, items/services
   - For reports/articles: main topic, key findings, conclusions
   - For letters/correspondence: sender, recipient, main purpose, key points
3. Present the summary in a readable format with key details clearly stated
4. Keep the summary concise but comprehensive (around 150-250 words)

Document content:
{text}

Summary:"""

        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower():
            return "Error: Please set your GEMINI_API_KEY environment variable to enable summarization."
        raise Exception(f"Failed to generate summary: {error_msg}")

def get_answer(text, question):
    """Answer questions about the document using Google Gemini."""
    try:
        model = get_gemini_model()
        
        # Truncate text if too long
        if len(text) > 15000:
            text = text[:15000] + "..."
        
        prompt = f"""You are a helpful document assistant. Answer the following question based ONLY on the information provided in the document below.

Instructions:
1. Answer the question directly and accurately
2. If the answer can be found in the document, provide it clearly
3. If the answer is not in the document, say "This information is not available in the document."
4. For yes/no questions, answer yes or no first, then provide supporting details from the document
5. Be concise but thorough

Document content:
{text}

Question: {question}

Answer:"""

        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower():
            return "Error: Please set your GEMINI_API_KEY environment variable to enable Q&A."
        raise Exception(f"Failed to answer question: {error_msg}")

def split_text_into_chunks(text, max_length=900):
    """Split text into chunks of maximum length based on words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks