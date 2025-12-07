from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.document_processor import (
    process_document, get_summary, get_answer, 
    get_document_metadata, calculate_reading_time, get_word_count
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///docsum.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"Upload directory '{app.config['UPLOAD_FOLDER']}' ensured to exist.")
except OSError as e:
    print(f"Error creating upload directory '{app.config['UPLOAD_FOLDER']}': {e}")

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    documents = db.relationship('Document', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    summary = db.Column(db.Text)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata fields
    domain = db.Column(db.String(100), default='Unknown')
    keywords = db.Column(db.Text, default='[]')
    entities = db.Column(db.Text, default='{}')
    reading_time = db.Column(db.Integer, default=1)
    word_count = db.Column(db.Integer, default=0)
    
    def get_keywords_list(self):
        try:
            return json.loads(self.keywords) if self.keywords else []
        except:
            return []
    
    def get_entities_dict(self):
        try:
            return json.loads(self.entities) if self.entities else {}
        except:
            return {}

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for('dashboard'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    documents = Document.query.filter_by(user_id=current_user.id).order_by(Document.upload_date.desc()).all()
    return render_template('dashboard.html', documents=documents)

@app.route('/analytics')
@login_required
def analytics():
    """Analytics dashboard showing document statistics."""
    documents = Document.query.filter_by(user_id=current_user.id).all()
    
    total_documents = len(documents)
    total_words = sum(doc.word_count or 0 for doc in documents)
    total_reading_time = sum(doc.reading_time or 0 for doc in documents)
    
    domain_counts = {}
    for doc in documents:
        domain = doc.domain or 'Unknown'
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    all_keywords = []
    for doc in documents:
        all_keywords.extend(doc.get_keywords_list())
    
    keyword_counts = {}
    for kw in all_keywords:
        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
    
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    recent_docs = documents[:5]
    
    monthly_data = {}
    for doc in documents:
        month_key = doc.upload_date.strftime('%Y-%m')
        monthly_data[month_key] = monthly_data.get(month_key, 0) + 1
    
    return render_template('analytics.html',
        total_documents=total_documents,
        total_words=total_words,
        total_reading_time=total_reading_time,
        domain_counts=domain_counts,
        top_keywords=top_keywords,
        recent_docs=recent_docs,
        monthly_data=monthly_data
    )

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'})
    
    if file:
        filename = secure_filename(file.filename)
        original_filename = filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'success': False, 'error': f"Failed to save file: {str(e)}"})

        try:
            text = process_document(filepath)
            summary = get_summary(text)
            metadata = get_document_metadata(text)
            reading_time = calculate_reading_time(text)
            word_count = get_word_count(text)

            document = Document(
                filename=original_filename,
                filepath=filepath,
                summary=summary,
                user_id=current_user.id,
                domain=metadata.get('domain', 'Unknown'),
                keywords=json.dumps(metadata.get('keywords', [])),
                entities=json.dumps(metadata.get('entities', {})),
                reading_time=reading_time,
                word_count=word_count
            )
            db.session.add(document)
            db.session.commit()

            return jsonify({'success': True, 'message': 'File uploaded and processed successfully!'})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f"Document processing failed: {str(e)}"})

    return jsonify({'success': False, 'error': 'Invalid file type.'})

@app.route('/document/<int:doc_id>')
@login_required
def get_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    return jsonify({
        'success': True,
        'summary': document.summary,
        'domain': document.domain,
        'keywords': document.get_keywords_list(),
        'entities': document.get_entities_dict(),
        'reading_time': document.reading_time,
        'word_count': document.word_count
    })

@app.route('/document/<int:doc_id>', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    try:
        if os.path.exists(document.filepath):
            os.remove(document.filepath)
        db.session.delete(document)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Document deleted successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'error': f"Failed to delete document: {str(e)}"})

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data.get('question')
    doc_id = data.get('doc_id')

    if not question:
        return jsonify({'success': False, 'error': 'Missing question.'})
    if not doc_id:
        return jsonify({'success': False, 'error': 'Missing document ID.'})

    try:
        doc_id = int(doc_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid document ID format.'})

    document = db.session.get(Document, doc_id)
    if not document:
        return jsonify({'success': False, 'error': 'Document not found.'})
    if document.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized access to document.'})

    try:
        text = process_document(document.filepath)
        answer = get_answer(text, question)
        return jsonify({'success': True, 'answer': answer})
    except FileNotFoundError:
        return jsonify({'success': False, 'error': f"Document file not found. Please re-upload."})
    except Exception as e:
        return jsonify({'success': False, 'error': f"QA failed: {str(e)}"})

@app.route('/reprocess')
@login_required
def reprocess_documents():
    """Re-process all documents for the current user to update metadata."""
    documents = Document.query.filter_by(user_id=current_user.id).all()
    processed = 0
    errors = 0
    
    for document in documents:
        try:
            text = process_document(document.filepath)
            metadata = get_document_metadata(text)
            
            document.domain = metadata.get('domain', 'Unknown')
            document.keywords = json.dumps(metadata.get('keywords', []))
            document.entities = json.dumps(metadata.get('entities', {}))
            document.reading_time = calculate_reading_time(text)
            document.word_count = get_word_count(text)
            document.summary = get_summary(text)
            
            processed += 1
        except Exception as e:
            print(f"Error reprocessing document {document.id}: {e}")
            errors += 1
            continue
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Reprocessed {processed} documents. {errors} errors.',
        'processed': processed,
        'errors': errors
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port)