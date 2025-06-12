import faiss
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModel
import torch
from .document_processor import chunk_text
import os
import json
from datetime import datetime

class DocumentStore:
    def __init__(self, store_dir='document_store'):
        self.store_dir = store_dir
        self.documents = {}
        self.metadata = {}
        os.makedirs(store_dir, exist_ok=True)
        self._load_documents()

    def _load_documents(self):
        """Load existing documents from disk"""
        metadata_file = os.path.join(self.store_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            for doc_id in self.metadata:
                doc_file = os.path.join(self.store_dir, f'{doc_id}.txt')
                if os.path.exists(doc_file):
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        self.documents[doc_id] = f.read()

    def _save_documents(self):
        """Save documents to disk"""
        for doc_id, content in self.documents.items():
            doc_file = os.path.join(self.store_dir, f'{doc_id}.txt')
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        with open(os.path.join(self.store_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)

    def add_document(self, content, user_id=None, filename=None):
        """Add a new document to the store"""
        doc_id = str(len(self.documents))
        self.documents[doc_id] = content
        self.metadata[doc_id] = {
            'user_id': user_id,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        self._save_documents()
        return doc_id

    def get_document(self, doc_id):
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)

    def get_user_documents(self, user_id):
        """Get all documents for a specific user"""
        return {
            doc_id: {
                'content': content,
                'metadata': self.metadata[doc_id]
            }
            for doc_id, content in self.documents.items()
            if self.metadata[doc_id]['user_id'] == user_id
        }

class QASystem:
    def __init__(self):
        # Initialize models
        self.qa_model_name = 'deepset/roberta-base-squad2'
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
        
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        
        # Initialize document store
        self.document_store = DocumentStore()
        
        # Initialize FAISS index
        self.index = None
        self.chunk_to_doc = {}  # Maps chunk index to document ID
        self.chunks = []

    def _create_embeddings(self, texts):
        """Create embeddings for the given texts"""
        embeddings = []
        for text in texts:
            inputs = self.embedding_tokenizer.encode(
                text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.embedding_model(inputs)
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                embeddings.append(embedding)
        return np.array(embeddings)

    def add_document(self, text, user_id=None, filename=None):
        """Add a document to the QA system"""
        # Store document
        doc_id = self.document_store.add_document(text, user_id, filename)
        
        # Split into chunks
        chunks = chunk_text(text, chunk_size=1000)
        
        # Create embeddings
        embeddings = self._create_embeddings(chunks)
        
        # Update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Add chunks to index
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        # Update chunk to document mapping
        for i in range(len(chunks)):
            self.chunk_to_doc[start_idx + i] = doc_id

    def answer_question(self, question, user_id=None):
        """Answer a question based on the stored documents"""
        if not self.chunks:
            return "No documents have been added to the system."

        # Create question embedding
        question_embedding = self._create_embeddings([question])[0]
        
        # Search for relevant chunks
        D, I = self.index.search(question_embedding.reshape(1, -1), 3)
        
        # Get relevant chunks and their documents
        relevant_chunks = []
        for idx in I[0]:
            if idx in self.chunk_to_doc:
                doc_id = self.chunk_to_doc[idx]
                if user_id is None or self.document_store.metadata[doc_id]['user_id'] == user_id:
                    relevant_chunks.append(self.chunks[idx])
        
        if not relevant_chunks:
            return "No relevant information found in the documents."

        # Combine relevant chunks
        context = " ".join(relevant_chunks)
        
        # Prepare input for QA model
        inputs = self.qa_tokenizer.encode_plus(
            question,
            context,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get answer
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            tokens = self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self.qa_tokenizer.convert_tokens_to_string(
                tokens[answer_start:answer_end + 1]
            )
        
        return answer if answer.strip() else "No specific answer found in the documents."

    def get_user_documents(self, user_id):
        """Get all documents for a specific user"""
        return self.document_store.get_user_documents(user_id)

# Create a global QA system instance
qa_system = QASystem()

def answer_question(question):
    """
    Global function to answer questions using the QA system.
    """
    return qa_system.answer_question(question) 