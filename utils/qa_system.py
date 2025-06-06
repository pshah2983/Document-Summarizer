import faiss
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .document_processor import chunk_text
import torch
import pickle
import os

# Initialize the model and tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

class QASystem:
    def __init__(self):
        self.index = None
        self.documents = []
        self.document_chunks = []
        
    def add_document(self, text):
        """
        Add a document to the QA system.
        """
        # Split document into chunks
        chunks = chunk_text(text)
        self.document_chunks.extend(chunks)
        
        # Create embeddings for chunks
        embeddings = self._create_embeddings(chunks)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings)
        
    def _create_embeddings(self, texts):
        """
        Create embeddings for the given texts using T5.
        """
        embeddings = []
        for text in texts:
            # Tokenize and get model output
            inputs = tokenizer.encode(
                text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model.encoder(inputs)
                # Use the [CLS] token embedding as the text embedding
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def answer_question(self, question, num_chunks=3):
        """
        Answer a question based on the stored documents.
        """
        if not self.document_chunks:
            return "No documents have been added to the system."
        
        # Create question embedding
        question_embedding = self._create_embeddings([question])[0]
        
        # Search for relevant chunks
        D, I = self.index.search(question_embedding.reshape(1, -1), num_chunks)
        
        # Get relevant chunks
        relevant_chunks = [self.document_chunks[i] for i in I[0]]
        
        # Prepare input for T5
        context = " ".join(relevant_chunks)
        input_text = f"question: {question} context: {context}"
        
        # Generate answer
        inputs = tokenizer.encode(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        answer_ids = model.generate(
            inputs,
            max_length=100,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer

# Create a global QA system instance
qa_system = QASystem()

def answer_question(question):
    """
    Global function to answer questions using the QA system.
    """
    return qa_system.answer_question(question) 