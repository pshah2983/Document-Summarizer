from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from .document_processor import chunk_text

# Initialize the model and tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_summary(text, max_length=150, min_length=50):
    """
    Generate a summary of the input text using T5 model.
    """
    # Split text into chunks if it's too long
    chunks = chunk_text(text)
    summaries = []
    
    for chunk in chunks:
        # Prepare the input text
        input_text = f"summarize: {chunk}"
        
        # Tokenize the input
        inputs = tokenizer.encode(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Combine all summaries
    if len(summaries) > 1:
        # If we have multiple summaries, summarize them again
        combined_summary = " ".join(summaries)
        return generate_summary(combined_summary, max_length, min_length)
    
    return summaries[0]

def extractive_summary(text, num_sentences=3):
    """
    Generate an extractive summary using sentence importance scoring.
    This is a fallback method if the T5 model fails.
    """
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Calculate word frequencies
    word_frequencies = Counter()
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word not in stop_words and word.isalnum():
                word_frequencies[word] += 1
    
    # Calculate sentence scores
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                score += word_frequencies[word]
        sentence_scores[i] = score
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    # Combine sentences
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    return summary 