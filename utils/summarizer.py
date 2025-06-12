from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from .document_processor import chunk_text

# Initialize the model and tokenizer
model_name = 't5-base'  # Using a larger model for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_summary(text, max_length=500, min_length=100):
    """
    Generate a summary of the input text using T5 model.
    """
    try:
        # Split text into chunks if it's too long
        chunks = chunk_text(text, chunk_size=1500)  # Increased chunk size
        summaries = []
        
        for chunk in chunks:
            if not chunk.strip():  # Skip empty chunks
                continue
                
            # Prepare the input text
            input_text = f"summarize: {chunk}"
            
            # Tokenize the input
            inputs = tokenizer.encode(
                input_text,
                max_length=1024,  # Increased max length
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate summary
            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=3.0,  # Increased length penalty
                num_beams=5,  # Increased beam size
                early_stopping=True,
                no_repeat_ngram_size=3  # Prevent repetition
            )
            
            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        # Combine all summaries
        if len(summaries) > 1:
            # If we have multiple summaries, summarize them again
            combined_summary = " ".join(summaries)
            return generate_summary(combined_summary, max_length, min_length)
        
        return summaries[0] if summaries else "No content to summarize."
        
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Error generating summary. Please try again."

def extractive_summary(text, num_sentences=5):
    """
    Generate an extractive summary using sentence importance scoring.
    This is a fallback method if the T5 model fails.
    """
    try:
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
        
        if not sentences:
            return "No content to summarize."
        
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
        num_sentences = min(num_sentences, len(sentences))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        # Combine sentences
        summary = ' '.join([sentences[i] for i, _ in top_sentences])
        return summary
        
    except Exception as e:
        print(f"Error in extractive summarization: {str(e)}")
        return "Error generating summary. Please try again." 