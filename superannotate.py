from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer
import torch.nn.functional as F
import numpy as np
import streamlit as st

def process_chunk(text, model, tokenizer):
    """Process a single chunk of text"""
    text = preprocessing_text(text)
    
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='longest',
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt"
    )
    
    _, logits = model(**tokens)
    proba = F.sigmoid(logits).squeeze(1).item()
    return proba

def analyze_text(text):
    """Analyze entire text in chunks"""
    # Move model and tokenizer loading inside the function
    with st.spinner("Loading model..."):
        model = RobertaClassifier.from_pretrained("SuperAnnotate/ai-detector")
        tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/ai-detector")
        model.eval()
    
    # Split text into chunks of roughly 500 words
    words = text.split()
    chunk_size = 500
    chunks = [' '.join(words[i:i + chunk_size]) 
             for i in range(0, len(words), chunk_size)]
    
    # Process each chunk
    chunk_scores = []
    for chunk in chunks:
        score = process_chunk(chunk, model, tokenizer)
        chunk_scores.append(score)
    
    # Calculate final probability (mean of all chunks)
    final_proba = np.mean(chunk_scores)
    
    return final_proba, chunk_scores

def main():
    st.title("AI Text Detector")
    st.write("This tool analyzes text to determine the probability of it being AI-generated.")
    
    # Text input area
    user_input = st.text_area("Enter your text here:", height=300)
    
    if st.button("Analyze"):
        if user_input:
            try:
                with st.spinner("Analyzing text..."):
                    final_score, chunk_scores = analyze_text(user_input)
                    
                    # Display results
                    st.subheader("Results")
                    st.write(f"Overall AI probability: {final_score:.2%}")
                    
                    # Show chunk scores in a bar chart
                    st.subheader("Chunk Analysis")
                    st.bar_chart(chunk_scores)
                    
                    # Add some interpretation
                    if final_score > 0.7:
                        st.warning("High probability of AI-generated content")
                    elif final_score > 0.4:
                        st.info("Moderate probability of AI-generated content")
                    else:
                        st.success("Low probability of AI-generated content")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter some text to analyze")

if __name__ == "__main__":
    main()
