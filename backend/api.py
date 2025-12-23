from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Optional

app = FastAPI(title="Sentiment Analysis API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and vectorizer
BASE_DIR = Path(__file__).resolve().parent
try:
    model = joblib.load(BASE_DIR / "models" / "svm_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None


# API Key authentication
API_KEY = os.getenv("API_KEY", "12345678")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


def clean_text(text):
    """
    Preprocess text using the same pipeline as training
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class TextInput(BaseModel):
    text: str
    clean: bool = True  # Option to skip cleaning if already cleaned

class BatchTextInput(BaseModel):
    texts: List[str]
    clean: bool = True

class SentimentResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float

@app.get("/")
def read_root():
    return {
        "message": "Sentiment Analysis API - Trained on Sentiment140 Dataset",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health"
        }
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(
    input_data: TextInput,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict sentiment for a single text
    
    Parameters:
    - text: The text to analyze
    - clean: Whether to apply text preprocessing (default: True)
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Clean the text if requested
        original_text = input_data.text
        cleaned = clean_text(input_data.text) if input_data.clean else input_data.text
        
        # Check if text is too short after cleaning
        if len(cleaned.strip()) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Text too short after preprocessing (minimum 3 characters required)"
            )
        
        # Vectorize the cleaned text
        text_vectorized = vectorizer.transform([cleaned])
        
        # Get prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence score
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vectorized)[0]
            confidence = float(max(proba))
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(text_vectorized)[0]
            # Convert decision function to probability-like score
            confidence = float(1 / (1 + np.exp(-abs(decision))))
        else:
            confidence = 1.0
        
        # Map prediction to sentiment label (assuming 0=negative, 1=positive)
        sentiment = "positive" if prediction == 1 else "negative"
        
        return SentimentResponse(
            text=original_text,
            cleaned_text=cleaned,
            sentiment=sentiment,
            confidence=confidence
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(
    input_data: BatchTextInput,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict sentiment for multiple texts
    
    Parameters:
    - texts: List of texts to analyze
    - clean: Whether to apply text preprocessing (default: True)
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Clean all texts if requested
        cleaned_texts = [
            clean_text(text) if input_data.clean else text 
            for text in input_data.texts
        ]
        
        # Filter out texts that are too short
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(cleaned_texts):
            if len(text.strip()) >= 3:
                valid_indices.append(i)
                valid_texts.append(text)
        
        if not valid_texts:
            raise HTTPException(
                status_code=400,
                detail="All texts are too short after preprocessing"
            )
        
        # Vectorize valid texts
        texts_vectorized = vectorizer.transform(valid_texts)
        
        # Get predictions
        predictions = model.predict(texts_vectorized)
        
        # Get confidence scores
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(texts_vectorized)
            confidences = [float(max(proba)) for proba in probas]
        elif hasattr(model, 'decision_function'):
            decisions = model.decision_function(texts_vectorized)
            confidences = [float(1 / (1 + np.exp(-abs(d)))) for d in decisions]
        else:
            confidences = [1.0] * len(predictions)
        
        # Format results
        results = []
        for idx, pred, conf in zip(valid_indices, predictions, confidences):
            sentiment = "positive" if pred == 1 else "negative"
            results.append({
                "index": idx,
                "text": input_data.texts[idx],
                "cleaned_text": cleaned_texts[idx],
                "sentiment": sentiment,
                "confidence": conf
            })
        
        # Add info about filtered texts
        filtered_count = len(input_data.texts) - len(valid_texts)
        
        return {
            "predictions": results,
            "total_processed": len(valid_texts),
            "total_filtered": filtered_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    """
    Check API health and model status
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }