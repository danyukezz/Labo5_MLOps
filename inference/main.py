"""
FastAPI application for Animals Classification Model Inference
Serves predictions for animal classification (pandas, cats, dogs)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
import os
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Animals Classification API",
    description="ML API for classifying animals (pandas, cats, dogs)",
    version="1.0.0"
)

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model")
IMG_SIZE = 224

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained TensorFlow model on startup"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        
        # Check if MODEL_PATH is a directory or file
        if os.path.isdir(MODEL_PATH):
            # Look for .keras file in the directory
            keras_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.keras')]
            if keras_files:
                model_file = os.path.join(MODEL_PATH, keras_files[0])
                logger.info(f"Found .keras file: {model_file}")
                model = tf.keras.models.load_model(model_file)
            else:
                # Try loading as SavedModel directory
                model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Direct file path
            model = tf.keras.models.load_model(MODEL_PATH)
            
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes liveness and readiness probes
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Animals Classification API",
        "version": "1.0.0",
        "description": "Classifies images as pandas, cats, or dogs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict animal class from an uploaded image
    
    Parameters:
    - file: Image file (JPEG, PNG)
    
    Returns:
    - predictions: Dict with class predictions and confidence scores
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess image
        image = image.convert("RGB")  # Ensure RGB format
        image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize to 0-1
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Class labels
        class_names = ["pandas", "cats", "dogs"]
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Build response with all class probabilities
        all_predictions = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
        
        logger.info(f"✅ Prediction: {predicted_class} ({confidence:.2%})")
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_predictions,
            "image_size": f"{IMG_SIZE}x{IMG_SIZE}"
        }
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "animals-classification",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "classes": ["pandas", "cats", "dogs"],
        "model_type": "TensorFlow Keras",
        "image_size": IMG_SIZE
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
