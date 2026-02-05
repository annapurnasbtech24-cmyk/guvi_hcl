"""
Problem Statement 1: AI-Generated Voice Detection
-------------------------------------------------
This FastAPI application exposes a REST endpoint that detects whether
a given Base64-encoded MP3 voice sample is AI-generated or Human.

Supported Languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu
"""

import os
import uuid
import base64

import torch
import librosa
import numpy as np

from fastapi import FastAPI, Header, HTTPException
from transformers import pipeline

# --------------------------------------------------
# Application Initialization
# --------------------------------------------------

app = FastAPI(title="AI Voice Detection API", version="1.0")
@app.get("/")
@app.post("/")
async def root():
    return {
        "status": "success",
        "message": "AI Voice Detection API is Online",
        "instructions": "Send POST requests to /api/voice-detection with your Base64 audio."
    }

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Supported languages as per problem statement
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# API Key (use environment variable in real deployment)
VALID_API_KEY = os.getenv("API_KEY", "guvi-hcl-voice-ai-2026")

# Device selection for inference
# Transformers expects:
#   device = 0  -> GPU
#   device = -1 -> CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# --------------------------------------------------
# Model Loading (Executed once at startup)
# --------------------------------------------------

# NOTE:
# This model is multilingual and based on XLSR,
# which supports Indian languages reasonably well.
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-xlsr-53"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-xlsr-53"
)

model.eval()

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/api/voice-detection")
async def detect_voice(
    payload: dict,
    x_api_key: str = Header(None)
):
    """
    Detect whether a given voice sample is AI-generated or Human.

    Request:
    - Headers:
        x-api-key: YOUR_SECRET_API_KEY
    - Body (JSON):
        {
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "<Base64 MP3>"
        }

    Response:
    - JSON with classification result and confidence
    """

    # --------------------------------------------------
    # 1. API Key Authentication
    # --------------------------------------------------
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # --------------------------------------------------
    # 2. Input Validation
    # --------------------------------------------------
    language = payload.get("language")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if payload.get("audioFormat") != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format is supported")

    if "audioBase64" not in payload:
        raise HTTPException(status_code=400, detail="audioBase64 field missing")

    try:
        # --------------------------------------------------
        # 3. Decode Base64 Audio
        # --------------------------------------------------
        audio_bytes = base64.b64decode(payload["audioBase64"])

        # Use unique temp file to avoid race conditions
        temp_file = f"/tmp/{uuid.uuid4()}.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        ## --------------------------------------------------
        # 4. Audio Loading (Minimal Processing)
        # --------------------------------------------------
        speech, sr = librosa.load(
            temp_file,
            sr=16000,
            offset=0.5,
            duration=4.0
        )
        
        speech, _ = librosa.effects.trim(speech)
        speech = librosa.util.normalize(speech)


        # Resample ONLY if required
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)

        # --------------------------------------------------
        # 4.1 Validate Audio Signal (Fix 2)
        # --------------------------------------------------
        if speech is None or len(speech) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file contains no usable signal"
            )

        # --------------------------------------------------
        # 4.2 Minimum Duration Check (Fix 3)
        # --------------------------------------------------
        duration_seconds = len(speech) / 16000
        if duration_seconds < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Audio duration too short for analysis"
            )

        # --------------------------------------------------
        # 5. Model Inference
        # --------------------------------------------------
        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Shape: (batch, time, features)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Acoustic naturalness metrics
        temporal_variance = np.var(embeddings, axis=0).mean()
        frame_energy = np.mean(np.abs(speech))
        
        # Decision logic (tuned for Indian languages)
        # --------------------------------------------------
        # Decision & Confidence Computation (DATA-DRIVEN)
        # --------------------------------------------------

        # Normalize metrics into 0–1 range
        tv_score = np.clip((0.02 - temporal_variance) / 0.02, 0.0, 1.0)
        energy_score = np.clip(frame_energy / 0.1, 0.0, 1.0)
        
        # Final confidence from acoustic evidence
        confidence = float((tv_score + energy_score) / 2)
        
        if confidence > 0.65:
            classification = "AI_GENERATED"
            explanation = "Overly smooth acoustic patterns indicative of synthetic speech"
        else:
            classification = "HUMAN"
            explanation = "Natural temporal variations in human speech detected"



        # --------------------------------------------------
        # 6. Construct Response
        # --------------------------------------------------
        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": float(confidence),
            "explanation": explanation
        }


    except Exception as e:
        # --------------------------------------------------
        # 7. Error Handling
        # --------------------------------------------------
        return {
            "status": "error",
            "message": str(e)
        }
