from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import Optional, Union
import numpy as np

from .models.text_detector import QuillBotLevelAIDetector
from .models.image_detector import AdvancedImageDetector
from .models.audio_detector import AudioAIDetector
from .models.video_detector import VideoAIDetector

app = FastAPI(title="AI Content Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    verdict: str
    confidence: float
    modality: str
    perplexity: Optional[float] = None  # Optional field for text detection

text_detector = QuillBotLevelAIDetector()
image_detector = AdvancedImageDetector()
audio_detector = AudioAIDetector()

# Load the model once at startup (if not already loaded)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../face_detector.pth')
CLASS_IDX_PATH = os.path.join(os.path.dirname(__file__), '../../class_to_idx.json')
if os.path.exists(MODEL_PATH):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    # Preprocessing (must match training)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Load class_to_idx mapping and invert it
    if os.path.exists(CLASS_IDX_PATH):
        with open(CLASS_IDX_PATH, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    else:
        idx_to_class = {0: "ai", 1: "real"}  # fallback
else:
    model = None
    preprocess = None
    idx_to_class = None

# Initialize VideoAIDetector
video_detector = VideoAIDetector(
    image_detector=image_detector,
    pytorch_model=model,
    preprocess=preprocess,
    idx_to_class=idx_to_class,
    frame_sample_rate=1  # 1 frame per second
)

@app.post("/detect/text", response_model=DetectionResult)
def detect_text(text: str = Form(...)):
    result = text_detector.is_ai_generated(text)
    # Ensure confidence is not None and convert to float
    confidence = result.get("confidence", 0.5)
    if confidence is None:
        confidence = 0.5
    return DetectionResult(
        verdict=result["verdict"],
        confidence=float(confidence),
        modality="text",
        perplexity=result.get("score")  # Or you can add more fields if needed
    )

@app.post("/detect/image", response_model=DetectionResult)
def detect_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    
    # Get heuristic prediction
    heuristic_label, heuristic_confidence = image_detector.predict(image_bytes)
    
    # Get PyTorch prediction if model is available
    pytorch_label = "unknown"
    pytorch_confidence = 0.0
    
    if model is not None and preprocess is not None and idx_to_class is not None:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = preprocess(img)
            # Ensure img_tensor is a torch.Tensor before calling unsqueeze
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.tensor(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                pytorch_confidence = float(probs[0, int(pred)].item())
            
            pred_int = int(pred)
            if pred_int in idx_to_class:
                pytorch_label = idx_to_class[pred_int]
            else:
                pytorch_label = "unknown"
        except Exception as e:
            print(f"PyTorch model error in image detection: {e}")
    
    # Ensemble logic - same as face detection
    if pytorch_label == heuristic_label:
        final_result = pytorch_label
        final_conf = max(pytorch_confidence, heuristic_confidence)
        method = "agreement"
    else:
        if pytorch_confidence > 0.8:
            final_result = pytorch_label
            final_conf = pytorch_confidence
            method = "pytorch_high_conf"
        elif heuristic_confidence > 0.8:
            final_result = heuristic_label
            final_conf = heuristic_confidence
            method = "heuristic_high_conf"
        else:
            # Use weighted average when both models disagree and neither is very confident
            if pytorch_label != "unknown":
                # Trust PyTorch more when available
                final_result = pytorch_label
                final_conf = pytorch_confidence * 0.7 + heuristic_confidence * 0.3
                method = "weighted_pytorch"
            else:
                # Fall back to heuristic if PyTorch unavailable
                final_result = heuristic_label
                final_conf = heuristic_confidence
                method = "heuristic_fallback"
    
    # Map internal label to user-friendly verdict
    if final_result == "ai":
        user_verdict = "AI-Generated Image"
    else:
        user_verdict = "Real Image"
    
    # Cap confidence at 95% for reliability
    final_conf = min(0.95, final_conf)
    
    print(f"Image detection - Heuristic: ({heuristic_label}, {heuristic_confidence:.2f}), PyTorch: ({pytorch_label}, {pytorch_confidence:.2f}), Final: ({final_result}, {final_conf:.2f}), Method: {method}")
    
    return DetectionResult(verdict=user_verdict, confidence=float(final_conf), modality="image")

@app.post("/detect/audio", response_model=DetectionResult)
def detect_audio(file: UploadFile = File(...)):
    audio_bytes = file.file.read()
    result = audio_detector.detect_audio(audio_bytes)
    
    # Map internal label to user-friendly verdict
    if result["verdict"] == "AI-Generated":
        user_verdict = "AI-Generated Audio"
    elif result["verdict"] == "Human-Generated":
        user_verdict = "Human-Generated Audio"
    else:
        user_verdict = result["verdict"]  # Keep "Uncertain", "Inconclusive", "Error"
    
    return DetectionResult(
        verdict=user_verdict, 
        confidence=float(result["confidence"]), 
        modality="audio"
    )

@app.post("/detect/video", response_model=DetectionResult)
def detect_video(file: UploadFile = File(...)):
    video_bytes = file.file.read()
    verdict, confidence, frame_results, message = video_detector.detect_video(video_bytes)
    # Map internal label to user-friendly verdict
    if verdict == "ai":
        user_verdict = "AI-Generated Video"
    elif verdict == "real":
        user_verdict = "Real Video"
    else:
        user_verdict = verdict  # fallback for unknown
    # Optionally, you can log or return the message for debugging
    return DetectionResult(verdict=user_verdict, confidence=float(confidence), modality="video")

@app.post("/detect-face-ai")
async def detect_face_ai(file: UploadFile = File(...)):
    if model is None or preprocess is None or idx_to_class is None:
        return JSONResponse({"error": "Model or class mapping not found. Please ensure face_detector.pth and class_to_idx.json are in the project root."}, status_code=500)
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        # Convert PIL Image to tensor and add batch dimension
        img_tensor = preprocess(img)
        # Ensure img_tensor is a torch.Tensor before calling unsqueeze
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)
        
        # PyTorch model prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            # Fix tensor indexing - use proper indexing for PyTorch tensors
            pytorch_confidence = float(probs[0, int(pred)].item())
        
        # Fix the idx_to_class access - ensure pred is an integer
        pred_int = int(pred)
        if pred_int in idx_to_class:
            pytorch_label = idx_to_class[pred_int]
        else:
            pytorch_label = str(pred_int)
        # Heuristic prediction
        heuristic_label, heuristic_confidence = image_detector.predict(contents)
        # Ensemble logic
        if pytorch_label == heuristic_label:
            final_result = pytorch_label
            final_conf = max(pytorch_confidence, heuristic_confidence)
            method = "agreement"
        else:
            if pytorch_confidence > 0.8:
                final_result = pytorch_label
                final_conf = pytorch_confidence
                method = "pytorch_high_conf"
            elif heuristic_confidence > 0.8:
                final_result = heuristic_label
                final_conf = heuristic_confidence
                method = "heuristic_high_conf"
            else:
                final_result = pytorch_label
                final_conf = max(pytorch_confidence, heuristic_confidence)
                method = "disagreement"
        return JSONResponse({
            "result": final_result,
            "confidence": final_conf,
            "pytorch_result": pytorch_label,
            "pytorch_confidence": pytorch_confidence,
            "heuristic_result": heuristic_label,
            "heuristic_confidence": heuristic_confidence,
            "method": method
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 