import cv2
import numpy as np
from PIL import Image
import io
import os
import torch
from collections import defaultdict
import time

# Disable quantum computing imports to fix confidence issues
QUANTUM_AVAILABLE = False

class VideoAIDetector:
    def __init__(self, image_detector, pytorch_model=None, preprocess=None, idx_to_class=None, frame_sample_rate=5):
        self.image_detector = image_detector
        self.pytorch_model = pytorch_model
        self.preprocess = preprocess
        self.idx_to_class = idx_to_class
        self.frame_sample_rate = frame_sample_rate
        # More conservative thresholds to reduce false AI detections
        self.ai_threshold = 0.7  # Increased from 0.6 - requires higher confidence for AI
        self.real_threshold = 0.65  # Slightly increased for real detection
        # Temporal smoothing window
        self.smoothing_window = 3
        # Minimum frames to analyze
        self.min_frames = 10
        
        # Disable quantum computing to fix confidence issues
        self.quantum_available = False
        print("Running in classical mode for reliable confidence scores")

    def _ensemble_vote(self, heuristic_result, pytorch_result):
        """Classical ensemble voting without quantum enhancement"""
        h_label, h_conf = heuristic_result
        p_label, p_conf = pytorch_result
            
        # If PyTorch model is available and very confident, trust it
        if p_conf > 0.75:  # Increased threshold from 0.6 - requires higher confidence
            return p_label, p_conf
        
        # If PyTorch model is moderately confident, use balanced approach
        if p_conf > 0.6:  # Increased threshold from 0.5
            # More balanced weighting to reduce bias
            heuristic_weight = 0.3  # Increased heuristic weight
            pytorch_weight = 0.7    # Decreased PyTorch weight
            
            ai_score = 0
            real_score = 0
            
            if h_label == "ai":
                ai_score += h_conf * heuristic_weight
            else:
                real_score += h_conf * heuristic_weight
                
            if p_label == "ai":
                ai_score += p_conf * pytorch_weight
            else:
                real_score += p_conf * pytorch_weight
                
            total = ai_score + real_score
            if total == 0:
                return "unknown", 0.0
                
            if ai_score > real_score:
                return "ai", min(0.8, ai_score / total)  # Reduced cap from 90% to 80%
            else:
                return "real", min(0.8, real_score / total)  # Reduced cap from 90% to 80%
        
        # If heuristic is very confident and PyTorch is not, trust heuristic
        # But be more conservative about AI detection
        if h_conf > 0.8:
            if h_label == "ai":
                # For AI detection, require even higher heuristic confidence
                if h_conf > 0.85:
                    return h_label, min(0.8, h_conf * 0.9)  # Reduced confidence for AI
                else:
                    # If not confident enough, default to real
                    return "real", 0.6
            else:
                # For real detection, trust heuristic more
                return h_label, min(0.85, h_conf)
        
        # Otherwise, use balanced weighted average with slight bias toward real (safer)
        heuristic_weight = 0.45  # Increased heuristic weight for more balance
        pytorch_weight = 0.55    # Decreased PyTorch weight to reduce AI bias
        
        ai_score = 0
        real_score = 0
        
        if h_label == "ai":
            ai_score += h_conf * heuristic_weight
        else:
            real_score += h_conf * heuristic_weight
            
        if p_label == "ai":
            ai_score += p_conf * pytorch_weight
        else:
            real_score += p_conf * pytorch_weight
            
        total = ai_score + real_score
        if total == 0:
            return "unknown", 0.0
            
        if ai_score > real_score:
            return "ai", min(0.85, ai_score / total)  # Cap confidence at 85%
        else:
            return "real", min(0.85, real_score / total)  # Cap confidence at 85%

    def _temporal_smoothing(self, frame_results):
        """Apply temporal smoothing to reduce flickering between predictions"""
        smoothed_results = []
        for i in range(len(frame_results)):
            start = max(0, i - self.smoothing_window)
            end = min(len(frame_results), i + self.smoothing_window + 1)
            window = frame_results[start:end]
            
            ai_count = sum(1 for label, _ in window if label == "ai")
            real_count = sum(1 for label, _ in window if label == "real")
            
            if ai_count > real_count:
                conf = sum(conf for label, conf in window if label == "ai") / ai_count
                smoothed_results.append(("ai", min(0.9, conf)))  # Reduced confidence cap from 0.95
            elif real_count > ai_count:
                conf = sum(conf for label, conf in window if label == "real") / real_count
                smoothed_results.append(("real", min(0.9, conf)))  # Reduced confidence cap from 0.95
            else:
                # Use the current frame's result if no clear majority
                current_label, current_conf = frame_results[i]
                smoothed_results.append((current_label, min(0.9, current_conf)))  # Reduced confidence cap
                
        return smoothed_results

    def _final_aggregation(self, smoothed_results):
        """Final aggregation without quantum enhancement"""
        if not smoothed_results:
            return "real", 0.5, smoothed_results, "No frames processed"
        
        ai_frames = [conf for label, conf in smoothed_results if label == "ai"]
        real_frames = [conf for label, conf in smoothed_results if label == "real"]
        
        ai_prop = len(ai_frames) / len(smoothed_results)
        real_prop = len(real_frames) / len(smoothed_results)
        
        # Calculate average confidence for each class
        ai_avg_conf = np.mean(ai_frames) if ai_frames else 0.0
        real_avg_conf = np.mean(real_frames) if real_frames else 0.0
        
        # More conservative decision making with higher thresholds
        if ai_prop > 0.75 and ai_avg_conf > 0.7:  # Increased thresholds for AI detection
            confidence = min(0.85, (ai_prop + ai_avg_conf) / 2)  # Reduced max confidence
            return "ai", confidence, smoothed_results, f"AI frames: {ai_prop:.1%}, avg conf: {ai_avg_conf:.2f}"
        elif real_prop > 0.7 and real_avg_conf > 0.65:  # Kept same for real detection
            confidence = min(0.85, (real_prop + real_avg_conf) / 2)  # Reduced max confidence
            return "real", confidence, smoothed_results, f"Real frames: {real_prop:.1%}, avg conf: {real_avg_conf:.2f}"
        else:
            # If no clear majority, use the class with higher average confidence
            # but be more conservative about the decision
            if ai_avg_conf > real_avg_conf and ai_avg_conf > 0.7:  # Increased threshold from 0.6
                confidence = min(0.75, ai_avg_conf * 0.8)  # Further reduced confidence for uncertain AI cases
                return "ai", confidence, smoothed_results, f"Uncertain - AI higher avg conf: {ai_avg_conf:.2f}"
            elif real_avg_conf > ai_avg_conf and real_avg_conf > 0.6:  # Kept same for real
                confidence = min(0.8, real_avg_conf * 0.85)  # Same confidence for uncertain real cases
                return "real", confidence, smoothed_results, f"Uncertain - Real higher avg conf: {real_avg_conf:.2f}"
            else:
                # If both are below threshold, default to real with low confidence
                return "real", 0.55, smoothed_results, f"Low confidence - AI: {ai_avg_conf:.2f}, Real: {real_avg_conf:.2f}"

    def detect_video(self, video_bytes):
        import time
        start_time = time.time()
        
        # Create temp file
        temp_filename = 'temp_video.mp4'
        with open(temp_filename, 'wb') as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return "error", 0.0, [], "Could not open video file"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames_to_analyze = min(20, total_frames)
        
        if max_frames_to_analyze < 1:
            cap.release()
            os.remove(temp_filename)
            return "real", 0.5, [], "No frames to process, defaulting to real with low confidence"
        
        frame_indices = np.linspace(0, total_frames - 1, max_frames_to_analyze, dtype=int)
        frame_results = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (224, 224))
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                continue
            
            frame_bytes = buffer.tobytes()
            heuristic_label, heuristic_conf = self.image_detector.predict(frame_bytes)
            
            pytorch_label = "unknown"
            pytorch_conf = 0.0
            
            if self.pytorch_model and self.preprocess and self.idx_to_class:
                try:
                    img = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        output = self.pytorch_model(img_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        pytorch_conf = float(probs[0, int(pred)].item())
                    
                    pred_int = int(pred)
                    if pred_int in self.idx_to_class:
                        pytorch_label = self.idx_to_class[pred_int]
                    else:
                        pytorch_label = "unknown"
                except Exception as e:
                    print(f"PyTorch model error: {e}")
            
            # Use classical ensemble voting without quantum enhancement
            final_label, final_conf = self._ensemble_vote(
                (heuristic_label, heuristic_conf),
                (pytorch_label, pytorch_conf)
            )
            
            print(f"Frame {idx}: Heuristic=({heuristic_label}, {heuristic_conf:.2f}), PyTorch=({pytorch_label}, {pytorch_conf:.2f}), Final=({final_label}, {final_conf:.2f})")
            frame_results.append((final_label, final_conf))
        
        cap.release()
        
        # Clean up temp file with retry logic
        for _ in range(5):
            try:
                os.remove(temp_filename)
                break
            except PermissionError:
                time.sleep(0.1)
        
        if not frame_results:
            return "real", 0.5, [], "No frames processed, defaulting to real with low confidence"
        
        # Apply temporal smoothing
        smoothed_results = self._temporal_smoothing(frame_results)
        
        # Apply final aggregation
        final_verdict, final_confidence, frame_results, message = self._final_aggregation(smoothed_results)
        
        # Final safety check: be more conservative about AI detection
        if final_verdict == "ai" and final_confidence < 0.75:
            # If AI detection confidence is not high enough, default to real
            final_verdict = "real"
            final_confidence = 0.65
            message += " - AI confidence too low, defaulting to real"
            print(f"Safety check: AI confidence {final_confidence:.2f} too low, defaulting to real")
        
        # Performance measurement
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Video processing completed in {processing_time:.2f} seconds")
        
        return final_verdict, final_confidence, frame_results, message

    def get_quantum_status(self):
        """Return status - now always classical"""
        return {
            "quantum_enabled": False,
            "quantum_device": "None",
            "quantum_circuits": [],
            "performance_boost": "Classical mode for reliable confidence scores"
        }
