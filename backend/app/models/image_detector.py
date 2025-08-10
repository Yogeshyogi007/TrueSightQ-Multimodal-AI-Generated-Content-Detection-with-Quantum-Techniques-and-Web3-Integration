from PIL import Image
import cv2
import numpy as np
import io
import os
from scipy.fft import dct

class AdvancedImageDetector:
    def __init__(self):
        pass

    def analyze_image_artifacts(self, image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Use scipy's DCT with proper type handling
        gray_float = np.float32(gray)
        dct_result = dct(dct(gray_float, axis=0), axis=1)
        # Ensure dct_result is a numpy array and handle indexing properly
        dct_result = np.asarray(dct_result)
        # Calculate DCT energy from high-frequency components
        dct_high_freq = dct_result[8:, 8:]
        dct_total = float(np.sum(np.abs(dct_result)))
        dct_high_freq_sum = float(np.sum(np.abs(dct_high_freq)))
        dct_energy = dct_high_freq_sum / dct_total if dct_total > 0 else 0.0
        
        noise = cv2.fastNlMeansDenoising(gray) - gray
        noise_std = np.std(noise)
        noise_entropy = self.calculate_entropy(noise)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:, :, 1])
        texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate frequency domain features
        center_freq = magnitude_spectrum[96:160, 96:160]
        high_freq = magnitude_spectrum[64:192, 64:192]
        
        center_energy = np.mean(center_freq)
        high_energy = np.mean(high_freq)
        freq_uniformity = center_energy / (high_energy + 1e-10)
        
        return {
            'dct_energy': dct_energy,
            'noise_std': noise_std,
            'noise_entropy': noise_entropy,
            'edge_density': edge_density,
            'color_std': color_std,
            'texture_variance': texture_variance,
            'freq_uniformity': freq_uniformity,
            'center_energy': center_energy,
            'high_energy': high_energy
        }

    def detect_face_anomalies(self, image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Fix cv2.data access - use a more reliable method to find the cascade file
        try:
            # Try to find the cascade file in common locations
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            face_cascade = None
            for path in cascade_paths:
                if os.path.exists(path):
                    face_cascade = cv2.CascadeClassifier(path)
                    break
            if face_cascade is None:
                # Fallback: create a simple face detector
                return {'face_detected': False, 'anomaly_score': 0.0, 'deepfake_indicators': []}
        except Exception:
            # If cascade loading fails, return no face detected
            return {'face_detected': False, 'anomaly_score': 0.0, 'deepfake_indicators': []}
        
        # Check if face_cascade was successfully loaded
        if face_cascade is None:
            return {'face_detected': False, 'anomaly_score': 0.0, 'deepfake_indicators': []}
            
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return {'face_detected': False, 'anomaly_score': 0.0, 'deepfake_indicators': []}
        anomaly_scores = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            left_half = face_roi[:, :w//2]
            right_half = cv2.flip(face_roi[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            symmetry_score = 1.0 / (1.0 + symmetry_diff / 255.0)
            skin_roi = face_roi[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            if skin_roi.size > 0:
                skin_texture = np.std(cv2.Laplacian(skin_roi, cv2.CV_64F))
                skin_anomaly = abs(skin_texture - 50) / 50
            else:
                skin_anomaly = 0.5
            eye_region = face_roi[int(h*0.3):int(h*0.5), int(w*0.2):int(w*0.8)]
            if eye_region.size > 0:
                eye_variance = np.var(eye_region)
                eye_anomaly = abs(eye_variance - 1000) / 1000
            else:
                eye_anomaly = 0.5
            lighting_variance = np.var(face_roi)
            lighting_anomaly = abs(lighting_variance - 2000) / 2000
            total_anomaly = (symmetry_score + skin_anomaly + eye_anomaly + lighting_anomaly) / 4
            anomaly_scores.append(total_anomaly)
        return {
            'face_detected': True,
            'anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0.0,
        }

    def analyze_color_distribution(self, image):
        img_array = np.array(image)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        # Fix histogram range parameter - use tuple instead of list
        r_hist = np.histogram(r.flatten(), bins=256, range=(0, 256))[0]
        g_hist = np.histogram(g.flatten(), bins=256, range=(0, 256))[0]
        b_hist = np.histogram(b.flatten(), bins=256, range=(0, 256))[0]
        r_entropy = self.calculate_entropy_from_hist(r_hist)
        g_entropy = self.calculate_entropy_from_hist(g_hist)
        b_entropy = self.calculate_entropy_from_hist(b_hist)
        color_uniformity = (r_entropy + g_entropy + b_entropy) / 3
        r_g_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        r_b_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        g_b_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        avg_correlation = (abs(r_g_corr) + abs(r_b_corr) + abs(g_b_corr)) / 3
        color_banding = self.detect_color_banding(img_array)
        return {
            'color_uniformity': color_uniformity,
            'color_correlation': avg_correlation,
            'color_banding': color_banding
        }

    def detect_color_banding(self, img_array):
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharp_transitions = np.sum(gradient_magnitude > np.mean(gradient_magnitude) + 2*np.std(gradient_magnitude))
        banding_score = sharp_transitions / gradient_magnitude.size
        return banding_score

    def calculate_entropy(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def calculate_entropy_from_hist(self, hist):
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def predict(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        artifacts = self.analyze_image_artifacts(img)
        face_analysis = self.detect_face_anomalies(img)
        color_analysis = self.analyze_color_distribution(img)
        
        # More balanced AI indicators with realistic thresholds
        ai_score = 0.0
        real_score = 0.0
        total_indicators = 0
        
        # DCT energy - high frequency artifacts often indicate AI generation
        if artifacts['dct_energy'] > 0.8:  # Much higher threshold - random noise can be high
            ai_score += 0.6
            real_score += 0.2
        elif artifacts['dct_energy'] < 0.1:  # Very low DCT energy suggests real
            ai_score += 0.2
            real_score += 0.6
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Noise entropy - AI images often have unnatural noise patterns
        if artifacts['noise_entropy'] < 2.0:  # Much lower threshold - random images can have low entropy
            ai_score += 0.7
            real_score += 0.2
        elif artifacts['noise_entropy'] > 6.0:  # Higher threshold for real images
            ai_score += 0.2
            real_score += 0.7
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Edge density - AI images may have unnatural edge patterns
        if artifacts['edge_density'] > 0.4:  # Much higher threshold - random images can have high edge density
            ai_score += 0.6
            real_score += 0.2
        elif artifacts['edge_density'] < 0.1:  # Lower threshold for real images
            ai_score += 0.2
            real_score += 0.6
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Color uniformity - AI images may have artificial color patterns
        if color_analysis['color_uniformity'] > 7.5:  # Much higher threshold
            ai_score += 0.7
            real_score += 0.2
        elif color_analysis['color_uniformity'] < 4.0:  # Lower threshold for real images
            ai_score += 0.2
            real_score += 0.7
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Color banding - AI images may have artificial color transitions
        if color_analysis['color_banding'] > 0.2:  # Higher threshold
            ai_score += 0.7
            real_score += 0.2
        elif color_analysis['color_banding'] < 0.02:  # Lower threshold for real images
            ai_score += 0.2
            real_score += 0.6
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Face analysis - if face detected, check for anomalies
        if face_analysis['face_detected']:
            if face_analysis['anomaly_score'] > 0.8:  # Higher threshold
                ai_score += 0.8
                real_score += 0.2
            elif face_analysis['anomaly_score'] < 0.2:  # Lower threshold for real images
                ai_score += 0.2
                real_score += 0.8
            else:
                ai_score += 0.4
                real_score += 0.4
        else:
            # No face detected - neutral indicator
            ai_score += 0.5
            real_score += 0.5
        total_indicators += 1
        
        # Texture variance - AI images may have artificial textures
        if artifacts['texture_variance'] < 50:  # Much lower threshold
            ai_score += 0.6
            real_score += 0.2
        elif artifacts['texture_variance'] > 200:  # Higher threshold for real images
            ai_score += 0.2
            real_score += 0.6
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Frequency uniformity - AI images may have artificial frequency patterns
        if artifacts['freq_uniformity'] > 0.95:  # Much higher threshold
            ai_score += 0.6
            real_score += 0.2
        elif artifacts['freq_uniformity'] < 0.5:  # Lower threshold for real images
            ai_score += 0.2
            real_score += 0.6
        else:
            ai_score += 0.4
            real_score += 0.4
        total_indicators += 1
        
        # Calculate average scores
        ai_probability = ai_score / total_indicators
        real_probability = real_score / total_indicators
        
        # Make decision based on which probability is higher
        if ai_probability > real_probability:
            verdict = 'ai'
            # Much more conservative confidence calculation for AI detection
            if ai_probability > 0.7:  # Only high confidence for AI
                confidence = min(0.75, 0.6 + (ai_probability - 0.7) * 0.5)
            else:
                confidence = min(0.65, 0.5 + (ai_probability - 0.5) * 0.3)  # Lower confidence for uncertain AI
        else:
            verdict = 'real'
            # More conservative confidence calculation for real
            confidence = min(0.8, 0.5 + (real_probability - 0.5) * 0.6)
        
        return verdict, round(confidence, 2)