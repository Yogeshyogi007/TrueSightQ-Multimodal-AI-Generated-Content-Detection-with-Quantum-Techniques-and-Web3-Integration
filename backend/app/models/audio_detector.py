import numpy as np
import librosa
import soundfile as sf
import io
import os
from scipy import signal
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class AudioAIDetector:
    def __init__(self):
        # Audio processing parameters
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 13
        
        # Detection thresholds (narrow Uncertain band)
        self.ai_threshold = 0.55
        self.real_threshold = 0.55
        
        # Initialize anomaly detection
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def _load_audio(self, audio_bytes):
        """Load audio from bytes and resample if necessary"""
        try:
            # Try to load with soundfile first
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Convert stereo to mono
        except:
            # Fallback to librosa
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio, self.sample_rate
    
    def _extract_spectral_features(self, audio):
        """Extract spectral features that help distinguish AI audio"""
        features = {}
        
        # Mel-frequency cepstral coefficients (MFCC)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        features['mfcc_entropy'] = entropy(np.abs(mfccs).flatten())
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def _extract_temporal_features(self, audio):
        """Extract temporal features that indicate AI generation"""
        features = {}
        
        # Root mean square energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_entropy'] = entropy(rms)
        
        # Onset strength (rhythm analysis)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Beat frames
        beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)[1]
        features['beat_count'] = len(beat_frames)
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = pitches[magnitudes > np.percentile(magnitudes, 95)]
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        return features
    
    def _detect_synthetic_artifacts(self, audio):
        """Detect artifacts common in AI-generated audio"""
        artifacts = {}
        
        # Check for unnaturally perfect consistency
        # AI audio often has very consistent amplitude
        amplitude_consistency = np.std(np.abs(audio))
        artifacts['amplitude_consistency'] = amplitude_consistency
        
        # Check for artificial silence gaps
        # AI audio might have unnaturally clean transitions
        silence_threshold = 0.01
        silence_mask = np.abs(audio) < silence_threshold
        silence_ratio = np.sum(silence_mask) / len(audio)
        artifacts['silence_ratio'] = silence_ratio
        
        # Check for unnatural frequency patterns
        # AI audio might have overly smooth frequency transitions
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Frequency smoothness (AI audio tends to be smoother)
        freq_smoothness = np.mean(np.diff(magnitude, axis=0))
        artifacts['freq_smoothness'] = freq_smoothness
        
        # Time smoothness
        time_smoothness = np.mean(np.diff(magnitude, axis=1))
        artifacts['time_smoothness'] = time_smoothness
        
        # Check for quantization artifacts
        # AI audio might have subtle quantization effects
        quantization_error = np.mean(np.abs(audio - np.round(audio * 32768) / 32768))
        artifacts['quantization_error'] = quantization_error
        
        return artifacts
    
    def _analyze_voice_characteristics(self, audio):
        """Analyze voice-specific characteristics"""
        voice_features = {}
        
        # Formant analysis (for speech-like audio)
        # Extract formants using LPC
        try:
            # Use a window for formant analysis
            window_size = int(0.025 * self.sample_rate)  # 25ms window
            hop_size = int(0.010 * self.sample_rate)     # 10ms hop
            
            formants = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                if np.std(window) > 0.01:  # Only analyze non-silent frames
                    # LPC analysis
                    lpc_coeffs = librosa.lpc(window, order=12)
                    roots = np.roots(lpc_coeffs)
                    # Find formant frequencies
                    angles = np.angle(roots)
                    freqs = angles * self.sample_rate / (2 * np.pi)
                    # Filter realistic formant frequencies (80-8000 Hz)
                    formant_freqs = freqs[(freqs > 80) & (freqs < 8000)]
                    if len(formant_freqs) > 0:
                        formants.extend(formant_freqs[:3])  # First 3 formants
            
            if formants:
                formants = np.array(formants)
                voice_features['formant_mean'] = np.mean(formants)
                voice_features['formant_std'] = np.std(formants)
                voice_features['formant_range'] = np.max(formants) - np.min(formants)
            else:
                voice_features['formant_mean'] = 0
                voice_features['formant_std'] = 0
                voice_features['formant_range'] = 0
        except:
            voice_features['formant_mean'] = 0
            voice_features['formant_std'] = 0
            voice_features['formant_range'] = 0
        
        # Jitter and shimmer (voice quality measures)
        # These are typically more natural in human speech
        voice_features['jitter'] = self._calculate_jitter(audio)
        voice_features['shimmer'] = self._calculate_shimmer(audio)
        
        return voice_features
    
    def _calculate_jitter(self, audio):
        """Calculate jitter (pitch period variation)"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 90)]
            
            if len(pitch_values) < 10:
                return 0
            
            # Calculate pitch period variations
            pitch_periods = 1 / pitch_values[pitch_values > 0]
            if len(pitch_periods) < 2:
                return 0
            
            jitter = np.std(pitch_periods) / np.mean(pitch_periods)
            return jitter
        except:
            return 0
    
    def _calculate_shimmer(self, audio):
        """Calculate shimmer (amplitude variation)"""
        try:
            # Use RMS energy as amplitude measure
            rms = librosa.feature.rms(y=audio)[0]
            if len(rms) < 10:
                return 0
            
            shimmer = np.std(rms) / np.mean(rms)
            return shimmer
        except:
            return 0
    
    def _ensemble_scoring(self, features):
        """Combine all features into a final AI detection score"""
        # Normalize features to 0-1 range
        normalized_features = {}
        
        # Spectral features (AI audio tends to have more consistent spectral features)
        normalized_features['spectral_consistency'] = min(1, features['spectral_centroid_std'] / 1000)
        normalized_features['mfcc_entropy'] = min(1, features['mfcc_entropy'] / 10)
        
        # Temporal features (AI audio tends to have more consistent timing)
        normalized_features['temporal_consistency'] = min(1, features['rms_std'] / 0.1)
        normalized_features['rhythm_consistency'] = min(1, features['onset_strength_std'] / 0.5)
        
        # Synthetic artifacts (higher values indicate AI)
        normalized_features['amplitude_consistency'] = min(1, features['amplitude_consistency'] / 0.1)
        normalized_features['silence_ratio'] = features['silence_ratio']
        normalized_features['freq_smoothness'] = min(1, abs(features['freq_smoothness']) / 0.1)
        normalized_features['time_smoothness'] = min(1, abs(features['time_smoothness']) / 0.1)
        
        # Voice characteristics (AI voice tends to have less natural variation)
        normalized_features['jitter'] = min(1, features['jitter'] / 0.1)
        normalized_features['shimmer'] = min(1, features['shimmer'] / 0.1)
        normalized_features['formant_variation'] = min(1, features['formant_std'] / 1000)
        
        # Weighted scoring
        weights = {
            'spectral_consistency': 0.15,
            'mfcc_entropy': 0.10,
            'temporal_consistency': 0.15,
            'rhythm_consistency': 0.10,
            'amplitude_consistency': 0.15,
            'silence_ratio': 0.05,
            'freq_smoothness': 0.10,
            'time_smoothness': 0.10,
            'jitter': 0.05,
            'shimmer': 0.05
        }
        
        # Calculate AI score (higher = more likely AI)
        ai_score = 0
        for feature, weight in weights.items():
            if feature in normalized_features:
                ai_score += normalized_features[feature] * weight
        
        return ai_score
    
    def detect_audio(self, audio_bytes):
        """Main method to detect AI-generated audio"""
        try:
            # Load and preprocess audio
            audio, sr = self._load_audio(audio_bytes)
            
            # Ensure minimum audio length
            if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                return {
                    "verdict": "Inconclusive",
                    "confidence": 0.5,
                    "score": 0.5,
                    "message": "Audio too short for reliable analysis"
                }
            
            # Extract all features
            spectral_features = self._extract_spectral_features(audio)
            temporal_features = self._extract_temporal_features(audio)
            artifacts = self._detect_synthetic_artifacts(audio)
            voice_features = self._analyze_voice_characteristics(audio)
            
            # Combine all features
            all_features = {**spectral_features, **temporal_features, **artifacts, **voice_features}
            
            # Calculate AI score
            ai_score = self._ensemble_scoring(all_features)
            
            # Determine verdict with an explicit Uncertain band
            low_real = 1 - self.real_threshold
            if ai_score >= self.ai_threshold:
                verdict = "AI-Generated"
                confidence = min(0.99, 0.7 + (ai_score - self.ai_threshold) * 2)
            elif ai_score <= low_real:
                verdict = "Real-Human Audio"
                confidence = min(0.99, 0.7 + (low_real - ai_score) * 2)
            else:
                verdict = "Uncertain Audio"
                # Keep uncertainty confidence in a narrow, modest band
                # Centered around ai_score ~ 0.5
                confidence = 0.4 + (1 - abs(ai_score - 0.5) * 2) * 0.2
            
            return {
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "score": round(ai_score, 4),
                "features": all_features,
                "message": f"AI score: {ai_score:.3f}"
            }
            
        except Exception as e:
            return {
                "verdict": "Error",
                "confidence": 0.0,
                "score": 0.0,
                "message": f"Error processing audio: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    detector = AudioAIDetector()
    # Test with a sample audio file
    # result = detector.detect_audio(audio_bytes)
    # print(f"Verdict: {result['verdict']}")
    # print(f"Confidence: {result['confidence']*100:.2f}%")
    # print(f"AI Score: {result['score']*100:.2f}%") 