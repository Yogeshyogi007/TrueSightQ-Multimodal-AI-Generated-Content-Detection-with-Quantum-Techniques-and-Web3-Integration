import torch
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sklearn.ensemble import IsolationForest
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import math
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class QuillBotLevelAIDetector:
    def __init__(self):
        # Load OpenAI detector model
        self.detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
        self.detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
        # Load GPT-2 for perplexity
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
        # Initialize statistical models
        self.isolation_forest = IsolationForest(contamination=0.1)
        # Predefined patterns and features
        self.ai_patterns = self._load_ai_patterns()
        self.human_patterns = self._load_human_patterns()

    def _load_ai_patterns(self):
        return {
            'overly_descriptive': [
                'shimmering', 'glistening', 'trembling', 'nestled between', 'bursting with',
                'wide with wonder', 'gentle ticking', 'warm glow', 'hidden beneath',
                'secret garden', 'forgotten door', 'curious girl', 'rolling hills'
            ],
            'perfect_structure': [
                'first', 'then', 'next', 'finally', 'after that', 'in the end',
                'as a result', 'consequently', 'meanwhile'
            ],
            'repetitive_starters': [
                'the', 'she', 'he', 'it', 'they', 'there', 'as', 'with', 'while'
            ],
            'generic_emotions': [
                'happy', 'sad', 'excited', 'angry', 'surprised', 'joyful',
                'fearful', 'content', 'peaceful'
            ]
        }

    def _load_human_patterns(self):
        return {
            'imperfections': [
                'kinda', 'sorta', 'maybe', 'probably', 'i think', 'i guess',
                'like', 'you know', 'well', 'actually', 'basically'
            ],
            'contractions': [
                "don't", "can't", "won't", "isn't", "wasn't", "didn't",
                "i'm", "you're", "they're", "we're", "it's"
            ],
            'interjections': [
                'uh', 'um', 'ah', 'oh', 'hmm', 'well', 'so', 'anyway'
            ],
            'specific_details': [
                'at 3pm', 'last tuesday', 'my cousin jake', 'the red one',
                'near the old', 'smelled like', 'tasted like'
            ]
        }

    def openai_detector_score(self, text):
        inputs = self.detector_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
            logits = outputs.logits
            # For binary classification, HuggingFace detector: label 1 = AI, label 0 = human
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        return prob  # Closer to 1 = AI, closer to 0 = human

    def calculate_perplexity(self, text):
        # Limit sequence length so GPT-2 position embeddings stay in range
        encodings = self.gpt_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = self.gpt_model(input_ids, labels=target_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    def calculate_burstiness(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        return np.var(sentence_lengths) / np.mean(sentence_lengths)

    def analyze_lexical_diversity(self, text):
        words = word_tokenize(text.lower())
        if len(words) < 10:
            return 0.5
        ttr = len(set(words)) / len(words)
        word_freq = Counter(words)
        rare_words = sum(1 for word, count in word_freq.items() if count == 1)
        rare_word_ratio = rare_words / len(words)
        pos_tags = [tag for word, tag in pos_tag(words)]
        pos_ttr = len(set(pos_tags)) / len(pos_tags)
        return 0.4*ttr + 0.3*rare_word_ratio + 0.3*pos_ttr

    def detect_ai_specific_patterns(self, text):
        text_lower = text.lower()
        ai_score = 0
        total_indicators = 0
        for phrase in self.ai_patterns['overly_descriptive']:
            if phrase in text_lower:
                ai_score += 1
                total_indicators += 1
        structure_words = sum(1 for word in self.ai_patterns['perfect_structure'] if word in text_lower)
        if structure_words > 3:
            ai_score += min(3, structure_words / 2)
            total_indicators += 1
        sentences = sent_tokenize(text)
        if len(sentences) > 3:
            starters = [s.split()[0].lower() for s in sentences if len(s.split()) > 0]
            starter_counts = Counter(starters)
            most_common = starter_counts.most_common(1)[0][1] if starter_counts else 0
            if most_common / len(sentences) > 0.5:
                ai_score += 2
                total_indicators += 1
        emotion_words = sum(1 for word in self.ai_patterns['generic_emotions'] if word in text_lower)
        if emotion_words > 2:
            ai_score += min(2, emotion_words)
            total_indicators += 1
        return ai_score / max(1, total_indicators)

    def detect_human_patterns(self, text):
        text_lower = text.lower()
        human_score = 0
        total_indicators = 0
        for phrase in self.human_patterns['imperfections']:
            if phrase in text_lower:
                human_score += 1
                total_indicators += 1
        contractions = sum(1 for word in self.human_patterns['contractions'] if word in text_lower)
        if contractions > 1:
            human_score += min(2, contractions)
            total_indicators += 1
        interjections = sum(1 for word in self.human_patterns['interjections'] if word in text_lower)
        if interjections > 0:
            human_score += interjections
            total_indicators += 1
        details = sum(1 for word in self.human_patterns['specific_details'] if word in text_lower)
        if details > 0:
            human_score += details * 2
            total_indicators += 1
        return human_score / max(1, total_indicators)

    def analyze_writing_style(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        length_var = np.var(sent_lengths)
        punctuation_ratio = len(re.findall(r'[,.!?;:]', text)) / len(word_tokenize(text))
        words = word_tokenize(text)
        if words:
            capitalized = sum(1 for word in words if word[0].isupper() and word not in ['I', 'I\'m'])
            cap_ratio = capitalized / len(words)
        else:
            cap_ratio = 0
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            para_lengths = [len(word_tokenize(p)) for p in paragraphs]
            para_var = np.var(para_lengths)
        else:
            para_var = 0
        style_score = 0.3 * (1 - min(1, length_var/50)) + \
                     0.2 * (1 - min(1, punctuation_ratio*10)) + \
                     0.2 * cap_ratio + \
                     0.3 * (1 - min(1, para_var/100))
        return style_score

    def detect_personal_experience(self, text):
        # Count first-person pronouns and references
        first_person = ["i", "my", "me", "mine", "we", "our", "us", "ours"]
        words = word_tokenize(text.lower())
        count = sum(1 for w in words if w in first_person)
        return min(1, count / max(1, len(words) / 20))  # Normalize: 1 if >5% of words are first-person

    def detect_connector_overuse(self, text):
        connectors = ["however", "moreover", "furthermore", "in conclusion", "therefore", "thus", "additionally", "consequently", "nevertheless", "on the other hand", "for example", "for instance"]
        words = word_tokenize(text.lower())
        count = sum(1 for w in words if w in connectors)
        return min(1, count / max(1, len(words) / 30))  # Normalize: 1 if >3% of words are connectors

    def sentence_structure_variety(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5
        starters = [s.split()[0].lower() for s in sentences if len(s.split()) > 0]
        unique_starters = len(set(starters))
        variety = unique_starters / len(sentences)
        return min(1, variety)  # 1 = high variety, 0 = all same

    def named_entity_count(self, text):
        # Simple NER using capitalized words not at sentence start
        words = word_tokenize(text)
        entities = [w for i, w in enumerate(words) if w[0].isupper() and i != 0 and w.lower() not in ["i"]]
        return min(1, len(entities) / max(1, len(words) / 20))  # Normalize: 1 if >5% of words are entities

    def is_ai_generated(self, text, threshold=0.6):
        text = text.strip()
        if len(text) < 50:
            return {"verdict": "Inconclusive", "confidence": 0.5, "score": 0.5}
        # Hard cap very long inputs to keep the detector stable and fast
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars]
        detector_prob = self.openai_detector_score(text)
        metrics = {
            'openai_detector_prob': detector_prob,
            'perplexity': self.calculate_perplexity(text),
            'burstiness': self.calculate_burstiness(text),
            'lexical_diversity': self.analyze_lexical_diversity(text),
            'ai_patterns': min(1, max(0, self.detect_ai_specific_patterns(text))),
            'human_patterns': min(1, max(0, self.detect_human_patterns(text))),
            'writing_style': self.analyze_writing_style(text),
            'personal_experience': self.detect_personal_experience(text),
            'connector_overuse': self.detect_connector_overuse(text),
            'sentence_variety': self.sentence_structure_variety(text),
            'named_entity_count': self.named_entity_count(text)
        }
        # Ensemble logic – always return a distinct AI vs Human verdict
        ai_score = 0.4 * detector_prob + 0.3 * metrics['ai_patterns'] + 0.3 * (1 - metrics['human_patterns'])
        # Slightly conservative threshold: >=0.6 → AI, otherwise Human
        if ai_score >= 0.6:
            verdict = "AI-Generated"
            confidence = round(ai_score, 4)
        else:
            verdict = "Human-Written"
            confidence = round(1 - ai_score, 4)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "score": round(ai_score, 4),
            "metrics": metrics
        }