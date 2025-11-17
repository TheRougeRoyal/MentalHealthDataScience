"""
Sentiment analysis module for text processing and crisis keyword detection.
"""
import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class SentimentScore:
    """Container for sentiment analysis results."""

    def __init__(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        compound: Optional[float] = None,
    ):
        """
        Initialize sentiment score.

        Args:
            valence: Pleasure/displeasure score (-1 to 1)
            arousal: Activation/deactivation score (-1 to 1)
            dominance: Control/lack of control score (-1 to 1)
            compound: Optional compound sentiment score (-1 to 1)
        """
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
        self.compound = compound

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }
        if self.compound is not None:
            result["compound"] = self.compound
        return result


class SentimentAnalyzer:
    """
    Analyzes sentiment from text responses using pre-trained models
    and detects crisis-related keywords.
    """

    # Crisis keywords for immediate flagging
    CRISIS_KEYWORDS = [
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "want to die",
        "better off dead",
        "no reason to live",
        "self-harm",
        "hurt myself",
        "cut myself",
        "overdose",
        "end it all",
        "can't go on",
        "hopeless",
        "worthless",
        "no way out",
    ]

    def __init__(self, use_transformers: bool = False, model_name: Optional[str] = None):
        """
        Initialize SentimentAnalyzer.

        Args:
            use_transformers: Whether to use transformer-based models
            model_name: Optional specific model name (e.g., 'distilbert-base-uncased')
        """
        self.use_transformers = use_transformers
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if use_transformers:
            self._load_transformer_model()
        else:
            self._load_lexicon_model()

    def _load_transformer_model(self) -> None:
        """Load transformer-based sentiment model."""
        try:
            from transformers import pipeline

            model_name = self.model_name or "distilbert-base-uncased-finetuned-sst-2-english"
            self.model = pipeline("sentiment-analysis", model=model_name)
            logger.info(f"Loaded transformer model: {model_name}")
        except ImportError:
            logger.warning(
                "transformers library not available. Falling back to lexicon-based approach."
            )
            self.use_transformers = False
            self._load_lexicon_model()
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}. Using lexicon-based approach.")
            self.use_transformers = False
            self._load_lexicon_model()

    def _load_lexicon_model(self) -> None:
        """Load lexicon-based sentiment analyzer (VADER-like)."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.model = SentimentIntensityAnalyzer()
            logger.info("Loaded VADER sentiment analyzer")
        except ImportError:
            logger.warning("vaderSentiment not available. Using simple lexicon-based approach.")
            self.model = None

    def analyze_sentiment(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of text and return valence, arousal, dominance scores.

        Args:
            text: Input text to analyze

        Returns:
            SentimentScore with valence, arousal, dominance values

        Note:
            Processing should complete within 500ms per text response
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided")
            return SentimentScore(valence=0.0, arousal=0.0, dominance=0.0, compound=0.0)

        text = text.strip()
        if not text:
            return SentimentScore(valence=0.0, arousal=0.0, dominance=0.0, compound=0.0)

        if self.use_transformers and self.model:
            return self._analyze_with_transformer(text)
        elif self.model:
            return self._analyze_with_vader(text)
        else:
            return self._analyze_with_simple_lexicon(text)

    def _analyze_with_transformer(self, text: str) -> SentimentScore:
        """Analyze sentiment using transformer model."""
        try:
            # Truncate text if too long (transformers have token limits)
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            result = self.model(text)[0]
            label = result["label"]
            score = result["score"]

            # Map to valence (-1 to 1)
            if label == "POSITIVE":
                valence = score
            else:
                valence = -score

            # Estimate arousal and dominance from text features
            arousal = self._estimate_arousal(text)
            dominance = self._estimate_dominance(text)

            return SentimentScore(
                valence=valence, arousal=arousal, dominance=dominance, compound=valence
            )
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            return SentimentScore(valence=0.0, arousal=0.0, dominance=0.0, compound=0.0)

    def _analyze_with_vader(self, text: str) -> SentimentScore:
        """Analyze sentiment using VADER."""
        try:
            scores = self.model.polarity_scores(text)
            compound = scores["compound"]

            # VADER compound score is already -1 to 1
            valence = compound

            # Estimate arousal and dominance
            arousal = self._estimate_arousal(text)
            dominance = self._estimate_dominance(text)

            return SentimentScore(
                valence=valence, arousal=arousal, dominance=dominance, compound=compound
            )
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return SentimentScore(valence=0.0, arousal=0.0, dominance=0.0, compound=0.0)

    def _analyze_with_simple_lexicon(self, text: str) -> SentimentScore:
        """Simple lexicon-based sentiment analysis as fallback."""
        text_lower = text.lower()

        # Simple positive/negative word lists
        positive_words = [
            "good",
            "great",
            "excellent",
            "happy",
            "joy",
            "love",
            "wonderful",
            "fantastic",
            "amazing",
            "better",
            "best",
            "positive",
            "hopeful",
            "optimistic",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "sad",
            "depressed",
            "anxious",
            "worried",
            "fear",
            "hate",
            "worst",
            "negative",
            "hopeless",
            "pessimistic",
            "miserable",
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total > 0:
            valence = (pos_count - neg_count) / total
        else:
            valence = 0.0

        arousal = self._estimate_arousal(text)
        dominance = self._estimate_dominance(text)

        return SentimentScore(
            valence=valence, arousal=arousal, dominance=dominance, compound=valence
        )

    def _estimate_arousal(self, text: str) -> float:
        """
        Estimate arousal (activation level) from text features.

        High arousal words: excited, anxious, angry, energetic
        Low arousal words: calm, tired, relaxed, bored
        """
        text_lower = text.lower()

        high_arousal_words = [
            "excited",
            "anxious",
            "angry",
            "energetic",
            "panic",
            "frantic",
            "intense",
            "urgent",
            "!",
        ]
        low_arousal_words = [
            "calm",
            "tired",
            "relaxed",
            "bored",
            "sleepy",
            "peaceful",
            "quiet",
            "still",
        ]

        high_count = sum(1 for word in high_arousal_words if word in text_lower)
        low_count = sum(1 for word in low_arousal_words if word in text_lower)

        # Count exclamation marks as high arousal indicator
        high_count += text.count("!")

        total = high_count + low_count
        if total > 0:
            arousal = (high_count - low_count) / total
        else:
            arousal = 0.0

        return arousal

    def _estimate_dominance(self, text: str) -> float:
        """
        Estimate dominance (sense of control) from text features.

        High dominance: confident, in control, powerful
        Low dominance: helpless, powerless, controlled
        """
        text_lower = text.lower()

        high_dominance_words = [
            "confident",
            "control",
            "powerful",
            "strong",
            "capable",
            "able",
            "can",
            "will",
        ]
        low_dominance_words = [
            "helpless",
            "powerless",
            "weak",
            "unable",
            "can't",
            "won't",
            "impossible",
            "trapped",
        ]

        high_count = sum(1 for word in high_dominance_words if word in text_lower)
        low_count = sum(1 for word in low_dominance_words if word in text_lower)

        total = high_count + low_count
        if total > 0:
            dominance = (high_count - low_count) / total
        else:
            dominance = 0.0

        return dominance

    def extract_emotion_scores(self, text: str) -> Dict[str, float]:
        """
        Extract emotion scores from text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping emotion names to scores
        """
        sentiment = self.analyze_sentiment(text)
        emotions = sentiment.to_dict()

        # Add basic emotion categories derived from VAD
        # High valence, high arousal = joy/excitement
        # Low valence, high arousal = anger/fear
        # Low valence, low arousal = sadness
        # High valence, low arousal = contentment

        if sentiment.valence > 0.3 and sentiment.arousal > 0.3:
            emotions["joy"] = min(sentiment.valence, sentiment.arousal)
        else:
            emotions["joy"] = 0.0

        if sentiment.valence < -0.3 and sentiment.arousal > 0.3:
            emotions["anxiety"] = min(abs(sentiment.valence), sentiment.arousal)
        else:
            emotions["anxiety"] = 0.0

        if sentiment.valence < -0.3 and sentiment.arousal < -0.3:
            emotions["sadness"] = min(abs(sentiment.valence), abs(sentiment.arousal))
        else:
            emotions["sadness"] = 0.0

        return emotions

    def detect_crisis_keywords(self, text: str) -> List[str]:
        """
        Detect crisis-related keywords for immediate flagging.

        Args:
            text: Input text to analyze

        Returns:
            List of detected crisis keywords
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        detected_keywords = []

        for keyword in self.CRISIS_KEYWORDS:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                detected_keywords.append(keyword)
                logger.warning(f"Crisis keyword detected: {keyword}")

        return detected_keywords

    def analyze_batch(
        self,
        df: DataFrame,
        text_column: str = "text_response",
        id_column: str = "anonymized_id",
    ) -> DataFrame:
        """
        Analyze sentiment for a batch of text responses.

        Args:
            df: Input DataFrame with text responses
            text_column: Column containing text to analyze
            id_column: Column containing anonymized identifiers

        Returns:
            DataFrame with sentiment scores added
        """
        if df.empty or text_column not in df.columns:
            logger.warning("Empty DataFrame or missing text column")
            return df

        df_result = df.copy()

        # Initialize result columns
        df_result["sentiment_valence"] = 0.0
        df_result["sentiment_arousal"] = 0.0
        df_result["sentiment_dominance"] = 0.0
        df_result["sentiment_compound"] = 0.0
        df_result["crisis_keywords"] = None
        df_result["crisis_flag"] = False

        for idx, row in df_result.iterrows():
            text = row[text_column]
            if pd.isna(text) or not text:
                continue

            # Analyze sentiment
            sentiment = self.analyze_sentiment(str(text))
            df_result.at[idx, "sentiment_valence"] = sentiment.valence
            df_result.at[idx, "sentiment_arousal"] = sentiment.arousal
            df_result.at[idx, "sentiment_dominance"] = sentiment.dominance
            if sentiment.compound is not None:
                df_result.at[idx, "sentiment_compound"] = sentiment.compound

            # Detect crisis keywords
            crisis_keywords = self.detect_crisis_keywords(str(text))
            if crisis_keywords:
                df_result.at[idx, "crisis_keywords"] = ", ".join(crisis_keywords)
                df_result.at[idx, "crisis_flag"] = True

        logger.info(
            f"Analyzed sentiment for {len(df_result)} text responses. "
            f"Crisis flags: {df_result['crisis_flag'].sum()}"
        )

        return df_result
