import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Using basic sentiment analysis.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER Sentiment not available. Using alternative methods.")

import re
import requests
from bs4 import BeautifulSoup
import json
import os
from dataclasses import dataclass
import logging
from collections import defaultdict, Counter
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

@dataclass
class SentimentConfig:
    sentiment_methods: List[str] = None
    news_sources: List[str] = None
    social_sources: List[str] = None
    update_frequency: int = 60  # minutes
    sentiment_window: int = 24  # hours
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.sentiment_methods is None:
            self.sentiment_methods = ['textblob', 'vader', 'custom', 'ml']
        if self.news_sources is None:
            self.news_sources = ['economic_times', 'moneycontrol', 'business_standard']
        if self.social_sources is None:
            self.social_sources = ['twitter', 'reddit']

class SentimentAnalyzer:
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.vader_analyzer = None
        self.ml_model = None
        self.vectorizer = None
        self.sentiment_cache = {}
        self.model_path = "models/sentiment_models"
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self._initialize_analyzers()
        
        # Market-specific keywords
        self.bullish_keywords = [
            'bull', 'bullish', 'rally', 'surge', 'gain', 'rise', 'up', 'positive',
            'growth', 'profit', 'earnings', 'beat', 'outperform', 'strong',
            'buy', 'upgrade', 'target', 'momentum', 'breakout', 'support'
        ]
        
        self.bearish_keywords = [
            'bear', 'bearish', 'fall', 'drop', 'decline', 'down', 'negative',
            'loss', 'miss', 'underperform', 'weak', 'sell', 'downgrade',
            'resistance', 'breakdown', 'crash', 'correction', 'volatility'
        ]
        
        self.neutral_keywords = [
            'hold', 'neutral', 'sideways', 'range', 'consolidation', 'stable',
            'unchanged', 'flat', 'mixed', 'cautious', 'wait', 'watch'
        ]
        
        # Indian market specific terms
        self.indian_market_terms = {
            'nifty': 1.0, 'sensex': 1.0, 'bse': 0.8, 'nse': 0.8,
            'fii': 0.9, 'dii': 0.9, 'sebi': 0.7, 'rbi': 0.9,
            'rupee': 0.6, 'inflation': 0.8, 'gdp': 0.9, 'budget': 0.8,
            'monsoon': 0.6, 'election': 0.7, 'policy': 0.7
        }
    
    def _initialize_analyzers(self):
        """Initialize sentiment analysis tools"""
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize ML model components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.ml_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            if not TEXTBLOB_AVAILABLE:
                return {'polarity': 0.0, 'subjectivity': 0.5, 'confidence': 0.0}
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to sentiment score
            if polarity > 0.1:
                sentiment = 'positive'
                score = polarity
            elif polarity < -0.1:
                sentiment = 'negative'
                score = abs(polarity)
            else:
                sentiment = 'neutral'
                score = 1 - abs(polarity)
            
            confidence = min(1.0, abs(polarity) + 0.3)
            
            return {
                'sentiment': sentiment,
                'score': score,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"TextBlob sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        try:
            if not VADER_AVAILABLE or not self.vader_analyzer:
                return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
            
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Classify sentiment
            if compound >= 0.05:
                sentiment = 'positive'
                score = compound
            elif compound <= -0.05:
                sentiment = 'negative'
                score = abs(compound)
            else:
                sentiment = 'neutral'
                score = 1 - abs(compound)
            
            confidence = min(1.0, abs(compound) + 0.2)
            
            return {
                'sentiment': sentiment,
                'score': score,
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"VADER sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def custom_sentiment(self, text: str) -> Dict[str, float]:
        """Custom rule-based sentiment analysis for financial text"""
        try:
            text_lower = text.lower()
            words = text_lower.split()
            
            bullish_count = sum(1 for word in words if any(keyword in word for keyword in self.bullish_keywords))
            bearish_count = sum(1 for word in words if any(keyword in word for keyword in self.bearish_keywords))
            neutral_count = sum(1 for word in words if any(keyword in word for keyword in self.neutral_keywords))
            
            # Weight by Indian market terms
            market_weight = 1.0
            for term, weight in self.indian_market_terms.items():
                if term in text_lower:
                    market_weight = max(market_weight, weight)
            
            # Calculate sentiment score
            total_sentiment_words = bullish_count + bearish_count + neutral_count
            
            if total_sentiment_words == 0:
                return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.1}
            
            bullish_ratio = bullish_count / total_sentiment_words
            bearish_ratio = bearish_count / total_sentiment_words
            neutral_ratio = neutral_count / total_sentiment_words
            
            # Apply market weight
            weighted_bullish = bullish_ratio * market_weight
            weighted_bearish = bearish_ratio * market_weight
            
            # Determine sentiment
            if weighted_bullish > weighted_bearish and weighted_bullish > neutral_ratio:
                sentiment = 'positive'
                score = weighted_bullish
            elif weighted_bearish > weighted_bullish and weighted_bearish > neutral_ratio:
                sentiment = 'negative'
                score = weighted_bearish
            else:
                sentiment = 'neutral'
                score = neutral_ratio
            
            confidence = min(1.0, (total_sentiment_words / len(words)) * market_weight)
            
            return {
                'sentiment': sentiment,
                'score': score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'market_weight': market_weight,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Custom sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def ml_sentiment(self, text: str) -> Dict[str, float]:
        """Machine learning based sentiment analysis"""
        try:
            if self.ml_model is None or self.vectorizer is None:
                return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
            
            # Preprocess and vectorize
            processed_text = self.preprocess_text(text)
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict sentiment
            prediction = self.ml_model.predict(text_vector)[0]
            probabilities = self.ml_model.predict_proba(text_vector)[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            # Map prediction to sentiment
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')
            
            return {
                'sentiment': sentiment,
                'score': confidence,
                'probabilities': {
                    'negative': probabilities[0],
                    'neutral': probabilities[1],
                    'positive': probabilities[2]
                },
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"ML sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def analyze_text(self, text: str, methods: List[str] = None) -> Dict:
        """Analyze sentiment using multiple methods"""
        methods = methods or self.config.sentiment_methods
        results = {}
        
        # Apply each sentiment analysis method
        for method in methods:
            if method == 'textblob':
                results['textblob'] = self.textblob_sentiment(text)
            elif method == 'vader':
                results['vader'] = self.vader_sentiment(text)
            elif method == 'custom':
                results['custom'] = self.custom_sentiment(text)
            elif method == 'ml':
                results['ml'] = self.ml_sentiment(text)
        
        # Calculate ensemble sentiment
        ensemble_result = self._calculate_ensemble_sentiment(results)
        results['ensemble'] = ensemble_result
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'methods': results,
            'final_sentiment': ensemble_result['sentiment'],
            'confidence': ensemble_result['confidence'],
            'analysis_time': datetime.now().isoformat()
        }
    
    def _calculate_ensemble_sentiment(self, results: Dict) -> Dict:
        """Calculate ensemble sentiment from multiple methods"""
        try:
            sentiments = []
            scores = []
            confidences = []
            
            # Weight different methods
            method_weights = {
                'textblob': 0.2,
                'vader': 0.3,
                'custom': 0.3,
                'ml': 0.2
            }
            
            weighted_positive = 0
            weighted_negative = 0
            weighted_neutral = 0
            total_weight = 0
            
            for method, result in results.items():
                if method in method_weights and 'sentiment' in result:
                    weight = method_weights[method]
                    confidence = result.get('confidence', 0.5)
                    
                    # Weight by confidence
                    adjusted_weight = weight * confidence
                    
                    if result['sentiment'] == 'positive':
                        weighted_positive += adjusted_weight
                    elif result['sentiment'] == 'negative':
                        weighted_negative += adjusted_weight
                    else:
                        weighted_neutral += adjusted_weight
                    
                    total_weight += adjusted_weight
                    confidences.append(confidence)
            
            if total_weight == 0:
                return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
            
            # Normalize weights
            weighted_positive /= total_weight
            weighted_negative /= total_weight
            weighted_neutral /= total_weight
            
            # Determine final sentiment
            if weighted_positive > weighted_negative and weighted_positive > weighted_neutral:
                final_sentiment = 'positive'
                final_score = weighted_positive
            elif weighted_negative > weighted_positive and weighted_negative > weighted_neutral:
                final_sentiment = 'negative'
                final_score = weighted_negative
            else:
                final_sentiment = 'neutral'
                final_score = weighted_neutral
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'sentiment': final_sentiment,
                'score': final_score,
                'confidence': ensemble_confidence,
                'distribution': {
                    'positive': weighted_positive,
                    'negative': weighted_negative,
                    'neutral': weighted_neutral
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Ensemble calculation failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def analyze_news_headlines(self, headlines: List[str], symbol: str = None) -> Dict:
        """Analyze sentiment of news headlines"""
        try:
            if not headlines:
                return {'error': 'No headlines provided'}
            
            headline_sentiments = []
            
            for headline in headlines:
                sentiment_result = self.analyze_text(headline)
                headline_sentiments.append({
                    'headline': headline,
                    'sentiment': sentiment_result['final_sentiment'],
                    'confidence': sentiment_result['confidence']
                })
            
            # Aggregate sentiments
            positive_count = sum(1 for h in headline_sentiments if h['sentiment'] == 'positive')
            negative_count = sum(1 for h in headline_sentiments if h['sentiment'] == 'negative')
            neutral_count = sum(1 for h in headline_sentiments if h['sentiment'] == 'neutral')
            
            total_headlines = len(headlines)
            
            # Calculate overall sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                overall_sentiment = 'positive'
                sentiment_strength = positive_count / total_headlines
            elif negative_count > positive_count and negative_count > neutral_count:
                overall_sentiment = 'negative'
                sentiment_strength = negative_count / total_headlines
            else:
                overall_sentiment = 'neutral'
                sentiment_strength = neutral_count / total_headlines
            
            # Calculate average confidence
            avg_confidence = np.mean([h['confidence'] for h in headline_sentiments])
            
            return {
                'symbol': symbol,
                'total_headlines': total_headlines,
                'overall_sentiment': overall_sentiment,
                'sentiment_strength': sentiment_strength,
                'confidence': avg_confidence,
                'distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count,
                    'positive_pct': (positive_count / total_headlines) * 100,
                    'negative_pct': (negative_count / total_headlines) * 100,
                    'neutral_pct': (neutral_count / total_headlines) * 100
                },
                'individual_sentiments': headline_sentiments,
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"News sentiment analysis failed: {e}")
            return {'error': str(e)}
    
    def get_market_sentiment_score(self, symbol: str = None) -> Dict:
        """Get overall market sentiment score"""
        try:
            # This would typically fetch from various sources
            # For now, we'll simulate market sentiment
            
            # Simulate news sentiment
            sample_headlines = [
                f"Nifty 50 shows strong momentum amid positive global cues",
                f"FII inflows boost market sentiment in Indian equities",
                f"Banking sector leads rally with strong quarterly results",
                f"Market volatility expected due to upcoming policy decisions",
                f"Retail investors show increased participation in equity markets"
            ]
            
            news_sentiment = self.analyze_news_headlines(sample_headlines, symbol)
            
            # Simulate social media sentiment (would be fetched from APIs)
            social_sentiment = {
                'platform': 'twitter',
                'sentiment': 'positive',
                'confidence': 0.72,
                'volume': 1250,
                'trending_topics': ['#Nifty50', '#BullRun', '#IndianMarkets']
            }
            
            # Calculate composite sentiment
            news_weight = 0.6
            social_weight = 0.4
            
            # Convert sentiments to numerical scores
            sentiment_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
            
            news_score = sentiment_to_score.get(news_sentiment.get('overall_sentiment', 'neutral'), 0)
            social_score = sentiment_to_score.get(social_sentiment.get('sentiment', 'neutral'), 0)
            
            composite_score = (news_score * news_weight + social_score * social_weight)
            
            # Determine overall sentiment
            if composite_score > 0.2:
                overall_sentiment = 'positive'
            elif composite_score < -0.2:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment,
                'composite_score': composite_score,
                'confidence': min(news_sentiment.get('confidence', 0.5), social_sentiment.get('confidence', 0.5)),
                'components': {
                    'news': news_sentiment,
                    'social_media': social_sentiment
                },
                'sentiment_trend': self._calculate_sentiment_trend(),
                'market_fear_greed_index': self._calculate_fear_greed_index(composite_score),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Market sentiment calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_sentiment_trend(self) -> Dict:
        """Calculate sentiment trend over time"""
        # Simulate historical sentiment data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        sentiments = np.random.choice(['positive', 'neutral', 'negative'], size=7, p=[0.4, 0.4, 0.2])
        
        trend_data = []
        for date, sentiment in zip(dates, sentiments):
            trend_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'sentiment': sentiment
            })
        
        # Calculate trend direction
        recent_positive = sum(1 for item in trend_data[-3:] if item['sentiment'] == 'positive')
        trend_direction = 'improving' if recent_positive >= 2 else 'declining' if recent_positive == 0 else 'stable'
        
        return {
            'trend_direction': trend_direction,
            'historical_data': trend_data
        }
    
    def _calculate_fear_greed_index(self, sentiment_score: float) -> Dict:
        """Calculate Fear & Greed Index based on sentiment"""
        # Normalize sentiment score to 0-100 scale
        normalized_score = ((sentiment_score + 1) / 2) * 100
        
        # Classify fear/greed level
        if normalized_score >= 75:
            level = 'Extreme Greed'
        elif normalized_score >= 55:
            level = 'Greed'
        elif normalized_score >= 45:
            level = 'Neutral'
        elif normalized_score >= 25:
            level = 'Fear'
        else:
            level = 'Extreme Fear'
        
        return {
            'index_value': round(normalized_score, 1),
            'level': level,
            'description': self._get_fear_greed_description(level)
        }
    
    def _get_fear_greed_description(self, level: str) -> str:
        """Get description for fear/greed level"""
        descriptions = {
            'Extreme Greed': 'Market sentiment is extremely bullish. Consider taking profits.',
            'Greed': 'Market sentiment is bullish. Monitor for potential reversals.',
            'Neutral': 'Market sentiment is balanced. Look for directional catalysts.',
            'Fear': 'Market sentiment is bearish. Look for buying opportunities.',
            'Extreme Fear': 'Market sentiment is extremely bearish. Consider contrarian positions.'
        }
        return descriptions.get(level, 'Market sentiment analysis unavailable.')
    
    def train_ml_model(self, training_data: List[Tuple[str, str]]) -> Dict:
        """Train ML model on labeled sentiment data"""
        try:
            if not training_data:
                return {'error': 'No training data provided'}
            
            # Prepare data
            texts = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Vectorize
            X = self.vectorizer.fit_transform(processed_texts)
            
            # Encode labels
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            y = [label_map.get(label, 1) for label in labels]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            self.save_ml_model()
            
            return {
                'accuracy': accuracy,
                'training_samples': len(training_data),
                'model_saved': True,
                'training_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            return {'error': str(e)}
    
    def save_ml_model(self):
        """Save trained ML model"""
        try:
            model_file = os.path.join(self.model_path, 'sentiment_ml_model.pkl')
            vectorizer_file = os.path.join(self.model_path, 'sentiment_vectorizer.pkl')
            
            joblib.dump(self.ml_model, model_file)
            joblib.dump(self.vectorizer, vectorizer_file)
            
            self.logger.info("ML sentiment model saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save ML model: {e}")
    
    def load_ml_model(self) -> bool:
        """Load trained ML model"""
        try:
            model_file = os.path.join(self.model_path, 'sentiment_ml_model.pkl')
            vectorizer_file = os.path.join(self.model_path, 'sentiment_vectorizer.pkl')
            
            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                self.ml_model = joblib.load(model_file)
                self.vectorizer = joblib.load(vectorizer_file)
                
                self.logger.info("ML sentiment model loaded")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create sentiment analyzer
    config = SentimentConfig(
        sentiment_methods=['textblob', 'vader', 'custom'],
        confidence_threshold=0.6
    )
    
    analyzer = SentimentAnalyzer(config)
    
    # Test with sample financial news
    sample_texts = [
        "Nifty 50 surges to new all-time high amid strong FII inflows and positive global cues",
        "Banking stocks decline on concerns over rising NPAs and regulatory pressures",
        "Market remains sideways as investors await RBI policy decision and inflation data",
        "IT sector outperforms with strong Q3 earnings and robust demand outlook",
        "Volatility expected to remain high due to upcoming election results"
    ]
    
    print("Sentiment Analysis Results:")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"\n{i}. Text: {result['text']}")
        print(f"   Sentiment: {result['final_sentiment']}")
        print(f"   Confidence: {result['confidence']:.2f}")
    
    # Test news headlines analysis
    print("\n\nNews Headlines Analysis:")
    print("=" * 50)
    
    headlines_result = analyzer.analyze_news_headlines(sample_texts, 'NIFTY')
    print(f"Overall Sentiment: {headlines_result['overall_sentiment']}")
    print(f"Confidence: {headlines_result['confidence']:.2f}")
    print(f"Distribution: {headlines_result['distribution']}")
    
    # Test market sentiment
    print("\n\nMarket Sentiment Score:")
    print("=" * 50)
    
    market_sentiment = analyzer.get_market_sentiment_score('NIFTY')
    print(f"Overall Sentiment: {market_sentiment['overall_sentiment']}")
    print(f"Composite Score: {market_sentiment['composite_score']:.2f}")
    print(f"Fear & Greed Index: {market_sentiment['market_fear_greed_index']}")