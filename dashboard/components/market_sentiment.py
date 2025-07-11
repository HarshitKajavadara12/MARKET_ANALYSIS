import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')
import requests
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class MarketSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Fear & Greed Index components weights
        self.fear_greed_weights = {
            'market_momentum': 0.25,
            'stock_price_strength': 0.20,
            'stock_price_breadth': 0.15,
            'put_call_ratio': 0.15,
            'market_volatility': 0.10,
            'safe_haven_demand': 0.10,
            'junk_bond_demand': 0.05
        }
        
        # Sentiment keywords for Indian markets
        self.bullish_keywords = [
            'rally', 'surge', 'gain', 'rise', 'bull', 'positive', 'growth', 'strong',
            'outperform', 'upgrade', 'buy', 'accumulate', 'breakout', 'momentum',
            'recovery', 'expansion', 'optimistic', 'confident', 'robust', 'healthy'
        ]
        
        self.bearish_keywords = [
            'fall', 'drop', 'decline', 'bear', 'negative', 'weak', 'crash', 'correction',
            'sell', 'downgrade', 'breakdown', 'pressure', 'concern', 'worry',
            'recession', 'slowdown', 'pessimistic', 'cautious', 'volatile', 'risk'
        ]
        
        # Indian market specific terms
        self.indian_market_terms = {
            'positive': ['nifty surge', 'sensex rally', 'fii inflow', 'dii buying', 'rupee strength'],
            'negative': ['nifty fall', 'sensex crash', 'fii outflow', 'rupee weakness', 'inflation concern']
        }
        
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'bullish_color': '#00ff88',
                'bearish_color': '#ff4444',
                'neutral_color': '#ffa500',
                'fear_color': '#ff6b6b',
                'greed_color': '#51cf66'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'bullish_color': '#26a69a',
                'bearish_color': '#ef5350',
                'neutral_color': '#ff9800',
                'fear_color': '#f44336',
                'greed_color': '#4caf50'
            }
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using multiple methods"""
        try:
            # TextBlob analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # Custom keyword-based analysis
            custom_score = self._keyword_sentiment_analysis(text)
            
            # Ensemble score (weighted average)
            ensemble_score = (
                textblob_polarity * 0.4 +
                vader_scores['compound'] * 0.4 +
                custom_score * 0.2
            )
            
            return {
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'vader_compound': vader_scores['compound'],
                'custom_score': custom_score,
                'ensemble_score': ensemble_score,
                'sentiment_label': self._get_sentiment_label(ensemble_score)
            }
        except Exception as e:
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_compound': 0.0,
                'custom_score': 0.0,
                'ensemble_score': 0.0,
                'sentiment_label': 'Neutral'
            }
    
    def _keyword_sentiment_analysis(self, text: str) -> float:
        """Custom keyword-based sentiment analysis for Indian markets"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        # Check Indian market specific terms
        for term in self.indian_market_terms['positive']:
            if term in text_lower:
                bullish_count += 2  # Higher weight for market-specific terms
        
        for term in self.indian_market_terms['negative']:
            if term in text_lower:
                bearish_count += 2
        
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total_keywords
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'Bullish'
        elif score < -0.1:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def calculate_fear_greed_index(self, market_data: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """Calculate Fear & Greed Index for Indian markets"""
        try:
            # Normalize each component to 0-100 scale
            components = {}
            
            # Market Momentum (based on price change)
            momentum = market_data.get('momentum', 0)
            components['market_momentum'] = max(0, min(100, 50 + momentum * 50))
            
            # Stock Price Strength (RSI-like)
            strength = market_data.get('strength', 50)
            components['stock_price_strength'] = max(0, min(100, strength))
            
            # Stock Price Breadth (advance/decline ratio)
            breadth = market_data.get('breadth', 0.5)
            components['stock_price_breadth'] = max(0, min(100, breadth * 100))
            
            # Put/Call Ratio (inverted - high put/call = fear)
            put_call = market_data.get('put_call_ratio', 1.0)
            components['put_call_ratio'] = max(0, min(100, 100 - (put_call - 0.5) * 100))
            
            # Market Volatility (VIX-like, inverted)
            volatility = market_data.get('volatility', 20)
            components['market_volatility'] = max(0, min(100, 100 - (volatility - 10) * 2))
            
            # Safe Haven Demand (bond yields, inverted)
            safe_haven = market_data.get('safe_haven', 0.5)
            components['safe_haven_demand'] = max(0, min(100, (1 - safe_haven) * 100))
            
            # Junk Bond Demand
            junk_bond = market_data.get('junk_bond', 0.5)
            components['junk_bond_demand'] = max(0, min(100, junk_bond * 100))
            
            # Calculate weighted Fear & Greed Index
            fear_greed_score = sum(
                components[component] * self.fear_greed_weights[component]
                for component in components
            )
            
            # Determine sentiment label
            if fear_greed_score >= 75:
                sentiment_label = 'Extreme Greed'
            elif fear_greed_score >= 55:
                sentiment_label = 'Greed'
            elif fear_greed_score >= 45:
                sentiment_label = 'Neutral'
            elif fear_greed_score >= 25:
                sentiment_label = 'Fear'
            else:
                sentiment_label = 'Extreme Fear'
            
            return {
                'score': fear_greed_score,
                'label': sentiment_label,
                'components': components
            }
            
        except Exception as e:
            return {
                'score': 50.0,
                'label': 'Neutral',
                'components': {k: 50.0 for k in self.fear_greed_weights.keys()}
            }
    
    def generate_sample_news(self) -> List[Dict[str, str]]:
        """Generate sample news headlines for demonstration"""
        sample_news = [
            {
                'headline': 'NIFTY 50 surges 2.5% on strong FII inflows and positive global cues',
                'source': 'Economic Times',
                'timestamp': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'headline': 'Banking stocks rally as RBI maintains accommodative stance',
                'source': 'Business Standard',
                'timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'headline': 'IT sector faces headwinds amid global recession fears',
                'source': 'Mint',
                'timestamp': (datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'headline': 'Rupee strengthens against dollar on improved trade deficit',
                'source': 'Financial Express',
                'timestamp': (datetime.now() - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'headline': 'Auto stocks decline on rising input costs and weak demand',
                'source': 'Moneycontrol',
                'timestamp': (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M')
            },
            {
                'headline': 'Pharma sector outperforms on strong export growth prospects',
                'source': 'Livemint',
                'timestamp': (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')
            }
        ]
        
        return sample_news
    
    def analyze_news_sentiment(self, news_list: List[Dict[str, str]]) -> Dict[str, Union[float, List]]:
        """Analyze sentiment of news headlines"""
        try:
            sentiments = []
            detailed_analysis = []
            
            for news in news_list:
                headline = news.get('headline', '')
                sentiment_data = self.analyze_text_sentiment(headline)
                
                sentiments.append(sentiment_data['ensemble_score'])
                detailed_analysis.append({
                    'headline': headline,
                    'source': news.get('source', 'Unknown'),
                    'timestamp': news.get('timestamp', ''),
                    'sentiment_score': sentiment_data['ensemble_score'],
                    'sentiment_label': sentiment_data['sentiment_label']
                })
            
            # Calculate overall sentiment metrics
            if sentiments:
                overall_sentiment = np.mean(sentiments)
                sentiment_std = np.std(sentiments)
                bullish_count = sum(1 for s in sentiments if s > 0.1)
                bearish_count = sum(1 for s in sentiments if s < -0.1)
                neutral_count = len(sentiments) - bullish_count - bearish_count
            else:
                overall_sentiment = 0.0
                sentiment_std = 0.0
                bullish_count = bearish_count = neutral_count = 0
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_std': sentiment_std,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'total_news': len(news_list),
                'detailed_analysis': detailed_analysis,
                'sentiment_distribution': sentiments
            }
            
        except Exception as e:
            return {
                'overall_sentiment': 0.0,
                'sentiment_std': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_news': 0,
                'detailed_analysis': [],
                'sentiment_distribution': []
            }
    
    def create_sentiment_gauge(self, sentiment_score: float, title: str = "Market Sentiment", theme: str = 'dark') -> go.Figure:
        """Create a sentiment gauge chart"""
        try:
            colors = self.chart_themes[theme]
            
            # Normalize sentiment score to 0-100 scale
            gauge_value = (sentiment_score + 1) * 50
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gauge_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title, 'font': {'color': colors['text_color']}},
                delta={'reference': 50, 'increasing': {'color': colors['bullish_color']}, 'decreasing': {'color': colors['bearish_color']}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': colors['text_color']},
                    'bar': {'color': colors['neutral_color']},
                    'steps': [
                        {'range': [0, 25], 'color': colors['bearish_color'], 'name': 'Extreme Fear'},
                        {'range': [25, 45], 'color': '#ff9999', 'name': 'Fear'},
                        {'range': [45, 55], 'color': colors['neutral_color'], 'name': 'Neutral'},
                        {'range': [55, 75], 'color': '#99ff99', 'name': 'Greed'},
                        {'range': [75, 100], 'color': colors['bullish_color'], 'name': 'Extreme Greed'}
                    ],
                    'threshold': {
                        'line': {'color': colors['text_color'], 'width': 4},
                        'thickness': 0.75,
                        'value': gauge_value
                    }
                }
            ))
            
            fig.update_layout(
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=300,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def create_sentiment_timeline(self, sentiment_data: List[Dict], theme: str = 'dark') -> go.Figure:
        """Create sentiment timeline chart"""
        try:
            colors = self.chart_themes[theme]
            
            if not sentiment_data:
                return go.Figure()
            
            df = pd.DataFrame(sentiment_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Create color mapping based on sentiment
            color_map = []
            for score in df['sentiment_score']:
                if score > 0.1:
                    color_map.append(colors['bullish_color'])
                elif score < -0.1:
                    color_map.append(colors['bearish_color'])
                else:
                    color_map.append(colors['neutral_color'])
            
            fig = go.Figure()
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sentiment_score'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color=colors['text_color'], width=2),
                marker=dict(color=color_map, size=8),
                hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<br>Time: %{x}<extra></extra>',
                text=df['headline'].str[:50] + '...'
            ))
            
            # Add reference lines
            fig.add_hline(y=0.1, line_dash="dash", line_color=colors['bullish_color'], annotation_text="Bullish Threshold")
            fig.add_hline(y=-0.1, line_dash="dash", line_color=colors['bearish_color'], annotation_text="Bearish Threshold")
            fig.add_hline(y=0, line_dash="dot", line_color=colors['neutral_color'], annotation_text="Neutral")
            
            fig.update_layout(
                title="News Sentiment Timeline",
                xaxis_title="Time",
                yaxis_title="Sentiment Score",
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=400,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def create_sentiment_distribution(self, sentiment_scores: List[float], theme: str = 'dark') -> go.Figure:
        """Create sentiment distribution histogram"""
        try:
            colors = self.chart_themes[theme]
            
            if not sentiment_scores:
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=sentiment_scores,
                nbinsx=20,
                name='Sentiment Distribution',
                marker_color=colors['neutral_color'],
                opacity=0.7
            ))
            
            # Add mean line
            mean_sentiment = np.mean(sentiment_scores)
            fig.add_vline(
                x=mean_sentiment,
                line_dash="dash",
                line_color=colors['text_color'],
                annotation_text=f"Mean: {mean_sentiment:.3f}"
            )
            
            fig.update_layout(
                title="Sentiment Score Distribution",
                xaxis_title="Sentiment Score",
                yaxis_title="Frequency",
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=300,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def generate_sample_market_data(self) -> Dict[str, float]:
        """Generate sample market data for Fear & Greed calculation"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        return {
            'momentum': np.random.normal(0.1, 0.3),  # Slight positive bias
            'strength': np.random.normal(55, 15),    # RSI-like
            'breadth': np.random.normal(0.6, 0.2),   # Advance/decline ratio
            'put_call_ratio': np.random.normal(0.8, 0.3),  # Put/call ratio
            'volatility': np.random.normal(18, 5),   # VIX-like
            'safe_haven': np.random.normal(0.4, 0.2),  # Bond demand
            'junk_bond': np.random.normal(0.6, 0.2)    # Junk bond demand
        }

# Streamlit interface functions
def render_market_sentiment_panel():
    """
    Render the market sentiment panel in Streamlit
    """
    st.subheader("📰 Market Sentiment Analysis")
    
    # Initialize sentiment analyzer
    sentiment_analyzer = MarketSentimentAnalyzer()
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Chart Theme", ['dark', 'light'], key='sentiment_theme')
        show_news_analysis = st.checkbox("Show News Analysis", value=True)
    
    with col2:
        show_fear_greed = st.checkbox("Show Fear & Greed Index", value=True)
        auto_refresh = st.checkbox("Auto Refresh (Demo)", value=False)
    
    return {
        'sentiment_analyzer': sentiment_analyzer,
        'theme': theme,
        'show_news_analysis': show_news_analysis,
        'show_fear_greed': show_fear_greed,
        'auto_refresh': auto_refresh
    }

def display_market_sentiment(config: Dict):
    """
    Display market sentiment analysis dashboard
    """
    try:
        sentiment_analyzer = config['sentiment_analyzer']
        theme = config['theme']
        
        # Generate sample data
        sample_news = sentiment_analyzer.generate_sample_news()
        sample_market_data = sentiment_analyzer.generate_sample_market_data()
        
        # Analyze news sentiment
        if config['show_news_analysis']:
            st.subheader("📊 News Sentiment Analysis")
            
            news_analysis = sentiment_analyzer.analyze_news_sentiment(sample_news)
            
            # Display sentiment metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_label = sentiment_analyzer._get_sentiment_label(news_analysis['overall_sentiment'])
                sentiment_color = {
                    'Bullish': '🟢',
                    'Bearish': '🔴',
                    'Neutral': '🟡'
                }.get(sentiment_label, '⚪')
                
                st.metric(
                    f"{sentiment_color} Overall Sentiment",
                    sentiment_label,
                    delta=f"{news_analysis['overall_sentiment']:.3f}"
                )
            
            with col2:
                st.metric(
                    "📈 Bullish News",
                    news_analysis['bullish_count'],
                    delta=f"{news_analysis['bullish_count']/news_analysis['total_news']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "📉 Bearish News",
                    news_analysis['bearish_count'],
                    delta=f"{news_analysis['bearish_count']/news_analysis['total_news']*100:.1f}%"
                )
            
            with col4:
                st.metric(
                    "📊 Neutral News",
                    news_analysis['neutral_count'],
                    delta=f"{news_analysis['neutral_count']/news_analysis['total_news']*100:.1f}%"
                )
            
            # Display sentiment timeline
            timeline_fig = sentiment_analyzer.create_sentiment_timeline(
                news_analysis['detailed_analysis'], theme
            )
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Display sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                dist_fig = sentiment_analyzer.create_sentiment_distribution(
                    news_analysis['sentiment_distribution'], theme
                )
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                # Display recent news with sentiment
                st.subheader("Recent News Headlines")
                
                for news_item in news_analysis['detailed_analysis'][:5]:
                    sentiment_emoji = {
                        'Bullish': '🟢',
                        'Bearish': '🔴',
                        'Neutral': '🟡'
                    }.get(news_item['sentiment_label'], '⚪')
                    
                    with st.expander(f"{sentiment_emoji} {news_item['headline'][:60]}..."):
                        st.write(f"**Source:** {news_item['source']}")
                        st.write(f"**Time:** {news_item['timestamp']}")
                        st.write(f"**Sentiment:** {news_item['sentiment_label']} ({news_item['sentiment_score']:.3f})")
                        st.write(f"**Full Headline:** {news_item['headline']}")
        
        # Display Fear & Greed Index
        if config['show_fear_greed']:
            st.subheader("😨😍 Fear & Greed Index")
            
            fear_greed_data = sentiment_analyzer.calculate_fear_greed_index(sample_market_data)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display Fear & Greed gauge
                gauge_fig = sentiment_analyzer.create_sentiment_gauge(
                    (fear_greed_data['score'] - 50) / 50,  # Convert to -1 to 1 scale
                    "Fear & Greed Index",
                    theme
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Display current reading
                st.metric(
                    "Current Reading",
                    fear_greed_data['label'],
                    delta=f"{fear_greed_data['score']:.1f}/100"
                )
            
            with col2:
                # Display component breakdown
                st.subheader("Index Components")
                
                components_df = pd.DataFrame([
                    {
                        'Component': component.replace('_', ' ').title(),
                        'Value': f"{value:.1f}",
                        'Weight': f"{sentiment_analyzer.fear_greed_weights[component]*100:.1f}%"
                    }
                    for component, value in fear_greed_data['components'].items()
                ])
                
                st.dataframe(components_df, use_container_width=True)
                
                # Component visualization
                components_fig = go.Figure()
                
                components_fig.add_trace(go.Bar(
                    x=list(fear_greed_data['components'].keys()),
                    y=list(fear_greed_data['components'].values()),
                    marker_color=[
                        sentiment_analyzer.chart_themes[theme]['bullish_color'] if v > 50 
                        else sentiment_analyzer.chart_themes[theme]['bearish_color']
                        for v in fear_greed_data['components'].values()
                    ],
                    text=[f"{v:.1f}" for v in fear_greed_data['components'].values()],
                    textposition='auto'
                ))
                
                components_fig.update_layout(
                    title="Component Scores",
                    xaxis_title="Components",
                    yaxis_title="Score (0-100)",
                    template='plotly_dark' if theme == 'dark' else 'plotly_white',
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(components_fig, use_container_width=True)
        
        # Market sentiment insights
        st.subheader("🔍 Sentiment Insights")
        
        insights = []
        
        if config['show_news_analysis']:
            overall_sentiment = news_analysis['overall_sentiment']
            if overall_sentiment > 0.2:
                insights.append("📈 **Strong positive sentiment** detected in recent news coverage")
            elif overall_sentiment < -0.2:
                insights.append("📉 **Strong negative sentiment** detected in recent news coverage")
            else:
                insights.append("📊 **Neutral sentiment** prevails in current news cycle")
        
        if config['show_fear_greed']:
            fg_score = fear_greed_data['score']
            if fg_score > 75:
                insights.append("😍 **Extreme Greed** - Market may be overheated, consider caution")
            elif fg_score > 55:
                insights.append("😊 **Greed** - Positive market sentiment, but watch for reversals")
            elif fg_score < 25:
                insights.append("😨 **Extreme Fear** - Potential buying opportunity for contrarians")
            elif fg_score < 45:
                insights.append("😟 **Fear** - Market uncertainty, consider defensive strategies")
            else:
                insights.append("😐 **Neutral** - Balanced market sentiment")
        
        for insight in insights:
            st.info(insight)
        
        # Auto-refresh functionality (demo)
        if config['auto_refresh']:
            st.info("🔄 Auto-refresh enabled (demo mode) - Data updates every 30 seconds")
            # In a real implementation, you would use st.rerun() with a timer
        
    except Exception as e:
        st.error(f"Error displaying market sentiment: {e}")

# Example usage
if __name__ == "__main__":
    st.title("Market Sentiment Analysis Demo")
    
    # Render sentiment panel
    config = render_market_sentiment_panel()
    display_market_sentiment(config)