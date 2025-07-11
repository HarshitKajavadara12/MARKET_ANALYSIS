#!/usr/bin/env python3
"""
News Sentiment Simulator - Indian Market Research Platform
Simulates news events and their impact on market sentiment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
from enum import Enum
import json

class NewsCategory(Enum):
    """News category types"""
    EARNINGS = "earnings"
    POLICY = "policy"
    CORPORATE = "corporate"
    ECONOMIC = "economic"
    GLOBAL = "global"
    SECTOR = "sector"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"
    MARKET = "market"
    TECHNOLOGY = "technology"

class SentimentType(Enum):
    """Sentiment types"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class ImpactLevel(Enum):
    """Impact level on markets"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass
class NewsEvent:
    """News event data structure"""
    timestamp: datetime
    headline: str
    content: str
    category: NewsCategory
    sentiment: SentimentType
    impact_level: ImpactLevel
    affected_symbols: List[str]
    affected_sectors: List[str]
    sentiment_score: float  # -1 to 1
    credibility_score: float  # 0 to 1
    source: str
    tags: List[str]
    market_impact_duration: int  # minutes
    
class NewsSentimentSimulator:
    """
    Simulates realistic news events and their market impact
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            
        self.news_templates = self._initialize_news_templates()
        self.sentiment_impact_matrix = self._initialize_sentiment_impact_matrix()
        self.sector_keywords = self._initialize_sector_keywords()
        self.company_data = self._initialize_company_data()
        self.news_sources = self._initialize_news_sources()
        
    def _initialize_news_templates(self) -> Dict:
        """Initialize news headline and content templates"""
        return {
            NewsCategory.EARNINGS: {
                SentimentType.VERY_POSITIVE: [
                    "{company} reports exceptional Q{quarter} results, beats estimates by {beat_pct}%",
                    "{company} delivers stellar quarterly performance with {growth_pct}% revenue growth",
                    "{company} announces record-breaking profits for Q{quarter}"
                ],
                SentimentType.POSITIVE: [
                    "{company} Q{quarter} earnings beat analyst expectations",
                    "{company} shows strong quarterly growth in key segments",
                    "{company} reports healthy profit margins in latest quarter"
                ],
                SentimentType.NEUTRAL: [
                    "{company} announces Q{quarter} results in line with estimates",
                    "{company} reports mixed quarterly performance",
                    "{company} Q{quarter} earnings meet market expectations"
                ],
                SentimentType.NEGATIVE: [
                    "{company} Q{quarter} earnings miss analyst estimates",
                    "{company} reports declining revenues in latest quarter",
                    "{company} faces margin pressure in Q{quarter} results"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "{company} reports massive Q{quarter} losses, cuts guidance",
                    "{company} announces significant writedowns in quarterly results",
                    "{company} Q{quarter} performance well below expectations"
                ]
            },
            NewsCategory.POLICY: {
                SentimentType.VERY_POSITIVE: [
                    "RBI announces major policy reforms to boost economic growth",
                    "Government unveils comprehensive economic stimulus package",
                    "Finance Ministry announces significant tax reforms for businesses"
                ],
                SentimentType.POSITIVE: [
                    "RBI maintains accommodative monetary policy stance",
                    "Government announces infrastructure spending boost",
                    "New policy measures to support MSME sector announced"
                ],
                SentimentType.NEUTRAL: [
                    "RBI keeps repo rate unchanged in latest policy review",
                    "Government maintains current fiscal policy stance",
                    "Central bank provides economic outlook update"
                ],
                SentimentType.NEGATIVE: [
                    "RBI raises concerns over inflation in policy statement",
                    "Government announces new regulatory restrictions",
                    "Central bank signals potential policy tightening"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "RBI announces emergency rate hike to combat inflation",
                    "Government imposes strict new regulations on financial sector",
                    "Central bank warns of significant economic headwinds"
                ]
            },
            NewsCategory.CORPORATE: {
                SentimentType.VERY_POSITIVE: [
                    "{company} announces major acquisition worth ₹{amount} crores",
                    "{company} secures largest contract in company history",
                    "{company} announces revolutionary product launch"
                ],
                SentimentType.POSITIVE: [
                    "{company} expands operations with new facility launch",
                    "{company} announces strategic partnership with global leader",
                    "{company} receives major order from key client"
                ],
                SentimentType.NEUTRAL: [
                    "{company} announces management changes",
                    "{company} provides business update to shareholders",
                    "{company} completes routine corporate restructuring"
                ],
                SentimentType.NEGATIVE: [
                    "{company} faces regulatory investigation",
                    "{company} announces plant closure due to losses",
                    "{company} reports significant customer loss"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "{company} under investigation for financial irregularities",
                    "{company} announces massive layoffs amid restructuring",
                    "{company} faces bankruptcy proceedings"
                ]
            },
            NewsCategory.ECONOMIC: {
                SentimentType.VERY_POSITIVE: [
                    "India GDP growth accelerates to {growth_rate}% in latest quarter",
                    "Manufacturing PMI hits multi-year high of {pmi_value}",
                    "Export growth surges to {export_growth}% year-on-year"
                ],
                SentimentType.POSITIVE: [
                    "Economic indicators show steady improvement",
                    "Industrial production grows {ip_growth}% in latest month",
                    "Services sector shows robust expansion"
                ],
                SentimentType.NEUTRAL: [
                    "Economic growth remains stable at {growth_rate}%",
                    "Mixed signals from key economic indicators",
                    "Economy shows signs of gradual recovery"
                ],
                SentimentType.NEGATIVE: [
                    "Economic growth slows to {growth_rate}% amid headwinds",
                    "Manufacturing activity contracts for second month",
                    "Trade deficit widens to {deficit_amount} billion USD"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "Economy enters technical recession with {decline_rate}% contraction",
                    "Unemployment rate surges to {unemployment_rate}%",
                    "Currency hits record low amid economic crisis"
                ]
            },
            NewsCategory.GLOBAL: {
                SentimentType.VERY_POSITIVE: [
                    "Global markets rally on positive economic data",
                    "Major breakthrough in international trade negotiations",
                    "Global central banks coordinate stimulus measures"
                ],
                SentimentType.POSITIVE: [
                    "US markets close higher on strong earnings",
                    "European markets show steady gains",
                    "Asian markets advance on positive sentiment"
                ],
                SentimentType.NEUTRAL: [
                    "Global markets trade mixed amid uncertainty",
                    "International markets show sideways movement",
                    "Mixed global cues impact market sentiment"
                ],
                SentimentType.NEGATIVE: [
                    "Global markets decline on recession fears",
                    "International trade tensions escalate",
                    "Global supply chain disruptions continue"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "Global financial crisis spreads across markets",
                    "International markets crash on pandemic fears",
                    "Global recession fears trigger massive selloff"
                ]
            },
            NewsCategory.SECTOR: {
                SentimentType.VERY_POSITIVE: [
                    "{sector} sector receives major government support package",
                    "{sector} companies report exceptional demand growth",
                    "Revolutionary technology breakthrough in {sector} sector"
                ],
                SentimentType.POSITIVE: [
                    "{sector} sector shows strong quarterly performance",
                    "Positive outlook for {sector} industry growth",
                    "{sector} companies benefit from policy changes"
                ],
                SentimentType.NEUTRAL: [
                    "{sector} sector maintains steady performance",
                    "Mixed results across {sector} companies",
                    "{sector} industry shows gradual improvement"
                ],
                SentimentType.NEGATIVE: [
                    "{sector} sector faces regulatory headwinds",
                    "Challenges mount for {sector} industry",
                    "{sector} companies report margin pressure"
                ],
                SentimentType.VERY_NEGATIVE: [
                    "{sector} sector hit by major crisis",
                    "Massive disruption in {sector} industry",
                    "{sector} companies face existential threat"
                ]
            }
        }
        
    def _initialize_sentiment_impact_matrix(self) -> Dict:
        """Initialize sentiment to market impact mapping"""
        return {
            SentimentType.VERY_POSITIVE: {
                'price_impact': (0.03, 0.08),  # 3-8% positive impact
                'volume_multiplier': (2.0, 4.0),
                'volatility_multiplier': (1.5, 2.5),
                'duration_minutes': (60, 240)
            },
            SentimentType.POSITIVE: {
                'price_impact': (0.01, 0.03),  # 1-3% positive impact
                'volume_multiplier': (1.3, 2.0),
                'volatility_multiplier': (1.2, 1.8),
                'duration_minutes': (30, 120)
            },
            SentimentType.NEUTRAL: {
                'price_impact': (-0.005, 0.005),  # Minimal impact
                'volume_multiplier': (0.9, 1.1),
                'volatility_multiplier': (1.0, 1.2),
                'duration_minutes': (15, 60)
            },
            SentimentType.NEGATIVE: {
                'price_impact': (-0.03, -0.01),  # 1-3% negative impact
                'volume_multiplier': (1.5, 2.5),
                'volatility_multiplier': (1.3, 2.0),
                'duration_minutes': (45, 150)
            },
            SentimentType.VERY_NEGATIVE: {
                'price_impact': (-0.08, -0.03),  # 3-8% negative impact
                'volume_multiplier': (2.5, 5.0),
                'volatility_multiplier': (2.0, 3.5),
                'duration_minutes': (90, 300)
            }
        }
        
    def _initialize_sector_keywords(self) -> Dict:
        """Initialize sector-specific keywords"""
        return {
            'Banking': ['bank', 'lending', 'credit', 'deposits', 'NPA', 'loan', 'interest rate'],
            'IT': ['software', 'technology', 'digital', 'cloud', 'AI', 'automation', 'outsourcing'],
            'Pharma': ['drug', 'medicine', 'healthcare', 'clinical', 'FDA', 'patent', 'pharmaceutical'],
            'Auto': ['automobile', 'vehicle', 'car', 'truck', 'EV', 'electric vehicle', 'automotive'],
            'FMCG': ['consumer goods', 'retail', 'brand', 'distribution', 'rural demand'],
            'Energy': ['oil', 'gas', 'petroleum', 'refinery', 'energy', 'fuel', 'crude'],
            'Metals': ['steel', 'iron', 'copper', 'aluminum', 'mining', 'commodity'],
            'Infrastructure': ['construction', 'roads', 'bridges', 'infrastructure', 'cement'],
            'Telecom': ['telecom', 'mobile', '5G', 'spectrum', 'network', 'connectivity'],
            'Real Estate': ['real estate', 'property', 'housing', 'construction', 'land']
        }
        
    def _initialize_company_data(self) -> Dict:
        """Initialize company data for news generation"""
        return {
            'RELIANCE': {'sector': 'Energy', 'market_cap': 'Large', 'volatility': 0.02},
            'TCS': {'sector': 'IT', 'market_cap': 'Large', 'volatility': 0.018},
            'HDFCBANK': {'sector': 'Banking', 'market_cap': 'Large', 'volatility': 0.025},
            'INFY': {'sector': 'IT', 'market_cap': 'Large', 'volatility': 0.022},
            'ICICIBANK': {'sector': 'Banking', 'market_cap': 'Large', 'volatility': 0.028},
            'HINDUNILVR': {'sector': 'FMCG', 'market_cap': 'Large', 'volatility': 0.015},
            'ITC': {'sector': 'FMCG', 'market_cap': 'Large', 'volatility': 0.020},
            'SBIN': {'sector': 'Banking', 'market_cap': 'Large', 'volatility': 0.035},
            'BHARTIARTL': {'sector': 'Telecom', 'market_cap': 'Large', 'volatility': 0.030},
            'ASIANPAINT': {'sector': 'Consumer Durables', 'market_cap': 'Large', 'volatility': 0.025},
            'MARUTI': {'sector': 'Auto', 'market_cap': 'Large', 'volatility': 0.028},
            'SUNPHARMA': {'sector': 'Pharma', 'market_cap': 'Large', 'volatility': 0.032},
            'NTPC': {'sector': 'Energy', 'market_cap': 'Large', 'volatility': 0.025},
            'POWERGRID': {'sector': 'Infrastructure', 'market_cap': 'Large', 'volatility': 0.022},
            'NESTLEIND': {'sector': 'FMCG', 'market_cap': 'Large', 'volatility': 0.018}
        }
        
    def _initialize_news_sources(self) -> Dict:
        """Initialize news sources with credibility scores"""
        return {
            'Economic Times': {'credibility': 0.9, 'bias': 0.1},
            'Business Standard': {'credibility': 0.85, 'bias': 0.05},
            'Financial Express': {'credibility': 0.8, 'bias': 0.1},
            'Mint': {'credibility': 0.85, 'bias': 0.0},
            'MoneyControl': {'credibility': 0.75, 'bias': 0.15},
            'CNBC TV18': {'credibility': 0.8, 'bias': 0.2},
            'Bloomberg Quint': {'credibility': 0.9, 'bias': 0.05},
            'Reuters India': {'credibility': 0.95, 'bias': 0.0},
            'PTI': {'credibility': 0.9, 'bias': 0.0},
            'Social Media': {'credibility': 0.3, 'bias': 0.4}
        }
        
    def generate_news_event(self,
                          timestamp: datetime,
                          category: Optional[NewsCategory] = None,
                          sentiment: Optional[SentimentType] = None,
                          affected_symbols: Optional[List[str]] = None) -> NewsEvent:
        """Generate a single news event"""
        
        # Random category if not specified
        if category is None:
            category = np.random.choice(list(NewsCategory))
            
        # Random sentiment if not specified
        if sentiment is None:
            sentiment_weights = {
                SentimentType.VERY_POSITIVE: 0.1,
                SentimentType.POSITIVE: 0.25,
                SentimentType.NEUTRAL: 0.3,
                SentimentType.NEGATIVE: 0.25,
                SentimentType.VERY_NEGATIVE: 0.1
            }
            sentiment = np.random.choice(
                list(sentiment_weights.keys()),
                p=list(sentiment_weights.values())
            )
            
        # Generate headline and content
        headline, content = self._generate_news_content(category, sentiment, affected_symbols)
        
        # Determine affected symbols and sectors
        if affected_symbols is None:
            affected_symbols = self._determine_affected_symbols(category, headline)
            
        affected_sectors = self._determine_affected_sectors(category, headline, affected_symbols)
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(sentiment, headline, content)
        
        # Determine impact level
        impact_level = self._determine_impact_level(category, sentiment, affected_symbols)
        
        # Select news source
        source = np.random.choice(list(self.news_sources.keys()))
        credibility_score = self.news_sources[source]['credibility']
        
        # Generate tags
        tags = self._generate_tags(category, headline, affected_sectors)
        
        # Determine market impact duration
        impact_params = self.sentiment_impact_matrix[sentiment]
        duration = np.random.randint(*impact_params['duration_minutes'])
        
        return NewsEvent(
            timestamp=timestamp,
            headline=headline,
            content=content,
            category=category,
            sentiment=sentiment,
            impact_level=impact_level,
            affected_symbols=affected_symbols,
            affected_sectors=affected_sectors,
            sentiment_score=sentiment_score,
            credibility_score=credibility_score,
            source=source,
            tags=tags,
            market_impact_duration=duration
        )
        
    def _generate_news_content(self,
                             category: NewsCategory,
                             sentiment: SentimentType,
                             affected_symbols: Optional[List[str]] = None) -> Tuple[str, str]:
        """Generate headline and content for news"""
        
        templates = self.news_templates.get(category, {})
        sentiment_templates = templates.get(sentiment, [])
        
        if not sentiment_templates:
            # Fallback to generic template
            headline = f"Market news: {category.value} update"
        else:
            template = np.random.choice(sentiment_templates)
            
            # Fill template variables
            if '{company}' in template and affected_symbols:
                company = np.random.choice(affected_symbols)
                template = template.replace('{company}', company)
            elif '{company}' in template:
                company = np.random.choice(list(self.company_data.keys()))
                template = template.replace('{company}', company)
                
            if '{sector}' in template:
                sector = np.random.choice(list(self.sector_keywords.keys()))
                template = template.replace('{sector}', sector)
                
            # Fill numeric placeholders
            template = template.replace('{quarter}', str(np.random.randint(1, 5)))
            template = template.replace('{beat_pct}', str(np.random.randint(5, 25)))
            template = template.replace('{growth_pct}', str(np.random.randint(10, 50)))
            template = template.replace('{amount}', str(np.random.randint(100, 10000)))
            template = template.replace('{growth_rate}', f"{np.random.uniform(2, 8):.1f}")
            template = template.replace('{pmi_value}', f"{np.random.uniform(50, 65):.1f}")
            template = template.replace('{export_growth}', f"{np.random.uniform(5, 25):.1f}")
            template = template.replace('{ip_growth}', f"{np.random.uniform(2, 12):.1f}")
            template = template.replace('{deficit_amount}', f"{np.random.uniform(10, 50):.1f}")
            template = template.replace('{decline_rate}', f"{np.random.uniform(1, 5):.1f}")
            template = template.replace('{unemployment_rate}', f"{np.random.uniform(6, 15):.1f}")
            
            headline = template
            
        # Generate content (expanded version of headline)
        content = self._expand_headline_to_content(headline, category, sentiment)
        
        return headline, content
        
    def _expand_headline_to_content(self,
                                  headline: str,
                                  category: NewsCategory,
                                  sentiment: SentimentType) -> str:
        """Expand headline into full news content"""
        
        content_parts = [headline + "."]
        
        # Add context based on category
        if category == NewsCategory.EARNINGS:
            content_parts.append("The company's financial performance was driven by strong operational efficiency and market demand.")
            if sentiment in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]:
                content_parts.append("Management expressed confidence about future growth prospects.")
            elif sentiment in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]:
                content_parts.append("The company cited challenging market conditions and increased competition.")
                
        elif category == NewsCategory.POLICY:
            content_parts.append("The policy decision is expected to have significant implications for the broader economy.")
            content_parts.append("Market participants are closely monitoring the implementation details.")
            
        elif category == NewsCategory.CORPORATE:
            content_parts.append("This development is part of the company's strategic growth initiative.")
            content_parts.append("Industry analysts are evaluating the potential impact on market dynamics.")
            
        # Add market impact statement
        if sentiment in [SentimentType.POSITIVE, SentimentType.VERY_POSITIVE]:
            content_parts.append("The news is expected to have a positive impact on investor sentiment.")
        elif sentiment in [SentimentType.NEGATIVE, SentimentType.VERY_NEGATIVE]:
            content_parts.append("Market participants are concerned about the potential negative implications.")
        else:
            content_parts.append("The market reaction to this development remains to be seen.")
            
        return " ".join(content_parts)
        
    def _determine_affected_symbols(self, category: NewsCategory, headline: str) -> List[str]:
        """Determine which symbols are affected by the news"""
        affected = []
        
        # Check if specific company mentioned in headline
        for symbol in self.company_data.keys():
            if symbol.lower() in headline.lower():
                affected.append(symbol)
                
        # If no specific company, determine based on category
        if not affected:
            if category == NewsCategory.POLICY:
                # Policy news affects banking and financial stocks
                affected = ['HDFCBANK', 'ICICIBANK', 'SBIN']
            elif category == NewsCategory.ECONOMIC:
                # Economic news affects broad market
                affected = list(self.company_data.keys())[:5]
            elif category == NewsCategory.GLOBAL:
                # Global news affects IT and export-oriented stocks
                affected = ['TCS', 'INFY', 'RELIANCE']
            else:
                # Random selection for other categories
                num_affected = np.random.randint(1, 4)
                affected = np.random.choice(
                    list(self.company_data.keys()),
                    size=num_affected,
                    replace=False
                ).tolist()
                
        return affected
        
    def _determine_affected_sectors(self,
                                  category: NewsCategory,
                                  headline: str,
                                  affected_symbols: List[str]) -> List[str]:
        """Determine which sectors are affected"""
        sectors = set()
        
        # Add sectors based on affected symbols
        for symbol in affected_symbols:
            if symbol in self.company_data:
                sectors.add(self.company_data[symbol]['sector'])
                
        # Add sectors based on keywords in headline
        headline_lower = headline.lower()
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in headline_lower for keyword in keywords):
                sectors.add(sector)
                
        return list(sectors)
        
    def _calculate_sentiment_score(self,
                                 sentiment: SentimentType,
                                 headline: str,
                                 content: str) -> float:
        """Calculate numerical sentiment score"""
        
        base_scores = {
            SentimentType.VERY_NEGATIVE: -0.8,
            SentimentType.NEGATIVE: -0.4,
            SentimentType.NEUTRAL: 0.0,
            SentimentType.POSITIVE: 0.4,
            SentimentType.VERY_POSITIVE: 0.8
        }
        
        base_score = base_scores[sentiment]
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(base_score + noise, -1.0, 1.0)
        
        return final_score
        
    def _determine_impact_level(self,
                              category: NewsCategory,
                              sentiment: SentimentType,
                              affected_symbols: List[str]) -> ImpactLevel:
        """Determine market impact level"""
        
        # Base impact based on sentiment
        sentiment_impact = {
            SentimentType.VERY_NEGATIVE: ImpactLevel.VERY_HIGH,
            SentimentType.NEGATIVE: ImpactLevel.HIGH,
            SentimentType.NEUTRAL: ImpactLevel.LOW,
            SentimentType.POSITIVE: ImpactLevel.MEDIUM,
            SentimentType.VERY_POSITIVE: ImpactLevel.HIGH
        }
        
        base_impact = sentiment_impact[sentiment]
        
        # Adjust based on category
        if category in [NewsCategory.POLICY, NewsCategory.ECONOMIC]:
            # Policy and economic news have higher impact
            if base_impact == ImpactLevel.MEDIUM:
                return ImpactLevel.HIGH
            elif base_impact == ImpactLevel.HIGH:
                return ImpactLevel.VERY_HIGH
                
        # Adjust based on number of affected symbols
        if len(affected_symbols) > 5:
            if base_impact in [ImpactLevel.LOW, ImpactLevel.MEDIUM]:
                return ImpactLevel.HIGH
                
        return base_impact
        
    def _generate_tags(self,
                     category: NewsCategory,
                     headline: str,
                     affected_sectors: List[str]) -> List[str]:
        """Generate relevant tags for the news"""
        tags = [category.value]
        
        # Add sector tags
        tags.extend(affected_sectors)
        
        # Add keyword-based tags
        headline_lower = headline.lower()
        
        keyword_tags = {
            'earnings': ['quarterly results', 'financial performance'],
            'acquisition': ['M&A', 'corporate action'],
            'policy': ['government', 'regulation'],
            'growth': ['expansion', 'business growth'],
            'crisis': ['risk', 'market stress'],
            'technology': ['innovation', 'digital transformation']
        }
        
        for keyword, tag_list in keyword_tags.items():
            if keyword in headline_lower:
                tags.extend(tag_list)
                
        return list(set(tags))  # Remove duplicates
        
    def generate_news_flow(self,
                         start_time: datetime,
                         end_time: datetime,
                         news_frequency: float = 0.5) -> List[NewsEvent]:
        """Generate a flow of news events over time"""
        
        events = []
        current_time = start_time
        
        while current_time < end_time:
            # Determine if news event occurs (Poisson process)
            if np.random.random() < news_frequency:
                event = self.generate_news_event(current_time)
                events.append(event)
                
            # Move to next minute
            current_time += timedelta(minutes=1)
            
        return events
        
    def generate_earnings_season(self,
                               start_date: datetime,
                               duration_days: int = 45) -> List[NewsEvent]:
        """Generate earnings season news flow"""
        
        events = []
        companies = list(self.company_data.keys())
        
        # Distribute earnings announcements over the season
        for i, company in enumerate(companies):
            # Random day within earnings season
            days_offset = np.random.randint(0, duration_days)
            announcement_date = start_date + timedelta(days=days_offset)
            
            # Random time during market hours
            hour = np.random.choice([9, 10, 11, 12, 13, 14, 15])
            minute = np.random.randint(0, 60)
            announcement_time = announcement_date.replace(hour=hour, minute=minute)
            
            # Generate earnings sentiment (more likely to be positive)
            sentiment_weights = {
                SentimentType.VERY_POSITIVE: 0.15,
                SentimentType.POSITIVE: 0.35,
                SentimentType.NEUTRAL: 0.25,
                SentimentType.NEGATIVE: 0.20,
                SentimentType.VERY_NEGATIVE: 0.05
            }
            
            sentiment = np.random.choice(
                list(sentiment_weights.keys()),
                p=list(sentiment_weights.values())
            )
            
            event = self.generate_news_event(
                timestamp=announcement_time,
                category=NewsCategory.EARNINGS,
                sentiment=sentiment,
                affected_symbols=[company]
            )
            
            events.append(event)
            
        return sorted(events, key=lambda x: x.timestamp)
        
    def calculate_market_impact(self, event: NewsEvent) -> Dict:
        """Calculate expected market impact of news event"""
        
        impact_params = self.sentiment_impact_matrix[event.sentiment]
        
        # Base impact
        price_impact_range = impact_params['price_impact']
        volume_mult_range = impact_params['volume_multiplier']
        volatility_mult_range = impact_params['volatility_multiplier']
        
        # Adjust for credibility
        credibility_factor = event.credibility_score
        
        # Adjust for impact level
        impact_multipliers = {
            ImpactLevel.MINIMAL: 0.5,
            ImpactLevel.LOW: 0.7,
            ImpactLevel.MEDIUM: 1.0,
            ImpactLevel.HIGH: 1.5,
            ImpactLevel.VERY_HIGH: 2.0
        }
        
        impact_mult = impact_multipliers[event.impact_level]
        
        # Calculate final impact
        price_impact = np.random.uniform(*price_impact_range) * credibility_factor * impact_mult
        volume_multiplier = np.random.uniform(*volume_mult_range) * credibility_factor
        volatility_multiplier = np.random.uniform(*volatility_mult_range)
        
        return {
            'price_impact_pct': price_impact * 100,
            'volume_multiplier': volume_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'duration_minutes': event.market_impact_duration,
            'affected_symbols': event.affected_symbols,
            'affected_sectors': event.affected_sectors
        }
        
    def to_dataframe(self, events: List[NewsEvent]) -> pd.DataFrame:
        """Convert news events to pandas DataFrame"""
        
        data = []
        for event in events:
            data.append({
                'timestamp': event.timestamp,
                'headline': event.headline,
                'category': event.category.value,
                'sentiment': event.sentiment.value,
                'impact_level': event.impact_level.value,
                'sentiment_score': event.sentiment_score,
                'credibility_score': event.credibility_score,
                'source': event.source,
                'affected_symbols': ','.join(event.affected_symbols),
                'affected_sectors': ','.join(event.affected_sectors),
                'tags': ','.join(event.tags),
                'duration_minutes': event.market_impact_duration
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        return df

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = NewsSentimentSimulator(seed=42)
    
    # Generate single news event
    event = simulator.generate_news_event(
        timestamp=datetime.now(),
        category=NewsCategory.EARNINGS,
        sentiment=SentimentType.POSITIVE
    )
    
    print(f"Generated News Event:")
    print(f"Headline: {event.headline}")
    print(f"Category: {event.category.value}")
    print(f"Sentiment: {event.sentiment.value} (Score: {event.sentiment_score:.2f})")
    print(f"Impact Level: {event.impact_level.value}")
    print(f"Source: {event.source}")
    print(f"Affected Symbols: {', '.join(event.affected_symbols)}")
    print(f"Affected Sectors: {', '.join(event.affected_sectors)}")
    
    # Calculate market impact
    impact = simulator.calculate_market_impact(event)
    print(f"\nExpected Market Impact:")
    print(f"Price Impact: {impact['price_impact_pct']:.2f}%")
    print(f"Volume Multiplier: {impact['volume_multiplier']:.2f}x")
    print(f"Volatility Multiplier: {impact['volatility_multiplier']:.2f}x")
    print(f"Duration: {impact['duration_minutes']} minutes")
    
    # Generate news flow
    start_time = datetime(2024, 1, 15, 9, 0)
    end_time = datetime(2024, 1, 15, 16, 0)
    
    news_flow = simulator.generate_news_flow(start_time, end_time, news_frequency=0.1)
    print(f"\nGenerated {len(news_flow)} news events during trading day")
    
    # Generate earnings season
    earnings_events = simulator.generate_earnings_season(
        start_date=datetime(2024, 1, 15),
        duration_days=45
    )
    print(f"\nGenerated {len(earnings_events)} earnings announcements")
    
    # Convert to DataFrame
    df = simulator.to_dataframe(news_flow + earnings_events)
    print(f"\nNews DataFrame shape: {df.shape}")
    
    if not df.empty:
        print(f"\nSentiment distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
            
        print(f"\nCategory distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}")