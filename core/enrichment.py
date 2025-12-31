"""
BTC Bot Pro - Data Enrichment Modülü
FAZA 6: Ek Veri Kaynakları ve Sentiment Analizi

Özellikler:
- Funding Rate analizi
- Open Interest tracking
- Liquidation data
- Fear & Greed Index
- Social sentiment (Twitter/Reddit)
- Whale alert monitoring
- On-chain metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# DATACLASSES
# ================================================================

class SentimentLevel(Enum):
    """Sentiment seviyesi"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class FundingData:
    """Funding rate verisi"""
    timestamp: datetime
    rate: float  # Funding rate (örn: 0.01 = %1)
    predicted_rate: float = 0
    average_8h: float = 0
    average_24h: float = 0
    
    @property
    def annual_rate(self) -> float:
        """Yıllık oran (8 saatlik funding * 3 * 365)"""
        return self.rate * 3 * 365 * 100


@dataclass
class OpenInterestData:
    """Open Interest verisi"""
    timestamp: datetime
    oi_value: float  # USD değeri
    oi_change_1h: float = 0
    oi_change_24h: float = 0
    long_short_ratio: float = 1.0
    top_trader_long_ratio: float = 0.5


@dataclass
class LiquidationData:
    """Liquidation verisi"""
    timestamp: datetime
    long_liquidations: float  # USD
    short_liquidations: float  # USD
    total_24h: float = 0
    largest_single: float = 0
    
    @property
    def net_liquidation(self) -> float:
        """Net likidite (pozitif = daha fazla long liq)"""
        return self.long_liquidations - self.short_liquidations


@dataclass
class SentimentData:
    """Sentiment verisi"""
    timestamp: datetime
    fear_greed_index: int  # 0-100
    fear_greed_level: SentimentLevel
    twitter_sentiment: float  # -1 to 1
    reddit_sentiment: float  # -1 to 1
    news_sentiment: float  # -1 to 1
    
    @property
    def composite_sentiment(self) -> float:
        """Birleşik sentiment (-1 to 1)"""
        fg_normalized = (self.fear_greed_index - 50) / 50
        return (
            fg_normalized * 0.4 +
            self.twitter_sentiment * 0.3 +
            self.reddit_sentiment * 0.2 +
            self.news_sentiment * 0.1
        )


@dataclass
class WhaleActivity:
    """Whale aktivitesi"""
    timestamp: datetime
    large_transactions: int  # Son 24 saat
    exchange_inflow: float  # BTC miktarı
    exchange_outflow: float  # BTC miktarı
    net_flow: float  # Pozitif = satış baskısı
    
    @property
    def flow_signal(self) -> str:
        """Flow sinyali"""
        if self.net_flow > 1000:
            return "BEARISH"
        elif self.net_flow < -1000:
            return "BULLISH"
        return "NEUTRAL"


@dataclass
class OnChainMetrics:
    """On-chain metrikleri"""
    timestamp: datetime
    active_addresses: int
    transaction_count: int
    hash_rate: float  # EH/s
    difficulty: float
    nvt_ratio: float  # Network Value to Transactions
    mvrv_ratio: float  # Market Value to Realized Value
    sopr: float  # Spent Output Profit Ratio


@dataclass
class EnrichedData:
    """Zenginleştirilmiş veri"""
    timestamp: datetime
    price: float
    
    # Derivatives
    funding: FundingData = None
    open_interest: OpenInterestData = None
    liquidations: LiquidationData = None
    
    # Sentiment
    sentiment: SentimentData = None
    
    # Whale/On-chain
    whale_activity: WhaleActivity = None
    on_chain: OnChainMetrics = None
    
    # Signals
    derivatives_signal: float = 0  # -1 to 1
    sentiment_signal: float = 0  # -1 to 1
    onchain_signal: float = 0  # -1 to 1
    composite_signal: float = 0  # -1 to 1


# ================================================================
# DATA PROVIDERS (Abstract)
# ================================================================

class DataProvider:
    """Veri sağlayıcı base class"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache: Dict = {}
        self.last_update: datetime = None
    
    def fetch(self) -> Optional[Dict]:
        """Veri çek (alt sınıflar implement eder)"""
        raise NotImplementedError
    
    def get_cached(self, max_age_seconds: int = 60) -> Optional[Dict]:
        """Cache'den al"""
        if self.last_update and self.cache:
            age = (datetime.now() - self.last_update).total_seconds()
            if age < max_age_seconds:
                return self.cache
        return None


class MockDataProvider(DataProvider):
    """
    Mock veri sağlayıcı (test ve demo için)
    
    Gerçekçi simüle edilmiş veri üretir
    """
    
    def __init__(self, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        self.base_price = 95000
        self.base_oi = 20_000_000_000  # $20B
    
    def generate_funding(self, price: float = None) -> FundingData:
        """Funding rate üret"""
        # Gerçekçi funding: genelde -0.1% ile +0.1% arası
        base_rate = np.random.normal(0.0001, 0.0003)
        
        # Fiyat değişimine göre adjust
        if price:
            momentum = (price - self.base_price) / self.base_price
            base_rate += momentum * 0.001
        
        return FundingData(
            timestamp=datetime.now(),
            rate=base_rate,
            predicted_rate=base_rate * 1.1,
            average_8h=base_rate * 0.95,
            average_24h=base_rate * 0.9
        )
    
    def generate_open_interest(self, price: float = None) -> OpenInterestData:
        """Open Interest üret"""
        # OI değişimi
        change = np.random.normal(0, 0.02)
        oi = self.base_oi * (1 + change)
        
        # Long/Short ratio
        ratio = np.random.normal(1.0, 0.1)
        ratio = max(0.5, min(2.0, ratio))
        
        return OpenInterestData(
            timestamp=datetime.now(),
            oi_value=oi,
            oi_change_1h=np.random.normal(0, 0.5),
            oi_change_24h=change * 100,
            long_short_ratio=ratio,
            top_trader_long_ratio=0.5 + np.random.normal(0, 0.05)
        )
    
    def generate_liquidations(self) -> LiquidationData:
        """Liquidation verisi üret"""
        # Genelde asimetrik
        base_liq = np.random.exponential(10_000_000)
        long_ratio = np.random.uniform(0.3, 0.7)
        
        return LiquidationData(
            timestamp=datetime.now(),
            long_liquidations=base_liq * long_ratio,
            short_liquidations=base_liq * (1 - long_ratio),
            total_24h=base_liq * 24,
            largest_single=base_liq * np.random.uniform(0.1, 0.5)
        )
    
    def generate_sentiment(self) -> SentimentData:
        """Sentiment verisi üret"""
        fg_index = int(np.random.normal(50, 20))
        fg_index = max(0, min(100, fg_index))
        
        if fg_index < 20:
            level = SentimentLevel.EXTREME_FEAR
        elif fg_index < 40:
            level = SentimentLevel.FEAR
        elif fg_index < 60:
            level = SentimentLevel.NEUTRAL
        elif fg_index < 80:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED
        
        return SentimentData(
            timestamp=datetime.now(),
            fear_greed_index=fg_index,
            fear_greed_level=level,
            twitter_sentiment=np.random.uniform(-0.5, 0.5),
            reddit_sentiment=np.random.uniform(-0.5, 0.5),
            news_sentiment=np.random.uniform(-0.3, 0.3)
        )
    
    def generate_whale_activity(self) -> WhaleActivity:
        """Whale aktivitesi üret"""
        inflow = np.random.exponential(500)
        outflow = np.random.exponential(500)
        
        return WhaleActivity(
            timestamp=datetime.now(),
            large_transactions=int(np.random.poisson(50)),
            exchange_inflow=inflow,
            exchange_outflow=outflow,
            net_flow=inflow - outflow
        )
    
    def generate_onchain(self) -> OnChainMetrics:
        """On-chain metrikleri üret"""
        return OnChainMetrics(
            timestamp=datetime.now(),
            active_addresses=int(np.random.normal(900000, 50000)),
            transaction_count=int(np.random.normal(300000, 20000)),
            hash_rate=np.random.normal(600, 20),  # EH/s
            difficulty=np.random.normal(80, 5),  # T
            nvt_ratio=np.random.normal(50, 10),
            mvrv_ratio=np.random.normal(1.5, 0.3),
            sopr=np.random.normal(1.0, 0.05)
        )


# ================================================================
# DATA AGGREGATOR
# ================================================================

class DataAggregator:
    """
    Tüm veri kaynaklarını birleştirir
    """
    
    def __init__(self, provider: DataProvider = None):
        self.provider = provider or MockDataProvider()
        self.history: List[EnrichedData] = []
        self.max_history = 1000
    
    def enrich(self, price: float) -> EnrichedData:
        """
        Fiyat verisini zenginleştir
        
        Args:
            price: Güncel fiyat
        
        Returns:
            EnrichedData
        """
        # Verileri topla
        funding = self.provider.generate_funding(price) if isinstance(self.provider, MockDataProvider) else None
        oi = self.provider.generate_open_interest(price) if isinstance(self.provider, MockDataProvider) else None
        liqs = self.provider.generate_liquidations() if isinstance(self.provider, MockDataProvider) else None
        sentiment = self.provider.generate_sentiment() if isinstance(self.provider, MockDataProvider) else None
        whale = self.provider.generate_whale_activity() if isinstance(self.provider, MockDataProvider) else None
        onchain = self.provider.generate_onchain() if isinstance(self.provider, MockDataProvider) else None
        
        # Sinyalleri hesapla
        derivatives_signal = self._calc_derivatives_signal(funding, oi, liqs)
        sentiment_signal = self._calc_sentiment_signal(sentiment)
        onchain_signal = self._calc_onchain_signal(onchain, whale)
        
        # Composite
        composite = (
            derivatives_signal * 0.4 +
            sentiment_signal * 0.3 +
            onchain_signal * 0.3
        )
        
        enriched = EnrichedData(
            timestamp=datetime.now(),
            price=price,
            funding=funding,
            open_interest=oi,
            liquidations=liqs,
            sentiment=sentiment,
            whale_activity=whale,
            on_chain=onchain,
            derivatives_signal=derivatives_signal,
            sentiment_signal=sentiment_signal,
            onchain_signal=onchain_signal,
            composite_signal=composite
        )
        
        # History
        self.history.append(enriched)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return enriched
    
    def _calc_derivatives_signal(self, funding: FundingData,
                                  oi: OpenInterestData,
                                  liqs: LiquidationData) -> float:
        """Derivatives sinyali hesapla"""
        signal = 0
        
        if funding:
            # Yüksek funding = short fırsatı
            if funding.rate > 0.001:
                signal -= 0.3
            elif funding.rate < -0.001:
                signal += 0.3
        
        if oi:
            # Yükselen OI + long ratio = dikkatli ol
            if oi.long_short_ratio > 1.2:
                signal -= 0.2
            elif oi.long_short_ratio < 0.8:
                signal += 0.2
        
        if liqs:
            # Çok long liq = dip yakın olabilir
            net = liqs.net_liquidation
            if net > 50_000_000:
                signal += 0.2
            elif net < -50_000_000:
                signal -= 0.2
        
        return max(-1, min(1, signal))
    
    def _calc_sentiment_signal(self, sentiment: SentimentData) -> float:
        """Sentiment sinyali hesapla"""
        if not sentiment:
            return 0
        
        # Contrarian: Extreme fear = buy, extreme greed = sell
        if sentiment.fear_greed_index < 20:
            return 0.5
        elif sentiment.fear_greed_index > 80:
            return -0.5
        
        return sentiment.composite_sentiment
    
    def _calc_onchain_signal(self, onchain: OnChainMetrics,
                             whale: WhaleActivity) -> float:
        """On-chain sinyali hesapla"""
        signal = 0
        
        if onchain:
            # MVRV > 3 = overvalued
            if onchain.mvrv_ratio > 3:
                signal -= 0.3
            elif onchain.mvrv_ratio < 1:
                signal += 0.3
            
            # SOPR < 1 = kapitülasyon
            if onchain.sopr < 0.95:
                signal += 0.2
        
        if whale:
            # Net inflow = satış baskısı
            if whale.net_flow > 2000:
                signal -= 0.2
            elif whale.net_flow < -2000:
                signal += 0.2
        
        return max(-1, min(1, signal))
    
    def get_average_signals(self, periods: int = 24) -> Dict[str, float]:
        """Son N periyodun ortalama sinyallerini al"""
        if len(self.history) < periods:
            periods = len(self.history)
        
        if periods == 0:
            return {'derivatives': 0, 'sentiment': 0, 'onchain': 0, 'composite': 0}
        
        recent = self.history[-periods:]
        
        return {
            'derivatives': np.mean([d.derivatives_signal for d in recent]),
            'sentiment': np.mean([d.sentiment_signal for d in recent]),
            'onchain': np.mean([d.onchain_signal for d in recent]),
            'composite': np.mean([d.composite_signal for d in recent])
        }


# ================================================================
# SIGNAL GENERATOR
# ================================================================

class EnrichedSignalGenerator:
    """
    Zenginleştirilmiş verilerden sinyal üret
    
    Teknik analiz + derivatives + sentiment
    """
    
    def __init__(self, aggregator: DataAggregator = None):
        self.aggregator = aggregator or DataAggregator()
    
    def generate(self, 
                 price: float,
                 technical_signal: float,
                 technical_weight: float = 0.6) -> Dict:
        """
        Birleşik sinyal üret
        
        Args:
            price: Güncel fiyat
            technical_signal: Teknik analiz sinyali (-1 to 1)
            technical_weight: Teknik analiz ağırlığı
        
        Returns:
            Dict with signal details
        """
        # Veriyi zenginleştir
        enriched = self.aggregator.enrich(price)
        
        # Ağırlıklı sinyal
        other_weight = 1 - technical_weight
        
        combined_signal = (
            technical_signal * technical_weight +
            enriched.composite_signal * other_weight
        )
        
        # Sinyal yorumla
        if combined_signal >= 0.3:
            signal = "STRONG_LONG"
            confidence = min(95, 50 + combined_signal * 45)
        elif combined_signal >= 0.1:
            signal = "LONG"
            confidence = min(80, 50 + combined_signal * 30)
        elif combined_signal <= -0.3:
            signal = "STRONG_SHORT"
            confidence = min(95, 50 + abs(combined_signal) * 45)
        elif combined_signal <= -0.1:
            signal = "SHORT"
            confidence = min(80, 50 + abs(combined_signal) * 30)
        else:
            signal = "HOLD"
            confidence = 50
        
        # Risk adjustment
        risk_multiplier = 1.0
        
        # Extreme sentiment = reduce risk
        if enriched.sentiment:
            if enriched.sentiment.fear_greed_index < 15 or enriched.sentiment.fear_greed_index > 85:
                risk_multiplier *= 0.7
        
        # High funding = reduce risk
        if enriched.funding and abs(enriched.funding.rate) > 0.001:
            risk_multiplier *= 0.8
        
        return {
            'signal': signal,
            'confidence': confidence,
            'combined_value': combined_signal,
            'technical_component': technical_signal,
            'enriched_component': enriched.composite_signal,
            'risk_multiplier': risk_multiplier,
            'details': {
                'derivatives_signal': enriched.derivatives_signal,
                'sentiment_signal': enriched.sentiment_signal,
                'onchain_signal': enriched.onchain_signal,
                'funding_rate': enriched.funding.rate if enriched.funding else 0,
                'fear_greed': enriched.sentiment.fear_greed_index if enriched.sentiment else 50
            }
        }


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_enriched_report(enriched: EnrichedData):
    """Zenginleştirilmiş veri raporunu yazdır"""
    print("\n" + "="*60)
    print("ZENGİNLEŞTİRİLMİŞ VERİ RAPORU")
    print("="*60)
    
    print(f"\nFiyat: ${enriched.price:,.2f}")
    print(f"Zaman: {enriched.timestamp}")
    
    if enriched.funding:
        print(f"\n{'FUNDING':^60}")
        print("-"*60)
        print(f"Rate:            {enriched.funding.rate*100:.4f}%")
        print(f"Yıllık:          {enriched.funding.annual_rate:.2f}%")
    
    if enriched.open_interest:
        print(f"\n{'OPEN INTEREST':^60}")
        print("-"*60)
        print(f"Değer:           ${enriched.open_interest.oi_value/1e9:.2f}B")
        print(f"24H Değişim:     {enriched.open_interest.oi_change_24h:+.2f}%")
        print(f"Long/Short:      {enriched.open_interest.long_short_ratio:.2f}")
    
    if enriched.sentiment:
        print(f"\n{'SENTIMENT':^60}")
        print("-"*60)
        print(f"Fear & Greed:    {enriched.sentiment.fear_greed_index}/100")
        print(f"Seviye:          {enriched.sentiment.fear_greed_level.value}")
    
    print(f"\n{'SİNYALLER':^60}")
    print("-"*60)
    print(f"Derivatives:     {enriched.derivatives_signal:+.2f}")
    print(f"Sentiment:       {enriched.sentiment_signal:+.2f}")
    print(f"On-chain:        {enriched.onchain_signal:+.2f}")
    print(f"Composite:       {enriched.composite_signal:+.2f}")
    
    print("\n" + "="*60)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Data Enrichment test ediliyor...\n")
    
    # 1. Mock Data Provider
    print("1. Mock Data Provider")
    provider = MockDataProvider(seed=42)
    
    funding = provider.generate_funding(95000)
    print(f"   Funding Rate: {funding.rate*100:.4f}%")
    print(f"   Yıllık: {funding.annual_rate:.2f}%")
    
    oi = provider.generate_open_interest(95000)
    print(f"   OI: ${oi.oi_value/1e9:.2f}B")
    print(f"   L/S Ratio: {oi.long_short_ratio:.2f}")
    
    sentiment = provider.generate_sentiment()
    print(f"   Fear & Greed: {sentiment.fear_greed_index}")
    
    # 2. Data Aggregator
    print("\n2. Data Aggregator")
    aggregator = DataAggregator(provider)
    
    for price in [94000, 95000, 96000, 94500, 95500]:
        enriched = aggregator.enrich(price)
    
    signals = aggregator.get_average_signals(5)
    print(f"   Avg Derivatives: {signals['derivatives']:+.2f}")
    print(f"   Avg Sentiment:   {signals['sentiment']:+.2f}")
    print(f"   Avg Composite:   {signals['composite']:+.2f}")
    
    # 3. Signal Generator
    print("\n3. Enriched Signal Generator")
    gen = EnrichedSignalGenerator(aggregator)
    
    result = gen.generate(price=95000, technical_signal=0.3)
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"   Risk Mult: {result['risk_multiplier']:.2f}x")
    
    print("\n✓ Data Enrichment testi başarılı!")
