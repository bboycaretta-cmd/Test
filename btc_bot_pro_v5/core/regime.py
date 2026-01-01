"""
BTC Bot Pro - Market Regime Detection
FAZA 5.2: Piyasa Rejimi Tespiti ve Adaptif Strateji

Özellikler:
- Market regime classification (Trend, Range, Volatile)
- Hidden Markov Model (HMM) based detection
- Volatility regime detection
- Trend strength analysis
- Adaptive strategy switching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# MARKET REGIMES
# ================================================================

class MarketRegime(Enum):
    """Piyasa rejimi"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class VolatilityRegime(Enum):
    """Volatilite rejimi"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RegimeState:
    """Rejim durumu"""
    regime: MarketRegime
    confidence: float  # 0-1
    volatility_regime: VolatilityRegime
    trend_strength: float  # -1 to 1
    
    # Detaylar
    adx: float = 0
    rsi: float = 50
    bb_width: float = 0
    atr_percentile: float = 50
    
    # Zaman serisi
    regime_duration: int = 0  # Bar sayısı
    last_regime_change: int = 0


@dataclass
class RegimeAnalysis:
    """Rejim analizi sonucu"""
    current_regime: RegimeState
    regime_history: List[MarketRegime] = field(default_factory=list)
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    suggested_strategy: str = "balanced"
    risk_multiplier: float = 1.0


# ================================================================
# REGIME DETECTOR
# ================================================================

class RegimeDetector:
    """
    Piyasa rejimi tespit sistemi
    
    Multiple indikator kullanarak rejim tespit eder:
    - ADX: Trend gücü
    - RSI: Momentum
    - Bollinger Band width: Volatilite
    - ATR: Volatilite
    - Price vs MA: Trend yönü
    """
    
    def __init__(self,
                 lookback: int = 100,
                 adx_period: int = 14,
                 rsi_period: int = 14,
                 bb_period: int = 20,
                 atr_period: int = 14):
        
        self.lookback = lookback
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.atr_period = atr_period
        
        # State
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: deque = deque(maxlen=500)
        self.regime_duration = 0
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX hesapla"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.adx_period).mean()
        
        # +DM, -DM
        up = high.diff()
        down = -low.diff()
        
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        
        # +DI, -DI
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / (atr + 1e-10))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """RSI hesapla"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_bb_width(self, close: pd.Series) -> pd.Series:
        """Bollinger Band width hesapla"""
        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        return (upper - lower) / mid * 100
    
    def _calculate_atr_percentile(self, high: pd.Series, low: pd.Series, 
                                   close: pd.Series) -> pd.Series:
        """ATR yüzdelik dilimi hesapla"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Rolling percentile
        def percentile_rank(x):
            return (x.rank() / len(x) * 100).iloc[-1]
        
        return atr.rolling(self.lookback).apply(percentile_rank, raw=False)
    
    def _classify_volatility(self, atr_percentile: float, 
                             bb_width: float) -> VolatilityRegime:
        """Volatilite rejimini sınıflandır"""
        # Composite score
        vol_score = (atr_percentile + bb_width * 10) / 2
        
        if vol_score < 20:
            return VolatilityRegime.VERY_LOW
        elif vol_score < 40:
            return VolatilityRegime.LOW
        elif vol_score < 60:
            return VolatilityRegime.NORMAL
        elif vol_score < 80:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _classify_regime(self, adx: float, plus_di: float, minus_di: float,
                         rsi: float, atr_pct: float) -> Tuple[MarketRegime, float]:
        """Rejimi sınıflandır"""
        
        # Trend yönü
        trend_direction = 1 if plus_di > minus_di else -1
        
        # High volatility check
        if atr_pct > 85:
            return MarketRegime.HIGH_VOLATILITY, 0.8
        
        # Low volatility check
        if atr_pct < 15 and adx < 20:
            return MarketRegime.LOW_VOLATILITY, 0.7
        
        # Strong trend
        if adx > 40:
            if trend_direction > 0:
                return MarketRegime.STRONG_UPTREND, min(0.9, adx / 50)
            else:
                return MarketRegime.STRONG_DOWNTREND, min(0.9, adx / 50)
        
        # Moderate trend
        elif adx > 25:
            if trend_direction > 0:
                return MarketRegime.UPTREND, min(0.8, adx / 40)
            else:
                return MarketRegime.DOWNTREND, min(0.8, adx / 40)
        
        # Ranging
        else:
            # RSI ile doğrula
            if 40 < rsi < 60:
                return MarketRegime.RANGING, 0.7
            elif rsi > 60:
                return MarketRegime.UPTREND, 0.5
            else:
                return MarketRegime.DOWNTREND, 0.5
    
    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Rejim tespit et
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            RegimeState
        """
        if len(df) < self.lookback:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0,
                volatility_regime=VolatilityRegime.NORMAL,
                trend_strength=0
            )
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # İndikatörleri hesapla
        adx, plus_di, minus_di = self._calculate_adx(high, low, close)
        rsi = self._calculate_rsi(close)
        bb_width = self._calculate_bb_width(close)
        atr_pct = self._calculate_atr_percentile(high, low, close)
        
        # Son değerler
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_bb_width = bb_width.iloc[-1]
        current_atr_pct = atr_pct.iloc[-1] if not pd.isna(atr_pct.iloc[-1]) else 50
        
        # Sınıflandır
        regime, confidence = self._classify_regime(
            current_adx, current_plus_di, current_minus_di,
            current_rsi, current_atr_pct
        )
        
        vol_regime = self._classify_volatility(current_atr_pct, current_bb_width)
        
        # Trend strength (-1 to 1)
        trend_strength = (current_plus_di - current_minus_di) / (current_plus_di + current_minus_di + 1e-10)
        trend_strength = trend_strength * min(1, current_adx / 30)
        
        # Regime duration
        if regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.regime_history.append(self.current_regime)
            self.regime_duration = 1
            self.current_regime = regime
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
            adx=current_adx,
            rsi=current_rsi,
            bb_width=current_bb_width,
            atr_percentile=current_atr_pct,
            regime_duration=self.regime_duration
        )
    
    def get_regime_probabilities(self, window: int = 50) -> Dict[str, float]:
        """Son N bar için rejim olasılıklarını hesapla"""
        if len(self.regime_history) < window:
            recent = list(self.regime_history)
        else:
            recent = list(self.regime_history)[-window:]
        
        if not recent:
            return {}
        
        counts = {}
        for r in recent:
            name = r.value
            counts[name] = counts.get(name, 0) + 1
        
        total = len(recent)
        return {k: v/total for k, v in counts.items()}


# ================================================================
# ADAPTIVE STRATEGY SELECTOR
# ================================================================

class AdaptiveStrategySelector:
    """
    Piyasa rejimine göre strateji seç
    
    Her rejim için optimal strateji önerir
    """
    
    # Rejim -> Strateji mapping
    REGIME_STRATEGIES = {
        MarketRegime.STRONG_UPTREND: {
            'strategy': 'trend_surfer',
            'position_multiplier': 1.2,
            'use_trailing': True,
            'sl_multiplier': 0.8,
            'tp_multiplier': 1.5
        },
        MarketRegime.UPTREND: {
            'strategy': 'momentum',
            'position_multiplier': 1.0,
            'use_trailing': True,
            'sl_multiplier': 1.0,
            'tp_multiplier': 1.2
        },
        MarketRegime.RANGING: {
            'strategy': 'mean_reversion',
            'position_multiplier': 0.8,
            'use_trailing': False,
            'sl_multiplier': 1.2,
            'tp_multiplier': 0.8
        },
        MarketRegime.DOWNTREND: {
            'strategy': 'conservative',
            'position_multiplier': 0.5,
            'use_trailing': True,
            'sl_multiplier': 1.2,
            'tp_multiplier': 1.0
        },
        MarketRegime.STRONG_DOWNTREND: {
            'strategy': 'ultra_safe',
            'position_multiplier': 0.3,
            'use_trailing': False,
            'sl_multiplier': 1.5,
            'tp_multiplier': 0.8
        },
        MarketRegime.HIGH_VOLATILITY: {
            'strategy': 'volatility',
            'position_multiplier': 0.6,
            'use_trailing': True,
            'sl_multiplier': 1.5,
            'tp_multiplier': 1.5
        },
        MarketRegime.LOW_VOLATILITY: {
            'strategy': 'breakout',
            'position_multiplier': 0.7,
            'use_trailing': False,
            'sl_multiplier': 0.8,
            'tp_multiplier': 2.0
        },
        MarketRegime.UNKNOWN: {
            'strategy': 'balanced',
            'position_multiplier': 0.5,
            'use_trailing': False,
            'sl_multiplier': 1.0,
            'tp_multiplier': 1.0
        }
    }
    
    # Volatilite -> Risk multiplier
    VOLATILITY_RISK = {
        VolatilityRegime.VERY_LOW: 1.3,
        VolatilityRegime.LOW: 1.1,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 0.7,
        VolatilityRegime.EXTREME: 0.4
    }
    
    def __init__(self, detector: RegimeDetector = None):
        self.detector = detector or RegimeDetector()
        self.current_strategy: str = "balanced"
        self.strategy_history: deque = deque(maxlen=100)
    
    def select(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Rejime göre strateji seç
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            RegimeAnalysis
        """
        # Rejim tespit
        regime_state = self.detector.detect(df)
        
        # Strateji seç
        regime_config = self.REGIME_STRATEGIES.get(
            regime_state.regime,
            self.REGIME_STRATEGIES[MarketRegime.UNKNOWN]
        )
        
        # Risk multiplier
        vol_risk = self.VOLATILITY_RISK.get(
            regime_state.volatility_regime,
            1.0
        )
        
        # Confidence ayarlaması
        confidence_factor = 0.5 + regime_state.confidence * 0.5
        
        # Final risk multiplier
        risk_multiplier = (
            regime_config['position_multiplier'] *
            vol_risk *
            confidence_factor
        )
        
        # Strategy update
        suggested = regime_config['strategy']
        
        if suggested != self.current_strategy:
            self.strategy_history.append({
                'from': self.current_strategy,
                'to': suggested,
                'regime': regime_state.regime.value,
                'confidence': regime_state.confidence
            })
            self.current_strategy = suggested
        
        # Rejim olasılıkları
        regime_probs = self.detector.get_regime_probabilities()
        
        return RegimeAnalysis(
            current_regime=regime_state,
            regime_history=list(self.detector.regime_history)[-20:],
            regime_probabilities=regime_probs,
            suggested_strategy=suggested,
            risk_multiplier=risk_multiplier
        )
    
    def get_strategy_adjustments(self, regime_state: RegimeState) -> Dict:
        """Strateji ayarlamalarını al"""
        config = self.REGIME_STRATEGIES.get(
            regime_state.regime,
            self.REGIME_STRATEGIES[MarketRegime.UNKNOWN]
        )
        
        return {
            'strategy': config['strategy'],
            'sl_multiplier': config['sl_multiplier'],
            'tp_multiplier': config['tp_multiplier'],
            'use_trailing': config['use_trailing'],
            'position_multiplier': config['position_multiplier']
        }


# ================================================================
# HMM-BASED REGIME DETECTOR (Simplified)
# ================================================================

class SimpleHMMRegimeDetector:
    """
    Simplified Hidden Markov Model benzeri rejim tespiti
    
    Transition probabilities kullanarak rejim değişimlerini modeller
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: Rejim sayısı (3 = Bull, Bear, Ranging)
        """
        self.n_regimes = n_regimes
        
        # Başlangıç olasılıkları
        self.initial_probs = np.ones(n_regimes) / n_regimes
        
        # Geçiş olasılıkları (satır = mevcut, sütun = sonraki)
        # Varsayılan: Rejimlerin kalıcı olma eğilimi
        self.transition_matrix = np.array([
            [0.90, 0.05, 0.05],  # Bull -> Bull, Bear, Ranging
            [0.05, 0.90, 0.05],  # Bear -> Bull, Bear, Ranging
            [0.10, 0.10, 0.80]   # Ranging -> Bull, Bear, Ranging
        ])
        
        # Mevcut durum
        self.current_state = 2  # Ranging ile başla
        self.state_probabilities = self.initial_probs.copy()
    
    def _calculate_emission_probs(self, returns: float, volatility: float) -> np.ndarray:
        """Emission olasılıklarını hesapla"""
        # Basitleştirilmiş model
        probs = np.zeros(self.n_regimes)
        
        # Bull (pozitif return, normal volatilite)
        bull_score = max(0, returns * 100) * (1 - min(1, volatility * 10))
        probs[0] = np.exp(bull_score)
        
        # Bear (negatif return, yüksek volatilite)
        bear_score = max(0, -returns * 100) * (1 + volatility * 5)
        probs[1] = np.exp(bear_score)
        
        # Ranging (düşük volatilite, düşük return)
        range_score = (1 - abs(returns * 100)) * (1 - min(1, volatility * 5))
        probs[2] = np.exp(range_score)
        
        # Normalize
        return probs / (probs.sum() + 1e-10)
    
    def update(self, returns: float, volatility: float) -> int:
        """
        Durumu güncelle
        
        Args:
            returns: Güncel getiri
            volatility: Güncel volatilite (ATR/price)
        
        Returns:
            Mevcut rejim index (0=Bull, 1=Bear, 2=Ranging)
        """
        # Emission probabilities
        emission = self._calculate_emission_probs(returns, volatility)
        
        # Forward step (simplified)
        # P(state_t | obs_1:t) ∝ emission(obs_t | state_t) * Σ P(state_t | state_t-1) * P(state_t-1 | obs_1:t-1)
        new_probs = np.zeros(self.n_regimes)
        
        for j in range(self.n_regimes):
            for i in range(self.n_regimes):
                new_probs[j] += self.transition_matrix[i, j] * self.state_probabilities[i]
            new_probs[j] *= emission[j]
        
        # Normalize
        new_probs /= (new_probs.sum() + 1e-10)
        self.state_probabilities = new_probs
        
        # En olası durum
        self.current_state = np.argmax(self.state_probabilities)
        
        return self.current_state
    
    def get_regime(self) -> Tuple[str, float]:
        """Mevcut rejimi ve güvenini döndür"""
        regimes = ['bull', 'bear', 'ranging']
        confidence = self.state_probabilities[self.current_state]
        return regimes[self.current_state], confidence
    
    def get_probabilities(self) -> Dict[str, float]:
        """Tüm rejim olasılıklarını döndür"""
        return {
            'bull': self.state_probabilities[0],
            'bear': self.state_probabilities[1],
            'ranging': self.state_probabilities[2]
        }


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_regime_report(analysis: RegimeAnalysis):
    """Rejim raporunu yazdır"""
    state = analysis.current_regime
    
    print("\n" + "="*60)
    print("MARKET REJİM RAPORU")
    print("="*60)
    
    print(f"\n{'MEVCUT REJİM':^60}")
    print("-"*60)
    
    regime_emoji = {
        MarketRegime.STRONG_UPTREND: "🚀",
        MarketRegime.UPTREND: "📈",
        MarketRegime.RANGING: "↔️",
        MarketRegime.DOWNTREND: "📉",
        MarketRegime.STRONG_DOWNTREND: "💥",
        MarketRegime.HIGH_VOLATILITY: "🌪️",
        MarketRegime.LOW_VOLATILITY: "😴",
        MarketRegime.UNKNOWN: "❓"
    }
    
    emoji = regime_emoji.get(state.regime, "")
    print(f"Rejim:           {emoji} {state.regime.value}")
    print(f"Güven:           {state.confidence*100:.1f}%")
    print(f"Trend Gücü:      {state.trend_strength:+.2f}")
    print(f"Süre:            {state.regime_duration} bar")
    
    print(f"\n{'VOLATİLİTE':^60}")
    print("-"*60)
    print(f"Rejim:           {state.volatility_regime.value}")
    print(f"ATR Yüzdelik:    {state.atr_percentile:.1f}%")
    print(f"BB Width:        {state.bb_width:.2f}%")
    
    print(f"\n{'İNDİKATÖRLER':^60}")
    print("-"*60)
    print(f"ADX:             {state.adx:.1f}")
    print(f"RSI:             {state.rsi:.1f}")
    
    print(f"\n{'STRATEJİ ÖNERİSİ':^60}")
    print("-"*60)
    print(f"Strateji:        {analysis.suggested_strategy}")
    print(f"Risk Çarpanı:    {analysis.risk_multiplier:.2f}x")
    
    print("\n" + "="*60)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Market Regime Detection test ediliyor...\n")
    
    # Test verisi
    np.random.seed(42)
    n = 500
    
    # Farklı rejimler simüle et
    prices = [50000]
    regimes = []
    
    for i in range(n):
        if i < 100:  # Uptrend
            change = np.random.randn() * 0.005 + 0.002
            regimes.append('uptrend')
        elif i < 200:  # Ranging
            change = np.random.randn() * 0.002
            regimes.append('ranging')
        elif i < 300:  # Downtrend
            change = np.random.randn() * 0.005 - 0.002
            regimes.append('downtrend')
        elif i < 400:  # High volatility
            change = np.random.randn() * 0.015
            regimes.append('volatile')
        else:  # Low volatility
            change = np.random.randn() * 0.001
            regimes.append('low_vol')
        
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices[1:])
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(n)) * 0.005),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    # 1. Regime Detection
    print("1. Regime Detection")
    detector = RegimeDetector()
    
    # Her 100 bar'da kontrol
    for i in [100, 200, 300, 400, 499]:
        state = detector.detect(df.iloc[:i+1])
        print(f"   Bar {i}: {state.regime.value} (conf: {state.confidence:.2f})")
    
    # 2. Adaptive Strategy
    print("\n2. Adaptive Strategy Selection")
    selector = AdaptiveStrategySelector()
    analysis = selector.select(df)
    
    print(f"   Suggested: {analysis.suggested_strategy}")
    print(f"   Risk Mult: {analysis.risk_multiplier:.2f}x")
    
    # 3. HMM Detector
    print("\n3. Simple HMM Regime")
    hmm = SimpleHMMRegimeDetector()
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(14).std().dropna()
    
    for i in [100, 200, 300, 400]:
        if i < len(returns) and i < len(volatility):
            hmm.update(returns.iloc[i], volatility.iloc[i])
            regime, conf = hmm.get_regime()
            print(f"   Bar {i}: {regime} (conf: {conf:.2f})")
    
    print("\n✓ Market Regime Detection testi başarılı!")
