"""
BTC Bot Pro - Feature Engineering Modülü
FAZA 4.1: Gelişmiş Teknik İndikatörler ve Feature Engineering

Özellikler:
- 50+ teknik indikatör
- Multi-timeframe features
- Volume profile
- Volatilite features
- Market structure detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# TEMEL İNDİKATÖRLER
# ================================================================

class TechnicalIndicators:
    """Teknik indikatör hesaplayıcı"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def stoch_rsi(series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI"""
        rsi = TechnicalIndicators.rsi(series, period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min() + 1e-10)
        k = stoch_rsi.rolling(smooth_k).mean() * 100
        d = k.rolling(smooth_d).mean()
        return k, d
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = mid + (std * std_dev)
        lower = mid - (std * std_dev)
        return upper, mid, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        # +DM, -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # +DI, -DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-10)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        tp = (high + low + close) / 3
        mf = tp * volume
        
        delta = tp.diff()
        pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
        neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        return mfi
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        sign = np.sign(close.diff())
        sign.iloc[0] = 0
        return (sign * volume).cumsum()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        mid = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        upper = mid + (atr_mult * atr)
        lower = mid - (atr_mult * atr)
        return upper, mid, lower
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels"""
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        mid = (upper + lower) / 2
        return upper, mid, lower
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Supertrend"""
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        
        return supertrend, direction
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Pivot Points"""
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        
        return {
            'pp': pp, 'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }


# ================================================================
# FEATURE ENGINEER
# ================================================================

class FeatureEngineer:
    """
    Feature engineering pipeline
    
    50+ features:
    - Price features
    - Trend indicators
    - Momentum indicators
    - Volatility indicators
    - Volume indicators
    - Pattern recognition
    """
    
    def __init__(self, include_advanced: bool = True):
        self.include_advanced = include_advanced
        self.ti = TechnicalIndicators()
        self.feature_names: List[str] = []
    
    def generate(self, df: pd.DataFrame, include_target: bool = False,
                 target_horizon: int = 4) -> pd.DataFrame:
        """
        Feature'ları oluştur
        
        Args:
            df: OHLCV DataFrame
            include_target: Hedef değişkeni ekle
            target_horizon: Tahmin süresi (saat)
        """
        data = df.copy()
        
        # Temel değişkenler
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        features = pd.DataFrame(index=data.index)
        
        # ==================== PRICE FEATURES ====================
        
        # Returns (farklı periyotlar)
        for period in [1, 2, 4, 8, 12, 24]:
            features[f'return_{period}h'] = close.pct_change(period) * 100
        
        # Log returns
        features['log_return'] = np.log(close / close.shift(1)) * 100
        
        # Price position
        features['price_position'] = (close - low) / (high - low + 1e-10)
        
        # Gap
        features['gap'] = (open_price - close.shift(1)) / close.shift(1) * 100
        
        # ==================== TREND INDICATORS ====================
        
        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            sma = self.ti.sma(close, period)
            ema = self.ti.ema(close, period)
            features[f'sma_{period}_dist'] = (close - sma) / close * 100
            features[f'ema_{period}_dist'] = (close - ema) / close * 100
        
        # MA Crossovers
        features['sma_7_21_cross'] = (self.ti.sma(close, 7) - self.ti.sma(close, 21)) / close * 100
        features['ema_12_26_cross'] = (self.ti.ema(close, 12) - self.ti.ema(close, 26)) / close * 100
        
        # MACD
        macd, signal, hist = self.ti.macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_hist_diff'] = hist.diff()
        
        # ADX
        adx, plus_di, minus_di = self.ti.adx(high, low, close)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di
        
        # Supertrend
        if self.include_advanced:
            st, direction = self.ti.supertrend(high, low, close)
            features['supertrend_dist'] = (close - st) / close * 100
            features['supertrend_dir'] = direction
        
        # ==================== MOMENTUM INDICATORS ====================
        
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self.ti.rsi(close, period)
        
        # Stochastic RSI
        stoch_k, stoch_d = self.ti.stoch_rsi(close)
        features['stoch_rsi_k'] = stoch_k
        features['stoch_rsi_d'] = stoch_d
        
        # Williams %R
        features['williams_r'] = self.ti.williams_r(high, low, close)
        
        # CCI
        features['cci'] = self.ti.cci(high, low, close)
        
        # MFI
        features['mfi'] = self.ti.mfi(high, low, close, volume)
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        # Rate of Change
        features['roc'] = (close - close.shift(10)) / close.shift(10) * 100
        
        # ==================== VOLATILITY INDICATORS ====================
        
        # ATR
        for period in [7, 14, 21]:
            atr = self.ti.atr(high, low, close, period)
            features[f'atr_{period}'] = atr
            features[f'atr_{period}_pct'] = atr / close * 100
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self.ti.bollinger_bands(close)
        features['bb_upper_dist'] = (bb_upper - close) / close * 100
        features['bb_lower_dist'] = (close - bb_lower) / close * 100
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # Keltner Channels
        if self.include_advanced:
            kc_upper, kc_mid, kc_lower = self.ti.keltner_channels(high, low, close)
            features['kc_position'] = (close - kc_lower) / (kc_upper - kc_lower + 1e-10)
        
        # Historical Volatility
        features['volatility_14'] = close.pct_change().rolling(14).std() * np.sqrt(24) * 100
        features['volatility_30'] = close.pct_change().rolling(30).std() * np.sqrt(24) * 100
        
        # True Range
        features['true_range_pct'] = (high - low) / close * 100
        
        # ==================== VOLUME INDICATORS ====================
        
        # Volume Moving Average
        features['volume_sma_ratio'] = volume / volume.rolling(20).mean()
        
        # OBV
        obv = self.ti.obv(close, volume)
        features['obv_change'] = obv.pct_change(5) * 100
        
        # Volume Price Trend
        features['vpt'] = (volume * close.pct_change()).cumsum()
        features['vpt_change'] = features['vpt'].pct_change(5) * 100
        
        # VWAP Distance
        if self.include_advanced:
            vwap = self.ti.vwap(high, low, close, volume)
            features['vwap_dist'] = (close - vwap) / close * 100
        
        # Volume Delta
        features['volume_delta'] = volume.diff()
        features['volume_ma_delta'] = volume.rolling(5).mean().diff()
        
        # ==================== PATTERN FEATURES ====================
        
        # Candlestick patterns
        features['body_size'] = abs(close - open_price) / (high - low + 1e-10)
        features['upper_shadow'] = (high - np.maximum(close, open_price)) / (high - low + 1e-10)
        features['lower_shadow'] = (np.minimum(close, open_price) - low) / (high - low + 1e-10)
        features['is_bullish'] = (close > open_price).astype(int)
        
        # Consecutive patterns
        features['consecutive_up'] = features['is_bullish'].rolling(3).sum()
        features['consecutive_down'] = 3 - features['consecutive_up']
        
        # Higher highs / Lower lows
        features['higher_high'] = (high > high.shift(1)).astype(int).rolling(5).sum()
        features['lower_low'] = (low < low.shift(1)).astype(int).rolling(5).sum()
        
        # ==================== TIME FEATURES ====================
        
        if 'timestamp' in data.columns:
            ts = pd.to_datetime(data['timestamp'])
            features['hour'] = ts.dt.hour
            features['day_of_week'] = ts.dt.dayofweek
            features['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
        
        # ==================== ICHIMOKU ====================
        
        if self.include_advanced:
            ichi = self.ti.ichimoku(high, low, close)
            features['tenkan_kijun_diff'] = (ichi['tenkan_sen'] - ichi['kijun_sen']) / close * 100
            features['price_vs_cloud'] = (close - (ichi['senkou_span_a'] + ichi['senkou_span_b']) / 2) / close * 100
        
        # ==================== TARGET ====================
        
        if include_target:
            features['target'] = close.shift(-target_horizon).pct_change(target_horizon) * 100
        
        # ==================== CLEAN UP ====================
        
        # Feature names
        self.feature_names = [col for col in features.columns if col != 'target']
        
        # Drop NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Feature isimlerini döndür"""
        return self.feature_names
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Basit feature importance (correlation based)"""
        importance = {}
        
        for col in features.columns:
            if col != 'target':
                corr = features[col].corr(target)
                importance[col] = abs(corr) if not np.isnan(corr) else 0
        
        df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return df


# ================================================================
# MULTI-TIMEFRAME FEATURES
# ================================================================

class MultiTimeframeFeatures:
    """Multi-timeframe feature generator"""
    
    def __init__(self, timeframes: List[str] = None):
        """
        Args:
            timeframes: Liste of timeframes ['1h', '4h', '1d']
        """
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.engineer = FeatureEngineer(include_advanced=False)
    
    def resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """OHLCV verisini resample et"""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Resample mapping
        tf_map = {
            '1h': '1h', '2h': '2h', '4h': '4h',
            '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '1w': '1W'
        }
        
        rule = tf_map.get(timeframe, timeframe)
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        
        return resampled
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe features oluştur"""
        
        # Base features (1h)
        base_features = self.engineer.generate(df)
        
        result = base_features.copy()
        
        for tf in self.timeframes:
            if tf == '1h':
                continue
            
            try:
                # Resample
                resampled = self.resample_ohlcv(df, tf)
                
                if len(resampled) < 20:
                    continue
                
                # Features
                tf_features = self.engineer.generate(resampled)
                
                # Rename columns
                tf_features = tf_features.add_suffix(f'_{tf}')
                
                # Align with base timeframe (forward fill)
                if 'timestamp' in resampled.columns:
                    tf_features.index = resampled['timestamp']
                    tf_features = tf_features.reindex(base_features.index, method='ffill')
                
                # Merge
                for col in tf_features.columns:
                    if col not in result.columns:
                        result[col] = tf_features[col].values[:len(result)]
            
            except Exception as e:
                print(f"Warning: Could not generate {tf} features: {e}")
                continue
        
        return result


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_sequences(features: np.ndarray, target: np.ndarray, 
                     lookback: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """Sequence oluştur (LSTM/GRU için)"""
    X, y = [], []
    
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)


def normalize_features(train: np.ndarray, val: np.ndarray = None, 
                       test: np.ndarray = None) -> Tuple:
    """Features'ları normalize et"""
    from sklearn.preprocessing import StandardScaler
    
    # Reshape for scaler
    n_samples, n_steps, n_features = train.shape
    train_flat = train.reshape(-1, n_features)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).reshape(train.shape)
    
    results = [train_scaled, scaler]
    
    if val is not None:
        val_flat = val.reshape(-1, n_features)
        val_scaled = scaler.transform(val_flat).reshape(val.shape)
        results.insert(1, val_scaled)
    
    if test is not None:
        test_flat = test.reshape(-1, n_features)
        test_scaled = scaler.transform(test_flat).reshape(test.shape)
        results.insert(-1, test_scaled)
    
    return tuple(results)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Feature Engineering test ediliyor...\n")
    
    # Test verisi
    import numpy as np
    np.random.seed(42)
    
    n = 500
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    prices = 50000 * np.cumprod(1 + np.random.randn(n) * 0.001)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n)) * 0.003),
        'low': prices * (1 - abs(np.random.randn(n)) * 0.003),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    # 1. Feature Engineer
    print("1. Feature Engineering")
    engineer = FeatureEngineer()
    features = engineer.generate(df, include_target=True)
    print(f"   Features: {len(engineer.get_feature_names())}")
    print(f"   Shape: {features.shape}")
    print(f"   Sample features: {engineer.get_feature_names()[:10]}")
    
    # 2. Feature Importance
    print("\n2. Feature Importance (top 10)")
    clean_features = features.dropna()
    if 'target' in clean_features.columns:
        importance = engineer.get_feature_importance(
            clean_features.drop('target', axis=1),
            clean_features['target']
        )
        print(importance.head(10).to_string(index=False))
    
    # 3. Sequence Creation
    print("\n3. Sequence Creation")
    feature_cols = [c for c in clean_features.columns if c != 'target']
    X, y = create_sequences(
        clean_features[feature_cols].values,
        clean_features['target'].values,
        lookback=48
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    print("\n✓ Feature Engineering testi başarılı!")
    print(f"   Toplam {len(engineer.get_feature_names())} feature oluşturuldu")
