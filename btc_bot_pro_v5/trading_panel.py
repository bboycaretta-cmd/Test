"""
BTC BOT PRO v5.0 - TAM FONKSİYONEL CANLI TRADİNG PANELİ
=========================================================
Tüm özellikler tek sayfada, tüm fonksiyonlar gerçekten çalışıyor.

Özellikler:
- Bakiye kaydetme/okuma (JSON)
- 20 strateji seçimi
- Canlı/simüle fiyat akışı
- 30 teknik indikatör
- Sinyal üretimi
- Paper trading simülasyonu
- İşlem geçmişi (SQLite)
- Performans metrikleri
- Event-driven backtest
- Grafikler (Lightweight Charts + Chart.js)

Kullanım:
    python trading_panel.py
    Tarayıcıda: http://127.0.0.1:5000
"""

import os
import json
import sqlite3
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading

# Flask
from flask import Flask, render_template_string, jsonify, request

# Numerical
import numpy as np
import pandas as pd

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DATABASE_FILE = os.path.join(BASE_DIR, "trading.db")

# Default settings
DEFAULT_SETTINGS = {
    "balance": 10000.0,
    "initial_balance": 10000.0,
    "active_strategy": "balanced",
    "stop_loss": 0.02,
    "take_profit": 0.03,
    "position_size": 0.5,
    "is_running": False,
    "current_position": None,
    "last_update": None
}

# 20 Trading Strategies
STRATEGIES = {
    "ultra_safe": {
        "name": "Ultra Safe",
        "category": "safe",
        "stop_loss": 0.01,
        "take_profit": 0.015,
        "position_size": 0.2,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "description": "Minimum risk, küçük pozisyonlar"
    },
    "conservative": {
        "name": "Conservative",
        "category": "safe",
        "stop_loss": 0.015,
        "take_profit": 0.02,
        "position_size": 0.3,
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "description": "Muhafazakar yaklaşım"
    },
    "safe_trend": {
        "name": "Safe Trend",
        "category": "safe",
        "stop_loss": 0.02,
        "take_profit": 0.025,
        "position_size": 0.35,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "description": "Güvenli trend takibi"
    },
    "dip_buyer": {
        "name": "Dip Buyer",
        "category": "safe",
        "stop_loss": 0.025,
        "take_profit": 0.03,
        "position_size": 0.4,
        "rsi_oversold": 25,
        "rsi_overbought": 65,
        "description": "Düşüşlerde alım"
    },
    "balanced": {
        "name": "Balanced",
        "category": "balanced",
        "stop_loss": 0.02,
        "take_profit": 0.03,
        "position_size": 0.5,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "description": "Dengeli risk/ödül"
    },
    "momentum": {
        "name": "Momentum",
        "category": "balanced",
        "stop_loss": 0.02,
        "take_profit": 0.035,
        "position_size": 0.5,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "description": "Momentum takibi"
    },
    "swing_trader": {
        "name": "Swing Trader",
        "category": "balanced",
        "stop_loss": 0.025,
        "take_profit": 0.04,
        "position_size": 0.45,
        "rsi_oversold": 32,
        "rsi_overbought": 68,
        "description": "Swing trading"
    },
    "trend_surfer": {
        "name": "Trend Surfer",
        "category": "balanced",
        "stop_loss": 0.03,
        "take_profit": 0.05,
        "position_size": 0.5,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "description": "Trend sürme"
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "category": "balanced",
        "stop_loss": 0.02,
        "take_profit": 0.025,
        "position_size": 0.45,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "description": "Ortalamaya dönüş"
    },
    "breakout_confirmed": {
        "name": "Breakout Confirmed",
        "category": "balanced",
        "stop_loss": 0.025,
        "take_profit": 0.04,
        "position_size": 0.5,
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "description": "Onaylı kırılım"
    },
    "breakout": {
        "name": "Breakout",
        "category": "aggressive",
        "stop_loss": 0.03,
        "take_profit": 0.05,
        "position_size": 0.6,
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "description": "Kırılım stratejisi"
    },
    "aggressive": {
        "name": "Aggressive",
        "category": "aggressive",
        "stop_loss": 0.035,
        "take_profit": 0.06,
        "position_size": 0.65,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "description": "Agresif trading"
    },
    "volatility_hunter": {
        "name": "Volatility Hunter",
        "category": "aggressive",
        "stop_loss": 0.04,
        "take_profit": 0.07,
        "position_size": 0.6,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "description": "Volatilite avcısı"
    },
    "news_reactor": {
        "name": "News Reactor",
        "category": "aggressive",
        "stop_loss": 0.035,
        "take_profit": 0.055,
        "position_size": 0.55,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "description": "Haber tepkisi"
    },
    "scalper": {
        "name": "Scalper",
        "category": "aggressive",
        "stop_loss": 0.015,
        "take_profit": 0.02,
        "position_size": 0.7,
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "description": "Scalping"
    },
    "grid_trader": {
        "name": "Grid Trader",
        "category": "aggressive",
        "stop_loss": 0.03,
        "take_profit": 0.04,
        "position_size": 0.5,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "description": "Grid trading"
    },
    "martingale_light": {
        "name": "Martingale Light",
        "category": "aggressive",
        "stop_loss": 0.04,
        "take_profit": 0.05,
        "position_size": 0.4,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "description": "Hafif martingale"
    },
    "yolo": {
        "name": "YOLO",
        "category": "extreme",
        "stop_loss": 0.05,
        "take_profit": 0.10,
        "position_size": 0.8,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "description": "Maksimum risk"
    },
    "leverage_king": {
        "name": "Leverage King",
        "category": "extreme",
        "stop_loss": 0.06,
        "take_profit": 0.12,
        "position_size": 0.85,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
        "description": "Yüksek kaldıraç"
    },
    "all_in": {
        "name": "All In",
        "category": "extreme",
        "stop_loss": 0.08,
        "take_profit": 0.15,
        "position_size": 0.95,
        "rsi_oversold": 15,
        "rsi_overbought": 85,
        "description": "Tüm sermaye"
    }
}

# ================================================================
# DATA CLASSES
# ================================================================

class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

@dataclass
class OHLCV:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Trade:
    id: int
    timestamp: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    strategy: str
    reason: str

@dataclass
class Position:
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str

# ================================================================
# SETTINGS MANAGER (JSON)
# ================================================================

class SettingsManager:
    """JSON tabanlı ayar yönetimi - Gerçekten kaydeder ve okur"""
    
    def __init__(self):
        self.settings = self.load()
    
    def load(self) -> dict:
        """Ayarları JSON'dan yükle"""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    # Eksik anahtarları default'tan ekle
                    for key, value in DEFAULT_SETTINGS.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
            except Exception as e:
                print(f"[WARN] Settings yüklenemedi: {e}")
        return DEFAULT_SETTINGS.copy()
    
    def save(self) -> bool:
        """Ayarları JSON'a kaydet"""
        try:
            self.settings['last_update'] = datetime.now().isoformat()
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[ERROR] Settings kaydedilemedi: {e}")
            return False
    
    def get(self, key: str, default=None):
        """Ayar değeri al"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value) -> bool:
        """Ayar değeri set et ve kaydet"""
        self.settings[key] = value
        return self.save()
    
    def update(self, updates: dict) -> bool:
        """Birden fazla ayarı güncelle"""
        self.settings.update(updates)
        return self.save()
    
    def get_balance(self) -> float:
        return float(self.settings.get('balance', 10000))
    
    def set_balance(self, balance: float) -> bool:
        return self.set('balance', balance)
    
    def get_strategy(self) -> str:
        return self.settings.get('active_strategy', 'balanced')
    
    def set_strategy(self, strategy: str) -> bool:
        if strategy in STRATEGIES:
            return self.set('active_strategy', strategy)
        return False

# ================================================================
# DATABASE MANAGER (SQLite)
# ================================================================

class DatabaseManager:
    """SQLite veritabanı yönetimi"""
    
    def __init__(self):
        self.db_path = DATABASE_FILE
        self.init_db()
    
    def get_connection(self):
        """Thread-safe connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def init_db(self):
        """Veritabanı tablolarını oluştur"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Trades tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                strategy TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        ''')
        
        # OHLCV tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT UNIQUE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            )
        ''')
        
        # Performance tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                balance REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                win_count INTEGER NOT NULL,
                total_pnl REAL NOT NULL
            )
        ''')
        
        # Signals tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                strategy TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: dict) -> int:
        """Trade kaydet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, side, entry_price, exit_price, quantity, pnl, pnl_percent, strategy, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['timestamp'],
            trade['side'],
            trade['entry_price'],
            trade['exit_price'],
            trade['quantity'],
            trade['pnl'],
            trade['pnl_percent'],
            trade['strategy'],
            trade['reason']
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    
    def get_trades(self, limit: int = 50) -> List[dict]:
        """Son trade'leri getir"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, side, entry_price, exit_price, quantity, pnl, pnl_percent, strategy, reason
            FROM trades ORDER BY id DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            trades.append({
                'id': row[0],
                'timestamp': row[1],
                'side': row[2],
                'entry_price': row[3],
                'exit_price': row[4],
                'quantity': row[5],
                'pnl': row[6],
                'pnl_percent': row[7],
                'strategy': row[8],
                'reason': row[9]
            })
        return trades
    
    def get_all_trades(self) -> List[dict]:
        """Tüm trade'leri getir"""
        return self.get_trades(limit=10000)
    
    def save_ohlcv(self, data: dict):
        """OHLCV verisi kaydet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ohlcv (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['timestamp'], data['open'], data['high'], data['low'], data['close'], data['volume']))
        conn.commit()
        conn.close()
    
    def save_ohlcv_bulk(self, data_list: List[dict]):
        """Bulk OHLCV kaydet"""
        conn = self.get_connection()
        cursor = conn.cursor()
        for data in data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (data['timestamp'], data['open'], data['high'], data['low'], data['close'], data['volume']))
        conn.commit()
        conn.close()
    
    def get_ohlcv(self, limit: int = 500) -> pd.DataFrame:
        """OHLCV verisi getir"""
        conn = self.get_connection()
        df = pd.read_sql_query(f'''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv ORDER BY timestamp DESC LIMIT {limit}
        ''', conn)
        conn.close()
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological
        return df
    
    def clear_trades(self):
        """Tüm trade'leri sil"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trades')
        conn.commit()
        conn.close()

# ================================================================
# TECHNICAL INDICATORS (30 Indicators)
# ================================================================

class TechnicalIndicators:
    """30 Teknik İndikatör Hesaplama"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
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
    def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        tp = (high + low + close) / 3
        mf = tp * volume
        
        positive_mf = []
        negative_mf = []
        
        for i in range(1, len(tp)):
            if tp.iloc[i] > tp.iloc[i-1]:
                positive_mf.append(mf.iloc[i])
                negative_mf.append(0)
            else:
                positive_mf.append(0)
                negative_mf.append(mf.iloc[i])
        
        positive_mf = pd.Series([0] + positive_mf, index=tp.index)
        negative_mf = pd.Series([0] + negative_mf, index=tp.index)
        
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf_sum / (negative_mf_sum + 1e-10)))
        return mfi
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Tüm 30 indikatörü hesapla"""
        df = df.copy()
        
        # Trend Indicators
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        
        # Momentum Indicators
        df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        df['rsi_7'] = TechnicalIndicators.rsi(df['close'], 7)
        
        macd_line, signal_line, macd_hist = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_hist
        
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        df['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        df['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        
        # Volatility Indicators
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        df['atr_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
        df['atr_7'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 7)
        
        # Volume Indicators
        df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        df['mfi'] = TechnicalIndicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # Trend Strength
        df['adx'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        
        # Price Action
        df['price_change'] = df['close'].pct_change() * 100
        df['price_range'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Custom Indicators
        df['trend_strength'] = (df['ema_12'] - df['ema_26']) / df['close'] * 100
        df['momentum_score'] = (df['rsi_14'] - 50) / 50  # -1 to 1
        
        return df

# ================================================================
# PRICE DATA GENERATOR (Simulated + Real API Ready)
# ================================================================

class PriceGenerator:
    """Fiyat verisi üretici - Simüle veya Gerçek API"""
    
    def __init__(self, mode: str = 'simulated'):
        self.mode = mode
        self.base_price = 95000
        self.current_price = self.base_price
        self.volatility = 0.002  # %0.2 per update
        self.trend = 0  # -1 to 1
        self.price_history = []
        self.last_update = time.time()
        
        # Generate initial history
        self._generate_initial_history()
    
    def _generate_initial_history(self, hours: int = 720):
        """720 saatlik (30 gün) başlangıç verisi üret"""
        self.price_history = []
        price = 85000  # 30 gün önce fiyat
        
        base_time = datetime.now() - timedelta(hours=hours)
        
        for i in range(hours):
            # Trend ve volatilite
            trend_change = random.uniform(-0.1, 0.1)
            self.trend = max(-1, min(1, self.trend + trend_change))
            
            # Fiyat değişimi
            change = random.gauss(0.0001 * self.trend, self.volatility)
            price = price * (1 + change)
            
            # High, Low hesapla
            range_pct = random.uniform(0.005, 0.02)
            high = price * (1 + range_pct / 2)
            low = price * (1 - range_pct / 2)
            open_price = price * (1 + random.uniform(-range_pct/4, range_pct/4))
            
            timestamp = (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
            
            self.price_history.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': round(random.uniform(1000, 5000), 2)
            })
        
        self.current_price = price
        self.base_price = price
    
    def get_current_price(self) -> dict:
        """Güncel fiyat al"""
        if self.mode == 'simulated':
            return self._get_simulated_price()
        else:
            return self._fetch_binance_price()
    
    def _get_simulated_price(self) -> dict:
        """Simüle fiyat üret"""
        # Trend değişimi
        trend_change = random.uniform(-0.05, 0.05)
        self.trend = max(-0.5, min(0.5, self.trend + trend_change))
        
        # Fiyat değişimi
        change = random.gauss(0.0002 * self.trend, self.volatility)
        self.current_price = self.current_price * (1 + change)
        
        # Aralık
        range_pct = random.uniform(0.001, 0.005)
        high = self.current_price * (1 + range_pct)
        low = self.current_price * (1 - range_pct)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        candle = {
            'timestamp': timestamp,
            'open': round(self.price_history[-1]['close'] if self.price_history else self.current_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(self.current_price, 2),
            'volume': round(random.uniform(1000, 5000), 2)
        }
        
        return candle
    
    def _fetch_binance_price(self) -> dict:
        """Binance API'den fiyat çek (Gerçek kullanım için)"""
        try:
            import requests
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()[0]
            
            return {
                'timestamp': datetime.fromtimestamp(data[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(data[1]),
                'high': float(data[2]),
                'low': float(data[3]),
                'close': float(data[4]),
                'volume': float(data[5])
            }
        except Exception as e:
            print(f"[WARN] Binance API error: {e}, using simulated")
            return self._get_simulated_price()
    
    def update_and_get_candle(self) -> dict:
        """Yeni mum üret ve history'e ekle"""
        candle = self.get_current_price()
        
        # Her saat başında yeni mum ekle (simülasyonda her update)
        current_hour = datetime.now().strftime('%Y-%m-%d %H:00:00')
        if self.price_history and self.price_history[-1]['timestamp'][:13] != current_hour[:13]:
            self.price_history.append(candle)
            # Max 720 candle tut
            if len(self.price_history) > 720:
                self.price_history = self.price_history[-720:]
        
        return candle
    
    def get_dataframe(self, limit: int = 200) -> pd.DataFrame:
        """DataFrame olarak veri al"""
        data = self.price_history[-limit:] if self.price_history else []
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

# ================================================================
# SIGNAL GENERATOR
# ================================================================

class SignalGenerator:
    """Sinyal üretici - Teknik analiz + Strateji kuralları"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.last_signal = Signal.HOLD
        self.last_confidence = 50
    
    def generate_signal(self, df: pd.DataFrame, strategy_name: str) -> dict:
        """Sinyal üret"""
        if df.empty or len(df) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 50,
                'price': 0,
                'reason': 'Yetersiz veri'
            }
        
        # İndikatörleri hesapla
        df = self.indicators.calculate_all(df)
        
        # Son değerleri al
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        strategy = STRATEGIES.get(strategy_name, STRATEGIES['balanced'])
        
        # Sinyal skorları
        long_score = 0
        short_score = 0
        reasons = []
        
        # RSI Sinyalleri
        rsi = last['rsi_14']
        if rsi < strategy['rsi_oversold']:
            long_score += 25
            reasons.append(f"RSI aşırı satım ({rsi:.1f})")
        elif rsi > strategy['rsi_overbought']:
            short_score += 25
            reasons.append(f"RSI aşırı alım ({rsi:.1f})")
        
        # MACD Sinyalleri
        if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            long_score += 20
            reasons.append("MACD yukarı kesişim")
        elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            short_score += 20
            reasons.append("MACD aşağı kesişim")
        
        # EMA Trend
        if last['ema_12'] > last['ema_26']:
            long_score += 15
            reasons.append("EMA yükseliş trendi")
        else:
            short_score += 15
            reasons.append("EMA düşüş trendi")
        
        # Bollinger Bands
        bb_pos = last['bb_position']
        if bb_pos < 0.2:
            long_score += 15
            reasons.append("BB alt bandında")
        elif bb_pos > 0.8:
            short_score += 15
            reasons.append("BB üst bandında")
        
        # Stochastic
        if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']:
            long_score += 10
            reasons.append("Stochastic aşırı satım dönüşü")
        elif last['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d']:
            short_score += 10
            reasons.append("Stochastic aşırı alım dönüşü")
        
        # Volume Confirmation
        if last['volume_ratio'] > 1.5:
            if long_score > short_score:
                long_score += 10
            else:
                short_score += 10
            reasons.append("Yüksek hacim onayı")
        
        # MFI
        if last['mfi'] < 30:
            long_score += 5
        elif last['mfi'] > 70:
            short_score += 5
        
        # Final Signal
        if long_score > short_score and long_score >= 40:
            signal = 'LONG'
            confidence = min(95, 50 + long_score)
        elif short_score > long_score and short_score >= 40:
            signal = 'SHORT'
            confidence = min(95, 50 + short_score)
        else:
            signal = 'HOLD'
            confidence = 50
        
        self.last_signal = Signal[signal]
        self.last_confidence = confidence
        
        return {
            'signal': signal,
            'confidence': confidence,
            'price': last['close'],
            'reason': ', '.join(reasons[:3]) if reasons else 'Nötr',
            'rsi': round(rsi, 1),
            'macd': round(last['macd'], 2),
            'bb_position': round(bb_pos * 100, 1),
            'trend': 'UP' if last['ema_12'] > last['ema_26'] else 'DOWN'
        }

# ================================================================
# TRADING ENGINE (Paper Trading)
# ================================================================

class TradingEngine:
    """Paper Trading Motor - Gerçek simülasyon"""
    
    def __init__(self, settings: SettingsManager, db: DatabaseManager):
        self.settings = settings
        self.db = db
        self.current_position = None
        self.load_position()
    
    def load_position(self):
        """Mevcut pozisyonu yükle"""
        pos_data = self.settings.get('current_position')
        if pos_data:
            self.current_position = Position(**pos_data)
    
    def save_position(self, position: Optional[Position]):
        """Pozisyonu kaydet"""
        if position:
            self.settings.set('current_position', {
                'side': position.side,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time
            })
        else:
            self.settings.set('current_position', None)
        self.current_position = position
    
    def open_position(self, side: str, price: float, strategy_name: str) -> dict:
        """Pozisyon aç"""
        if self.current_position:
            return {'success': False, 'message': 'Zaten açık pozisyon var'}
        
        strategy = STRATEGIES.get(strategy_name, STRATEGIES['balanced'])
        balance = self.settings.get_balance()
        
        # Position size
        position_value = balance * strategy['position_size']
        quantity = position_value / price
        
        # Slippage (gerçekçilik için)
        slippage = price * 0.0005 * (1 if side == 'LONG' else -1)
        entry_price = price + slippage
        
        # Stop Loss ve Take Profit
        if side == 'LONG':
            stop_loss = entry_price * (1 - strategy['stop_loss'])
            take_profit = entry_price * (1 + strategy['take_profit'])
        else:
            stop_loss = entry_price * (1 + strategy['stop_loss'])
            take_profit = entry_price * (1 - strategy['take_profit'])
        
        # Pozisyon oluştur
        position = Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        self.save_position(position)
        
        return {
            'success': True,
            'message': f'{side} pozisyon açıldı',
            'position': {
                'side': side,
                'entry_price': round(entry_price, 2),
                'quantity': round(quantity, 6),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'value': round(position_value, 2)
            }
        }
    
    def check_position(self, current_price: float) -> Optional[dict]:
        """Pozisyonu kontrol et - SL/TP tetikleme"""
        if not self.current_position:
            return None
        
        pos = self.current_position
        
        # Stop Loss kontrolü
        if pos.side == 'LONG' and current_price <= pos.stop_loss:
            return self.close_position(current_price, 'stop_loss')
        elif pos.side == 'SHORT' and current_price >= pos.stop_loss:
            return self.close_position(current_price, 'stop_loss')
        
        # Take Profit kontrolü
        if pos.side == 'LONG' and current_price >= pos.take_profit:
            return self.close_position(current_price, 'take_profit')
        elif pos.side == 'SHORT' and current_price <= pos.take_profit:
            return self.close_position(current_price, 'take_profit')
        
        return None
    
    def close_position(self, price: float, reason: str = 'manual') -> dict:
        """Pozisyonu kapat"""
        if not self.current_position:
            return {'success': False, 'message': 'Açık pozisyon yok'}
        
        pos = self.current_position
        
        # Slippage
        slippage = price * 0.0005 * (-1 if pos.side == 'LONG' else 1)
        exit_price = price + slippage
        
        # PnL hesapla
        if pos.side == 'LONG':
            pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_percent = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_percent = (pos.entry_price - exit_price) / pos.entry_price * 100
        
        # Commission (0.1%)
        commission = (pos.entry_price + exit_price) * pos.quantity * 0.001
        pnl -= commission
        
        # Trade kaydet
        trade = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'side': pos.side,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'quantity': pos.quantity,
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'strategy': self.settings.get_strategy(),
            'reason': reason
        }
        
        self.db.save_trade(trade)
        
        # Bakiye güncelle
        new_balance = self.settings.get_balance() + pnl
        self.settings.set_balance(new_balance)
        
        # Pozisyonu temizle
        self.save_position(None)
        
        return {
            'success': True,
            'message': f'Pozisyon kapatıldı ({reason})',
            'trade': trade,
            'new_balance': round(new_balance, 2)
        }
    
    def get_position_pnl(self, current_price: float) -> dict:
        """Açık pozisyon PnL hesapla"""
        if not self.current_position:
            return None
        
        pos = self.current_position
        
        if pos.side == 'LONG':
            pnl = (current_price - pos.entry_price) * pos.quantity
            pnl_percent = (current_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity
            pnl_percent = (pos.entry_price - current_price) / pos.entry_price * 100
        
        return {
            'side': pos.side,
            'entry_price': round(pos.entry_price, 2),
            'current_price': round(current_price, 2),
            'quantity': round(pos.quantity, 6),
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'stop_loss': round(pos.stop_loss, 2),
            'take_profit': round(pos.take_profit, 2),
            'entry_time': pos.entry_time
        }

# ================================================================
# PERFORMANCE TRACKER
# ================================================================

class PerformanceTracker:
    """Performans metrikleri hesaplama"""
    
    def __init__(self, db: DatabaseManager, settings: SettingsManager):
        self.db = db
        self.settings = settings
    
    def calculate_metrics(self) -> dict:
        """Tüm performans metriklerini hesapla"""
        trades = self.db.get_all_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_profit': 0,
                'total_loss': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'current_balance': self.settings.get_balance(),
                'initial_balance': self.settings.get('initial_balance', 10000),
                'roi': 0
            }
        
        # Basic stats
        total_trades = len(trades)
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]
        
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL stats
        total_pnl = sum(t['pnl'] for t in trades)
        total_profit = sum(t['pnl'] for t in winning) if winning else 0
        total_loss = abs(sum(t['pnl'] for t in losing)) if losing else 0
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        average_win = (total_profit / winning_trades) if winning_trades > 0 else 0
        average_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
        
        best_trade = max(t['pnl'] for t in trades) if trades else 0
        worst_trade = min(t['pnl'] for t in trades) if trades else 0
        
        # Drawdown hesapla
        initial_balance = self.settings.get('initial_balance', 10000)
        current_balance = self.settings.get_balance()
        
        # Running balance ile drawdown
        running_balance = initial_balance
        peak_balance = initial_balance
        max_drawdown = 0
        
        for trade in trades:
            running_balance += trade['pnl']
            if running_balance > peak_balance:
                peak_balance = running_balance
            drawdown = (peak_balance - running_balance) / peak_balance * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # ROI
        roi = ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        
        # Sharpe Ratio (simplified)
        pnl_list = [t['pnl_percent'] for t in trades]
        if len(pnl_list) > 1:
            avg_return = np.mean(pnl_list)
            std_return = np.std(pnl_list)
            sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'total_profit': round(total_profit, 2),
            'total_loss': round(total_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
            'average_win': round(average_win, 2),
            'average_loss': round(average_loss, 2),
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'current_balance': round(current_balance, 2),
            'initial_balance': initial_balance,
            'roi': round(roi, 2)
        }

# ================================================================
# BACKTEST ENGINE
# ================================================================

class BacktestEngine:
    """Event-driven backtest motoru"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.signal_gen = SignalGenerator()
    
    def run_backtest(self, data: pd.DataFrame, strategy_name: str, initial_balance: float = 10000) -> dict:
        """Backtest çalıştır"""
        if data.empty or len(data) < 50:
            return {'error': 'Yetersiz veri'}
        
        strategy = STRATEGIES.get(strategy_name, STRATEGIES['balanced'])
        
        # İndikatörleri hesapla
        df = self.indicators.calculate_all(data.copy())
        df = df.dropna()
        
        if len(df) < 20:
            return {'error': 'İndikatör hesaplaması sonrası yetersiz veri'}
        
        # Backtest state
        balance = initial_balance
        position = None
        trades = []
        equity_curve = [initial_balance]
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            price = current['close']
            
            # Pozisyon varsa kontrol et
            if position:
                # Stop Loss
                if position['side'] == 'LONG' and price <= position['stop_loss']:
                    pnl = (price - position['entry']) * position['qty']
                    balance += pnl - (position['entry'] + price) * position['qty'] * 0.001
                    trades.append({
                        'side': 'LONG',
                        'entry': position['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'reason': 'stop_loss'
                    })
                    position = None
                elif position['side'] == 'SHORT' and price >= position['stop_loss']:
                    pnl = (position['entry'] - price) * position['qty']
                    balance += pnl - (position['entry'] + price) * position['qty'] * 0.001
                    trades.append({
                        'side': 'SHORT',
                        'entry': position['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'reason': 'stop_loss'
                    })
                    position = None
                
                # Take Profit
                elif position and position['side'] == 'LONG' and price >= position['take_profit']:
                    pnl = (price - position['entry']) * position['qty']
                    balance += pnl - (position['entry'] + price) * position['qty'] * 0.001
                    trades.append({
                        'side': 'LONG',
                        'entry': position['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'reason': 'take_profit'
                    })
                    position = None
                elif position and position['side'] == 'SHORT' and price <= position['take_profit']:
                    pnl = (position['entry'] - price) * position['qty']
                    balance += pnl - (position['entry'] + price) * position['qty'] * 0.001
                    trades.append({
                        'side': 'SHORT',
                        'entry': position['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'reason': 'take_profit'
                    })
                    position = None
            
            # Sinyal kontrol (pozisyon yoksa)
            if not position:
                rsi = current['rsi_14']
                macd = current['macd']
                macd_signal = current['macd_signal']
                prev_macd = prev['macd']
                prev_macd_signal = prev['macd_signal']
                
                # LONG sinyal
                if rsi < strategy['rsi_oversold'] or (macd > macd_signal and prev_macd <= prev_macd_signal):
                    position_value = balance * strategy['position_size']
                    qty = position_value / price
                    position = {
                        'side': 'LONG',
                        'entry': price,
                        'qty': qty,
                        'stop_loss': price * (1 - strategy['stop_loss']),
                        'take_profit': price * (1 + strategy['take_profit'])
                    }
                
                # SHORT sinyal
                elif rsi > strategy['rsi_overbought'] or (macd < macd_signal and prev_macd >= prev_macd_signal):
                    position_value = balance * strategy['position_size']
                    qty = position_value / price
                    position = {
                        'side': 'SHORT',
                        'entry': price,
                        'qty': qty,
                        'stop_loss': price * (1 + strategy['stop_loss']),
                        'take_profit': price * (1 - strategy['take_profit'])
                    }
            
            equity_curve.append(balance)
        
        # Son pozisyonu kapat
        if position:
            price = df.iloc[-1]['close']
            if position['side'] == 'LONG':
                pnl = (price - position['entry']) * position['qty']
            else:
                pnl = (position['entry'] - price) * position['qty']
            balance += pnl
            trades.append({
                'side': position['side'],
                'entry': position['entry'],
                'exit': price,
                'pnl': pnl,
                'reason': 'end_of_test'
            })
        
        # Sonuçları hesapla
        total_trades = len(trades)
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)
        total_profit = sum(t['pnl'] for t in winning) if winning else 0
        total_loss = abs(sum(t['pnl'] for t in losing)) if losing else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 999
        
        # Max Drawdown
        peak = initial_balance
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Buy & Hold karşılaştırma
        start_price = df.iloc[50]['close']
        end_price = df.iloc[-1]['close']
        buy_hold_return = (end_price - start_price) / start_price * 100
        strategy_return = (balance - initial_balance) / initial_balance * 100
        
        return {
            'strategy': strategy_name,
            'initial_balance': initial_balance,
            'final_balance': round(balance, 2),
            'total_return': round(strategy_return, 2),
            'total_trades': total_trades,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_dd, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'vs_buy_hold': round(strategy_return - buy_hold_return, 2),
            'trades': trades[-10:],  # Son 10 trade
            'equity_curve': equity_curve[::max(1, len(equity_curve)//50)]  # 50 nokta
        }

# ================================================================
# MAIN APPLICATION CLASS
# ================================================================

class TradingBot:
    """Ana bot sınıfı - Tüm bileşenleri yönetir"""
    
    def __init__(self):
        self.settings = SettingsManager()
        self.db = DatabaseManager()
        self.price_gen = PriceGenerator(mode='simulated')
        self.signal_gen = SignalGenerator()
        self.trading_engine = TradingEngine(self.settings, self.db)
        self.performance = PerformanceTracker(self.db, self.settings)
        self.backtest = BacktestEngine()
        
        self.is_running = False
        self.last_signal = {'signal': 'HOLD', 'confidence': 50}
        self.last_price = 0
        
        # OHLCV verisini database'e kaydet
        self._init_price_data()
    
    def _init_price_data(self):
        """Başlangıç fiyat verisini kaydet"""
        if self.price_gen.price_history:
            self.db.save_ohlcv_bulk(self.price_gen.price_history)
    
    def start(self):
        """Botu başlat"""
        self.is_running = True
        self.settings.set('is_running', True)
        return {'success': True, 'message': 'Bot başlatıldı'}
    
    def stop(self):
        """Botu durdur"""
        self.is_running = False
        self.settings.set('is_running', False)
        return {'success': True, 'message': 'Bot durduruldu'}
    
    def reset(self):
        """Botu sıfırla"""
        self.stop()
        
        # Bakiye sıfırla
        initial = self.settings.get('initial_balance', 10000)
        self.settings.set('balance', initial)
        
        # Pozisyonu kapat
        self.trading_engine.save_position(None)
        
        # Trade'leri temizle
        self.db.clear_trades()
        
        return {'success': True, 'message': 'Bot sıfırlandı'}
    
    def update(self) -> dict:
        """Tek güncelleme döngüsü"""
        # Fiyat güncelle
        candle = self.price_gen.update_and_get_candle()
        self.last_price = candle['close']
        
        # OHLCV kaydet
        self.db.save_ohlcv(candle)
        
        result = {
            'price': candle['close'],
            'change': ((candle['close'] - candle['open']) / candle['open'] * 100),
            'high': candle['high'],
            'low': candle['low'],
            'signal': self.last_signal,
            'position': None,
            'trade': None,
            'balance': self.settings.get_balance(),
            'timestamp': candle['timestamp']
        }
        
        if not self.is_running:
            return result
        
        # Pozisyon kontrolü (SL/TP)
        trade_result = self.trading_engine.check_position(candle['close'])
        if trade_result and trade_result.get('success'):
            result['trade'] = trade_result.get('trade')
            result['balance'] = trade_result.get('new_balance')
        
        # Mevcut pozisyon bilgisi
        pos_pnl = self.trading_engine.get_position_pnl(candle['close'])
        if pos_pnl:
            result['position'] = pos_pnl
        
        # DataFrame oluştur ve sinyal üret
        df = self.price_gen.get_dataframe(200)
        if not df.empty and len(df) >= 50:
            strategy_name = self.settings.get_strategy()
            signal = self.signal_gen.generate_signal(df, strategy_name)
            self.last_signal = signal
            result['signal'] = signal
            
            # Pozisyon yoksa ve sinyal varsa trade aç
            if not self.trading_engine.current_position and signal['signal'] != 'HOLD':
                if signal['confidence'] >= 60:  # Minimum güven
                    trade_open = self.trading_engine.open_position(
                        signal['signal'], 
                        candle['close'],
                        strategy_name
                    )
                    if trade_open.get('success'):
                        result['position'] = trade_open.get('position')
        
        return result
    
    def get_status(self) -> dict:
        """Tam durum bilgisi"""
        candle = self.price_gen.get_current_price()
        df = self.price_gen.get_dataframe(200)
        
        # Sinyal
        strategy_name = self.settings.get_strategy()
        signal = self.signal_gen.generate_signal(df, strategy_name) if len(df) >= 50 else {'signal': 'HOLD', 'confidence': 50}
        
        # Pozisyon
        pos_pnl = self.trading_engine.get_position_pnl(candle['close'])
        
        # Performans
        metrics = self.performance.calculate_metrics()
        
        # Son trade'ler
        trades = self.db.get_trades(20)
        
        return {
            'is_running': self.is_running,
            'price': {
                'current': candle['close'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'change': round((candle['close'] - candle['open']) / candle['open'] * 100, 2),
                'timestamp': candle['timestamp']
            },
            'signal': signal,
            'position': pos_pnl,
            'balance': self.settings.get_balance(),
            'initial_balance': self.settings.get('initial_balance', 10000),
            'strategy': strategy_name,
            'strategy_name': STRATEGIES[strategy_name]['name'],
            'metrics': metrics,
            'trades': trades
        }
    
    def run_backtest(self, strategy_name: str, months: int = 1) -> dict:
        """Backtest çalıştır"""
        # Veri al
        hours = months * 720  # 30 gün × 24 saat
        df = self.price_gen.get_dataframe(min(hours, len(self.price_gen.price_history)))
        
        if df.empty or len(df) < 100:
            return {'error': 'Yetersiz veri'}
        
        initial = self.settings.get('initial_balance', 10000)
        result = self.backtest.run_backtest(df, strategy_name, initial)
        
        return result
    
    def get_chart_data(self, limit: int = 100) -> dict:
        """Grafik verisi"""
        df = self.price_gen.get_dataframe(limit)
        
        if df.empty:
            return {'candles': [], 'trades': []}
        
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
        
        # Trade marker'ları
        trades = self.db.get_trades(50)
        markers = []
        for trade in trades:
            markers.append({
                'time': trade['timestamp'],
                'position': 'belowBar' if trade['side'] == 'LONG' else 'aboveBar',
                'color': '#26a69a' if trade['pnl'] > 0 else '#ef5350',
                'shape': 'arrowUp' if trade['side'] == 'LONG' else 'arrowDown',
                'text': f"{trade['side']} ${trade['pnl']:.0f}"
            })
        
        return {'candles': candles, 'markers': markers}

# ================================================================
# FLASK APPLICATION
# ================================================================

app = Flask(__name__)
bot = None  # Global bot instance

def get_bot():
    """Bot instance al veya oluştur"""
    global bot
    if bot is None:
        bot = TradingBot()
    return bot

# ================================================================
# HTML TEMPLATE
# ================================================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Bot Pro v5.0 - Trading Panel</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #f7931a;
        }
        
        .price-display {
            text-align: right;
        }
        
        .price-main {
            font-size: 32px;
            font-weight: bold;
            color: #fff;
        }
        
        .price-change {
            font-size: 16px;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 5px;
        }
        
        .price-change.positive { background: rgba(38, 166, 154, 0.3); color: #26a69a; }
        .price-change.negative { background: rgba(239, 83, 80, 0.3); color: #ef5350; }
        
        /* Cards Grid */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .card-title {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .card-value {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }
        
        .card-sub {
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }
        
        /* Signal Card */
        .signal-card {
            text-align: center;
        }
        
        .signal-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 18px;
        }
        
        .signal-LONG { background: rgba(38, 166, 154, 0.3); color: #26a69a; }
        .signal-SHORT { background: rgba(239, 83, 80, 0.3); color: #ef5350; }
        .signal-HOLD { background: rgba(255, 193, 7, 0.3); color: #ffc107; }
        
        .confidence-bar {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }
        
        /* Chart Container */
        .chart-container {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .chart-title {
            font-size: 18px;
            font-weight: bold;
        }
        
        #priceChart {
            width: 100%;
            height: 350px;
        }
        
        /* Settings Panel */
        .panels-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .panel-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 14px;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #f7931a;
        }
        
        /* Buttons */
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #f7931a, #ff6b00);
            color: #fff;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(247, 147, 26, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #26a69a, #00897b);
            color: #fff;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ef5350, #c62828);
            color: #fff;
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        
        /* Control Panel */
        .control-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .control-buttons {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .control-buttons .btn {
            width: auto;
            padding: 12px 30px;
        }
        
        .status-bar {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 10px 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-dot.running { background: #26a69a; }
        .status-dot.stopped { background: #ef5350; animation: none; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Trade History */
        .trade-history {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .trade-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .trade-table th,
        .trade-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .trade-table th {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        
        .trade-table tr:hover {
            background: rgba(255,255,255,0.05);
        }
        
        .side-badge {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .side-LONG { background: rgba(38, 166, 154, 0.3); color: #26a69a; }
        .side-SHORT { background: rgba(239, 83, 80, 0.3); color: #ef5350; }
        
        .pnl-positive { color: #26a69a; }
        .pnl-negative { color: #ef5350; }
        
        /* Performance Grid */
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .perf-item {
            text-align: center;
        }
        
        .perf-label {
            font-size: 12px;
            color: #888;
        }
        
        .perf-value {
            font-size: 20px;
            font-weight: bold;
            color: #fff;
            margin-top: 5px;
        }
        
        /* Backtest Results */
        .backtest-result {
            margin-top: 15px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            display: none;
        }
        
        .backtest-result.show {
            display: block;
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 13px;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
        }
        
        /* Position Info */
        .position-info {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            font-size: 13px;
        }
        
        .position-info.hidden {
            display: none;
        }
        
        /* Toast Notification */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            color: #fff;
            font-weight: bold;
            z-index: 1000;
            transform: translateX(400px);
            transition: transform 0.3s;
        }
        
        .toast.show {
            transform: translateX(0);
        }
        
        .toast.success { background: #26a69a; }
        .toast.error { background: #ef5350; }
        .toast.info { background: #2196f3; }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .cards-grid { grid-template-columns: repeat(2, 1fr); }
            .panels-grid { grid-template-columns: repeat(2, 1fr); }
            .performance-grid { grid-template-columns: repeat(3, 1fr); }
        }
        
        @media (max-width: 768px) {
            .cards-grid { grid-template-columns: 1fr; }
            .panels-grid { grid-template-columns: 1fr; }
            .performance-grid { grid-template-columns: repeat(2, 1fr); }
            .control-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">₿ BTC Bot Pro v5.0</div>
            <div class="price-display">
                <div class="price-main" id="currentPrice">$95,234.56</div>
                <div class="price-change positive" id="priceChange">+2.34%</div>
            </div>
        </div>
        
        <!-- Stats Cards -->
        <div class="cards-grid">
            <div class="card signal-card">
                <div class="card-title">📡 Aktif Sinyal</div>
                <div class="signal-badge signal-HOLD" id="signalBadge">HOLD</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill" style="width: 50%; background: #ffc107;"></div>
                </div>
                <div class="card-sub" id="signalReason">Sinyal bekleniyor...</div>
            </div>
            
            <div class="card">
                <div class="card-title">💰 Bakiye</div>
                <div class="card-value" id="balance">$10,000.00</div>
                <div class="card-sub" id="balancePnl">PnL: $0.00 (0%)</div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 Pozisyon</div>
                <div class="card-value" id="positionStatus">Yok</div>
                <div class="position-info hidden" id="positionInfo">
                    <div>Entry: <span id="posEntry">-</span></div>
                    <div>PnL: <span id="posPnl">-</span></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📈 Performans</div>
                <div class="card-value" id="winRate">0%</div>
                <div class="card-sub" id="totalTrades">0 işlem</div>
            </div>
        </div>
        
        <!-- Chart -->
        <div class="chart-container">
            <div class="chart-header">
                <div class="chart-title">📊 BTC/USDT Fiyat Grafiği</div>
                <div id="chartTime">Son güncelleme: -</div>
            </div>
            <div id="priceChart"></div>
        </div>
        
        <!-- Panels -->
        <div class="panels-grid">
            <!-- Settings Panel -->
            <div class="panel">
                <div class="panel-title">⚙️ Ayarlar</div>
                <div class="form-group">
                    <label>Bakiye (USD)</label>
                    <input type="number" id="inputBalance" value="10000" min="100" step="100">
                </div>
                <div class="form-group">
                    <label>Stop Loss (%)</label>
                    <input type="number" id="inputSL" value="2" min="0.5" max="10" step="0.5">
                </div>
                <div class="form-group">
                    <label>Take Profit (%)</label>
                    <input type="number" id="inputTP" value="3" min="0.5" max="15" step="0.5">
                </div>
                <button class="btn btn-primary" onclick="saveSettings()">💾 Kaydet</button>
            </div>
            
            <!-- Strategy Panel -->
            <div class="panel">
                <div class="panel-title">🎯 Strateji</div>
                <div class="form-group">
                    <label>Aktif Strateji</label>
                    <select id="strategySelect">
                        <option value="ultra_safe">🛡️ Ultra Safe</option>
                        <option value="conservative">🔒 Conservative</option>
                        <option value="safe_trend">📈 Safe Trend</option>
                        <option value="dip_buyer">📉 Dip Buyer</option>
                        <option value="balanced" selected>⚖️ Balanced</option>
                        <option value="momentum">🚀 Momentum</option>
                        <option value="swing_trader">🔄 Swing Trader</option>
                        <option value="trend_surfer">🏄 Trend Surfer</option>
                        <option value="mean_reversion">↩️ Mean Reversion</option>
                        <option value="breakout_confirmed">✅ Breakout Confirmed</option>
                        <option value="breakout">💥 Breakout</option>
                        <option value="aggressive">⚡ Aggressive</option>
                        <option value="volatility_hunter">🎯 Volatility Hunter</option>
                        <option value="news_reactor">📰 News Reactor</option>
                        <option value="scalper">⏱️ Scalper</option>
                        <option value="grid_trader">📊 Grid Trader</option>
                        <option value="martingale_light">🎲 Martingale Light</option>
                        <option value="yolo">🔥 YOLO</option>
                        <option value="leverage_king">👑 Leverage King</option>
                        <option value="all_in">💎 All In</option>
                    </select>
                </div>
                <div class="card-sub" id="strategyDesc">Dengeli risk/ödül stratejisi</div>
                <button class="btn btn-primary" onclick="applyStrategy()" style="margin-top: 15px;">✅ Uygula</button>
            </div>
            
            <!-- Backtest Panel -->
            <div class="panel">
                <div class="panel-title">📈 Backtest</div>
                <div class="form-group">
                    <label>Test Süresi</label>
                    <select id="backtestPeriod">
                        <option value="1">1 Ay</option>
                        <option value="3">3 Ay</option>
                        <option value="6">6 Ay</option>
                    </select>
                </div>
                <button class="btn btn-primary" onclick="runBacktest()">▶️ Backtest Yap</button>
                <div class="backtest-result" id="backtestResult">
                    <div class="result-grid">
                        <div class="result-item"><span>Getiri:</span><span id="btReturn">-</span></div>
                        <div class="result-item"><span>İşlem:</span><span id="btTrades">-</span></div>
                        <div class="result-item"><span>Win Rate:</span><span id="btWinRate">-</span></div>
                        <div class="result-item"><span>Max DD:</span><span id="btMaxDD">-</span></div>
                        <div class="result-item"><span>PF:</span><span id="btPF">-</span></div>
                        <div class="result-item"><span>vs HODL:</span><span id="btVsHodl">-</span></div>
                    </div>
                </div>
            </div>
            
            <!-- Quick Stats -->
            <div class="panel">
                <div class="panel-title">📊 Hızlı İstatistik</div>
                <div class="form-group">
                    <label>Toplam Kar</label>
                    <div class="card-value" id="totalProfit" style="font-size: 20px;">$0.00</div>
                </div>
                <div class="form-group">
                    <label>Toplam Zarar</label>
                    <div class="card-value" id="totalLoss" style="font-size: 20px; color: #ef5350;">$0.00</div>
                </div>
                <div class="form-group">
                    <label>Profit Factor</label>
                    <div class="card-value" id="profitFactor" style="font-size: 20px;">0.00</div>
                </div>
            </div>
        </div>
        
        <!-- Control Panel -->
        <div class="control-panel">
            <div class="chart-title" style="margin-bottom: 15px;">⚡ Kontrol Paneli</div>
            <div class="control-buttons">
                <button class="btn btn-success" id="btnStart" onclick="startBot()">▶️ BAŞLAT</button>
                <button class="btn btn-danger" id="btnStop" onclick="stopBot()" disabled>⏹️ DURDUR</button>
                <button class="btn btn-secondary" onclick="resetBot()">🔄 SIFIRLA</button>
            </div>
            <div class="status-bar">
                <div class="status-indicator">
                    <div class="status-dot stopped" id="statusDot"></div>
                    <span id="statusText">Durduruldu</span>
                </div>
                <div>|</div>
                <div>Sonraki güncelleme: <span id="nextUpdate">-</span></div>
                <div>|</div>
                <div>Strateji: <span id="activeStrategy">Balanced</span></div>
            </div>
        </div>
        
        <!-- Trade History -->
        <div class="trade-history">
            <div class="chart-title" style="margin-bottom: 15px;">📜 İşlem Geçmişi</div>
            <table class="trade-table">
                <thead>
                    <tr>
                        <th>Zaman</th>
                        <th>Yön</th>
                        <th>Giriş</th>
                        <th>Çıkış</th>
                        <th>Miktar</th>
                        <th>K/Z</th>
                        <th>Sebep</th>
                    </tr>
                </thead>
                <tbody id="tradesBody">
                    <tr>
                        <td colspan="7" style="text-align: center; color: #888;">Henüz işlem yok</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Performance Metrics -->
        <div class="performance-grid">
            <div class="perf-item">
                <div class="perf-label">Toplam İşlem</div>
                <div class="perf-value" id="perfTrades">0</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">Kazanan</div>
                <div class="perf-value" id="perfWins" style="color: #26a69a;">0</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">Kaybeden</div>
                <div class="perf-value" id="perfLosses" style="color: #ef5350;">0</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">Win Rate</div>
                <div class="perf-value" id="perfWinRate">0%</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">Max Drawdown</div>
                <div class="perf-value" id="perfMaxDD">0%</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">ROI</div>
                <div class="perf-value" id="perfROI">0%</div>
            </div>
        </div>
    </div>
    
    <!-- Toast -->
    <div class="toast" id="toast"></div>
    
    <script>
        // ================================================================
        // GLOBAL STATE
        // ================================================================
        let chart = null;
        let candleSeries = null;
        let isRunning = false;
        let updateInterval = null;
        let countdown = 5;
        
        // ================================================================
        // INITIALIZATION
        // ================================================================
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            loadInitialData();
            
            // Strategy description update
            document.getElementById('strategySelect').addEventListener('change', updateStrategyDesc);
        });
        
        function initChart() {
            const container = document.getElementById('priceChart');
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 350,
                layout: {
                    background: { type: 'solid', color: 'transparent' },
                    textColor: '#888'
                },
                grid: {
                    vertLines: { color: 'rgba(255,255,255,0.1)' },
                    horzLines: { color: 'rgba(255,255,255,0.1)' }
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal
                },
                rightPriceScale: {
                    borderColor: 'rgba(255,255,255,0.2)'
                },
                timeScale: {
                    borderColor: 'rgba(255,255,255,0.2)',
                    timeVisible: true
                }
            });
            
            candleSeries = chart.addCandlestickSeries({
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350'
            });
            
            // Responsive
            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth });
            });
        }
        
        // ================================================================
        // API FUNCTIONS
        // ================================================================
        async function apiCall(endpoint, method = 'GET', data = null) {
            try {
                const options = {
                    method: method,
                    headers: { 'Content-Type': 'application/json' }
                };
                if (data) options.body = JSON.stringify(data);
                
                const response = await fetch(endpoint, options);
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                showToast('Bağlantı hatası!', 'error');
                return null;
            }
        }
        
        async function loadInitialData() {
            const status = await apiCall('/api/status');
            if (status) {
                updateUI(status);
                loadChartData();
            }
        }
        
        async function loadChartData() {
            const data = await apiCall('/api/chart');
            if (data && data.candles) {
                const chartData = data.candles.map(c => ({
                    time: Math.floor(new Date(c.time).getTime() / 1000),
                    open: c.open,
                    high: c.high,
                    low: c.low,
                    close: c.close
                }));
                candleSeries.setData(chartData);
            }
        }
        
        // ================================================================
        // UI UPDATE FUNCTIONS
        // ================================================================
        function updateUI(status) {
            // Price
            const price = status.price;
            document.getElementById('currentPrice').textContent = '$' + price.current.toLocaleString('en-US', {minimumFractionDigits: 2});
            
            const changeEl = document.getElementById('priceChange');
            changeEl.textContent = (price.change >= 0 ? '+' : '') + price.change.toFixed(2) + '%';
            changeEl.className = 'price-change ' + (price.change >= 0 ? 'positive' : 'negative');
            
            // Signal
            const signal = status.signal;
            const signalBadge = document.getElementById('signalBadge');
            signalBadge.textContent = signal.signal;
            signalBadge.className = 'signal-badge signal-' + signal.signal;
            
            const confidence = signal.confidence || 50;
            const confFill = document.getElementById('confidenceFill');
            confFill.style.width = confidence + '%';
            confFill.style.background = signal.signal === 'LONG' ? '#26a69a' : 
                                        signal.signal === 'SHORT' ? '#ef5350' : '#ffc107';
            
            document.getElementById('signalReason').textContent = signal.reason || 'Sinyal bekleniyor...';
            
            // Balance
            const balance = status.balance;
            const initial = status.initial_balance;
            const pnl = balance - initial;
            const pnlPct = (pnl / initial * 100);
            
            document.getElementById('balance').textContent = '$' + balance.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('balancePnl').textContent = `PnL: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%)`;
            document.getElementById('balancePnl').style.color = pnl >= 0 ? '#26a69a' : '#ef5350';
            
            // Position
            const position = status.position;
            if (position) {
                document.getElementById('positionStatus').textContent = position.side + ' 📊';
                document.getElementById('positionStatus').style.color = position.side === 'LONG' ? '#26a69a' : '#ef5350';
                document.getElementById('positionInfo').classList.remove('hidden');
                document.getElementById('posEntry').textContent = '$' + position.entry_price.toLocaleString();
                document.getElementById('posPnl').textContent = (position.pnl >= 0 ? '+' : '') + '$' + position.pnl.toFixed(2);
                document.getElementById('posPnl').style.color = position.pnl >= 0 ? '#26a69a' : '#ef5350';
            } else {
                document.getElementById('positionStatus').textContent = 'Yok';
                document.getElementById('positionStatus').style.color = '#888';
                document.getElementById('positionInfo').classList.add('hidden');
            }
            
            // Performance
            const metrics = status.metrics;
            document.getElementById('winRate').textContent = metrics.win_rate + '%';
            document.getElementById('totalTrades').textContent = metrics.total_trades + ' işlem';
            document.getElementById('totalProfit').textContent = '$' + metrics.total_profit.toFixed(2);
            document.getElementById('totalLoss').textContent = '$' + metrics.total_loss.toFixed(2);
            document.getElementById('profitFactor').textContent = metrics.profit_factor.toFixed(2);
            
            // Performance Grid
            document.getElementById('perfTrades').textContent = metrics.total_trades;
            document.getElementById('perfWins').textContent = metrics.winning_trades;
            document.getElementById('perfLosses').textContent = metrics.losing_trades;
            document.getElementById('perfWinRate').textContent = metrics.win_rate + '%';
            document.getElementById('perfMaxDD').textContent = '-' + metrics.max_drawdown + '%';
            document.getElementById('perfROI').textContent = (metrics.roi >= 0 ? '+' : '') + metrics.roi + '%';
            document.getElementById('perfROI').style.color = metrics.roi >= 0 ? '#26a69a' : '#ef5350';
            
            // Strategy
            document.getElementById('activeStrategy').textContent = status.strategy_name;
            document.getElementById('strategySelect').value = status.strategy;
            
            // Trades
            updateTradesTable(status.trades);
            
            // Running state
            isRunning = status.is_running;
            updateRunningState();
            
            // Time
            document.getElementById('chartTime').textContent = 'Son: ' + price.timestamp;
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('tradesBody');
            
            if (!trades || trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #888;">Henüz işlem yok</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.map(t => `
                <tr>
                    <td>${t.timestamp.split(' ')[1] || t.timestamp}</td>
                    <td><span class="side-badge side-${t.side}">${t.side}</span></td>
                    <td>$${t.entry_price.toLocaleString()}</td>
                    <td>$${t.exit_price.toLocaleString()}</td>
                    <td>${t.quantity.toFixed(4)}</td>
                    <td class="${t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}</td>
                    <td>${t.reason}</td>
                </tr>
            `).join('');
        }
        
        function updateRunningState() {
            const btnStart = document.getElementById('btnStart');
            const btnStop = document.getElementById('btnStop');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (isRunning) {
                btnStart.disabled = true;
                btnStop.disabled = false;
                statusDot.className = 'status-dot running';
                statusText.textContent = 'Çalışıyor';
            } else {
                btnStart.disabled = false;
                btnStop.disabled = true;
                statusDot.className = 'status-dot stopped';
                statusText.textContent = 'Durduruldu';
            }
        }
        
        // ================================================================
        // CONTROL FUNCTIONS
        // ================================================================
        async function startBot() {
            const result = await apiCall('/api/start', 'POST');
            if (result && result.success) {
                isRunning = true;
                updateRunningState();
                showToast('Bot başlatıldı!', 'success');
                startUpdateLoop();
            }
        }
        
        async function stopBot() {
            const result = await apiCall('/api/stop', 'POST');
            if (result && result.success) {
                isRunning = false;
                updateRunningState();
                showToast('Bot durduruldu!', 'info');
                stopUpdateLoop();
            }
        }
        
        async function resetBot() {
            if (!confirm('Tüm veriler sıfırlanacak. Emin misiniz?')) return;
            
            const result = await apiCall('/api/reset', 'POST');
            if (result && result.success) {
                isRunning = false;
                updateRunningState();
                showToast('Bot sıfırlandı!', 'info');
                stopUpdateLoop();
                loadInitialData();
            }
        }
        
        function startUpdateLoop() {
            if (updateInterval) clearInterval(updateInterval);
            
            countdown = 5;
            updateInterval = setInterval(async () => {
                countdown--;
                document.getElementById('nextUpdate').textContent = countdown + 's';
                
                if (countdown <= 0) {
                    countdown = 5;
                    const result = await apiCall('/api/update', 'POST');
                    if (result) {
                        // Update price on chart
                        if (candleSeries) {
                            const time = Math.floor(new Date(result.timestamp).getTime() / 1000);
                            candleSeries.update({
                                time: time,
                                open: result.price * 0.999,
                                high: result.high || result.price * 1.001,
                                low: result.low || result.price * 0.999,
                                close: result.price
                            });
                        }
                        
                        // Update UI
                        const status = await apiCall('/api/status');
                        if (status) updateUI(status);
                        
                        // Trade notification
                        if (result.trade) {
                            const t = result.trade;
                            const msg = t.pnl >= 0 ? 
                                `✅ ${t.side} kapatıldı: +$${t.pnl.toFixed(2)}` :
                                `❌ ${t.side} kapatıldı: -$${Math.abs(t.pnl).toFixed(2)}`;
                            showToast(msg, t.pnl >= 0 ? 'success' : 'error');
                        }
                    }
                }
            }, 1000);
        }
        
        function stopUpdateLoop() {
            if (updateInterval) {
                clearInterval(updateInterval);
                updateInterval = null;
            }
            document.getElementById('nextUpdate').textContent = '-';
        }
        
        // ================================================================
        // SETTINGS FUNCTIONS
        // ================================================================
        async function saveSettings() {
            const balance = parseFloat(document.getElementById('inputBalance').value);
            const sl = parseFloat(document.getElementById('inputSL').value) / 100;
            const tp = parseFloat(document.getElementById('inputTP').value) / 100;
            
            if (balance < 100) {
                showToast('Minimum bakiye $100!', 'error');
                return;
            }
            
            const result = await apiCall('/api/settings', 'POST', {
                balance: balance,
                stop_loss: sl,
                take_profit: tp
            });
            
            if (result && result.success) {
                showToast('Ayarlar kaydedildi!', 'success');
                loadInitialData();
            }
        }
        
        async function applyStrategy() {
            const strategy = document.getElementById('strategySelect').value;
            
            const result = await apiCall('/api/strategy', 'POST', {
                strategy: strategy
            });
            
            if (result && result.success) {
                showToast('Strateji uygulandı: ' + result.strategy_name, 'success');
                loadInitialData();
            }
        }
        
        function updateStrategyDesc() {
            const select = document.getElementById('strategySelect');
            const descriptions = {
                'ultra_safe': '🛡️ Minimum risk, küçük pozisyonlar',
                'conservative': '🔒 Muhafazakar yaklaşım',
                'safe_trend': '📈 Güvenli trend takibi',
                'dip_buyer': '📉 Düşüşlerde alım',
                'balanced': '⚖️ Dengeli risk/ödül',
                'momentum': '🚀 Momentum takibi',
                'swing_trader': '🔄 Swing trading',
                'trend_surfer': '🏄 Trend sürme',
                'mean_reversion': '↩️ Ortalamaya dönüş',
                'breakout_confirmed': '✅ Onaylı kırılım',
                'breakout': '💥 Kırılım stratejisi',
                'aggressive': '⚡ Agresif trading',
                'volatility_hunter': '🎯 Volatilite avcısı',
                'news_reactor': '📰 Haber tepkisi',
                'scalper': '⏱️ Scalping',
                'grid_trader': '📊 Grid trading',
                'martingale_light': '🎲 Hafif martingale',
                'yolo': '🔥 Maksimum risk',
                'leverage_king': '👑 Yüksek kaldıraç',
                'all_in': '💎 Tüm sermaye'
            };
            document.getElementById('strategyDesc').textContent = descriptions[select.value] || '';
        }
        
        // ================================================================
        // BACKTEST FUNCTIONS
        // ================================================================
        async function runBacktest() {
            const strategy = document.getElementById('strategySelect').value;
            const months = parseInt(document.getElementById('backtestPeriod').value);
            
            showToast('Backtest başlatıldı...', 'info');
            
            const result = await apiCall('/api/backtest', 'POST', {
                strategy: strategy,
                months: months
            });
            
            if (result && !result.error) {
                document.getElementById('btReturn').textContent = (result.total_return >= 0 ? '+' : '') + result.total_return + '%';
                document.getElementById('btReturn').style.color = result.total_return >= 0 ? '#26a69a' : '#ef5350';
                document.getElementById('btTrades').textContent = result.total_trades;
                document.getElementById('btWinRate').textContent = result.win_rate + '%';
                document.getElementById('btMaxDD').textContent = '-' + result.max_drawdown + '%';
                document.getElementById('btPF').textContent = result.profit_factor;
                document.getElementById('btVsHodl').textContent = (result.vs_buy_hold >= 0 ? '+' : '') + result.vs_buy_hold + '%';
                document.getElementById('btVsHodl').style.color = result.vs_buy_hold >= 0 ? '#26a69a' : '#ef5350';
                
                document.getElementById('backtestResult').classList.add('show');
                showToast('Backtest tamamlandı!', 'success');
            } else {
                showToast('Backtest hatası: ' + (result.error || 'Bilinmeyen'), 'error');
            }
        }
        
        // ================================================================
        // UTILITY FUNCTIONS
        // ================================================================
        function showToast(message, type = 'info') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast ' + type + ' show';
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }
    </script>
</body>
</html>
'''

# ================================================================
# FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    bot = get_bot()
    return jsonify(bot.get_status())

@app.route('/api/update', methods=['POST'])
def api_update():
    bot = get_bot()
    result = bot.update()
    return jsonify(result)

@app.route('/api/start', methods=['POST'])
def api_start():
    bot = get_bot()
    result = bot.start()
    return jsonify(result)

@app.route('/api/stop', methods=['POST'])
def api_stop():
    bot = get_bot()
    result = bot.stop()
    return jsonify(result)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    bot = get_bot()
    result = bot.reset()
    return jsonify(result)

@app.route('/api/settings', methods=['POST'])
def api_settings():
    bot = get_bot()
    data = request.json
    
    if 'balance' in data:
        # Initial balance da güncelle (reset için)
        if bot.settings.get('balance') == bot.settings.get('initial_balance', 10000):
            bot.settings.set('initial_balance', data['balance'])
        bot.settings.set('balance', data['balance'])
    
    if 'stop_loss' in data:
        bot.settings.set('stop_loss', data['stop_loss'])
    
    if 'take_profit' in data:
        bot.settings.set('take_profit', data['take_profit'])
    
    return jsonify({'success': True, 'message': 'Ayarlar kaydedildi'})

@app.route('/api/strategy', methods=['POST'])
def api_strategy():
    bot = get_bot()
    data = request.json
    strategy = data.get('strategy', 'balanced')
    
    if strategy in STRATEGIES:
        bot.settings.set_strategy(strategy)
        return jsonify({
            'success': True, 
            'strategy': strategy,
            'strategy_name': STRATEGIES[strategy]['name']
        })
    
    return jsonify({'success': False, 'message': 'Geçersiz strateji'})

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    bot = get_bot()
    data = request.json
    strategy = data.get('strategy', 'balanced')
    months = data.get('months', 1)
    
    result = bot.run_backtest(strategy, months)
    return jsonify(result)

@app.route('/api/chart')
def api_chart():
    bot = get_bot()
    data = bot.get_chart_data(200)
    return jsonify(data)

@app.route('/api/trades')
def api_trades():
    bot = get_bot()
    trades = bot.db.get_trades(50)
    return jsonify({'trades': trades})

# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*60)
    print("🚀 BTC BOT PRO v5.0 - TRADING PANEL")
    print("="*60)
    print("\n✅ Sistem başlatılıyor...")
    print("📊 30 teknik indikatör aktif")
    print("📈 20 strateji hazır")
    print("💾 SQLite + JSON veri saklama")
    print("\n🌐 Dashboard: http://127.0.0.1:5000")
    print("\n⚠️  Durdurmak için CTRL+C\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()
