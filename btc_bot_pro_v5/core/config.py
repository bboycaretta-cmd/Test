"""
BTC Bot Pro - Konfigürasyon Modülü
FAZA 1.3: Gelişmiş config yönetimi

Özellikler:
- YAML/JSON config dosyası
- Environment variables desteği
- Config validation
- Default değerler
- Runtime config değişikliği
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Config dizini
CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
CONFIG_JSON = os.path.join(CONFIG_DIR, "config.json")


# ================================================================
# DATACLASS CONFIG YAPISI
# ================================================================

@dataclass
class DatabaseConfig:
    """Veritabanı ayarları"""
    path: str = "data/btc_bot.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    cleanup_days: int = 90


@dataclass
class APIConfig:
    """API ayarları"""
    binance_url: str = "https://api.binance.com/api/v3"
    binance_ws: str = "wss://stream.binance.com:9443/ws"
    timeout: int = 10
    max_retry: int = 3
    rate_limit_per_minute: int = 1200


@dataclass
class ModelConfig:
    """ML model ayarları"""
    coin: str = "BTCUSDT"
    timeframe: str = "1h"
    lookback_window: int = 48
    prediction_horizon: int = 4
    threshold: float = 0.5
    data_months: int = 12
    train_ratio: float = 0.65
    val_ratio: float = 0.20
    test_ratio: float = 0.15
    epochs: int = 150
    batch_size: int = 64
    patience: int = 15
    learning_rate: float = 0.0005
    gru_units_1: int = 64
    gru_units_2: int = 32
    dense_units: int = 16
    dropout: float = 0.35
    l2_reg: float = 0.001
    random_seed: int = 42


@dataclass
class TradingConfig:
    """Trading ayarları"""
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    spread: float = 0.0002
    stop_loss: float = 0.02
    take_profit: float = 0.03
    trailing_stop: float = 0.015
    trailing_enabled: bool = True
    max_daily_loss: float = 0.05
    max_position_size: float = 0.50
    risk_per_trade: float = 0.02


@dataclass
class BacktestConfig:
    """Backtest ayarları"""
    default_months: int = 3
    initial_balance: float = 10000.0
    random_seed: int = 42
    use_spread: bool = True
    use_slippage: bool = True
    realistic_fills: bool = True
    monte_carlo_runs: int = 100
    walk_forward_windows: int = 5


@dataclass
class RiskConfig:
    """Risk yönetimi ayarları"""
    max_drawdown_limit: float = 0.20
    consecutive_loss_limit: int = 5
    daily_loss_limit: float = 0.05
    var_confidence: float = 0.95
    position_sizing_method: str = "fixed_fractional"  # kelly, optimal_f, fixed_fractional
    kelly_fraction: float = 0.25


@dataclass
class NotificationConfig:
    """Bildirim ayarları"""
    telegram_enabled: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""
    email_enabled: bool = False
    email_smtp: str = ""
    email_from: str = ""
    email_to: str = ""
    notify_on_trade: bool = True
    notify_on_signal: bool = False
    notify_on_error: bool = True


@dataclass
class LogConfig:
    """Log ayarları"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    console_output: bool = True
    file_output: bool = True
    max_size_mb: int = 10
    backup_count: int = 5
    json_format: bool = True


@dataclass 
class UIConfig:
    """UI ayarları"""
    currency: str = "USD"  # USD, TRY
    theme: str = "default"
    refresh_rate_ms: int = 500
    chart_candles: int = 50
    table_rows: int = 20


# ================================================================
# STRATEJI TANIMLARI
# ================================================================

STRATEGIES = {
    # DÜŞÜK RİSK
    'ultra_safe': {
        'name': 'Ultra Güvenli',
        'description': 'Çok düşük risk, nadir işlem',
        'stop_loss': 0.01, 'take_profit': 0.015,
        'threshold': 1.2, 'position_size': 0.2,
        'risk_score': 1,
    },
    'conservative': {
        'name': 'Konservatif',
        'description': 'Düşük risk, az işlem',
        'stop_loss': 0.015, 'take_profit': 0.02,
        'threshold': 1.0, 'position_size': 0.3,
        'risk_score': 2,
    },
    'safe_trend': {
        'name': 'Güvenli Trend',
        'description': 'Sadece güçlü trendlerde',
        'stop_loss': 0.02, 'take_profit': 0.03,
        'threshold': 0.9, 'position_size': 0.35,
        'risk_score': 2,
    },
    'dip_buyer': {
        'name': 'Dip Alıcı',
        'description': 'Sadece düşüşte al',
        'stop_loss': 0.025, 'take_profit': 0.035,
        'threshold': 0.8, 'position_size': 0.3,
        'risk_score': 2, 'long_only': True,
    },
    
    # ORTA RİSK
    'balanced': {
        'name': 'Dengeli',
        'description': 'Orta risk, dengeli getiri',
        'stop_loss': 0.02, 'take_profit': 0.03,
        'threshold': 0.5, 'position_size': 0.5,
        'risk_score': 5,
    },
    'swing': {
        'name': 'Swing Trader',
        'description': 'Orta vadeli dalgalanmalar',
        'stop_loss': 0.025, 'take_profit': 0.04,
        'threshold': 0.6, 'position_size': 0.45,
        'risk_score': 5,
    },
    'momentum': {
        'name': 'Momentum',
        'description': 'Momentum takibi',
        'stop_loss': 0.02, 'take_profit': 0.035,
        'threshold': 0.55, 'position_size': 0.5,
        'risk_score': 5,
    },
    'breakout': {
        'name': 'Breakout',
        'description': 'Kırılım stratejisi',
        'stop_loss': 0.022, 'take_profit': 0.038,
        'threshold': 0.65, 'position_size': 0.45,
        'risk_score': 5,
    },
    'mean_reversion': {
        'name': 'Mean Reversion',
        'description': 'Ortalamaya dönüş',
        'stop_loss': 0.018, 'take_profit': 0.028,
        'threshold': 0.5, 'position_size': 0.4,
        'risk_score': 4,
    },
    'rsi_trader': {
        'name': 'RSI Trader',
        'description': 'RSI bazlı işlem',
        'stop_loss': 0.02, 'take_profit': 0.032,
        'threshold': 0.55, 'position_size': 0.45,
        'risk_score': 4,
    },
    'macd_cross': {
        'name': 'MACD Cross',
        'description': 'MACD kesişim',
        'stop_loss': 0.022, 'take_profit': 0.034,
        'threshold': 0.5, 'position_size': 0.5,
        'risk_score': 5,
    },
    'bollinger': {
        'name': 'Bollinger Bands',
        'description': 'BB bazlı işlem',
        'stop_loss': 0.02, 'take_profit': 0.03,
        'threshold': 0.45, 'position_size': 0.45,
        'risk_score': 4,
    },
    
    # YÜKSEK RİSK
    'aggressive': {
        'name': 'Agresif',
        'description': 'Yüksek risk, yüksek getiri',
        'stop_loss': 0.03, 'take_profit': 0.05,
        'threshold': 0.3, 'position_size': 0.7,
        'risk_score': 7,
    },
    'trend_surfer': {
        'name': 'Trend Surfer',
        'description': 'Güçlü trendlerde agresif',
        'stop_loss': 0.035, 'take_profit': 0.06,
        'threshold': 0.4, 'position_size': 0.65,
        'risk_score': 7,
    },
    'volatility': {
        'name': 'Volatilite Avcısı',
        'description': 'Yüksek volatilitede işlem',
        'stop_loss': 0.04, 'take_profit': 0.07,
        'threshold': 0.35, 'position_size': 0.6,
        'risk_score': 8,
    },
    'grid': {
        'name': 'Grid Trader',
        'description': 'Grid trading',
        'stop_loss': 0.025, 'take_profit': 0.035,
        'threshold': 0.25, 'position_size': 0.55,
        'risk_score': 6,
    },
    
    # ÇOK YÜKSEK RİSK
    'scalper': {
        'name': 'Scalper',
        'description': 'Hızlı al-sat, çok işlem',
        'stop_loss': 0.01, 'take_profit': 0.015,
        'threshold': 0.2, 'position_size': 0.6,
        'risk_score': 8,
    },
    'yolo': {
        'name': 'YOLO',
        'description': 'Maksimum risk',
        'stop_loss': 0.05, 'take_profit': 0.1,
        'threshold': 0.15, 'position_size': 0.9,
        'risk_score': 10,
    },
    'leverage_3x': {
        'name': 'Leverage 3x',
        'description': '3x kaldıraç simülasyonu',
        'stop_loss': 0.02, 'take_profit': 0.06,
        'threshold': 0.25, 'position_size': 0.8,
        'risk_score': 9, 'leverage': 3,
    },
    'ai_adaptive': {
        'name': 'AI Adaptive',
        'description': 'ML optimize strateji',
        'stop_loss': 0.022, 'take_profit': 0.038,
        'threshold': 0.45, 'position_size': 0.55,
        'risk_score': 6, 'adaptive': True,
    },
}


# ================================================================
# ANA CONFIG SINIFI
# ================================================================

@dataclass
class Config:
    """Ana konfigürasyon sınıfı"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Metadata
    version: str = "5.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfigManager:
    """Config yöneticisi"""
    
    _instance = None
    _config: Config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> Config:
        """Config'i yükle"""
        # Önce YAML dene
        if os.path.exists(CONFIG_FILE):
            return self._load_yaml()
        
        # Sonra JSON dene
        if os.path.exists(CONFIG_JSON):
            return self._load_json()
        
        # Default config oluştur ve kaydet
        config = Config()
        self._save_yaml(config)
        return config
    
    def _load_yaml(self) -> Config:
        """YAML'dan yükle"""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return self._dict_to_config(data)
        except Exception as e:
            print(f"YAML yükleme hatası: {e}")
            return Config()
    
    def _load_json(self) -> Config:
        """JSON'dan yükle"""
        try:
            with open(CONFIG_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._dict_to_config(data)
        except Exception as e:
            print(f"JSON yükleme hatası: {e}")
            return Config()
    
    def _dict_to_config(self, data: Dict) -> Config:
        """Dict'i Config'e çevir"""
        config = Config()
        
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'trading' in data:
            config.trading = TradingConfig(**data['trading'])
        if 'backtest' in data:
            config.backtest = BacktestConfig(**data['backtest'])
        if 'risk' in data:
            config.risk = RiskConfig(**data['risk'])
        if 'notification' in data:
            config.notification = NotificationConfig(**data['notification'])
        if 'log' in data:
            config.log = LogConfig(**data['log'])
        if 'ui' in data:
            config.ui = UIConfig(**data['ui'])
        if 'version' in data:
            config.version = data['version']
        
        return config
    
    def _save_yaml(self, config: Config):
        """YAML'a kaydet"""
        data = asdict(config)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def _save_json(self, config: Config):
        """JSON'a kaydet"""
        data = asdict(config)
        with open(CONFIG_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @property
    def config(self) -> Config:
        """Config'i getir"""
        return self._config
    
    def save(self, format: str = 'yaml'):
        """Config'i kaydet"""
        self._config.updated_at = datetime.now().isoformat()
        
        if format == 'yaml':
            self._save_yaml(self._config)
        else:
            self._save_json(self._config)
    
    def reload(self):
        """Config'i yeniden yükle"""
        self._config = self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Değer getir (dot notation: 'model.threshold')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Değer ayarla (dot notation)"""
        keys = key.split('.')
        obj = self._config
        
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return False
        
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)
            return True
        
        return False
    
    def get_strategy(self, name: str) -> Optional[Dict]:
        """Strateji ayarlarını getir"""
        return STRATEGIES.get(name)
    
    def get_all_strategies(self) -> Dict:
        """Tüm stratejileri getir"""
        return STRATEGIES
    
    def validate(self) -> List[str]:
        """Config'i doğrula, hataları döndür"""
        errors = []
        
        # Model validasyonu
        if self._config.model.threshold < 0 or self._config.model.threshold > 2:
            errors.append("model.threshold 0-2 arasında olmalı")
        
        if self._config.model.train_ratio + self._config.model.val_ratio + self._config.model.test_ratio != 1.0:
            errors.append("train_ratio + val_ratio + test_ratio = 1.0 olmalı")
        
        # Trading validasyonu
        if self._config.trading.stop_loss >= self._config.trading.take_profit:
            errors.append("stop_loss < take_profit olmalı")
        
        if self._config.trading.max_position_size > 1.0:
            errors.append("max_position_size <= 1.0 olmalı")
        
        # Risk validasyonu
        if self._config.risk.max_drawdown_limit > 0.5:
            errors.append("max_drawdown_limit <= 0.5 olmalı")
        
        return errors


# ================================================================
# GLOBAL CONFIG INSTANCE
# ================================================================

config_manager = ConfigManager()
config = config_manager.config


# ================================================================
# YARDIMCI FONKSİYONLAR
# ================================================================

def get_config() -> Config:
    """Global config'i getir"""
    return config_manager.config

def get_strategy(name: str) -> Optional[Dict]:
    """Strateji getir"""
    return config_manager.get_strategy(name)

def get_all_strategies() -> Dict:
    """Tüm stratejileri getir"""
    return STRATEGIES

def save_config(format: str = 'yaml'):
    """Config'i kaydet"""
    config_manager.save(format)

def reload_config():
    """Config'i yeniden yükle"""
    config_manager.reload()


# ================================================================
# ENVIRONMENT VARIABLES
# ================================================================

def load_env_config():
    """Environment variable'lardan config yükle"""
    env_mappings = {
        'BTC_BOT_LOG_LEVEL': ('log.level', str),
        'BTC_BOT_CURRENCY': ('ui.currency', str),
        'BTC_BOT_TELEGRAM_TOKEN': ('notification.telegram_token', str),
        'BTC_BOT_TELEGRAM_CHAT': ('notification.telegram_chat_id', str),
        'BTC_BOT_INITIAL_BALANCE': ('trading.initial_balance', float),
    }
    
    for env_key, (config_key, type_fn) in env_mappings.items():
        value = os.environ.get(env_key)
        if value:
            try:
                config_manager.set(config_key, type_fn(value))
            except:
                pass


# Environment'tan yükle
load_env_config()


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Config test ediliyor...\n")
    
    # Config'i göster
    print(f"Version: {config.version}")
    print(f"Model threshold: {config.model.threshold}")
    print(f"Trading SL: {config.trading.stop_loss}")
    print(f"Log level: {config.log.level}")
    
    # Strateji test
    strat = get_strategy('balanced')
    print(f"\nBalanced strateji: {strat}")
    
    # Validation
    errors = config_manager.validate()
    if errors:
        print(f"\nValidation hataları: {errors}")
    else:
        print("\n✓ Config geçerli!")
    
    # Kaydet
    save_config('yaml')
    print(f"\nConfig kaydedildi: {CONFIG_FILE}")
    
    print("\n✓ Config testi başarılı!")
