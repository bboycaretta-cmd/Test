"""
BTC Bot Pro - Core Modülü v5.0
FAZA 1-7 TAMAMLANDI

Modüller:
- database: SQLite veritabanı yönetimi
- logger: Gelişmiş loglama sistemi  
- config: Konfigürasyon yönetimi (20 strateji)
- backtest: Event-driven backtest engine
- optimization: Walk-forward, Monte Carlo, Stress testing
- risk: Risk metrikleri ve pozisyon yönetimi
- features: 80+ teknik indikatör
- ml_model: GRU/LSTM/Attention modelleri
- genetic: Genetik algoritma optimizasyonu
- regime: Market regime detection
- enrichment: Data enrichment (funding, sentiment)
- dashboard: Web dashboard + REST API
"""

# FAZA 1: Altyapı
from .database import Database, db
from .logger import (
    BotLogger, TradeLogger, 
    logger, trade_logger,
    log_execution_time, log_exceptions
)
from .config import (
    Config, ConfigManager,
    config, config_manager,
    get_strategy, STRATEGIES
)

# FAZA 2: Backtest
from .backtest import (
    BacktestEngine, BacktestResult,
    MarketData, Position, Trade,
    PositionSide, SlippageModel, SpreadModel
)
from .optimization import (
    WalkForwardOptimizer, WalkForwardResult,
    MonteCarloSimulator, MonteCarloResult,
    StressTester, StressTestResult,
    print_backtest_report
)

# FAZA 3: Risk
from .risk import (
    RiskCalculator, RiskMetrics,
    PositionSizer, PositionSize, PositionSizingMethod,
    RiskManager, RiskLimits, RiskLevel
)

# FAZA 4: ML
from .features import (
    FeatureEngineer, TechnicalIndicators,
    create_sequences, normalize_features
)
from .ml_model import (
    GRUModel, LSTMModel, AttentionModel, EnsembleModel,
    ModelConfig, create_model, generate_signal,
    TF_AVAILABLE, OPTUNA_AVAILABLE
)

# FAZA 5: Strateji Optimizasyonu
from .genetic import (
    GeneticOptimizer, GAConfig,
    ParameterSpace, OptimizationResult,
    StrategyOptimizer
)
from .regime import (
    RegimeDetector, MarketRegime, RegimeState,
    AdaptiveStrategySelector, RegimeAnalysis,
    VolatilityRegime
)

# FAZA 6: Data Enrichment
from .enrichment import (
    DataAggregator, MockDataProvider,
    FundingData, OpenInterestData, SentimentData,
    EnrichedSignalGenerator, EnrichedData,
    SentimentLevel
)

# FAZA 7: Web Dashboard
from .dashboard import (
    BotDashboard, create_dashboard,
    FLASK_AVAILABLE
)

__version__ = "5.0.0"
__all__ = [
    # Database
    'Database', 'db',
    
    # Logger
    'logger', 'trade_logger',
    
    # Config
    'config', 'get_strategy', 'STRATEGIES',
    
    # Backtest
    'BacktestEngine', 'BacktestResult', 'PositionSide',
    
    # Optimization
    'WalkForwardOptimizer', 'MonteCarloSimulator', 'StressTester',
    
    # Risk
    'RiskCalculator', 'PositionSizer', 'RiskManager',
    
    # Features
    'FeatureEngineer', 'TechnicalIndicators',
    
    # ML
    'GRUModel', 'LSTMModel', 'create_model', 'generate_signal',
    'TF_AVAILABLE',
    
    # Genetic
    'GeneticOptimizer', 'StrategyOptimizer',
    
    # Regime
    'RegimeDetector', 'MarketRegime', 'AdaptiveStrategySelector',
    
    # Enrichment
    'DataAggregator', 'EnrichedSignalGenerator',
    
    # Dashboard
    'BotDashboard', 'create_dashboard', 'FLASK_AVAILABLE'
]
