"""
BTC Bot Pro - Loglama Modülü
FAZA 1.2: Gelişmiş loglama sistemi

Özellikler:
- Log seviyeleri (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Ayrı log dosyaları (trade.log, error.log, system.log)
- Log rotation (max 10MB, 5 yedek)
- JSON formatında structured logging
- Konsol ve dosya output
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import traceback

# Log dizini
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log dosyaları
SYSTEM_LOG = os.path.join(LOG_DIR, "system.log")
TRADE_LOG = os.path.join(LOG_DIR, "trade.log")
ERROR_LOG = os.path.join(LOG_DIR, "error.log")
DEBUG_LOG = os.path.join(LOG_DIR, "debug.log")

# Log ayarları
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


class JsonFormatter(logging.Formatter):
    """JSON formatında log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Ekstra veriler
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        
        # Exception bilgisi
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Renkli konsol formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Zaman formatı
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Seviye kısaltması
        level_short = {
            'DEBUG': 'DBG',
            'INFO': 'INF',
            'WARNING': 'WRN',
            'ERROR': 'ERR',
            'CRITICAL': 'CRT'
        }.get(record.levelname, record.levelname[:3])
        
        # Mesaj
        msg = record.getMessage()
        
        return f"{color}[{timestamp}] [{level_short}]{reset} {msg}"


class BotLogger:
    """Ana logger sınıfı"""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    def __init__(self, name: str = 'btc_bot', level: str = 'INFO'):
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Logger'ı kur"""
        if self.name in self._loggers:
            return self._loggers[self.name]
        
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # En düşük seviye
        logger.handlers = []  # Mevcut handler'ları temizle
        
        # Konsol handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
        # System log (JSON)
        system_handler = RotatingFileHandler(
            SYSTEM_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
        )
        system_handler.setLevel(logging.INFO)
        system_handler.setFormatter(JsonFormatter())
        logger.addHandler(system_handler)
        
        # Error log (JSON)
        error_handler = RotatingFileHandler(
            ERROR_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        logger.addHandler(error_handler)
        
        # Debug log (sadece DEBUG mode'da)
        if self.level <= logging.DEBUG:
            debug_handler = RotatingFileHandler(
                DEBUG_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(JsonFormatter())
            logger.addHandler(debug_handler)
        
        self._loggers[self.name] = logger
        return logger
    
    def _log_with_data(self, level: int, msg: str, data: Optional[Dict] = None):
        """Ekstra veri ile log"""
        record = logging.LogRecord(
            name=self.name,
            level=level,
            pathname='',
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None
        )
        if data:
            record.extra_data = data
        self.logger.handle(record)
    
    def debug(self, msg: str, data: Optional[Dict] = None):
        """Debug log"""
        if data:
            self._log_with_data(logging.DEBUG, msg, data)
        else:
            self.logger.debug(msg)
    
    def info(self, msg: str, data: Optional[Dict] = None):
        """Info log"""
        if data:
            self._log_with_data(logging.INFO, msg, data)
        else:
            self.logger.info(msg)
    
    def warning(self, msg: str, data: Optional[Dict] = None):
        """Warning log"""
        if data:
            self._log_with_data(logging.WARNING, msg, data)
        else:
            self.logger.warning(msg)
    
    def error(self, msg: str, data: Optional[Dict] = None, exc_info: bool = False):
        """Error log"""
        if data:
            self._log_with_data(logging.ERROR, msg, data)
        else:
            self.logger.error(msg, exc_info=exc_info)
    
    def critical(self, msg: str, data: Optional[Dict] = None, exc_info: bool = True):
        """Critical log"""
        if data:
            self._log_with_data(logging.CRITICAL, msg, data)
        else:
            self.logger.critical(msg, exc_info=exc_info)
    
    def exception(self, msg: str, data: Optional[Dict] = None):
        """Exception log (traceback dahil)"""
        self.logger.exception(msg)


class TradeLogger(BotLogger):
    """Trade-specific logger"""
    
    def __init__(self):
        super().__init__('trade', 'INFO')
        self._setup_trade_handler()
    
    def _setup_trade_handler(self):
        """Trade log handler ekle"""
        trade_handler = RotatingFileHandler(
            TRADE_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(trade_handler)
    
    def trade_open(self, account_id: str, side: str, price: float, 
                   quantity: float, strategy: str, **kwargs):
        """Trade açılış logu"""
        self.info(f"TRADE OPEN: {side} @ ${price:,.2f}", {
            'type': 'TRADE_OPEN',
            'account_id': account_id,
            'side': side,
            'price': price,
            'quantity': quantity,
            'strategy': strategy,
            **kwargs
        })
    
    def trade_close(self, account_id: str, side: str, entry_price: float,
                    exit_price: float, pnl: float, reason: str, **kwargs):
        """Trade kapanış logu"""
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        self.info(f"TRADE CLOSE: {side} | Entry: ${entry_price:,.2f} | Exit: ${exit_price:,.2f} | PnL: {pnl_str} | {reason}", {
            'type': 'TRADE_CLOSE',
            'account_id': account_id,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            **kwargs
        })
    
    def signal(self, signal_type: str, price: float, prediction: float,
               confidence: float, strategy: str, **kwargs):
        """Sinyal logu"""
        self.info(f"SIGNAL: {signal_type} @ ${price:,.2f} | Pred: {prediction:+.2f}% | Conf: {confidence:.0f}%", {
            'type': 'SIGNAL',
            'signal_type': signal_type,
            'price': price,
            'prediction': prediction,
            'confidence': confidence,
            'strategy': strategy,
            **kwargs
        })
    
    def sl_hit(self, account_id: str, price: float, sl_price: float, pnl: float):
        """Stop-loss tetiklendi"""
        self.warning(f"STOP-LOSS HIT @ ${price:,.2f} (SL: ${sl_price:,.2f}) | PnL: ${pnl:,.2f}", {
            'type': 'SL_HIT',
            'account_id': account_id,
            'price': price,
            'sl_price': sl_price,
            'pnl': pnl
        })
    
    def tp_hit(self, account_id: str, price: float, tp_price: float, pnl: float):
        """Take-profit tetiklendi"""
        self.info(f"TAKE-PROFIT HIT @ ${price:,.2f} (TP: ${tp_price:,.2f}) | PnL: ${pnl:,.2f}", {
            'type': 'TP_HIT',
            'account_id': account_id,
            'price': price,
            'tp_price': tp_price,
            'pnl': pnl
        })


def log_execution_time(func):
    """Fonksiyon çalışma süresini logla (decorator)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start).total_seconds()
        
        if elapsed > 1:  # 1 saniyeden uzun sürenleri logla
            logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        
        return result
    return wrapper


def log_exceptions(func):
    """Exception'ları otomatik logla (decorator)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper


# Global logger instances
logger = BotLogger('btc_bot', 'INFO')
trade_logger = TradeLogger()


# ==================== YARDIMCI FONKSİYONLAR ====================

def set_log_level(level: str):
    """Log seviyesini değiştir"""
    global logger
    logger = BotLogger('btc_bot', level)

def get_log_files() -> Dict[str, str]:
    """Log dosya yollarını getir"""
    return {
        'system': SYSTEM_LOG,
        'trade': TRADE_LOG,
        'error': ERROR_LOG,
        'debug': DEBUG_LOG
    }

def clear_logs():
    """Tüm log dosyalarını temizle"""
    for log_file in [SYSTEM_LOG, TRADE_LOG, ERROR_LOG, DEBUG_LOG]:
        if os.path.exists(log_file):
            open(log_file, 'w').close()

def get_recent_logs(log_type: str = 'system', lines: int = 100) -> list:
    """Son logları getir"""
    log_files = {
        'system': SYSTEM_LOG,
        'trade': TRADE_LOG,
        'error': ERROR_LOG,
        'debug': DEBUG_LOG
    }
    
    log_file = log_files.get(log_type, SYSTEM_LOG)
    
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return all_lines[-lines:]
    except:
        return []


# ==================== TEST ====================
if __name__ == "__main__":
    print("Logger test ediliyor...\n")
    
    # Temel loglar
    logger.debug("Bu bir debug mesajı")
    logger.info("Bu bir info mesajı")
    logger.warning("Bu bir warning mesajı")
    logger.error("Bu bir error mesajı")
    
    # Data ile log
    logger.info("Trade açıldı", {
        'side': 'LONG',
        'price': 95000,
        'quantity': 0.1
    })
    
    # Trade logger
    trade_logger.trade_open(
        account_id='test_001',
        side='LONG',
        price=95000,
        quantity=0.1,
        strategy='dengeli'
    )
    
    trade_logger.signal(
        signal_type='LONG',
        price=95000,
        prediction=1.5,
        confidence=75,
        strategy='momentum'
    )
    
    trade_logger.trade_close(
        account_id='test_001',
        side='LONG',
        entry_price=95000,
        exit_price=96000,
        pnl=100,
        reason='TP'
    )
    
    # Decorator test
    @log_execution_time
    def slow_function():
        import time
        time.sleep(0.1)
        return "done"
    
    slow_function()
    
    print("\n✓ Logger testi başarılı!")
    print(f"\nLog dosyaları: {LOG_DIR}")
