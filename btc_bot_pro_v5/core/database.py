"""
BTC Bot Pro - Veritabanı Modülü
FAZA 1.1: SQLite veritabanı yönetimi

Tablolar:
- trades: İşlem kayıtları
- signals: Sinyal geçmişi
- performance: Günlük performans
- market_data: Fiyat cache
- accounts: Hesap bilgileri
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# Veritabanı dosya yolu
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "btc_bot.db")


class Database:
    """SQLite veritabanı yöneticisi"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = DB_PATH
        self._create_tables()
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Context manager ile connection yönetimi"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Dict-like erişim
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _create_tables(self):
        """Tabloları oluştur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT DEFAULT 'BTCUSDT',
                    side TEXT NOT NULL,  -- LONG/SHORT
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',  -- OPEN/CLOSED
                    exit_reason TEXT,  -- SIGNAL/SL/TP/MANUAL
                    strategy TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    notes TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Signals tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT DEFAULT 'BTCUSDT',
                    signal_type TEXT NOT NULL,  -- LONG/SHORT/BEKLE
                    prediction REAL,
                    confidence REAL,
                    price REAL NOT NULL,
                    strategy TEXT,
                    indicators TEXT,  -- JSON (RSI, MACD, vb.)
                    executed INTEGER DEFAULT 0,  -- 0/1
                    trade_id INTEGER,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            """)
            
            # Performance tablosu (günlük özet)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    starting_balance REAL,
                    ending_balance REAL,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL,
                    profit_factor REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    best_trade REAL,
                    worst_trade REAL,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(account_id, date)
                )
            """)
            
            # Market data cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Accounts tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy TEXT DEFAULT 'dengeli',
                    currency TEXT DEFAULT 'USD',
                    initial_balance REAL DEFAULT 10000,
                    current_balance REAL DEFAULT 10000,
                    total_pnl REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    settings TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # İndeksler
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_account_date ON performance(account_id, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol, timeframe, timestamp)")
    
    # ==================== TRADE İŞLEMLERİ ====================
    
    def insert_trade(self, trade: Dict) -> int:
        """Yeni trade ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    account_id, timestamp, symbol, side, entry_price,
                    quantity, status, strategy, entry_time, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('account_id'),
                trade.get('timestamp', datetime.now().isoformat()),
                trade.get('symbol', 'BTCUSDT'),
                trade.get('side'),
                trade.get('entry_price'),
                trade.get('quantity'),
                'OPEN',
                trade.get('strategy'),
                trade.get('entry_time', datetime.now().isoformat()),
                json.dumps(trade.get('metadata', {}))
            ))
            return cursor.lastrowid
    
    def close_trade(self, trade_id: int, exit_data: Dict) -> bool:
        """Trade'i kapat"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    pnl = ?,
                    pnl_percent = ?,
                    commission = ?,
                    slippage = ?,
                    status = 'CLOSED',
                    exit_reason = ?,
                    exit_time = ?
                WHERE id = ?
            """, (
                exit_data.get('exit_price'),
                exit_data.get('pnl'),
                exit_data.get('pnl_percent'),
                exit_data.get('commission', 0),
                exit_data.get('slippage', 0),
                exit_data.get('exit_reason'),
                exit_data.get('exit_time', datetime.now().isoformat()),
                trade_id
            ))
            return cursor.rowcount > 0
    
    def get_trades(self, account_id: str, limit: int = 100, 
                   status: str = None, start_date: str = None) -> List[Dict]:
        """Trade'leri getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE account_id = ?"
            params = [account_id]
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_open_trade(self, account_id: str) -> Optional[Dict]:
        """Açık pozisyonu getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades 
                WHERE account_id = ? AND status = 'OPEN'
                ORDER BY timestamp DESC LIMIT 1
            """, (account_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== SIGNAL İŞLEMLERİ ====================
    
    def insert_signal(self, signal: Dict) -> int:
        """Yeni sinyal ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, signal_type, prediction,
                    confidence, price, strategy, indicators, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get('timestamp', datetime.now().isoformat()),
                signal.get('symbol', 'BTCUSDT'),
                signal.get('signal_type'),
                signal.get('prediction'),
                signal.get('confidence'),
                signal.get('price'),
                signal.get('strategy'),
                json.dumps(signal.get('indicators', {})),
                json.dumps(signal.get('metadata', {}))
            ))
            return cursor.lastrowid
    
    def get_signals(self, limit: int = 100, signal_type: str = None) -> List[Dict]:
        """Sinyalleri getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if signal_type:
                cursor.execute("""
                    SELECT * FROM signals 
                    WHERE signal_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (signal_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM signals 
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== PERFORMANCE İŞLEMLERİ ====================
    
    def save_daily_performance(self, perf: Dict) -> bool:
        """Günlük performansı kaydet"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance (
                    account_id, date, starting_balance, ending_balance,
                    pnl, pnl_percent, total_trades, winning_trades,
                    losing_trades, win_rate, max_drawdown, sharpe_ratio,
                    profit_factor, avg_win, avg_loss, best_trade, worst_trade, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                perf.get('account_id'),
                perf.get('date', datetime.now().strftime('%Y-%m-%d')),
                perf.get('starting_balance'),
                perf.get('ending_balance'),
                perf.get('pnl', 0),
                perf.get('pnl_percent', 0),
                perf.get('total_trades', 0),
                perf.get('winning_trades', 0),
                perf.get('losing_trades', 0),
                perf.get('win_rate', 0),
                perf.get('max_drawdown', 0),
                perf.get('sharpe_ratio'),
                perf.get('profit_factor'),
                perf.get('avg_win'),
                perf.get('avg_loss'),
                perf.get('best_trade'),
                perf.get('worst_trade'),
                json.dumps(perf.get('metadata', {}))
            ))
            return True
    
    def get_performance_history(self, account_id: str, days: int = 30) -> List[Dict]:
        """Performans geçmişi getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT * FROM performance 
                WHERE account_id = ? AND date >= ?
                ORDER BY date DESC
            """, (account_id, start_date))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== ACCOUNT İŞLEMLERİ ====================
    
    def create_account(self, account: Dict) -> str:
        """Yeni hesap oluştur"""
        import random
        random_suffix = random.randint(1000, 9999)
        account_id = account.get('id', f"acc_{datetime.now():%Y%m%d%H%M%S}_{random_suffix}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO accounts (
                    id, name, strategy, currency, initial_balance,
                    current_balance, settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id,
                account.get('name'),
                account.get('strategy', 'dengeli'),
                account.get('currency', 'USD'),
                account.get('initial_balance', 10000),
                account.get('initial_balance', 10000),
                json.dumps(account.get('settings', {}))
            ))
            return account_id
    
    def get_account(self, account_id: str) -> Optional[Dict]:
        """Hesap bilgisi getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_account(self, account_id: str, updates: Dict) -> bool:
        """Hesap güncelle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['current_balance', 'total_pnl', 'total_trades', 
                          'winning_trades', 'losing_trades', 'max_drawdown',
                          'strategy', 'settings', 'name']:
                    set_clauses.append(f"{key} = ?")
                    values.append(json.dumps(value) if key == 'settings' else value)
            
            if not set_clauses:
                return False
            
            set_clauses.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(account_id)
            
            cursor.execute(f"""
                UPDATE accounts SET {', '.join(set_clauses)}
                WHERE id = ?
            """, values)
            
            return cursor.rowcount > 0
    
    def get_all_accounts(self) -> List[Dict]:
        """Tüm hesapları getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_account(self, account_id: str) -> bool:
        """Hesap sil"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Önce ilişkili verileri sil
            cursor.execute("DELETE FROM trades WHERE account_id = ?", (account_id,))
            cursor.execute("DELETE FROM performance WHERE account_id = ?", (account_id,))
            cursor.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            return cursor.rowcount > 0
    
    # ==================== MARKET DATA İŞLEMLERİ ====================
    
    def cache_market_data(self, data: List[Dict], symbol: str = 'BTCUSDT', 
                         timeframe: str = '1h') -> int:
        """Market verisini cache'le"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            count = 0
            
            for row in data:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data (
                            symbol, timeframe, timestamp, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timeframe,
                        row.get('timestamp'),
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('close'),
                        row.get('volume')
                    ))
                    count += 1
                except:
                    pass
            
            return count
    
    def get_cached_data(self, symbol: str = 'BTCUSDT', timeframe: str = '1h',
                       start_time: str = None, limit: int = 1000) -> List[Dict]:
        """Cache'li veriyi getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if start_time:
                cursor.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                    ORDER BY timestamp ASC LIMIT ?
                """, (symbol, timeframe, start_time, limit))
            else:
                cursor.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, timeframe, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== YARDIMCI METODLAR ====================
    
    def get_statistics(self, account_id: str) -> Dict:
        """Hesap istatistikleri"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trade istatistikleri
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing,
                    SUM(pnl) as total_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades 
                WHERE account_id = ? AND status = 'CLOSED'
            """, (account_id,))
            
            row = cursor.fetchone()
            
            return {
                'total_trades': row['total_trades'] or 0,
                'winning_trades': row['winning'] or 0,
                'losing_trades': row['losing'] or 0,
                'total_pnl': row['total_pnl'] or 0,
                'avg_win': row['avg_win'] or 0,
                'avg_loss': row['avg_loss'] or 0,
                'best_trade': row['best_trade'] or 0,
                'worst_trade': row['worst_trade'] or 0,
                'win_rate': (row['winning'] / row['total_trades'] * 100) if row['total_trades'] else 0
            }
    
    def cleanup_old_data(self, days: int = 90):
        """Eski verileri temizle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Eski sinyalleri sil
            cursor.execute("DELETE FROM signals WHERE timestamp < ?", (cutoff,))
            
            # Eski market verisini sil
            cursor.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff,))
            
            return cursor.rowcount
    
    def vacuum(self):
        """Veritabanını optimize et"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")


# Singleton instance
db = Database()


# ==================== TEST ====================
if __name__ == "__main__":
    print("Veritabanı test ediliyor...")
    
    # Hesap oluştur
    acc_id = db.create_account({
        'name': 'Test Hesap',
        'strategy': 'dengeli',
        'initial_balance': 10000
    })
    print(f"Hesap oluşturuldu: {acc_id}")
    
    # Trade ekle
    trade_id = db.insert_trade({
        'account_id': acc_id,
        'side': 'LONG',
        'entry_price': 95000,
        'quantity': 0.1,
        'strategy': 'dengeli'
    })
    print(f"Trade eklendi: {trade_id}")
    
    # Trade kapat
    db.close_trade(trade_id, {
        'exit_price': 96000,
        'pnl': 100,
        'pnl_percent': 1.05,
        'exit_reason': 'TP'
    })
    print("Trade kapatıldı")
    
    # İstatistikler
    stats = db.get_statistics(acc_id)
    print(f"İstatistikler: {stats}")
    
    print("\n✓ Veritabanı testi başarılı!")
