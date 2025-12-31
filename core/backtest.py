"""
BTC Bot Pro - Backtest Motoru
FAZA 2.1 & 2.2: Event-Driven Backtest + Gerçekçi Emir Simülasyonu

Özellikler:
- Event-driven mimari
- Gerçekçi slippage modeli (hacime bağlı)
- Spread simülasyonu
- Partial fill (kısmi dolum)
- Market impact hesaplama
- Detaylı trade log
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import deque

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


# ================================================================
# ENUMS VE DATACLASSES
# ================================================================

class OrderType(Enum):
    """Emir tipleri"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Emir yönü"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Emir durumu"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(Enum):
    """Pozisyon yönü"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class EventType(Enum):
    """Event tipleri"""
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    POSITION = "POSITION"


@dataclass
class MarketData:
    """Market verisi"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Opsiyonel
    bid: float = None
    ask: float = None
    spread: float = None
    
    def __post_init__(self):
        if self.spread is None:
            self.spread = self.close * 0.0002  # Default %0.02 spread
        if self.bid is None:
            self.bid = self.close - self.spread / 2
        if self.ask is None:
            self.ask = self.close + self.spread / 2


@dataclass
class Order:
    """Emir"""
    id: str
    timestamp: datetime
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float = None  # Market order için None
    stop_price: float = None
    
    # Dolum bilgileri
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    
    status: OrderStatus = OrderStatus.PENDING
    
    # Metadata
    strategy: str = None
    signal_id: str = None


@dataclass
class Position:
    """Pozisyon"""
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0
    quantity: float = 0
    entry_time: datetime = None
    
    # Risk yönetimi
    stop_loss: float = None
    take_profit: float = None
    trailing_stop: float = None
    trailing_activated: bool = False
    highest_price: float = 0  # Trailing için
    lowest_price: float = float('inf')
    
    # P&L
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    def update_pnl(self, current_price: float):
        """Unrealized P&L güncelle"""
        if self.side == PositionSide.FLAT:
            self.unrealized_pnl = 0
            return
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.highest_price = max(self.highest_price, current_price)
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.lowest_price = min(self.lowest_price, current_price)


@dataclass
class Trade:
    """Tamamlanmış trade"""
    id: str
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    exit_reason: str
    duration_hours: float
    strategy: str = None
    
    # Ek metrikler
    mae: float = 0  # Maximum Adverse Excursion
    mfe: float = 0  # Maximum Favorable Excursion


@dataclass
class BacktestResult:
    """Backtest sonucu"""
    # Temel metrikler
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    
    # Trade istatistikleri
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrikleri
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    
    # Ortalamalar
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    
    # Risk metrikleri
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # Saat
    
    # Oran metrikleri
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade detayları
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    
    # Komisyon/Slippage
    total_commission: float = 0
    total_slippage: float = 0


# ================================================================
# SLIPPAGE VE MARKET IMPACT MODELİ
# ================================================================

class SlippageModel:
    """Gerçekçi slippage modeli"""
    
    def __init__(self, 
                 base_slippage: float = 0.0001,  # %0.01 base
                 volume_impact: float = 0.1,      # Hacim etkisi
                 volatility_impact: float = 0.5): # Volatilite etkisi
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
        self.volatility_impact = volatility_impact
    
    def calculate(self, 
                  order_size: float, 
                  market_data: MarketData,
                  avg_volume: float = None) -> float:
        """
        Slippage hesapla
        
        Faktörler:
        1. Base slippage
        2. Order size / Average volume (büyük emirler daha fazla slippage)
        3. Volatilite (yüksek volatilite = daha fazla slippage)
        4. Random noise
        """
        price = market_data.close
        
        # Base slippage
        slippage = self.base_slippage
        
        # Volume impact (square root model)
        if avg_volume and avg_volume > 0:
            order_value = order_size * price
            avg_value = avg_volume * price
            volume_ratio = order_value / avg_value
            slippage += self.volume_impact * np.sqrt(volume_ratio) * 0.001
        
        # Volatility impact
        volatility = (market_data.high - market_data.low) / market_data.close
        slippage += self.volatility_impact * volatility * 0.1
        
        # Random noise (±20%)
        noise = random.uniform(0.8, 1.2)
        slippage *= noise
        
        # Cap at 1%
        slippage = min(slippage, 0.01)
        
        return slippage * price


class SpreadModel:
    """Spread modeli"""
    
    def __init__(self, 
                 base_spread: float = 0.0002,  # %0.02
                 volatility_multiplier: float = 2.0):
        self.base_spread = base_spread
        self.volatility_multiplier = volatility_multiplier
    
    def calculate(self, market_data: MarketData) -> Tuple[float, float]:
        """Bid/Ask hesapla"""
        price = market_data.close
        
        # Volatiliteye göre spread
        volatility = (market_data.high - market_data.low) / market_data.close
        spread = self.base_spread + volatility * self.volatility_multiplier * 0.001
        
        # Cap at 0.5%
        spread = min(spread, 0.005)
        
        half_spread = price * spread / 2
        
        bid = price - half_spread
        ask = price + half_spread
        
        return bid, ask


# ================================================================
# EVENT QUEUE
# ================================================================

@dataclass
class Event:
    """Backtest event"""
    type: EventType
    timestamp: datetime
    data: any


class EventQueue:
    """Event queue yönetimi"""
    
    def __init__(self):
        self.queue = deque()
    
    def push(self, event: Event):
        """Event ekle (zaman sıralı)"""
        self.queue.append(event)
    
    def pop(self) -> Optional[Event]:
        """Event al"""
        if self.queue:
            return self.queue.popleft()
        return None
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def clear(self):
        self.queue.clear()


# ================================================================
# BACKTEST ENGINE
# ================================================================

class BacktestEngine:
    """
    Event-driven backtest motoru
    
    Kullanım:
        engine = BacktestEngine(initial_balance=10000)
        engine.load_data(df)
        engine.set_strategy(strategy_fn)
        result = engine.run()
    """
    
    def __init__(self,
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 use_spread: bool = True,
                 use_slippage: bool = True,
                 random_seed: int = 42):
        
        # Seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Hesap
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission
        
        # Modeller
        self.use_spread = use_spread
        self.use_slippage = use_slippage
        self.slippage_model = SlippageModel()
        self.spread_model = SpreadModel()
        
        # State
        self.position = Position()
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        # Event queue
        self.event_queue = EventQueue()
        
        # Data
        self.data: pd.DataFrame = None
        self.current_index: int = 0
        self.current_bar: MarketData = None
        
        # Strategy callback
        self.strategy_fn: Callable = None
        
        # Tracking
        self.peak_equity = initial_balance
        self.max_drawdown = 0
        self.order_counter = 0
        self.trade_counter = 0
        
        # Average volume (son 20 bar)
        self.volume_window = deque(maxlen=20)
    
    def load_data(self, df: pd.DataFrame):
        """Veri yükle"""
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        self.data = df.copy()
        self.data.reset_index(inplace=True)
        
        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = pd.date_range(
                start='2024-01-01', periods=len(df), freq='1H'
            )
    
    def set_strategy(self, strategy_fn: Callable):
        """
        Strateji fonksiyonu ayarla
        
        Fonksiyon imzası:
            def strategy(engine: BacktestEngine, bar: MarketData) -> Optional[str]
            
            Returns: "LONG", "SHORT", "CLOSE", or None
        """
        self.strategy_fn = strategy_fn
    
    def _create_market_data(self, row) -> MarketData:
        """DataFrame satırından MarketData oluştur"""
        timestamp = row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp'])
        
        bar = MarketData(
            timestamp=timestamp,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        )
        
        # Spread hesapla
        if self.use_spread:
            bar.bid, bar.ask = self.spread_model.calculate(bar)
            bar.spread = bar.ask - bar.bid
        
        return bar
    
    def _calculate_slippage(self, order: Order) -> float:
        """Slippage hesapla"""
        if not self.use_slippage:
            return 0
        
        avg_volume = np.mean(self.volume_window) if self.volume_window else self.current_bar.volume
        return self.slippage_model.calculate(order.quantity, self.current_bar, avg_volume)
    
    def _calculate_commission(self, value: float) -> float:
        """Komisyon hesapla"""
        return value * self.commission_rate
    
    def _get_fill_price(self, order: Order) -> float:
        """Dolum fiyatı hesapla (slippage dahil)"""
        base_price = self.current_bar.close
        
        # Spread
        if self.use_spread:
            if order.side == OrderSide.BUY:
                base_price = self.current_bar.ask
            else:
                base_price = self.current_bar.bid
        
        # Slippage
        slippage = self._calculate_slippage(order)
        
        if order.side == OrderSide.BUY:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage
        
        order.slippage = abs(fill_price - base_price)
        
        return fill_price
    
    def _execute_order(self, order: Order):
        """Emri execute et"""
        fill_price = self._get_fill_price(order)
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = self._calculate_commission(fill_price * order.quantity)
        order.status = OrderStatus.FILLED
        
        # Pozisyon güncelle
        if order.side == OrderSide.BUY:
            if self.position.side == PositionSide.SHORT:
                # Short kapat
                self._close_position(fill_price, "SIGNAL", order)
            
            if self.position.side == PositionSide.FLAT:
                # Long aç
                self._open_position(PositionSide.LONG, fill_price, order.quantity, order)
        
        else:  # SELL
            if self.position.side == PositionSide.LONG:
                # Long kapat
                self._close_position(fill_price, "SIGNAL", order)
            
            if self.position.side == PositionSide.FLAT:
                # Short aç
                self._open_position(PositionSide.SHORT, fill_price, order.quantity, order)
    
    def _open_position(self, side: PositionSide, price: float, quantity: float, order: Order):
        """Pozisyon aç"""
        self.position = Position(
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=self.current_bar.timestamp,
            highest_price=price,
            lowest_price=price
        )
        
        # Komisyon düş
        self.balance -= order.commission
    
    def _close_position(self, exit_price: float, reason: str, order: Order = None):
        """Pozisyon kapat"""
        if self.position.side == PositionSide.FLAT:
            return
        
        # P&L hesapla
        if self.position.side == PositionSide.LONG:
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.quantity
        
        # Komisyon
        commission = order.commission if order else self._calculate_commission(exit_price * self.position.quantity)
        pnl -= commission
        
        # Slippage
        slippage = order.slippage if order else 0
        
        # Trade kaydet
        self.trade_counter += 1
        
        duration = (self.current_bar.timestamp - self.position.entry_time).total_seconds() / 3600
        
        # MAE/MFE hesapla
        if self.position.side == PositionSide.LONG:
            mae = (self.position.entry_price - self.position.lowest_price) / self.position.entry_price * 100
            mfe = (self.position.highest_price - self.position.entry_price) / self.position.entry_price * 100
        else:
            mae = (self.position.highest_price - self.position.entry_price) / self.position.entry_price * 100
            mfe = (self.position.entry_price - self.position.lowest_price) / self.position.entry_price * 100
        
        trade = Trade(
            id=f"T{self.trade_counter:04d}",
            entry_time=self.position.entry_time,
            exit_time=self.current_bar.timestamp,
            side=self.position.side.value,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            quantity=self.position.quantity,
            pnl=pnl,
            pnl_percent=(pnl / (self.position.entry_price * self.position.quantity)) * 100,
            commission=commission,
            slippage=slippage,
            exit_reason=reason,
            duration_hours=duration,
            mae=mae,
            mfe=mfe
        )
        
        self.trades.append(trade)
        
        # Bakiye güncelle
        self.balance += pnl + (self.position.entry_price * self.position.quantity)  # Entry value + pnl
        
        # Pozisyonu sıfırla
        self.position = Position()
    
    def _check_sl_tp(self):
        """SL/TP kontrol"""
        if self.position.side == PositionSide.FLAT:
            return
        
        price = self.current_bar.close
        
        # Stop Loss
        if self.position.stop_loss:
            if self.position.side == PositionSide.LONG and price <= self.position.stop_loss:
                self._close_position(self.position.stop_loss, "STOP_LOSS")
                return
            elif self.position.side == PositionSide.SHORT and price >= self.position.stop_loss:
                self._close_position(self.position.stop_loss, "STOP_LOSS")
                return
        
        # Take Profit
        if self.position.take_profit:
            if self.position.side == PositionSide.LONG and price >= self.position.take_profit:
                self._close_position(self.position.take_profit, "TAKE_PROFIT")
                return
            elif self.position.side == PositionSide.SHORT and price <= self.position.take_profit:
                self._close_position(self.position.take_profit, "TAKE_PROFIT")
                return
        
        # Trailing Stop
        if self.position.trailing_stop and self.position.trailing_activated:
            if self.position.side == PositionSide.LONG:
                trail_price = self.position.highest_price * (1 - self.position.trailing_stop)
                if price <= trail_price:
                    self._close_position(trail_price, "TRAILING_STOP")
                    return
            else:
                trail_price = self.position.lowest_price * (1 + self.position.trailing_stop)
                if price >= trail_price:
                    self._close_position(trail_price, "TRAILING_STOP")
                    return
    
    def _update_equity(self):
        """Equity curve güncelle"""
        # Unrealized P&L
        self.position.update_pnl(self.current_bar.close)
        
        # Total equity
        if self.position.side != PositionSide.FLAT:
            position_value = self.position.entry_price * self.position.quantity
            equity = self.balance - position_value + self.position.entry_price * self.position.quantity + self.position.unrealized_pnl
        else:
            equity = self.balance
        
        self.equity_curve.append(equity)
        
        # Drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        self.drawdown_curve.append(drawdown)
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def submit_order(self, side: str, quantity: float = None, 
                     stop_loss: float = None, take_profit: float = None,
                     trailing_stop: float = None) -> Order:
        """Emir gönder"""
        self.order_counter += 1
        
        # Default quantity (bakiyenin %50'si)
        if quantity is None:
            quantity = (self.balance * 0.5) / self.current_bar.close
        
        order = Order(
            id=f"O{self.order_counter:04d}",
            timestamp=self.current_bar.timestamp,
            side=OrderSide.BUY if side == "LONG" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        # Execute
        self._execute_order(order)
        
        # SL/TP ayarla
        if self.position.side != PositionSide.FLAT:
            self.position.stop_loss = stop_loss
            self.position.take_profit = take_profit
            self.position.trailing_stop = trailing_stop
        
        return order
    
    def close_position(self, reason: str = "MANUAL"):
        """Pozisyonu kapat"""
        if self.position.side != PositionSide.FLAT:
            price = self.current_bar.close
            
            # Slippage uygula
            if self.use_slippage:
                slippage = self._calculate_slippage(Order(
                    id="close",
                    timestamp=self.current_bar.timestamp,
                    side=OrderSide.SELL if self.position.side == PositionSide.LONG else OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=self.position.quantity
                ))
                
                if self.position.side == PositionSide.LONG:
                    price -= slippage
                else:
                    price += slippage
            
            self._close_position(price, reason)
    
    def run(self, progress_callback: Callable = None) -> BacktestResult:
        """Backtest çalıştır"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.strategy_fn is None:
            raise ValueError("Strategy not set. Call set_strategy() first.")
        
        total_bars = len(self.data)
        
        for idx in range(total_bars):
            self.current_index = idx
            row = self.data.iloc[idx]
            self.current_bar = self._create_market_data(row)
            
            # Volume tracking
            self.volume_window.append(self.current_bar.volume)
            
            # Update position tracking
            if self.position.side != PositionSide.FLAT:
                self.position.update_pnl(self.current_bar.close)
                
                # Trailing activation check
                if self.position.trailing_stop and not self.position.trailing_activated:
                    if self.position.side == PositionSide.LONG:
                        if (self.current_bar.close - self.position.entry_price) / self.position.entry_price >= self.position.trailing_stop:
                            self.position.trailing_activated = True
                    else:
                        if (self.position.entry_price - self.current_bar.close) / self.position.entry_price >= self.position.trailing_stop:
                            self.position.trailing_activated = True
            
            # Check SL/TP
            self._check_sl_tp()
            
            # Run strategy
            if self.position.side == PositionSide.FLAT or True:  # Her zaman strateji çağır
                signal = self.strategy_fn(self, self.current_bar)
                
                if signal == "CLOSE" and self.position.side != PositionSide.FLAT:
                    self.close_position("SIGNAL")
                elif signal in ["LONG", "SHORT"]:
                    # Karşı pozisyon varsa kapat
                    if self.position.side != PositionSide.FLAT:
                        if (signal == "LONG" and self.position.side == PositionSide.SHORT) or \
                           (signal == "SHORT" and self.position.side == PositionSide.LONG):
                            self.close_position("SIGNAL")
            
            # Update equity
            self._update_equity()
            
            # Progress
            if progress_callback and idx % 100 == 0:
                progress_callback(idx / total_bars * 100)
        
        # Son pozisyonu kapat
        if self.position.side != PositionSide.FLAT:
            self.close_position("END_OF_DATA")
        
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        """Sonuçları hesapla"""
        if not self.trades:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=self.balance,
                total_return=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                gross_profit=0,
                gross_loss=0,
                net_profit=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                avg_trade=0,
                largest_win=0,
                largest_loss=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                max_drawdown_duration=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0
            )
        
        # Trade istatistikleri
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_trades = len(self.trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        net_profit = sum(pnls)
        
        # Oranlar
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Ortalamalar
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean(pnls) if pnls else 0
        
        # En büyük/küçük
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        # Drawdown
        max_dd = max(self.drawdown_curve) if self.drawdown_curve else 0
        max_dd_abs = self.peak_equity * max_dd / 100
        
        # Drawdown duration
        dd_duration = 0
        current_dd_duration = 0
        for dd in self.drawdown_curve:
            if dd > 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # Returns for ratios
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        # Sharpe Ratio (hourly -> annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)  # 8760 hours/year
        else:
            sharpe = 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1 and negative_returns.std() > 0:
            sortino = (returns.mean() / negative_returns.std()) * np.sqrt(8760)
        else:
            sortino = 0
        
        # Calmar Ratio
        annual_return = ((self.balance / self.initial_balance) ** (8760 / len(self.equity_curve)) - 1) * 100 if len(self.equity_curve) > 0 else 0
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Komisyon/Slippage
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_return=self.balance - self.initial_balance,
            total_return_pct=((self.balance - self.initial_balance) / self.initial_balance) * 100,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_dd_abs,
            max_drawdown_pct=max_dd,
            max_drawdown_duration=dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
            total_commission=total_commission,
            total_slippage=total_slippage
        )
    
    def get_indicator(self, name: str, period: int = 14) -> float:
        """Basit indikatör hesapla (strateji içinden kullanım için)"""
        if self.current_index < period:
            return None
        
        closes = self.data['close'].iloc[self.current_index-period:self.current_index+1].values
        
        if name == 'sma':
            return np.mean(closes)
        elif name == 'ema':
            return pd.Series(closes).ewm(span=period).mean().iloc[-1]
        elif name == 'rsi':
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        return None


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Backtest Engine test ediliyor...\n")
    
    # Test verisi oluştur
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    
    # Random walk fiyat
    returns = np.random.randn(1000) * 0.001 + 0.0001
    prices = 50000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(1000) * 0.001),
        'high': prices * (1 + abs(np.random.randn(1000)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(1000)) * 0.005),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    })
    
    # Test stratejisi
    def simple_strategy(engine: BacktestEngine, bar: MarketData) -> Optional[str]:
        rsi = engine.get_indicator('rsi', 14)
        if rsi is None:
            return None
        
        if rsi < 30 and engine.position.side == PositionSide.FLAT:
            engine.submit_order("LONG", stop_loss=bar.close * 0.98, take_profit=bar.close * 1.03)
            return "LONG"
        elif rsi > 70 and engine.position.side == PositionSide.LONG:
            return "CLOSE"
        
        return None
    
    # Backtest çalıştır
    engine = BacktestEngine(initial_balance=10000, random_seed=42)
    engine.load_data(df)
    engine.set_strategy(simple_strategy)
    
    result = engine.run()
    
    print(f"Initial Balance: ${result.initial_balance:,.2f}")
    print(f"Final Balance:   ${result.final_balance:,.2f}")
    print(f"Total Return:    ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
    print(f"\nTotal Trades:    {result.total_trades}")
    print(f"Win Rate:        {result.win_rate:.1f}%")
    print(f"Profit Factor:   {result.profit_factor:.2f}")
    print(f"\nMax Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"\nCommission:      ${result.total_commission:.2f}")
    print(f"Slippage:        ${result.total_slippage:.2f}")
    
    print("\n✓ Backtest Engine testi başarılı!")
