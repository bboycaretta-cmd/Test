"""
BTC Bot Pro - Risk Yönetim Modülü
FAZA 3: Profesyonel Risk Kontrol ve Pozisyon Yönetimi

Özellikler:
- Risk Metrikleri (VaR, CVaR, Sharpe, Sortino, Calmar)
- Position Sizing (Kelly, Optimal f, Fixed Fractional)
- Risk Limitleri (Daily loss, Consecutive loss, Max position)
- Drawdown Recovery Analizi
- Risk Dashboard
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# ENUMS VE DATACLASSES
# ================================================================

class PositionSizingMethod(Enum):
    """Pozisyon boyutlandırma metodları"""
    FIXED_AMOUNT = "fixed_amount"           # Sabit miktar
    FIXED_FRACTIONAL = "fixed_fractional"   # Sabit oran
    KELLY = "kelly"                         # Kelly Criterion
    OPTIMAL_F = "optimal_f"                 # Optimal f
    VOLATILITY_BASED = "volatility_based"   # Volatilite bazlı
    VAR_BASED = "var_based"                 # VaR bazlı


class RiskLevel(Enum):
    """Risk seviyesi"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6


@dataclass
class RiskMetrics:
    """Risk metrikleri"""
    # Return metrikleri
    total_return: float = 0
    annualized_return: float = 0
    daily_return_mean: float = 0
    daily_return_std: float = 0
    
    # Risk metrikleri
    var_95: float = 0          # Value at Risk %95
    var_99: float = 0          # Value at Risk %99
    cvar_95: float = 0         # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0
    
    # Drawdown metrikleri
    max_drawdown: float = 0
    avg_drawdown: float = 0
    max_drawdown_duration: int = 0  # Gün
    current_drawdown: float = 0
    
    # Ratio metrikleri
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    omega_ratio: float = 0
    
    # Trade metrikleri
    win_rate: float = 0
    profit_factor: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    avg_win_loss_ratio: float = 0
    
    # Risk skoru
    risk_score: float = 0  # 1-100
    risk_level: RiskLevel = RiskLevel.MODERATE


@dataclass
class PositionSize:
    """Pozisyon boyutu hesaplama sonucu"""
    method: PositionSizingMethod
    size: float              # BTC miktarı
    size_usd: float          # USD değeri
    size_percent: float      # Bakiye yüzdesi
    risk_amount: float       # Risk edilen miktar (USD)
    stop_loss_distance: float = 0
    reason: str = ""


@dataclass
class RiskLimits:
    """Risk limitleri"""
    max_position_size: float = 0.5      # Bakiyenin max %50'si
    max_daily_loss: float = 0.05        # Günlük max %5 kayıp
    max_weekly_loss: float = 0.10       # Haftalık max %10 kayıp
    max_drawdown: float = 0.20          # Max %20 drawdown
    max_consecutive_losses: int = 5     # Max 5 ardışık kayıp
    max_var_exposure: float = 0.02      # Max %2 VaR
    min_win_rate: float = 0.40          # Min %40 win rate (uyarı için)
    
    # Mevcut durum
    current_daily_loss: float = 0
    current_weekly_loss: float = 0
    current_drawdown: float = 0
    consecutive_losses: int = 0
    
    # Limitler aşıldı mı?
    daily_limit_hit: bool = False
    weekly_limit_hit: bool = False
    drawdown_limit_hit: bool = False
    consecutive_limit_hit: bool = False


@dataclass
class RiskAlert:
    """Risk uyarısı"""
    timestamp: datetime
    level: str  # WARNING, CRITICAL
    metric: str
    current_value: float
    limit_value: float
    message: str


# ================================================================
# RISK METRICS CALCULATOR
# ================================================================

class RiskCalculator:
    """Risk metriklerini hesapla"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Yıllık risksiz getiri oranı (default %5)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/365) - 1
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Value at Risk hesapla (Historical method)
        
        Args:
            returns: Getiri serisi
            confidence: Güven seviyesi (0.95 veya 0.99)
        
        Returns:
            VaR değeri (negatif = kayıp)
        """
        if len(returns) == 0:
            return 0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall) hesapla
        
        VaR'dan daha kötü senaryoların ortalaması
        """
        if len(returns) == 0:
            return 0
        
        var = -self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return abs(cvar) if not np.isnan(cvar) else abs(var)
    
    def calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 365) -> float:
        """
        Sharpe Ratio hesapla
        
        (Mean Return - Risk Free Rate) / Std Dev
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        
        excess_return = np.mean(returns) - self.daily_rf
        sharpe = (excess_return / np.std(returns)) * np.sqrt(periods_per_year)
        return sharpe
    
    def calculate_sortino(self, returns: np.ndarray, periods_per_year: int = 365) -> float:
        """
        Sortino Ratio hesapla
        
        Sadece downside volatiliteyi kullanır
        """
        if len(returns) < 2:
            return 0
        
        excess_return = np.mean(returns) - self.daily_rf
        downside = returns[returns < 0]
        
        if len(downside) < 2 or np.std(downside) == 0:
            return 0
        
        sortino = (excess_return / np.std(downside)) * np.sqrt(periods_per_year)
        return sortino
    
    def calculate_calmar(self, total_return: float, max_drawdown: float, 
                         periods: int = 365) -> float:
        """
        Calmar Ratio hesapla
        
        Annualized Return / Max Drawdown
        """
        if max_drawdown == 0:
            return 0
        
        # Yıllık getiri
        annualized = ((1 + total_return) ** (365 / max(periods, 1))) - 1
        
        return annualized / max_drawdown
    
    def calculate_omega(self, returns: np.ndarray, threshold: float = 0) -> float:
        """
        Omega Ratio hesapla
        
        Probability weighted ratio of gains vs losses
        """
        if len(returns) == 0:
            return 0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_losses = np.sum(losses)
        if sum_losses == 0:
            return float('inf')
        
        return np.sum(gains) / sum_losses
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, float, int]:
        """
        Drawdown hesapla
        
        Returns:
            (max_drawdown, avg_drawdown, max_duration)
        """
        if len(equity_curve) == 0:
            return 0, 0, 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        
        max_dd = np.max(drawdown)
        avg_dd = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0
        
        # Max duration
        in_drawdown = drawdown > 0
        if not np.any(in_drawdown):
            max_duration = 0
        else:
            # Consecutive True değerlerin maksimumu
            changes = np.diff(np.concatenate([[0], in_drawdown.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                max_duration = max(ends - starts)
            else:
                max_duration = 0
        
        return max_dd, avg_dd, max_duration
    
    def calculate_all(self, 
                      equity_curve: List[float],
                      trades: List[Dict] = None,
                      periods_per_year: int = 8760) -> RiskMetrics:
        """Tüm risk metriklerini hesapla"""
        
        equity = np.array(equity_curve)
        
        if len(equity) < 2:
            return RiskMetrics()
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        
        # Basic return metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        annualized_return = ((1 + total_return) ** (periods_per_year / len(equity))) - 1
        
        # Risk metrics
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Drawdown
        max_dd, avg_dd, max_duration = self.calculate_drawdown(equity)
        current_dd = (np.max(equity) - equity[-1]) / np.max(equity)
        
        # Ratios
        sharpe = self.calculate_sharpe(returns, periods_per_year)
        sortino = self.calculate_sortino(returns, periods_per_year)
        calmar = self.calculate_calmar(total_return, max_dd, len(equity))
        omega = self.calculate_omega(returns)
        
        # Trade metrics
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        largest_win = 0
        largest_loss = 0
        
        if trades:
            pnls = [t.get('pnl', t.pnl if hasattr(t, 'pnl') else 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            if pnls:
                win_rate = len(wins) / len(pnls)
                
                gross_profit = sum(wins) if wins else 0
                gross_loss = abs(sum(losses)) if losses else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                
                largest_win = max(pnls)
                largest_loss = min(pnls)
        
        avg_win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Risk score (1-100)
        risk_score = self._calculate_risk_score(
            max_dd, var_95, sharpe, win_rate, profit_factor
        )
        
        risk_level = self._get_risk_level(risk_score)
        
        return RiskMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_return_mean=np.mean(returns),
            daily_return_std=np.std(returns),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_duration,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=avg_win_loss,
            risk_score=risk_score,
            risk_level=risk_level
        )
    
    def _calculate_risk_score(self, max_dd: float, var: float, sharpe: float,
                              win_rate: float, profit_factor: float) -> float:
        """Risk skoru hesapla (1-100, düşük = daha riskli)"""
        score = 50  # Base score
        
        # Drawdown impact (-30 to +10)
        if max_dd > 0.30:
            score -= 30
        elif max_dd > 0.20:
            score -= 20
        elif max_dd > 0.10:
            score -= 10
        elif max_dd < 0.05:
            score += 10
        
        # Sharpe impact (-20 to +20)
        if sharpe < 0:
            score -= 20
        elif sharpe < 0.5:
            score -= 10
        elif sharpe > 2:
            score += 20
        elif sharpe > 1:
            score += 10
        
        # Win rate impact (-10 to +10)
        if win_rate < 0.3:
            score -= 10
        elif win_rate > 0.6:
            score += 10
        
        # Profit factor impact (-10 to +10)
        if profit_factor < 1:
            score -= 10
        elif profit_factor > 2:
            score += 10
        
        return max(1, min(100, score))
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Risk seviyesi belirle"""
        if score >= 80:
            return RiskLevel.VERY_LOW
        elif score >= 65:
            return RiskLevel.LOW
        elif score >= 50:
            return RiskLevel.MODERATE
        elif score >= 35:
            return RiskLevel.HIGH
        elif score >= 20:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME


# ================================================================
# POSITION SIZING
# ================================================================

class PositionSizer:
    """Pozisyon boyutlandırma"""
    
    def __init__(self, 
                 method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL,
                 max_position_pct: float = 0.5,
                 risk_per_trade: float = 0.02):
        """
        Args:
            method: Boyutlandırma metodu
            max_position_pct: Maximum pozisyon yüzdesi
            risk_per_trade: Trade başına risk yüzdesi
        """
        self.method = method
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
    
    def calculate(self,
                  balance: float,
                  price: float,
                  stop_loss_pct: float = 0.02,
                  win_rate: float = 0.5,
                  avg_win_loss_ratio: float = 1.5,
                  volatility: float = 0.02) -> PositionSize:
        """
        Pozisyon boyutu hesapla
        
        Args:
            balance: Mevcut bakiye
            price: Güncel fiyat
            stop_loss_pct: Stop loss yüzdesi
            win_rate: Kazanma oranı (Kelly için)
            avg_win_loss_ratio: Ort. kazanç/kayıp oranı (Kelly için)
            volatility: Volatilite (ATR bazlı)
        """
        
        if self.method == PositionSizingMethod.FIXED_AMOUNT:
            return self._fixed_amount(balance, price)
        
        elif self.method == PositionSizingMethod.FIXED_FRACTIONAL:
            return self._fixed_fractional(balance, price, stop_loss_pct)
        
        elif self.method == PositionSizingMethod.KELLY:
            return self._kelly(balance, price, win_rate, avg_win_loss_ratio)
        
        elif self.method == PositionSizingMethod.OPTIMAL_F:
            return self._optimal_f(balance, price, win_rate, avg_win_loss_ratio)
        
        elif self.method == PositionSizingMethod.VOLATILITY_BASED:
            return self._volatility_based(balance, price, volatility)
        
        elif self.method == PositionSizingMethod.VAR_BASED:
            return self._var_based(balance, price, volatility)
        
        return self._fixed_fractional(balance, price, stop_loss_pct)
    
    def _fixed_amount(self, balance: float, price: float) -> PositionSize:
        """Sabit miktar (bakiyenin %50'si)"""
        size_usd = balance * self.max_position_pct
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.FIXED_AMOUNT,
            size=size,
            size_usd=size_usd,
            size_percent=self.max_position_pct,
            risk_amount=size_usd * self.risk_per_trade,
            reason=f"Fixed {self.max_position_pct*100:.0f}% of balance"
        )
    
    def _fixed_fractional(self, balance: float, price: float, 
                          stop_loss_pct: float) -> PositionSize:
        """
        Fixed Fractional Position Sizing
        
        Risk Amount = Balance * Risk%
        Position Size = Risk Amount / Stop Loss Distance
        """
        risk_amount = balance * self.risk_per_trade
        stop_distance = price * stop_loss_pct
        
        size_usd = risk_amount / stop_loss_pct
        size_usd = min(size_usd, balance * self.max_position_pct)
        
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.FIXED_FRACTIONAL,
            size=size,
            size_usd=size_usd,
            size_percent=size_usd / balance,
            risk_amount=risk_amount,
            stop_loss_distance=stop_distance,
            reason=f"Risk {self.risk_per_trade*100:.1f}% with {stop_loss_pct*100:.1f}% SL"
        )
    
    def _kelly(self, balance: float, price: float,
               win_rate: float, avg_win_loss_ratio: float) -> PositionSize:
        """
        Kelly Criterion
        
        f* = W - (1-W)/R
        
        Where:
            W = Win rate
            R = Win/Loss ratio
        """
        if avg_win_loss_ratio <= 0:
            return self._fixed_fractional(balance, price, 0.02)
        
        # Full Kelly
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        
        # Half Kelly (daha güvenli)
        kelly = kelly * 0.5
        
        # Sınırla
        kelly = max(0, min(kelly, self.max_position_pct))
        
        size_usd = balance * kelly
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.KELLY,
            size=size,
            size_usd=size_usd,
            size_percent=kelly,
            risk_amount=size_usd * self.risk_per_trade,
            reason=f"Half Kelly: {kelly*100:.1f}% (WR={win_rate*100:.0f}%, R={avg_win_loss_ratio:.1f})"
        )
    
    def _optimal_f(self, balance: float, price: float,
                   win_rate: float, avg_win_loss_ratio: float) -> PositionSize:
        """
        Optimal f (Ralph Vince)
        
        Kelly'nin bir varyasyonu, max geometric growth için optimize edilmiş
        """
        if avg_win_loss_ratio <= 0 or win_rate <= 0:
            return self._fixed_fractional(balance, price, 0.02)
        
        # Simplified optimal f
        # f = ((1 + R) * W - 1) / R
        f = ((1 + avg_win_loss_ratio) * win_rate - 1) / avg_win_loss_ratio
        
        # Conservative adjustment
        f = f * 0.3  # %30 of optimal
        
        f = max(0, min(f, self.max_position_pct))
        
        size_usd = balance * f
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.OPTIMAL_F,
            size=size,
            size_usd=size_usd,
            size_percent=f,
            risk_amount=size_usd * self.risk_per_trade,
            reason=f"Optimal f: {f*100:.1f}%"
        )
    
    def _volatility_based(self, balance: float, price: float,
                          volatility: float) -> PositionSize:
        """
        Volatilite Bazlı Position Sizing
        
        Yüksek volatilite = Küçük pozisyon
        """
        # Base position = %50
        base_pct = self.max_position_pct
        
        # Volatility adjustment
        # Normal volatility = %2, higher = reduce position
        vol_factor = 0.02 / max(volatility, 0.005)
        vol_factor = min(vol_factor, 2.0)  # Max 2x
        
        adjusted_pct = base_pct * vol_factor
        adjusted_pct = min(adjusted_pct, self.max_position_pct)
        
        size_usd = balance * adjusted_pct
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.VOLATILITY_BASED,
            size=size,
            size_usd=size_usd,
            size_percent=adjusted_pct,
            risk_amount=size_usd * volatility,
            reason=f"Volatility adjusted: {adjusted_pct*100:.1f}% (vol={volatility*100:.2f}%)"
        )
    
    def _var_based(self, balance: float, price: float,
                   volatility: float) -> PositionSize:
        """
        VaR Bazlı Position Sizing
        
        Position size öyle ki VaR limiti aşılmasın
        """
        max_var = balance * 0.02  # Max %2 günlük VaR
        
        # VaR ≈ Position * 1.65 * Daily Volatility (95% confidence)
        # Position = VaR / (1.65 * Volatility)
        
        size_usd = max_var / (1.65 * max(volatility, 0.005))
        size_usd = min(size_usd, balance * self.max_position_pct)
        
        size = size_usd / price
        
        return PositionSize(
            method=PositionSizingMethod.VAR_BASED,
            size=size,
            size_usd=size_usd,
            size_percent=size_usd / balance,
            risk_amount=max_var,
            reason=f"VaR-based: max ${max_var:.0f} daily risk"
        )


# ================================================================
# RISK MANAGER
# ================================================================

class RiskManager:
    """
    Ana risk yönetim sınıfı
    
    - Limit kontrolü
    - Alert sistemi
    - Pozisyon onayı
    """
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.calculator = RiskCalculator()
        self.sizer = PositionSizer()
        self.alerts: List[RiskAlert] = []
        
        # Tracking
        self.daily_pnl: float = 0
        self.weekly_pnl: float = 0
        self.consecutive_losses: int = 0
        self.last_reset_date: datetime = datetime.now().date()
        self.last_week_reset: datetime = datetime.now()
    
    def check_limits(self) -> Tuple[bool, List[str]]:
        """
        Risk limitlerini kontrol et
        
        Returns:
            (trade_allowed, list of reasons if not)
        """
        reasons = []
        
        # Daily loss check
        daily_loss_pct = abs(self.daily_pnl) / max(1, self.limits.max_daily_loss)
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.limits.max_daily_loss:
            self.limits.daily_limit_hit = True
            reasons.append(f"Daily loss limit hit: {self.daily_pnl*100:.2f}%")
        
        # Weekly loss check
        if self.weekly_pnl < 0 and abs(self.weekly_pnl) >= self.limits.max_weekly_loss:
            self.limits.weekly_limit_hit = True
            reasons.append(f"Weekly loss limit hit: {self.weekly_pnl*100:.2f}%")
        
        # Drawdown check
        if self.limits.current_drawdown >= self.limits.max_drawdown:
            self.limits.drawdown_limit_hit = True
            reasons.append(f"Max drawdown hit: {self.limits.current_drawdown*100:.2f}%")
        
        # Consecutive losses check
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            self.limits.consecutive_limit_hit = True
            reasons.append(f"Consecutive losses: {self.consecutive_losses}")
        
        can_trade = len(reasons) == 0
        
        return can_trade, reasons
    
    def update_after_trade(self, pnl: float, pnl_pct: float, is_win: bool):
        """Trade sonrası güncelle"""
        
        # Reset daily if new day
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0
            self.last_reset_date = today
            self.limits.daily_limit_hit = False
        
        # Reset weekly if new week
        if (datetime.now() - self.last_week_reset).days >= 7:
            self.weekly_pnl = 0
            self.last_week_reset = datetime.now()
            self.limits.weekly_limit_hit = False
        
        # Update PnL
        self.daily_pnl += pnl_pct
        self.weekly_pnl += pnl_pct
        
        # Consecutive losses
        if is_win:
            self.consecutive_losses = 0
            self.limits.consecutive_limit_hit = False
        else:
            self.consecutive_losses += 1
        
        # Alerts
        self._check_alerts()
    
    def update_drawdown(self, current_equity: float, peak_equity: float):
        """Drawdown güncelle"""
        if peak_equity > 0:
            self.limits.current_drawdown = (peak_equity - current_equity) / peak_equity
    
    def _check_alerts(self):
        """Alert'leri kontrol et"""
        
        # Daily loss warning (%50 of limit)
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.limits.max_daily_loss * 0.5:
            self._add_alert(
                "WARNING", "daily_loss",
                abs(self.daily_pnl), self.limits.max_daily_loss,
                f"Daily loss at {abs(self.daily_pnl)*100:.1f}% ({self.limits.max_daily_loss*100:.0f}% limit)"
            )
        
        # Drawdown warning
        if self.limits.current_drawdown >= self.limits.max_drawdown * 0.7:
            level = "CRITICAL" if self.limits.current_drawdown >= self.limits.max_drawdown * 0.9 else "WARNING"
            self._add_alert(
                level, "drawdown",
                self.limits.current_drawdown, self.limits.max_drawdown,
                f"Drawdown at {self.limits.current_drawdown*100:.1f}%"
            )
        
        # Consecutive losses warning
        if self.consecutive_losses >= self.limits.max_consecutive_losses - 2:
            self._add_alert(
                "WARNING", "consecutive_losses",
                self.consecutive_losses, self.limits.max_consecutive_losses,
                f"{self.consecutive_losses} consecutive losses"
            )
    
    def _add_alert(self, level: str, metric: str, current: float, 
                   limit: float, message: str):
        """Alert ekle"""
        alert = RiskAlert(
            timestamp=datetime.now(),
            level=level,
            metric=metric,
            current_value=current,
            limit_value=limit,
            message=message
        )
        self.alerts.append(alert)
        
        # Son 100 alert'i tut
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_position_size(self, balance: float, price: float,
                          stop_loss_pct: float = 0.02,
                          win_rate: float = 0.5,
                          avg_win_loss: float = 1.5,
                          volatility: float = 0.02) -> PositionSize:
        """Onaylı pozisyon boyutu al"""
        
        # Limit kontrolü
        can_trade, reasons = self.check_limits()
        
        if not can_trade:
            return PositionSize(
                method=self.sizer.method,
                size=0,
                size_usd=0,
                size_percent=0,
                risk_amount=0,
                reason=f"Trading blocked: {'; '.join(reasons)}"
            )
        
        # Pozisyon hesapla
        position = self.sizer.calculate(
            balance=balance,
            price=price,
            stop_loss_pct=stop_loss_pct,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss,
            volatility=volatility
        )
        
        # Max position limit
        if position.size_percent > self.limits.max_position_size:
            position.size_percent = self.limits.max_position_size
            position.size_usd = balance * self.limits.max_position_size
            position.size = position.size_usd / price
            position.reason += f" (capped at {self.limits.max_position_size*100:.0f}%)"
        
        return position
    
    def get_recent_alerts(self, count: int = 10) -> List[RiskAlert]:
        """Son alert'leri getir"""
        return self.alerts[-count:]
    
    def get_status(self) -> Dict:
        """Risk durumunu getir"""
        can_trade, reasons = self.check_limits()
        
        return {
            'can_trade': can_trade,
            'blocked_reasons': reasons,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'current_drawdown': self.limits.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'limits': {
                'daily_hit': self.limits.daily_limit_hit,
                'weekly_hit': self.limits.weekly_limit_hit,
                'drawdown_hit': self.limits.drawdown_limit_hit,
                'consecutive_hit': self.limits.consecutive_limit_hit
            },
            'alert_count': len(self.alerts)
        }


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_risk_report(metrics: RiskMetrics):
    """Risk raporunu yazdır"""
    print("\n" + "="*60)
    print("RİSK RAPORU")
    print("="*60)
    
    print(f"\n{'GETİRİ METRİKLERİ':^60}")
    print("-"*60)
    print(f"Toplam Getiri:       {metrics.total_return*100:>10.2f}%")
    print(f"Yıllık Getiri:       {metrics.annualized_return*100:>10.2f}%")
    
    print(f"\n{'RİSK METRİKLERİ':^60}")
    print("-"*60)
    print(f"VaR (95%):           {metrics.var_95*100:>10.2f}%")
    print(f"VaR (99%):           {metrics.var_99*100:>10.2f}%")
    print(f"CVaR (95%):          {metrics.cvar_95*100:>10.2f}%")
    print(f"Max Drawdown:        {metrics.max_drawdown*100:>10.2f}%")
    print(f"Avg Drawdown:        {metrics.avg_drawdown*100:>10.2f}%")
    
    print(f"\n{'RATIO METRİKLERİ':^60}")
    print("-"*60)
    print(f"Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")
    print(f"Omega Ratio:         {metrics.omega_ratio:>10.2f}")
    
    print(f"\n{'TRADE METRİKLERİ':^60}")
    print("-"*60)
    print(f"Win Rate:            {metrics.win_rate*100:>10.1f}%")
    print(f"Profit Factor:       {metrics.profit_factor:>10.2f}")
    print(f"Avg Win/Loss:        {metrics.avg_win_loss_ratio:>10.2f}")
    
    print(f"\n{'RİSK DEĞERLENDİRME':^60}")
    print("-"*60)
    print(f"Risk Skoru:          {metrics.risk_score:>10.0f}/100")
    print(f"Risk Seviyesi:       {metrics.risk_level.name:>10}")
    
    print("\n" + "="*60)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Risk Management test ediliyor...\n")
    
    # Test verisi
    np.random.seed(42)
    equity = [10000]
    for _ in range(100):
        ret = np.random.randn() * 0.02 + 0.001
        equity.append(equity[-1] * (1 + ret))
    
    trades = [
        {'pnl': 100}, {'pnl': -50}, {'pnl': 75},
        {'pnl': -30}, {'pnl': 120}, {'pnl': -80},
        {'pnl': 90}, {'pnl': 60}, {'pnl': -40}, {'pnl': 150}
    ]
    
    # 1. Risk Calculator
    print("1. Risk Metrics")
    calc = RiskCalculator()
    metrics = calc.calculate_all(equity, trades)
    print(f"   Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"   Max DD: {metrics.max_drawdown*100:.2f}%")
    print(f"   Risk Score: {metrics.risk_score:.0f}")
    
    # 2. Position Sizer
    print("\n2. Position Sizing")
    sizer = PositionSizer(method=PositionSizingMethod.KELLY)
    pos = sizer.calculate(
        balance=10000,
        price=95000,
        win_rate=0.6,
        avg_win_loss_ratio=1.5
    )
    print(f"   Kelly Size: {pos.size:.4f} BTC (${pos.size_usd:,.0f})")
    print(f"   Reason: {pos.reason}")
    
    # 3. Risk Manager
    print("\n3. Risk Manager")
    manager = RiskManager()
    
    # Simüle trade
    manager.update_after_trade(-200, -0.02, False)
    manager.update_after_trade(-150, -0.015, False)
    manager.update_after_trade(-100, -0.01, False)
    
    status = manager.get_status()
    print(f"   Can Trade: {status['can_trade']}")
    print(f"   Daily PnL: {status['daily_pnl']*100:.2f}%")
    print(f"   Consecutive Losses: {status['consecutive_losses']}")
    
    # Position with limits
    pos2 = manager.get_position_size(
        balance=10000,
        price=95000,
        stop_loss_pct=0.02
    )
    print(f"   Approved Size: {pos2.size:.4f} BTC")
    
    print("\n✓ Risk Management testi başarılı!")
