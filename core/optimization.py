"""
BTC Bot Pro - İleri Düzey Backtest Araçları
FAZA 2.3 & 2.4: Walk-Forward Optimization + Monte Carlo Simülasyonu

Özellikler:
- Walk-forward optimization (overfitting koruması)
- Monte Carlo simülasyonu (güven aralıkları)
- Stress testing
- Out-of-sample validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import copy

from .backtest import BacktestEngine, BacktestResult, PositionSide, MarketData

# Seed
np.random.seed(42)
random.seed(42)


# ================================================================
# WALK-FORWARD OPTIMIZATION
# ================================================================

@dataclass
class WalkForwardWindow:
    """Walk-forward penceresi"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult = None
    test_result: BacktestResult = None
    best_params: Dict = None


@dataclass
class WalkForwardResult:
    """Walk-forward sonucu"""
    windows: List[WalkForwardWindow]
    
    # Aggregated metrics
    total_return: float
    total_return_pct: float
    avg_train_return: float
    avg_test_return: float
    
    # Robustness
    consistency_score: float  # Test/Train return ratio
    degradation: float  # In-sample vs out-of-sample fark
    
    # Risk
    avg_max_drawdown: float
    worst_drawdown: float
    
    # Trade stats
    total_trades: int
    avg_win_rate: float


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization
    
    Overfitting'i önlemek için:
    1. Veriyi eğitim/test pencerelerine böl
    2. Her pencerede optimize et ve test et
    3. Out-of-sample performansı ölç
    """
    
    def __init__(self,
                 n_windows: int = 5,
                 train_ratio: float = 0.7,
                 overlap: bool = False):
        """
        Args:
            n_windows: Pencere sayısı
            train_ratio: Eğitim verisi oranı (her pencerede)
            overlap: Pencereler örtüşsün mü?
        """
        self.n_windows = n_windows
        self.train_ratio = train_ratio
        self.overlap = overlap
    
    def _create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Pencereleri oluştur"""
        windows = []
        n = len(data)
        
        if self.overlap:
            # Örtüşen pencereler
            window_size = n // (self.n_windows // 2 + 1)
            step = window_size // 2
            
            for i in range(self.n_windows):
                start = i * step
                end = min(start + window_size, n)
                
                if end - start < 100:  # Minimum veri
                    break
                
                train_end_idx = start + int((end - start) * self.train_ratio)
                
                windows.append(WalkForwardWindow(
                    window_id=i,
                    train_start=data.index[start] if hasattr(data.index[0], 'strftime') else data.iloc[start]['timestamp'],
                    train_end=data.index[train_end_idx] if hasattr(data.index[0], 'strftime') else data.iloc[train_end_idx]['timestamp'],
                    test_start=data.index[train_end_idx] if hasattr(data.index[0], 'strftime') else data.iloc[train_end_idx]['timestamp'],
                    test_end=data.index[end-1] if hasattr(data.index[0], 'strftime') else data.iloc[end-1]['timestamp']
                ))
        else:
            # Ardışık pencereler
            window_size = n // self.n_windows
            
            for i in range(self.n_windows):
                start = i * window_size
                end = min((i + 1) * window_size, n)
                
                if end - start < 100:
                    break
                
                train_end_idx = start + int((end - start) * self.train_ratio)
                
                windows.append(WalkForwardWindow(
                    window_id=i,
                    train_start=data.index[start] if hasattr(data.index[0], 'strftime') else data.iloc[start]['timestamp'],
                    train_end=data.index[train_end_idx] if hasattr(data.index[0], 'strftime') else data.iloc[train_end_idx]['timestamp'],
                    test_start=data.index[train_end_idx] if hasattr(data.index[0], 'strftime') else data.iloc[train_end_idx]['timestamp'],
                    test_end=data.index[end-1] if hasattr(data.index[0], 'strftime') else data.iloc[end-1]['timestamp']
                ))
        
        return windows
    
    def _get_window_data(self, data: pd.DataFrame, start, end) -> pd.DataFrame:
        """Pencere verisini al"""
        if 'timestamp' in data.columns:
            mask = (data['timestamp'] >= start) & (data['timestamp'] <= end)
        else:
            mask = (data.index >= start) & (data.index <= end)
        
        return data[mask].copy()
    
    def run(self, 
            data: pd.DataFrame, 
            strategy_fn: Callable,
            initial_balance: float = 10000,
            optimize_fn: Callable = None,
            progress_callback: Callable = None) -> WalkForwardResult:
        """
        Walk-forward optimization çalıştır
        
        Args:
            data: OHLCV DataFrame
            strategy_fn: Strateji fonksiyonu
            initial_balance: Başlangıç bakiyesi
            optimize_fn: Parametre optimizasyon fonksiyonu (opsiyonel)
            progress_callback: İlerleme callback'i
        """
        windows = self._create_windows(data)
        
        for i, window in enumerate(windows):
            if progress_callback:
                progress_callback(i / len(windows) * 100, f"Window {i+1}/{len(windows)}")
            
            # Train data
            train_data = self._get_window_data(data, window.train_start, window.train_end)
            
            # Optimize (eğer fonksiyon verilmişse)
            if optimize_fn:
                best_params = optimize_fn(train_data, strategy_fn, initial_balance)
                window.best_params = best_params
            
            # Train backtest
            engine = BacktestEngine(initial_balance=initial_balance, random_seed=42)
            engine.load_data(train_data)
            engine.set_strategy(strategy_fn)
            window.train_result = engine.run()
            
            # Test data
            test_data = self._get_window_data(data, window.test_start, window.test_end)
            
            if len(test_data) > 50:
                # Test backtest
                engine = BacktestEngine(initial_balance=initial_balance, random_seed=42)
                engine.load_data(test_data)
                engine.set_strategy(strategy_fn)
                window.test_result = engine.run()
        
        return self._calculate_results(windows)
    
    def _calculate_results(self, windows: List[WalkForwardWindow]) -> WalkForwardResult:
        """Sonuçları hesapla"""
        train_returns = []
        test_returns = []
        drawdowns = []
        win_rates = []
        total_trades = 0
        
        for w in windows:
            if w.train_result:
                train_returns.append(w.train_result.total_return_pct)
            
            if w.test_result:
                test_returns.append(w.test_result.total_return_pct)
                drawdowns.append(w.test_result.max_drawdown_pct)
                win_rates.append(w.test_result.win_rate)
                total_trades += w.test_result.total_trades
        
        avg_train = np.mean(train_returns) if train_returns else 0
        avg_test = np.mean(test_returns) if test_returns else 0
        
        # Consistency score (test/train ratio)
        consistency = avg_test / avg_train if avg_train > 0 else 0
        
        # Degradation (in-sample vs out-of-sample)
        degradation = avg_train - avg_test
        
        # Combined return (test sonuçları)
        total_return_pct = sum(test_returns) if test_returns else 0
        
        return WalkForwardResult(
            windows=windows,
            total_return=total_return_pct * 100,  # Yaklaşık
            total_return_pct=total_return_pct,
            avg_train_return=avg_train,
            avg_test_return=avg_test,
            consistency_score=consistency,
            degradation=degradation,
            avg_max_drawdown=np.mean(drawdowns) if drawdowns else 0,
            worst_drawdown=max(drawdowns) if drawdowns else 0,
            total_trades=total_trades,
            avg_win_rate=np.mean(win_rates) if win_rates else 0
        )


# ================================================================
# MONTE CARLO SİMÜLASYONU
# ================================================================

@dataclass
class MonteCarloResult:
    """Monte Carlo sonucu"""
    n_simulations: int
    
    # Return dağılımı
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    
    # Percentiles
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    # Drawdown dağılımı
    mean_drawdown: float
    max_drawdown_95: float  # %95 güven aralığı
    
    # Probability
    profit_probability: float  # Kar olasılığı
    ruin_probability: float  # %20+ kayıp olasılığı
    
    # Tüm simülasyon sonuçları
    returns: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    equity_curves: List[List[float]] = field(default_factory=list)


class MonteCarloSimulator:
    """
    Monte Carlo Simülasyonu
    
    Trade sırasını rastgeleleştirerek:
    1. Stratejinin şans faktörüne bağımlılığını ölç
    2. Gerçekçi güven aralıkları hesapla
    3. Worst-case senaryoları simüle et
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def run_from_trades(self, 
                        trades: List[Dict], 
                        initial_balance: float = 10000) -> MonteCarloResult:
        """
        Mevcut trade listesinden Monte Carlo simülasyonu
        
        Args:
            trades: Trade listesi (pnl içermeli)
            initial_balance: Başlangıç bakiyesi
        """
        if not trades:
            return None
        
        # Trade P&L'leri
        pnls = [t['pnl'] if isinstance(t, dict) else t.pnl for t in trades]
        
        returns = []
        drawdowns = []
        equity_curves = []
        
        for sim in range(self.n_simulations):
            # Trade sırasını karıştır
            shuffled_pnls = pnls.copy()
            random.shuffle(shuffled_pnls)
            
            # Equity curve hesapla
            equity = [initial_balance]
            peak = initial_balance
            max_dd = 0
            
            for pnl in shuffled_pnls:
                new_equity = equity[-1] + pnl
                equity.append(new_equity)
                
                if new_equity > peak:
                    peak = new_equity
                
                dd = (peak - new_equity) / peak * 100
                max_dd = max(max_dd, dd)
            
            final_return = (equity[-1] - initial_balance) / initial_balance * 100
            returns.append(final_return)
            drawdowns.append(max_dd)
            
            if sim < 100:  # İlk 100 curve'ü sakla
                equity_curves.append(equity)
        
        returns = np.array(returns)
        drawdowns = np.array(drawdowns)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            mean_return=np.mean(returns),
            median_return=np.median(returns),
            std_return=np.std(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            percentile_5=np.percentile(returns, 5),
            percentile_25=np.percentile(returns, 25),
            percentile_75=np.percentile(returns, 75),
            percentile_95=np.percentile(returns, 95),
            mean_drawdown=np.mean(drawdowns),
            max_drawdown_95=np.percentile(drawdowns, 95),
            profit_probability=np.mean(returns > 0) * 100,
            ruin_probability=np.mean(returns < -20) * 100,
            returns=returns.tolist(),
            drawdowns=drawdowns.tolist(),
            equity_curves=equity_curves
        )
    
    def run_from_backtest(self,
                          data: pd.DataFrame,
                          strategy_fn: Callable,
                          initial_balance: float = 10000,
                          n_simulations: int = None,
                          progress_callback: Callable = None) -> MonteCarloResult:
        """
        Backtest sonuçlarından Monte Carlo simülasyonu
        
        Her simülasyonda farklı başlangıç noktası kullanır
        """
        if n_simulations:
            self.n_simulations = n_simulations
        
        returns = []
        drawdowns = []
        
        n = len(data)
        min_length = n // 2  # Minimum %50 veri kullan
        
        for sim in range(self.n_simulations):
            if progress_callback and sim % 100 == 0:
                progress_callback(sim / self.n_simulations * 100)
            
            # Rastgele başlangıç ve uzunluk
            start_idx = random.randint(0, n - min_length)
            length = random.randint(min_length, n - start_idx)
            
            subset = data.iloc[start_idx:start_idx + length].copy()
            
            # Backtest
            engine = BacktestEngine(
                initial_balance=initial_balance,
                random_seed=sim  # Her simülasyon için farklı seed
            )
            engine.load_data(subset)
            engine.set_strategy(strategy_fn)
            result = engine.run()
            
            returns.append(result.total_return_pct)
            drawdowns.append(result.max_drawdown_pct)
        
        returns = np.array(returns)
        drawdowns = np.array(drawdowns)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            mean_return=np.mean(returns),
            median_return=np.median(returns),
            std_return=np.std(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            percentile_5=np.percentile(returns, 5),
            percentile_25=np.percentile(returns, 25),
            percentile_75=np.percentile(returns, 75),
            percentile_95=np.percentile(returns, 95),
            mean_drawdown=np.mean(drawdowns),
            max_drawdown_95=np.percentile(drawdowns, 95),
            profit_probability=np.mean(returns > 0) * 100,
            ruin_probability=np.mean(returns < -20) * 100,
            returns=returns.tolist(),
            drawdowns=drawdowns.tolist()
        )


# ================================================================
# STRESS TESTING
# ================================================================

@dataclass
class StressTestResult:
    """Stress test sonucu"""
    scenario_name: str
    description: str
    baseline_return: float
    stressed_return: float
    impact: float  # % değişim
    max_drawdown: float
    survived: bool  # Bakiye > 0 kaldı mı?


class StressTester:
    """
    Stress Testing
    
    Black swan senaryoları:
    1. Flash crash (%10-30 ani düşüş)
    2. Volatilite spike
    3. Likidite krizi
    4. Ardışık kayıp serisi
    """
    
    SCENARIOS = {
        'flash_crash_10': {
            'description': '%10 ani düşüş',
            'price_impact': -0.10,
            'duration_bars': 3
        },
        'flash_crash_20': {
            'description': '%20 ani düşüş',
            'price_impact': -0.20,
            'duration_bars': 5
        },
        'flash_crash_30': {
            'description': '%30 ani düşüş (2020 Mart)',
            'price_impact': -0.30,
            'duration_bars': 10
        },
        'volatility_spike_2x': {
            'description': '2x volatilite artışı',
            'volatility_multiplier': 2.0,
            'duration_bars': 24
        },
        'volatility_spike_3x': {
            'description': '3x volatilite artışı',
            'volatility_multiplier': 3.0,
            'duration_bars': 24
        },
        'liquidity_crisis': {
            'description': 'Likidite krizi (spread %1)',
            'spread_multiplier': 50,  # Normal %0.02 -> %1
            'slippage_multiplier': 10,
            'duration_bars': 48
        },
        'consecutive_losses': {
            'description': '10 ardışık kayıp',
            'force_losses': 10
        }
    }
    
    def __init__(self):
        pass
    
    def _apply_flash_crash(self, data: pd.DataFrame, impact: float, 
                           duration: int, start_idx: int = None) -> pd.DataFrame:
        """Flash crash uygula"""
        df = data.copy()
        n = len(df)
        
        if start_idx is None:
            start_idx = n // 2
        
        # Crash
        crash_factor = 1 + impact
        for i in range(start_idx, min(start_idx + duration, n)):
            progress = (i - start_idx) / duration
            factor = 1 - (1 - crash_factor) * progress
            
            df.iloc[i, df.columns.get_loc('open')] *= factor
            df.iloc[i, df.columns.get_loc('high')] *= factor
            df.iloc[i, df.columns.get_loc('low')] *= factor
            df.iloc[i, df.columns.get_loc('close')] *= factor
        
        # Recovery (yavaş)
        recovery_duration = duration * 5
        for i in range(start_idx + duration, min(start_idx + duration + recovery_duration, n)):
            progress = (i - start_idx - duration) / recovery_duration
            factor = crash_factor + (1 - crash_factor) * progress
            
            df.iloc[i, df.columns.get_loc('open')] *= factor
            df.iloc[i, df.columns.get_loc('high')] *= factor
            df.iloc[i, df.columns.get_loc('low')] *= factor
            df.iloc[i, df.columns.get_loc('close')] *= factor
        
        return df
    
    def _apply_volatility_spike(self, data: pd.DataFrame, multiplier: float,
                                duration: int, start_idx: int = None) -> pd.DataFrame:
        """Volatilite spike uygula"""
        df = data.copy()
        n = len(df)
        
        if start_idx is None:
            start_idx = n // 2
        
        for i in range(start_idx, min(start_idx + duration, n)):
            close = df.iloc[i]['close']
            current_range = df.iloc[i]['high'] - df.iloc[i]['low']
            new_range = current_range * multiplier
            
            df.iloc[i, df.columns.get_loc('high')] = close + new_range / 2
            df.iloc[i, df.columns.get_loc('low')] = close - new_range / 2
        
        return df
    
    def run_scenario(self,
                     data: pd.DataFrame,
                     strategy_fn: Callable,
                     scenario_name: str,
                     initial_balance: float = 10000) -> StressTestResult:
        """Tek senaryo çalıştır"""
        scenario = self.SCENARIOS.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Baseline backtest
        engine = BacktestEngine(initial_balance=initial_balance, random_seed=42)
        engine.load_data(data)
        engine.set_strategy(strategy_fn)
        baseline_result = engine.run()
        
        # Stressed data oluştur
        if 'price_impact' in scenario:
            stressed_data = self._apply_flash_crash(
                data, 
                scenario['price_impact'],
                scenario['duration_bars']
            )
        elif 'volatility_multiplier' in scenario:
            stressed_data = self._apply_volatility_spike(
                data,
                scenario['volatility_multiplier'],
                scenario['duration_bars']
            )
        else:
            stressed_data = data.copy()
        
        # Stressed backtest
        engine = BacktestEngine(
            initial_balance=initial_balance,
            random_seed=42,
            use_spread='spread_multiplier' in scenario,
            use_slippage='slippage_multiplier' in scenario
        )
        
        if 'spread_multiplier' in scenario:
            engine.spread_model.base_spread *= scenario['spread_multiplier']
        if 'slippage_multiplier' in scenario:
            engine.slippage_model.base_slippage *= scenario['slippage_multiplier']
        
        engine.load_data(stressed_data)
        engine.set_strategy(strategy_fn)
        stressed_result = engine.run()
        
        # Impact hesapla
        impact = stressed_result.total_return_pct - baseline_result.total_return_pct
        
        return StressTestResult(
            scenario_name=scenario_name,
            description=scenario['description'],
            baseline_return=baseline_result.total_return_pct,
            stressed_return=stressed_result.total_return_pct,
            impact=impact,
            max_drawdown=stressed_result.max_drawdown_pct,
            survived=stressed_result.final_balance > 0
        )
    
    def run_all_scenarios(self,
                          data: pd.DataFrame,
                          strategy_fn: Callable,
                          initial_balance: float = 10000,
                          progress_callback: Callable = None) -> List[StressTestResult]:
        """Tüm senaryoları çalıştır"""
        results = []
        scenarios = list(self.SCENARIOS.keys())
        
        for i, scenario_name in enumerate(scenarios):
            if progress_callback:
                progress_callback(i / len(scenarios) * 100, scenario_name)
            
            try:
                result = self.run_scenario(data, strategy_fn, scenario_name, initial_balance)
                results.append(result)
            except Exception as e:
                print(f"Scenario {scenario_name} failed: {e}")
        
        return results


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_backtest_report(result: BacktestResult):
    """Backtest raporunu yazdır"""
    print("\n" + "="*60)
    print("BACKTEST RAPORU")
    print("="*60)
    
    print(f"\n{'PERFORMANS':^60}")
    print("-"*60)
    print(f"Başlangıç Bakiye:    ${result.initial_balance:>12,.2f}")
    print(f"Son Bakiye:          ${result.final_balance:>12,.2f}")
    print(f"Net Kar/Zarar:       ${result.total_return:>12,.2f} ({result.total_return_pct:+.2f}%)")
    
    print(f"\n{'TRADE İSTATİSTİKLERİ':^60}")
    print("-"*60)
    print(f"Toplam Trade:        {result.total_trades:>12}")
    print(f"Kazanan:             {result.winning_trades:>12}")
    print(f"Kaybeden:            {result.losing_trades:>12}")
    print(f"Win Rate:            {result.win_rate:>11.1f}%")
    print(f"Profit Factor:       {result.profit_factor:>12.2f}")
    
    print(f"\n{'RİSK METRİKLERİ':^60}")
    print("-"*60)
    print(f"Max Drawdown:        {result.max_drawdown_pct:>11.2f}%")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
    print(f"Sortino Ratio:       {result.sortino_ratio:>12.2f}")
    print(f"Calmar Ratio:        {result.calmar_ratio:>12.2f}")
    
    print(f"\n{'MALİYETLER':^60}")
    print("-"*60)
    print(f"Toplam Komisyon:     ${result.total_commission:>12,.2f}")
    print(f"Toplam Slippage:     ${result.total_slippage:>12,.2f}")
    
    print("\n" + "="*60)


def print_monte_carlo_report(result: MonteCarloResult):
    """Monte Carlo raporunu yazdır"""
    print("\n" + "="*60)
    print("MONTE CARLO SİMÜLASYONU")
    print("="*60)
    
    print(f"\nSimülasyon Sayısı: {result.n_simulations}")
    
    print(f"\n{'GETİRİ DAĞILIMI':^60}")
    print("-"*60)
    print(f"Ortalama:            {result.mean_return:>11.2f}%")
    print(f"Medyan:              {result.median_return:>11.2f}%")
    print(f"Std Sapma:           {result.std_return:>11.2f}%")
    print(f"Min:                 {result.min_return:>11.2f}%")
    print(f"Max:                 {result.max_return:>11.2f}%")
    
    print(f"\n{'GÜVEN ARALIKLARI':^60}")
    print("-"*60)
    print(f"%5 Percentile:       {result.percentile_5:>11.2f}%")
    print(f"%25 Percentile:      {result.percentile_25:>11.2f}%")
    print(f"%75 Percentile:      {result.percentile_75:>11.2f}%")
    print(f"%95 Percentile:      {result.percentile_95:>11.2f}%")
    
    print(f"\n{'RİSK ANALİZİ':^60}")
    print("-"*60)
    print(f"Ortalama Drawdown:   {result.mean_drawdown:>11.2f}%")
    print(f"%95 Max Drawdown:    {result.max_drawdown_95:>11.2f}%")
    print(f"Kar Olasılığı:       {result.profit_probability:>11.1f}%")
    print(f"Ruin Olasılığı:      {result.ruin_probability:>11.1f}%")
    
    print("\n" + "="*60)


def print_walk_forward_report(result: WalkForwardResult):
    """Walk-forward raporunu yazdır"""
    print("\n" + "="*60)
    print("WALK-FORWARD OPTİMİZASYON")
    print("="*60)
    
    print(f"\nPencere Sayısı: {len(result.windows)}")
    
    print(f"\n{'PENCERE SONUÇLARI':^60}")
    print("-"*60)
    print(f"{'#':>3} {'Train %':>10} {'Test %':>10} {'DD %':>10}")
    print("-"*60)
    
    for w in result.windows:
        train_ret = w.train_result.total_return_pct if w.train_result else 0
        test_ret = w.test_result.total_return_pct if w.test_result else 0
        dd = w.test_result.max_drawdown_pct if w.test_result else 0
        print(f"{w.window_id:>3} {train_ret:>10.2f} {test_ret:>10.2f} {dd:>10.2f}")
    
    print(f"\n{'ÖZET METRİKLER':^60}")
    print("-"*60)
    print(f"Toplam Getiri:       {result.total_return_pct:>11.2f}%")
    print(f"Ort. Train Getiri:   {result.avg_train_return:>11.2f}%")
    print(f"Ort. Test Getiri:    {result.avg_test_return:>11.2f}%")
    print(f"Tutarlılık Skoru:    {result.consistency_score:>11.2f}")
    print(f"Degradasyon:         {result.degradation:>11.2f}%")
    print(f"Ort. Win Rate:       {result.avg_win_rate:>11.1f}%")
    
    print("\n" + "="*60)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Advanced Backtest Tools test ediliyor...\n")
    
    # Test verisi
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1H')
    returns = np.random.randn(2000) * 0.001 + 0.00005
    prices = 50000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(2000) * 0.001),
        'high': prices * (1 + abs(np.random.randn(2000)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(2000)) * 0.005),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 2000)
    })
    
    # Test stratejisi
    def test_strategy(engine, bar):
        rsi = engine.get_indicator('rsi', 14)
        if rsi is None:
            return None
        
        if rsi < 30 and engine.position.side == PositionSide.FLAT:
            engine.submit_order("LONG", stop_loss=bar.close * 0.98, take_profit=bar.close * 1.03)
            return "LONG"
        elif rsi > 70 and engine.position.side == PositionSide.LONG:
            return "CLOSE"
        return None
    
    # 1. Basic Backtest
    print("1. Basic Backtest")
    engine = BacktestEngine(initial_balance=10000, random_seed=42)
    engine.load_data(df)
    engine.set_strategy(test_strategy)
    result = engine.run()
    print(f"   Return: {result.total_return_pct:+.2f}%, Trades: {result.total_trades}")
    
    # 2. Monte Carlo
    print("\n2. Monte Carlo Simülasyonu")
    mc = MonteCarloSimulator(n_simulations=100)
    if result.trades:
        trades = [{'pnl': t.pnl} for t in result.trades]
        mc_result = mc.run_from_trades(trades, 10000)
        print(f"   Mean Return: {mc_result.mean_return:+.2f}%")
        print(f"   Profit Prob: {mc_result.profit_probability:.1f}%")
        print(f"   %95 Max DD:  {mc_result.max_drawdown_95:.2f}%")
    
    # 3. Walk-Forward
    print("\n3. Walk-Forward Optimization")
    wf = WalkForwardOptimizer(n_windows=3)
    wf_result = wf.run(df, test_strategy, 10000)
    print(f"   Avg Test Return: {wf_result.avg_test_return:+.2f}%")
    print(f"   Consistency: {wf_result.consistency_score:.2f}")
    
    # 4. Stress Test
    print("\n4. Stress Testing")
    st = StressTester()
    stress_result = st.run_scenario(df, test_strategy, 'flash_crash_20', 10000)
    print(f"   Scenario: {stress_result.description}")
    print(f"   Baseline: {stress_result.baseline_return:+.2f}%")
    print(f"   Stressed: {stress_result.stressed_return:+.2f}%")
    print(f"   Impact: {stress_result.impact:+.2f}%")
    
    print("\n✓ Advanced Backtest Tools testi başarılı!")
