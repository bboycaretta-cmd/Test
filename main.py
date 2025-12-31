"""
BTC Bot Pro v5.0 - Ana Çalıştırma Scripti
==========================================

Kullanım:
    python main.py test      # Tüm modülleri test et
    python main.py backtest  # Backtest çalıştır
    python main.py live      # Canlı sinyal takibi
    python main.py dashboard # Web dashboard başlat
"""

import sys
import os
import time
from datetime import datetime

# Core modülünü import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all():
    """Tüm modülleri test et"""
    print("\n" + "="*60)
    print("🧪 BTC BOT PRO v5.0 - MODÜL TESTİ")
    print("="*60)
    
    errors = []
    
    # 1. Database
    print("\n📦 Database testi...")
    try:
        from core import db
        db.init()
        print("   ✅ Database OK")
    except Exception as e:
        errors.append(f"Database: {e}")
        print(f"   ❌ {e}")
    
    # 2. Config
    print("\n⚙️ Config testi...")
    try:
        from core import config, STRATEGIES
        print(f"   ✅ {len(STRATEGIES)} strateji yüklendi")
        print(f"   📋 Stratejiler: {', '.join(list(STRATEGIES.keys())[:5])}...")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"   ❌ {e}")
    
    # 3. Feature Engineering
    print("\n📊 Feature Engineering testi...")
    try:
        import numpy as np
        import pandas as pd
        from core import FeatureEngineer
        
        # Test verisi oluştur
        np.random.seed(42)
        n = 100
        prices = 50000 * np.cumprod(1 + np.random.randn(n) * 0.01)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        engineer = FeatureEngineer()
        features = engineer.generate(df)
        print(f"   ✅ {len(engineer.get_feature_names())} feature üretildi")
    except Exception as e:
        errors.append(f"Features: {e}")
        print(f"   ❌ {e}")
    
    # 4. Backtest Engine
    print("\n🔄 Backtest Engine testi...")
    try:
        from core import BacktestEngine, PositionSide
        
        engine = BacktestEngine(initial_balance=10000)
        engine.load_data(df)
        
        def simple_strategy(engine, bar):
            rsi = engine.get_indicator('rsi', 14)
            if rsi and rsi < 30 and engine.position.side == PositionSide.FLAT:
                return 'LONG'
            elif rsi and rsi > 70 and engine.position.side == PositionSide.LONG:
                return 'CLOSE'
            return None
        
        engine.set_strategy(simple_strategy)
        result = engine.run()
        print(f"   ✅ Backtest tamamlandı")
        print(f"   📈 Getiri: {result.total_return_pct:+.2f}%")
        print(f"   📊 İşlem: {result.total_trades}")
    except Exception as e:
        errors.append(f"Backtest: {e}")
        print(f"   ❌ {e}")
    
    # 5. Risk Management
    print("\n🛡️ Risk Management testi...")
    try:
        from core import RiskCalculator, PositionSizer
        
        calc = RiskCalculator()
        equity = [10000 + i*10 + np.random.randn()*50 for i in range(100)]
        metrics = calc.calculate_all(equity)
        print(f"   ✅ Risk metrikleri hesaplandı")
        print(f"   📉 Max DD: {metrics.max_drawdown*100:.2f}%")
        print(f"   📊 Sharpe: {metrics.sharpe_ratio:.2f}")
    except Exception as e:
        errors.append(f"Risk: {e}")
        print(f"   ❌ {e}")
    
    # 6. Regime Detection
    print("\n🎯 Market Regime testi...")
    try:
        from core import RegimeDetector, AdaptiveStrategySelector
        
        detector = RegimeDetector()
        state = detector.detect(df)
        print(f"   ✅ Rejim tespit edildi: {state.regime.value}")
        print(f"   📊 ADX: {state.adx:.1f}, RSI: {state.rsi:.1f}")
    except Exception as e:
        errors.append(f"Regime: {e}")
        print(f"   ❌ {e}")
    
    # 7. Signal Generator
    print("\n📡 Signal Generator testi...")
    try:
        from core import generate_signal
        
        for pred in [1.5, -0.8, 0.2]:
            signal = generate_signal(pred)
            print(f"   {pred:+.1f}% → {signal.signal} (güven: {signal.confidence:.0f}%)")
    except Exception as e:
        errors.append(f"Signal: {e}")
        print(f"   ❌ {e}")
    
    # Sonuç
    print("\n" + "="*60)
    if errors:
        print(f"❌ {len(errors)} HATA BULUNDU:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
    
    return len(errors) == 0


def run_backtest():
    """Detaylı backtest çalıştır"""
    print("\n" + "="*60)
    print("📊 BTC BOT PRO - BACKTEST")
    print("="*60)
    
    import numpy as np
    import pandas as pd
    from core import BacktestEngine, PositionSide, FeatureEngineer
    
    # Veri oluştur (gerçek veri için Binance API kullanılabilir)
    print("\n📥 Test verisi oluşturuluyor...")
    np.random.seed(42)
    n = 1000  # 1000 saat (~41 gün)
    
    # Gerçekçi fiyat hareketi simüle et
    returns = np.random.randn(n) * 0.015  # %1.5 volatilite
    trend = np.linspace(0, 0.1, n)  # Hafif yukarı trend
    prices = 90000 * np.cumprod(1 + returns + trend/n)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
        'open': prices * (1 + np.random.randn(n) * 0.002),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.008),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.008),
        'close': prices,
        'volume': np.random.uniform(500, 2000, n)
    })
    
    print(f"   ✅ {n} bar veri oluşturuldu")
    print(f"   📅 {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    print(f"   💰 Fiyat: ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")
    
    # Strateji
    print("\n🎯 Strateji: RSI + EMA Crossover")
    
    def strategy(engine, bar):
        rsi = engine.get_indicator('rsi', 14)
        ema_fast = engine.get_indicator('ema', 12)
        ema_slow = engine.get_indicator('ema', 26)
        
        if not all([rsi, ema_fast, ema_slow]):
            return None
        
        if engine.position.side == PositionSide.FLAT:
            # Giriş koşulları
            if rsi < 35 and ema_fast > ema_slow:
                engine.submit_order('LONG', 
                    stop_loss=bar.close * 0.97,
                    take_profit=bar.close * 1.05)
                return 'LONG'
        
        elif engine.position.side == PositionSide.LONG:
            # Çıkış koşulları
            if rsi > 70 or ema_fast < ema_slow:
                return 'CLOSE'
        
        return None
    
    # Backtest
    print("\n⏳ Backtest çalışıyor...")
    engine = BacktestEngine(
        initial_balance=10000,
        commission=0.001,
        slippage_pct=0.0005
    )
    engine.load_data(df)
    engine.set_strategy(strategy)
    result = engine.run()
    
    # Sonuçlar
    print("\n" + "-"*60)
    print("📈 BACKTEST SONUÇLARI")
    print("-"*60)
    print(f"""
    💰 Başlangıç:      ${result.initial_balance:,.2f}
    💰 Final:          ${result.final_balance:,.2f}
    📊 Getiri:         {result.total_return_pct:+.2f}%
    
    📋 Toplam İşlem:   {result.total_trades}
    ✅ Kazanan:        {result.winning_trades}
    ❌ Kaybeden:       {result.losing_trades}
    🎯 Win Rate:       {result.win_rate:.1f}%
    
    📉 Max Drawdown:   {result.max_drawdown_pct:.2f}%
    📊 Sharpe Ratio:   {result.sharpe_ratio:.2f}
    💹 Profit Factor:  {result.profit_factor:.2f}
    
    💸 Komisyon:       ${result.total_commission:.2f}
    """)
    
    # Buy & Hold karşılaştırma
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"    🆚 Buy & Hold:    {buy_hold:+.2f}%")
    print(f"    📊 Fark:          {result.total_return_pct - buy_hold:+.2f}%")
    print("-"*60 + "\n")


def run_live_signals():
    """Canlı sinyal takibi (simülasyon)"""
    print("\n" + "="*60)
    print("📡 BTC BOT PRO - CANLI SİNYAL TAKİBİ")
    print("="*60)
    print("\n⚠️  Bu bir simülasyondur. Gerçek trade yapmaz.")
    print("    Çıkmak için CTRL+C\n")
    
    import numpy as np
    import pandas as pd
    from core import FeatureEngineer, generate_signal, RegimeDetector
    
    np.random.seed(int(time.time()))
    
    # Başlangıç verisi
    n = 100
    base_price = 95000
    prices = [base_price]
    
    detector = RegimeDetector()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Fiyat güncelle (rastgele yürüyüş)
            change = np.random.randn() * 0.003  # %0.3 volatilite
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
            if len(prices) > 200:
                prices = prices[-200:]
            
            # DataFrame oluştur
            df = pd.DataFrame({
                'timestamp': pd.date_range(end=datetime.now(), periods=len(prices), freq='1h'),
                'open': np.array(prices) * 0.999,
                'high': np.array(prices) * 1.002,
                'low': np.array(prices) * 0.998,
                'close': np.array(prices),
                'volume': np.random.uniform(500, 1500, len(prices))
            })
            
            # Feature hesapla
            engineer = FeatureEngineer(include_advanced=False)
            features = engineer.generate(df)
            
            # Basit tahmin (RSI bazlı)
            rsi = features['rsi_14'].iloc[-1] if 'rsi_14' in features else 50
            
            if rsi < 30:
                pred = 1.5
            elif rsi > 70:
                pred = -1.5
            else:
                pred = (50 - rsi) / 50
            
            # Sinyal
            signal = generate_signal(pred)
            
            # Rejim
            if len(df) >= 100:
                regime = detector.detect(df)
                regime_str = regime.regime.value
            else:
                regime_str = "calculating..."
            
            # Ekrana yazdır
            change_pct = (new_price / prices[-2] - 1) * 100 if len(prices) > 1 else 0
            change_symbol = "🟢" if change_pct >= 0 else "🔴"
            
            signal_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(signal.signal, "⚪")
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"BTC: ${new_price:,.2f} {change_symbol}{change_pct:+.2f}% | "
                  f"RSI: {rsi:.0f} | "
                  f"Sinyal: {signal_emoji} {signal.signal} ({signal.confidence:.0f}%) | "
                  f"Rejim: {regime_str[:10]}     ", end="", flush=True)
            
            time.sleep(2)  # 2 saniye bekle
            
    except KeyboardInterrupt:
        print("\n\n👋 Sinyal takibi durduruldu.\n")


def run_dashboard():
    """Web dashboard başlat"""
    print("\n" + "="*60)
    print("🌐 BTC BOT PRO - WEB DASHBOARD")
    print("="*60)
    
    try:
        from core import WebDashboard, FLASK_AVAILABLE
        
        if not FLASK_AVAILABLE:
            print("\n❌ Flask yüklü değil!")
            print("   Yüklemek için: pip install flask")
            return
        
        dashboard = WebDashboard(host='127.0.0.1', port=5000)
        
        # Demo veri
        dashboard.update_data(
            price=95234.56,
            change=2.34,
            signal='LONG',
            confidence=75,
            balance=10500,
            pnl=500,
            position='LONG @ $94500',
            trades=[
                {'date': '2024-01-15 14:30', 'side': 'LONG', 'entry': '94000', 
                 'exit': '95500', 'pnl': 150, 'reason': 'TP'},
                {'date': '2024-01-15 10:15', 'side': 'SHORT', 'entry': '95000', 
                 'exit': '94200', 'pnl': 80, 'reason': 'Signal'},
                {'date': '2024-01-14 22:00', 'side': 'LONG', 'entry': '93500', 
                 'exit': '94800', 'pnl': 130, 'reason': 'TP'},
            ],
            metrics={
                'total_trades': 45,
                'win_rate': 62.5,
                'profit_factor': 1.85,
                'max_dd': 8.5,
                'sharpe': 1.42,
                'sortino': 1.95
            }
        )
        
        print("\n✅ Dashboard başlatılıyor...")
        print("   🌐 URL: http://127.0.0.1:5000")
        print("   📡 API: http://127.0.0.1:5000/api/docs")
        print("\n   Durdurmak için CTRL+C\n")
        
        dashboard.run(debug=False)
        
    except Exception as e:
        print(f"\n❌ Dashboard hatası: {e}")


def show_help():
    """Yardım göster"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    🤖 BTC BOT PRO v5.0                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  KULLANIM:                                                       ║
║    python main.py <komut>                                        ║
║                                                                  ║
║  KOMUTLAR:                                                       ║
║    test       Tüm modülleri test et                              ║
║    backtest   Backtest çalıştır                                  ║
║    live       Canlı sinyal takibi (simülasyon)                   ║
║    dashboard  Web dashboard başlat                               ║
║    help       Bu yardımı göster                                  ║
║                                                                  ║
║  ÖRNEKLER:                                                       ║
║    python main.py test                                           ║
║    python main.py backtest                                       ║
║    python main.py dashboard                                      ║
║                                                                  ║
║  GEREKSİNİMLER:                                                  ║
║    pip install numpy pandas scikit-learn requests flask          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


# Ana program
if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "test":
        test_all()
    elif command == "backtest":
        run_backtest()
    elif command == "live":
        run_live_signals()
    elif command == "dashboard":
        run_dashboard()
    elif command in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"❌ Bilinmeyen komut: {command}")
        show_help()
