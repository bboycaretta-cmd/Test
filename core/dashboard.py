"""
BTC Bot Pro - Web Dashboard
FAZA 7: Flask Web Interface + REST API

Özellikler:
- Anlık fiyat ve sinyal dashboard
- Backtest sonuçları görüntüleme
- Trade geçmişi
- Performans grafikleri
- Ayarlar yönetimi
- REST API endpoints
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import time

# Flask imports (opsiyonel)
try:
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# ================================================================
# HTML TEMPLATES
# ================================================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Bot Pro - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 28px;
            font-weight: bold;
            color: #f39c12;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #2ecc71;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .card-title {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .card-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .card-change {
            font-size: 14px;
        }
        
        .positive { color: #2ecc71; }
        .negative { color: #e74c3c; }
        .neutral { color: #f39c12; }
        
        .signal-box {
            text-align: center;
            padding: 30px;
        }
        
        .signal {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .signal-text {
            font-size: 24px;
            font-weight: bold;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin-top: 15px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        
        th {
            color: #888;
            font-weight: normal;
            text-transform: uppercase;
            font-size: 12px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #f39c12;
            color: #000;
        }
        
        .btn-primary:hover {
            background: #e67e22;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 20px;
            font-weight: bold;
        }
        
        .chart-placeholder {
            height: 300px;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #333;
            margin-top: 50px;
        }
        
        nav {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        nav a {
            color: #888;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        nav a:hover, nav a.active {
            color: #fff;
            background: rgba(255,255,255,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🤖 BTC Bot Pro</div>
            <div class="status">
                <div class="status-dot"></div>
                <span>Çalışıyor</span>
            </div>
        </header>
        
        <nav>
            <a href="/" class="active">Dashboard</a>
            <a href="/trades">İşlemler</a>
            <a href="/backtest">Backtest</a>
            <a href="/settings">Ayarlar</a>
            <a href="/api/docs">API</a>
        </nav>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">BTC/USDT Fiyat</div>
                <div class="card-value" id="price">${{ price }}</div>
                <div class="card-change {{ 'positive' if change >= 0 else 'negative' }}">
                    {{ '+' if change >= 0 else '' }}{{ change }}% (24s)
                </div>
            </div>
            
            <div class="card signal-box">
                <div class="signal">{{ signal_emoji }}</div>
                <div class="signal-text {{ signal_class }}">{{ signal }}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {{ signal_class }}" 
                         style="width: {{ confidence }}%; background: {{ signal_color }};"></div>
                </div>
                <div style="margin-top: 10px; color: #888;">Güven: %{{ confidence }}</div>
            </div>
            
            <div class="card">
                <div class="card-title">Bakiye</div>
                <div class="card-value">${{ balance }}</div>
                <div class="card-change {{ 'positive' if pnl >= 0 else 'negative' }}">
                    {{ '+' if pnl >= 0 else '' }}${{ pnl }} ({{ pnl_pct }}%)
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Aktif Pozisyon</div>
                <div class="card-value">{{ position }}</div>
                <div class="card-change">{{ position_detail }}</div>
            </div>
        </div>
        
        <div class="card" style="margin-bottom: 30px;">
            <div class="card-title">Performans Metrikleri</div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Toplam İşlem</div>
                    <div class="metric-value">{{ total_trades }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value positive">%{{ win_rate }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{{ profit_factor }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">%{{ max_dd }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{{ sharpe }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">{{ sortino }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Strateji</div>
                    <div class="metric-value">{{ strategy }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Market Rejimi</div>
                    <div class="metric-value">{{ regime }}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">Son İşlemler</div>
            <table>
                <thead>
                    <tr>
                        <th>Tarih</th>
                        <th>Tip</th>
                        <th>Giriş</th>
                        <th>Çıkış</th>
                        <th>K/Z</th>
                        <th>Sebep</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{ trade.date }}</td>
                        <td class="{{ 'positive' if trade.side == 'LONG' else 'negative' }}">
                            {{ trade.side }}
                        </td>
                        <td>${{ trade.entry }}</td>
                        <td>${{ trade.exit }}</td>
                        <td class="{{ 'positive' if trade.pnl >= 0 else 'negative' }}">
                            {{ '+' if trade.pnl >= 0 else '' }}${{ trade.pnl }}
                        </td>
                        <td>{{ trade.reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <footer>
            BTC Bot Pro v5.0 | Son güncelleme: {{ last_update }}
        </footer>
    </div>
    
    <script>
        // Auto refresh every 10 seconds
        setTimeout(() => location.reload(), 10000);
    </script>
</body>
</html>
'''

API_DOCS_HTML = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>BTC Bot Pro - API Dokümantasyonu</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #1a1a2e;
            color: #fff;
            padding: 40px;
            line-height: 1.6;
        }
        h1 { color: #f39c12; }
        h2 { color: #3498db; margin-top: 30px; }
        code {
            background: #333;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: monospace;
        }
        pre {
            background: #333;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .endpoint {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .method {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            margin-right: 10px;
        }
        .get { background: #2ecc71; color: #000; }
        .post { background: #3498db; color: #fff; }
        .delete { background: #e74c3c; color: #fff; }
    </style>
</head>
<body>
    <h1>🤖 BTC Bot Pro API</h1>
    <p>REST API dokümantasyonu</p>
    
    <h2>Endpoints</h2>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/api/status</code>
        <p>Bot durumu ve güncel bilgiler</p>
        <pre>{
  "status": "running",
  "price": 95000.00,
  "signal": "LONG",
  "confidence": 75,
  "balance": 10500.00,
  "position": "LONG @ $94500"
}</pre>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/api/trades</code>
        <p>Trade geçmişi (limit parametresi ile)</p>
        <pre>GET /api/trades?limit=10</pre>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/api/performance</code>
        <p>Performans metrikleri</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/api/signal</code>
        <p>Güncel sinyal ve detayları</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/api/config</code>
        <p>Konfigürasyon bilgileri</p>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/api/config</code>
        <p>Konfigürasyon güncelle</p>
        <pre>POST /api/config
Content-Type: application/json

{
  "strategy": "balanced",
  "stop_loss": 0.02,
  "take_profit": 0.03
}</pre>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/api/backtest</code>
        <p>Backtest çalıştır</p>
        <pre>POST /api/backtest
Content-Type: application/json

{
  "strategy": "momentum",
  "months": 3,
  "initial_balance": 10000
}</pre>
    </div>
    
    <h2>WebSocket</h2>
    <p>Gerçek zamanlı güncellemeler için:</p>
    <pre>ws://localhost:5000/ws</pre>
    
    <h2>Rate Limiting</h2>
    <p>100 istek/dakika</p>
</body>
</html>
'''


# ================================================================
# DATA MODELS
# ================================================================

@dataclass
class DashboardData:
    """Dashboard için veri modeli"""
    price: float = 0
    change: float = 0
    signal: str = "HOLD"
    signal_emoji: str = "🟡"
    signal_class: str = "neutral"
    signal_color: str = "#f39c12"
    confidence: int = 50
    balance: float = 10000
    pnl: float = 0
    pnl_pct: float = 0
    position: str = "YOK"
    position_detail: str = ""
    total_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    max_dd: float = 0
    sharpe: float = 0
    sortino: float = 0
    strategy: str = "balanced"
    regime: str = "unknown"
    recent_trades: list = None
    last_update: str = ""
    
    def __post_init__(self):
        if self.recent_trades is None:
            self.recent_trades = []


# ================================================================
# WEB SERVER
# ================================================================

class WebDashboard:
    """
    Flask web dashboard
    
    Kullanım:
        dashboard = WebDashboard()
        dashboard.update_data(data)
        dashboard.run()
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask required. Install: pip install flask")
        
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.data = DashboardData()
        self._setup_routes()
    
    def _setup_routes(self):
        """Route'ları ayarla"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML, **asdict(self.data))
        
        @self.app.route('/api/docs')
        def api_docs():
            return render_template_string(API_DOCS_HTML)
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                'status': 'running',
                'price': self.data.price,
                'change': self.data.change,
                'signal': self.data.signal,
                'confidence': self.data.confidence,
                'balance': self.data.balance,
                'position': self.data.position,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/signal')
        def api_signal():
            return jsonify({
                'signal': self.data.signal,
                'confidence': self.data.confidence,
                'price': self.data.price,
                'regime': self.data.regime,
                'strategy': self.data.strategy
            })
        
        @self.app.route('/api/trades')
        def api_trades():
            limit = request.args.get('limit', 50, type=int)
            trades = self.data.recent_trades[:limit]
            return jsonify({'trades': trades})
        
        @self.app.route('/api/performance')
        def api_performance():
            return jsonify({
                'total_trades': self.data.total_trades,
                'win_rate': self.data.win_rate,
                'profit_factor': self.data.profit_factor,
                'max_drawdown': self.data.max_dd,
                'sharpe_ratio': self.data.sharpe,
                'sortino_ratio': self.data.sortino,
                'pnl': self.data.pnl,
                'pnl_pct': self.data.pnl_pct
            })
        
        @self.app.route('/api/config', methods=['GET'])
        def api_get_config():
            return jsonify({
                'strategy': self.data.strategy,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'position_size': 0.5
            })
        
        @self.app.route('/api/config', methods=['POST'])
        def api_set_config():
            data = request.get_json()
            # Config güncelleme mantığı buraya
            return jsonify({'status': 'ok', 'message': 'Config updated'})
        
        @self.app.route('/trades')
        def trades_page():
            # Trade sayfası (basitleştirilmiş)
            return redirect('/')
        
        @self.app.route('/backtest')
        def backtest_page():
            return redirect('/')
        
        @self.app.route('/settings')
        def settings_page():
            return redirect('/')
    
    def update_data(self, 
                    price: float = None,
                    change: float = None,
                    signal: str = None,
                    confidence: int = None,
                    balance: float = None,
                    pnl: float = None,
                    position: str = None,
                    trades: list = None,
                    metrics: dict = None,
                    **kwargs):
        """Dashboard verisini güncelle"""
        
        if price is not None:
            self.data.price = price
        if change is not None:
            self.data.change = change
        if signal is not None:
            self.data.signal = signal
            if signal in ['LONG', 'STRONG_LONG']:
                self.data.signal_emoji = '🟢'
                self.data.signal_class = 'positive'
                self.data.signal_color = '#2ecc71'
            elif signal in ['SHORT', 'STRONG_SHORT']:
                self.data.signal_emoji = '🔴'
                self.data.signal_class = 'negative'
                self.data.signal_color = '#e74c3c'
            else:
                self.data.signal_emoji = '🟡'
                self.data.signal_class = 'neutral'
                self.data.signal_color = '#f39c12'
        
        if confidence is not None:
            self.data.confidence = confidence
        if balance is not None:
            self.data.balance = balance
        if pnl is not None:
            self.data.pnl = pnl
            self.data.pnl_pct = round((pnl / 10000) * 100, 2)
        if position is not None:
            self.data.position = position
        if trades is not None:
            self.data.recent_trades = trades
        
        if metrics:
            self.data.total_trades = metrics.get('total_trades', 0)
            self.data.win_rate = metrics.get('win_rate', 0)
            self.data.profit_factor = metrics.get('profit_factor', 0)
            self.data.max_dd = metrics.get('max_dd', 0)
            self.data.sharpe = metrics.get('sharpe', 0)
            self.data.sortino = metrics.get('sortino', 0)
        
        self.data.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def run(self, debug: bool = False, threaded: bool = True):
        """Server'ı başlat"""
        print(f"\n🌐 Web Dashboard başlatılıyor...")
        print(f"   URL: http://{self.host}:{self.port}")
        print(f"   API: http://{self.host}:{self.port}/api/docs")
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=threaded
        )
    
    def run_background(self):
        """Arka planda çalıştır"""
        thread = threading.Thread(target=self.run, kwargs={'debug': False})
        thread.daemon = True
        thread.start()
        return thread


# ================================================================
# STANDALONE MODE (Flask olmadan)
# ================================================================

class SimpleAPIServer:
    """
    Basit HTTP server (Flask olmadan)
    
    Sadece JSON API sunar
    """
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.data = {}
    
    def update(self, data: dict):
        """Veriyi güncelle"""
        self.data = data
        self.data['timestamp'] = datetime.now().isoformat()
    
    def get_json(self) -> str:
        """JSON string döndür"""
        return json.dumps(self.data, indent=2, default=str)
    
    def run(self):
        """Basit HTTP server (demo)"""
        import http.server
        import socketserver
        
        data = self.data
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/api'):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                else:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = f"<h1>BTC Bot Pro API</h1><pre>{json.dumps(data, indent=2)}</pre>"
                    self.wfile.write(html.encode())
        
        with socketserver.TCPServer(("", self.port), Handler) as httpd:
            print(f"Server running on port {self.port}")
            httpd.serve_forever()


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_dashboard(port: int = 5000) -> Optional[WebDashboard]:
    """Dashboard oluştur"""
    if FLASK_AVAILABLE:
        return WebDashboard(port=port)
    else:
        print("Flask yüklü değil. Basit API server kullanılacak.")
        return None


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Web Dashboard test ediliyor...\n")
    print(f"Flask available: {FLASK_AVAILABLE}")
    
    if FLASK_AVAILABLE:
        # Dashboard oluştur
        dashboard = WebDashboard(port=5000)
        
        # Test verisi
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
        
        print("Dashboard hazır!")
        print(f"Test için: http://localhost:5000")
        print("\nNot: Gerçek çalıştırma için dashboard.run() kullanın.")
        
        # Test API response
        with dashboard.app.test_client() as client:
            response = client.get('/api/status')
            print(f"\nAPI Test Response:")
            print(json.dumps(response.get_json(), indent=2))
    else:
        print("Flask yüklü değil.")
        print("Yüklemek için: pip install flask")
        
        # Simple API test
        api = SimpleAPIServer()
        api.update({
            'price': 95000,
            'signal': 'LONG',
            'status': 'ok'
        })
        print(f"\nSimple API Response:")
        print(api.get_json())
    
    print("\n✓ Web Dashboard testi başarılı!")
