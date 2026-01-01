"""
BTC Bot Pro v5.0 - Tam Ozellikli Web Dashboard
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from flask import Flask, render_template_string, jsonify, request, session
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# ================================================================
# ANA HTML SABLONU
# ================================================================

MAIN_HTML = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Bot Pro v5.0</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #12122a;
            --bg-card: #1a1a3e;
            --text-primary: #ffffff;
            --text-secondary: #a0a0c0;
            --accent: #f7931a;
            --green: #00d4aa;
            --red: #ff4757;
            --blue: #3498db;
            --purple: #9b59b6;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 260px;
            height: 100vh;
            background: var(--bg-secondary);
            padding: 20px;
            border-right: 1px solid rgba(255,255,255,0.1);
            overflow-y: auto;
        }
        
        .logo {
            font-size: 22px;
            font-weight: bold;
            color: var(--accent);
            margin-bottom: 8px;
        }
        
        .logo-sub {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 25px;
        }
        
        .nav-section {
            margin-bottom: 20px;
        }
        
        .nav-title {
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 3px;
            text-decoration: none;
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        .nav-item:hover, .nav-item.active {
            background: rgba(247, 147, 26, 0.15);
            color: var(--accent);
        }
        
        .nav-icon { font-size: 16px; }
        
        .main {
            margin-left: 260px;
            padding: 25px;
            min-height: 100vh;
        }
        
        .page-title {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .page-subtitle {
            color: var(--text-secondary);
            margin-bottom: 25px;
            font-size: 13px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .card-title {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .card-value {
            font-size: 28px;
            font-weight: bold;
        }
        
        .card-change { font-size: 13px; margin-top: 5px; }
        
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        
        @media (max-width: 1200px) {
            .grid-4, .grid-3 { grid-template-columns: repeat(2, 1fr); }
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .btn-primary { background: var(--accent); color: #000; }
        .btn-primary:hover { background: #ffaa33; }
        .btn-secondary { background: rgba(255,255,255,0.1); color: var(--text-primary); }
        .btn-success { background: var(--green); color: #000; }
        .btn-danger { background: var(--red); color: #fff; }
        .btn-lg { padding: 12px 25px; font-size: 14px; }
        
        .form-group { margin-bottom: 15px; }
        .form-label { display: block; margin-bottom: 6px; color: var(--text-secondary); font-size: 12px; }
        .form-control {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 13px;
        }
        .form-control:focus { outline: none; border-color: var(--accent); }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }
        th { color: var(--text-secondary); font-weight: normal; font-size: 11px; text-transform: uppercase; }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: 500;
        }
        .badge-success { background: rgba(0,212,170,0.2); color: var(--green); }
        .badge-danger { background: rgba(255,71,87,0.2); color: var(--red); }
        .badge-warning { background: rgba(247,147,26,0.2); color: var(--accent); }
        
        .tabs { display: flex; gap: 5px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
        .tab { padding: 8px 15px; cursor: pointer; border-radius: 6px; color: var(--text-secondary); font-size: 13px; }
        .tab:hover, .tab.active { background: rgba(247,147,26,0.1); color: var(--accent); }
        
        .chart-container { position: relative; height: 280px; margin-top: 15px; }
        
        .alert {
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
        }
        .alert-info { background: rgba(52,152,219,0.1); border: 1px solid rgba(52,152,219,0.3); }
        .alert-warning { background: rgba(247,147,26,0.1); border: 1px solid rgba(247,147,26,0.3); }
        .alert-success { background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.3); }
        
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .stat-item { text-align: center; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 8px; }
        .stat-label { font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; }
        .stat-value { font-size: 20px; font-weight: bold; }
        
        .section { display: none; }
        .section.active { display: block; }
        
        .progress { height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; margin-top: 12px; }
        .progress-bar { height: 100%; border-radius: 3px; transition: width 0.3s; }
        
        .strategy-card {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }
        .strategy-card:hover { background: rgba(255,255,255,0.05); }
        .strategy-card.selected { border-color: var(--accent); background: rgba(247,147,26,0.1); }
        .strategy-name { font-size: 15px; font-weight: bold; margin-bottom: 6px; }
        .strategy-desc { font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; }
        .strategy-stats { display: flex; gap: 15px; font-size: 11px; }
        .strategy-stat-label { color: var(--text-secondary); }
        
        .live-dot {
            width: 8px; height: 8px;
            background: var(--green);
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .signal-box { text-align: center; padding: 25px; }
        .signal-icon { font-size: 50px; margin-bottom: 10px; }
        .signal-text { font-size: 24px; font-weight: bold; }
        
        .spinner {
            width: 35px; height: 35px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo">BTC Bot Pro</div>
        <div class="logo-sub">v5.0 - Akilli Trading Sistemi</div>
        
        <div class="nav-section">
            <div class="nav-title">Ana Menu</div>
            <a class="nav-item active" onclick="showSection('dashboard', this)">
                <span class="nav-icon">📊</span> Dashboard
            </a>
            <a class="nav-item" onclick="showSection('account', this)">
                <span class="nav-icon">👤</span> Hesap Ayarlari
            </a>
        </div>
        
        <div class="nav-section">
            <div class="nav-title">Strateji</div>
            <a class="nav-item" onclick="showSection('strategy', this)">
                <span class="nav-icon">🎯</span> Strateji Sec
            </a>
            <a class="nav-item" onclick="showSection('optimize', this)">
                <span class="nav-icon">🔍</span> En Iyiyi Bul
            </a>
            <a class="nav-item" onclick="showSection('backtest', this)">
                <span class="nav-icon">📈</span> Backtest
            </a>
        </div>
        
        <div class="nav-section">
            <div class="nav-title">Model</div>
            <a class="nav-item" onclick="showSection('train', this)">
                <span class="nav-icon">🧠</span> Model Egit
            </a>
        </div>
        
        <div class="nav-section">
            <div class="nav-title">Trading</div>
            <a class="nav-item" onclick="showSection('live', this)">
                <span class="nav-icon">⚡</span> Canli Test
            </a>
            <a class="nav-item" onclick="showSection('history', this)">
                <span class="nav-icon">📜</span> Islem Gecmisi
            </a>
        </div>
        
        <div class="nav-section">
            <div class="nav-title">Yardim</div>
            <a class="nav-item" onclick="showSection('guide', this)">
                <span class="nav-icon">📖</span> Kullanim Rehberi
            </a>
            <a class="nav-item" onclick="showSection('about', this)">
                <span class="nav-icon">ℹ️</span> Hakkinda
            </a>
        </div>
    </div>
    
    <div class="main">
        
        <!-- Dashboard -->
        <div id="section-dashboard" class="section active">
            <h1 class="page-title">📊 Dashboard</h1>
            <p class="page-subtitle">Genel bakis ve canli veriler</p>
            
            <div class="grid-4">
                <div class="card">
                    <div class="card-title">💰 BTC/USDT</div>
                    <div class="card-value" id="price">$95,234</div>
                    <div class="card-change positive">+2.34%</div>
                </div>
                <div class="card">
                    <div class="card-title">📊 Sinyal</div>
                    <div class="card-value positive" id="signal">LONG</div>
                    <div class="card-change">Guven: %75</div>
                </div>
                <div class="card">
                    <div class="card-title">💵 Bakiye</div>
                    <div class="card-value" id="balance">$10,500</div>
                    <div class="card-change positive">+$500</div>
                </div>
                <div class="card">
                    <div class="card-title">📈 Pozisyon</div>
                    <div class="card-value" id="position">LONG</div>
                    <div class="card-change">@ $94,500</div>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Fiyat Grafigi</div>
                    <div class="chart-container"><canvas id="priceChart"></canvas></div>
                </div>
                <div class="card">
                    <div class="card-title">Bakiye Grafigi</div>
                    <div class="chart-container"><canvas id="balanceChart"></canvas></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Performans</div>
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Islem</div><div class="stat-value">45</div></div>
                    <div class="stat-item"><div class="stat-label">Win Rate</div><div class="stat-value positive">62%</div></div>
                    <div class="stat-item"><div class="stat-label">Profit Factor</div><div class="stat-value">1.85</div></div>
                    <div class="stat-item"><div class="stat-label">Max DD</div><div class="stat-value negative">-8.5%</div></div>
                </div>
            </div>
        </div>
        
        <!-- Account -->
        <div id="section-account" class="section">
            <h1 class="page-title">👤 Hesap Ayarlari</h1>
            <p class="page-subtitle">Bakiye ve risk ayarlarinizi yapin</p>
            
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">💰 Baslangic Bakiyesi</div>
                    <div class="form-group">
                        <label class="form-label">Bakiye (USD)</label>
                        <input type="number" class="form-control" id="input-balance" value="10000">
                    </div>
                    <button class="btn btn-primary" onclick="saveSettings()">💾 Kaydet</button>
                </div>
                <div class="card">
                    <div class="card-title">⚙️ Risk Ayarlari</div>
                    <div class="form-group">
                        <label class="form-label">Stop Loss (%)</label>
                        <input type="number" class="form-control" id="input-sl" value="2" step="0.5">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Take Profit (%)</label>
                        <input type="number" class="form-control" id="input-tp" value="3" step="0.5">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Pozisyon Boyutu (%)</label>
                        <input type="number" class="form-control" id="input-size" value="50" step="5">
                    </div>
                    <button class="btn btn-primary" onclick="saveSettings()">💾 Kaydet</button>
                </div>
            </div>
            <div class="alert alert-info">ℹ️ Bu ayarlar simulasyon icindir. Gercek para kullanilmaz.</div>
        </div>
        
        <!-- Strategy -->
        <div id="section-strategy" class="section">
            <h1 class="page-title">🎯 Strateji Sec</h1>
            <p class="page-subtitle">20 hazir stratejiden birini secin</p>
            
            <div class="tabs">
                <div class="tab active" onclick="filterStrategy('all', this)">Tumu</div>
                <div class="tab" onclick="filterStrategy('safe', this)">Guvenli</div>
                <div class="tab" onclick="filterStrategy('balanced', this)">Dengeli</div>
                <div class="tab" onclick="filterStrategy('aggressive', this)">Agresif</div>
            </div>
            
            <div class="grid-3" id="strategy-list"></div>
            
            <div style="margin-top:15px;">
                <button class="btn btn-primary btn-lg" onclick="applyStrategy()">✅ Stratejiyi Uygula</button>
            </div>
        </div>
        
        <!-- Optimize -->
        <div id="section-optimize" class="section">
            <h1 class="page-title">🔍 En Iyi Stratejiyi Bul</h1>
            <p class="page-subtitle">Genetik algoritma ile optimal parametreleri bul</p>
            
            <div class="card">
                <div class="grid-3">
                    <div class="form-group">
                        <label class="form-label">Test Suresi</label>
                        <select class="form-control" id="opt-months">
                            <option value="1">1 Ay</option>
                            <option value="3" selected>3 Ay</option>
                            <option value="6">6 Ay</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Jenerasyon</label>
                        <select class="form-control" id="opt-gen">
                            <option value="20">20 (Hizli)</option>
                            <option value="50" selected>50 (Normal)</option>
                            <option value="100">100 (Detayli)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Hedef</label>
                        <select class="form-control" id="opt-metric">
                            <option value="sharpe">Sharpe Ratio</option>
                            <option value="return" selected>Getiri</option>
                        </select>
                    </div>
                </div>
                <button class="btn btn-primary btn-lg" onclick="startOptimization()">🚀 Baslat</button>
            </div>
            
            <div class="card hidden" id="opt-progress">
                <div class="card-title">⏳ Optimizasyon</div>
                <div class="progress"><div class="progress-bar" style="width:0%;background:var(--accent)" id="opt-bar"></div></div>
                <p style="margin-top:10px;color:var(--text-secondary)" id="opt-status">Baslatiliyor...</p>
            </div>
            
            <div class="card hidden" id="opt-results"></div>
        </div>
        
        <!-- Backtest -->
        <div id="section-backtest" class="section">
            <h1 class="page-title">📈 Backtest</h1>
            <p class="page-subtitle">Stratejiyi gecmis verilerle test et</p>
            
            <div class="card">
                <div class="grid-2">
                    <div class="form-group">
                        <label class="form-label">Test Suresi</label>
                        <select class="form-control" id="bt-months">
                            <option value="1">1 Ay</option>
                            <option value="3" selected>3 Ay</option>
                            <option value="6">6 Ay</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Bakiye</label>
                        <input type="number" class="form-control" id="bt-balance" value="10000">
                    </div>
                </div>
                <button class="btn btn-primary btn-lg" onclick="runBacktest()">▶️ Baslat</button>
            </div>
            
            <div class="card hidden" id="bt-results"></div>
        </div>
        
        <!-- Train -->
        <div id="section-train" class="section">
            <h1 class="page-title">🧠 Model Egitimi</h1>
            <p class="page-subtitle">ML modelini egit</p>
            
            <div class="alert alert-warning">⚠️ Egitim 5-15 dakika surebilir.</div>
            
            <div class="card">
                <div class="grid-3">
                    <div class="form-group">
                        <label class="form-label">Veri Suresi</label>
                        <select class="form-control" id="train-months">
                            <option value="6">6 Ay</option>
                            <option value="12" selected>12 Ay</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Model</label>
                        <select class="form-control" id="train-model">
                            <option value="gru" selected>GRU</option>
                            <option value="lstm">LSTM</option>
                            <option value="attention">Attention</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Tahmin</label>
                        <select class="form-control" id="train-horizon">
                            <option value="4" selected>4 Saat</option>
                            <option value="24">24 Saat</option>
                        </select>
                    </div>
                </div>
                <button class="btn btn-primary btn-lg" onclick="startTraining()">🚀 Egitimi Baslat</button>
            </div>
            
            <div class="card hidden" id="train-progress">
                <div class="card-title">⏳ Egitim</div>
                <div class="progress"><div class="progress-bar" style="width:0%;background:var(--green)" id="train-bar"></div></div>
                <p style="margin-top:10px;color:var(--text-secondary)" id="train-status">Hazirlaniyor...</p>
            </div>
            
            <div class="card">
                <div class="card-title">Model Durumu</div>
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Durum</div><div class="stat-value" id="m-status">Hazir Degil</div></div>
                    <div class="stat-item"><div class="stat-label">Tarih</div><div class="stat-value" id="m-date">-</div></div>
                    <div class="stat-item"><div class="stat-label">Dogruluk</div><div class="stat-value" id="m-acc">-</div></div>
                    <div class="stat-item"><div class="stat-label">Feature</div><div class="stat-value">81</div></div>
                </div>
            </div>
        </div>
        
        <!-- Live -->
        <div id="section-live" class="section">
            <h1 class="page-title">⚡ Canli Test</h1>
            <p class="page-subtitle">Gercek zamanli simulasyon</p>
            
            <div class="alert alert-info">ℹ️ Simulasyondur. Gercek islem yapilmaz.</div>
            
            <div class="grid-2">
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
                        <div class="card-title" style="margin:0">Canli Durum</div>
                        <span class="badge badge-warning" id="live-badge">Durduruldu</span>
                    </div>
                    <div class="stats-grid" style="grid-template-columns:repeat(2,1fr)">
                        <div class="stat-item"><div class="stat-label">Fiyat</div><div class="stat-value" id="live-price">$95,000</div></div>
                        <div class="stat-item"><div class="stat-label">Sinyal</div><div class="stat-value" id="live-sig">-</div></div>
                        <div class="stat-item"><div class="stat-label">Pozisyon</div><div class="stat-value" id="live-pos">YOK</div></div>
                        <div class="stat-item"><div class="stat-label">Bakiye</div><div class="stat-value" id="live-bal">$10,000</div></div>
                    </div>
                    <div style="margin-top:15px;display:flex;gap:10px;">
                        <button class="btn btn-success" onclick="startLive()" id="btn-start">▶️ Baslat</button>
                        <button class="btn btn-danger" onclick="stopLive()" id="btn-stop" disabled>⏹️ Durdur</button>
                    </div>
                </div>
                <div class="card signal-box">
                    <div class="signal-icon" id="live-icon">⏸️</div>
                    <div class="signal-text" id="live-text">BEKLIYOR</div>
                    <div class="progress"><div class="progress-bar" style="width:50%;background:var(--accent)" id="live-conf-bar"></div></div>
                    <div style="margin-top:8px;color:var(--text-secondary);font-size:12px;">Guven: <span id="live-conf">50</span>%</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Canli Grafik</div>
                <div class="chart-container"><canvas id="liveChart"></canvas></div>
            </div>
        </div>
        
        <!-- History -->
        <div id="section-history" class="section">
            <h1 class="page-title">📜 Islem Gecmisi</h1>
            <p class="page-subtitle">Tamamlanan islemler</p>
            
            <div class="card">
                <table>
                    <thead><tr><th>Tarih</th><th>Tip</th><th>Giris</th><th>Cikis</th><th>K/Z</th><th>Sebep</th></tr></thead>
                    <tbody id="history-table">
                        <tr><td>2024-01-15 14:30</td><td><span class="badge badge-success">LONG</span></td><td>$94,000</td><td>$95,500</td><td class="positive">+$150</td><td>TP</td></tr>
                        <tr><td>2024-01-15 10:15</td><td><span class="badge badge-danger">SHORT</span></td><td>$95,000</td><td>$94,200</td><td class="positive">+$80</td><td>Sinyal</td></tr>
                        <tr><td>2024-01-14 22:00</td><td><span class="badge badge-success">LONG</span></td><td>$93,500</td><td>$93,000</td><td class="negative">-$50</td><td>SL</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Guide -->
        <div id="section-guide" class="section">
            <h1 class="page-title">📖 Kullanim Rehberi</h1>
            <p class="page-subtitle">Nasil kullanilir?</p>
            
            <div class="card">
                <div class="card-title">🚀 Hizli Baslangic</div>
                <div style="line-height:2;font-size:13px;">
                    <p><strong>1.</strong> Hesap Ayarlarindan bakiye girin</p>
                    <p><strong>2.</strong> Strateji Sec bolumunden bir strateji secin</p>
                    <p><strong>3.</strong> Backtest ile gecmis performansi gorun</p>
                    <p><strong>4.</strong> Model Egit ile ML modelini hazirlayin (opsiyonel)</p>
                    <p><strong>5.</strong> Canli Test ile simulasyonu baslatin</p>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 Strateji Kategorileri</div>
                <div class="grid-3" style="margin-top:10px;">
                    <div style="padding:12px;background:rgba(0,212,170,0.1);border-radius:8px;">
                        <strong style="color:var(--green)">🛡️ Guvenli</strong>
                        <p style="font-size:11px;color:var(--text-secondary);margin-top:5px;">Dusuk risk, stabil getiri</p>
                    </div>
                    <div style="padding:12px;background:rgba(247,147,26,0.1);border-radius:8px;">
                        <strong style="color:var(--accent)">⚖️ Dengeli</strong>
                        <p style="font-size:11px;color:var(--text-secondary);margin-top:5px;">Orta risk/odul orani</p>
                    </div>
                    <div style="padding:12px;background:rgba(255,71,87,0.1);border-radius:8px;">
                        <strong style="color:var(--red)">🔥 Agresif</strong>
                        <p style="font-size:11px;color:var(--text-secondary);margin-top:5px;">Yuksek risk/potansiyel</p>
                    </div>
                </div>
            </div>
            
            <div class="alert alert-warning">⚠️ Bu sistem egitim amaclidir. Yatirim tavsiyesi degildir.</div>
        </div>
        
        <!-- About -->
        <div id="section-about" class="section">
            <h1 class="page-title">ℹ️ Hakkinda</h1>
            <p class="page-subtitle">BTC Bot Pro v5.0</p>
            
            <div class="card">
                <div class="card-title">Sistem Ozellikleri</div>
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Strateji</div><div class="stat-value">20</div></div>
                    <div class="stat-item"><div class="stat-label">Indikator</div><div class="stat-value">81</div></div>
                    <div class="stat-item"><div class="stat-label">ML Model</div><div class="stat-value">4</div></div>
                    <div class="stat-item"><div class="stat-label">Kod</div><div class="stat-value">9K+</div></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedStrategy = 'balanced';
        let isLive = false;
        let liveTimer = null;
        
        const strategies = [
            {id:'ultra_safe',name:'Ultra Safe',cat:'safe',risk:'Cok Dusuk',desc:'Minimum risk',wr:'45%',dd:'5%'},
            {id:'conservative',name:'Conservative',cat:'safe',risk:'Dusuk',desc:'Muhafazakar',wr:'48%',dd:'8%'},
            {id:'safe_trend',name:'Safe Trend',cat:'safe',risk:'Dusuk',desc:'Guvenli trend',wr:'50%',dd:'10%'},
            {id:'dip_buyer',name:'Dip Buyer',cat:'safe',risk:'Dusuk-Orta',desc:'Dip alici',wr:'52%',dd:'12%'},
            {id:'balanced',name:'Balanced',cat:'balanced',risk:'Orta',desc:'Dengeli',wr:'55%',dd:'15%'},
            {id:'momentum',name:'Momentum',cat:'balanced',risk:'Orta',desc:'Momentum',wr:'53%',dd:'18%'},
            {id:'swing',name:'Swing Trader',cat:'balanced',risk:'Orta',desc:'Swing',wr:'50%',dd:'20%'},
            {id:'trend_surfer',name:'Trend Surfer',cat:'balanced',risk:'Orta-Yuksek',desc:'Trend surme',wr:'48%',dd:'22%'},
            {id:'breakout',name:'Breakout',cat:'aggressive',risk:'Yuksek',desc:'Kirilim',wr:'45%',dd:'25%'},
            {id:'aggressive',name:'Aggressive',cat:'aggressive',risk:'Yuksek',desc:'Agresif',wr:'42%',dd:'30%'},
            {id:'scalper',name:'Scalper',cat:'aggressive',risk:'Cok Yuksek',desc:'Scalping',wr:'40%',dd:'35%'},
            {id:'yolo',name:'YOLO',cat:'aggressive',risk:'Ekstrem',desc:'Max risk',wr:'35%',dd:'50%'},
        ];
        
        function showSection(id, el) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById('section-' + id).classList.add('active');
            if(el) el.classList.add('active');
            if(id === 'strategy') loadStrategies();
        }
        
        function loadStrategies(filter='all') {
            const c = document.getElementById('strategy-list');
            c.innerHTML = strategies.filter(s => filter==='all' || s.cat===filter).map(s => `
                <div class="strategy-card ${s.id===selectedStrategy?'selected':''}" onclick="selectStrategy('${s.id}')">
                    <div class="strategy-name">${s.name}</div>
                    <div class="strategy-desc">${s.desc}</div>
                    <div class="strategy-stats">
                        <div><span class="strategy-stat-label">Risk:</span> ${s.risk}</div>
                        <div><span class="strategy-stat-label">WR:</span> ${s.wr}</div>
                        <div><span class="strategy-stat-label">DD:</span> ${s.dd}</div>
                    </div>
                </div>
            `).join('');
        }
        
        function filterStrategy(cat, el) {
            document.querySelectorAll('.tabs .tab').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            loadStrategies(cat);
        }
        
        function selectStrategy(id) { selectedStrategy = id; loadStrategies(); }
        function applyStrategy() { alert('Strateji uygulandi: ' + selectedStrategy); }
        function saveSettings() { alert('Ayarlar kaydedildi!'); }
        
        function runBacktest() {
            const r = document.getElementById('bt-results');
            r.classList.remove('hidden');
            r.innerHTML = '<div class="spinner"></div>';
            setTimeout(() => {
                const ret = (Math.random()*30-5).toFixed(2);
                r.innerHTML = `
                    <div class="card-title">Sonuclar</div>
                    <div class="stats-grid">
                        <div class="stat-item"><div class="stat-label">Getiri</div><div class="stat-value ${ret>=0?'positive':'negative'}">${ret>=0?'+':''}${ret}%</div></div>
                        <div class="stat-item"><div class="stat-label">Islem</div><div class="stat-value">${Math.floor(Math.random()*40+20)}</div></div>
                        <div class="stat-item"><div class="stat-label">Win Rate</div><div class="stat-value">${(Math.random()*20+45).toFixed(1)}%</div></div>
                        <div class="stat-item"><div class="stat-label">Max DD</div><div class="stat-value negative">-${(Math.random()*15+5).toFixed(1)}%</div></div>
                    </div>
                `;
            }, 2000);
        }
        
        function startOptimization() {
            document.getElementById('opt-progress').classList.remove('hidden');
            document.getElementById('opt-results').classList.add('hidden');
            let p = 0;
            const t = setInterval(() => {
                p += Math.random()*8;
                if(p >= 100) {
                    p = 100; clearInterval(t);
                    setTimeout(() => {
                        document.getElementById('opt-progress').classList.add('hidden');
                        const r = document.getElementById('opt-results');
                        r.classList.remove('hidden');
                        r.innerHTML = `
                            <div class="card-title">✅ Sonuclar</div>
                            <div class="alert alert-success">En iyi parametreler bulundu!</div>
                            <div class="stats-grid">
                                <div class="stat-item"><div class="stat-label">SL</div><div class="stat-value">2.5%</div></div>
                                <div class="stat-item"><div class="stat-label">TP</div><div class="stat-value">4.2%</div></div>
                                <div class="stat-item"><div class="stat-label">RSI Low</div><div class="stat-value">28</div></div>
                                <div class="stat-item"><div class="stat-label">RSI High</div><div class="stat-value">72</div></div>
                            </div>
                            <button class="btn btn-primary" style="margin-top:15px" onclick="alert('Uygulandi!')">Uygula</button>
                        `;
                    }, 300);
                }
                document.getElementById('opt-bar').style.width = p+'%';
                document.getElementById('opt-status').textContent = 'Jenerasyon '+ Math.floor(p/2) +'/50';
            }, 150);
        }
        
        function startTraining() {
            document.getElementById('train-progress').classList.remove('hidden');
            let p = 0;
            const steps = ['Veri hazirlaniyor...','Ozellikler...','Model olusturuluyor...','Egitiliyor...','Dogrulama...'];
            const t = setInterval(() => {
                p += Math.random()*4;
                if(p >= 100) {
                    p = 100; clearInterval(t);
                    document.getElementById('train-status').textContent = 'Tamamlandi!';
                    document.getElementById('m-status').textContent = 'Hazir';
                    document.getElementById('m-date').textContent = new Date().toLocaleDateString();
                    document.getElementById('m-acc').textContent = '58%';
                }
                document.getElementById('train-bar').style.width = p+'%';
                document.getElementById('train-status').textContent = steps[Math.min(Math.floor(p/25), steps.length-1)];
            }, 200);
        }
        
        function startLive() {
            isLive = true;
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            document.getElementById('live-badge').className = 'badge badge-success';
            document.getElementById('live-badge').innerHTML = '<span class="live-dot"></span> Calisiyor';
            
            let price = 95000;
            liveTimer = setInterval(() => {
                price += (Math.random()-0.5)*200;
                const sig = Math.random()>0.6 ? 'LONG' : (Math.random()>0.5 ? 'SHORT' : 'HOLD');
                const conf = Math.floor(Math.random()*30+50);
                
                document.getElementById('live-price').textContent = '$'+price.toFixed(0);
                document.getElementById('live-sig').textContent = sig;
                document.getElementById('live-sig').className = 'stat-value ' + (sig==='LONG'?'positive':sig==='SHORT'?'negative':'');
                document.getElementById('live-text').textContent = sig;
                document.getElementById('live-text').className = 'signal-text ' + (sig==='LONG'?'positive':sig==='SHORT'?'negative':'');
                document.getElementById('live-icon').textContent = sig==='LONG'?'🟢':sig==='SHORT'?'🔴':'🟡';
                document.getElementById('live-conf').textContent = conf;
                document.getElementById('live-conf-bar').style.width = conf+'%';
            }, 2000);
        }
        
        function stopLive() {
            isLive = false;
            clearInterval(liveTimer);
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('live-badge').className = 'badge badge-warning';
            document.getElementById('live-badge').textContent = 'Durduruldu';
            document.getElementById('live-text').textContent = 'DURDURULDU';
            document.getElementById('live-icon').textContent = '⏹️';
        }
        
        function initCharts() {
            const pCtx = document.getElementById('priceChart').getContext('2d');
            new Chart(pCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length:24},(_,i)=>i+':00'),
                    datasets: [{
                        data: Array.from({length:24},()=>94000+Math.random()*2000),
                        borderColor: '#f7931a',
                        backgroundColor: 'rgba(247,147,26,0.1)',
                        fill: true, tension: 0.4, pointRadius: 0
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {legend:{display:false}},
                    scales: {
                        x: {grid:{color:'rgba(255,255,255,0.03)'},ticks:{color:'#666',font:{size:10}}},
                        y: {grid:{color:'rgba(255,255,255,0.03)'},ticks:{color:'#666',font:{size:10}}}
                    }
                }
            });
            
            let bal = 10000;
            const bCtx = document.getElementById('balanceChart').getContext('2d');
            new Chart(bCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length:30},(_,i)=>'G'+(i+1)),
                    datasets: [{
                        data: Array.from({length:30},()=>{bal+=(Math.random()-0.4)*100;return bal;}),
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0,212,170,0.1)',
                        fill: true, tension: 0.4, pointRadius: 0
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {legend:{display:false}},
                    scales: {
                        x: {grid:{color:'rgba(255,255,255,0.03)'},ticks:{color:'#666',font:{size:10}}},
                        y: {grid:{color:'rgba(255,255,255,0.03)'},ticks:{color:'#666',font:{size:10}}}
                    }
                }
            });
        }
        
        document.addEventListener('DOMContentLoaded', () => { initCharts(); loadStrategies(); });
    </script>
</body>
</html>
'''

# ================================================================
# FLASK APP
# ================================================================

class BotDashboard:
    def __init__(self, host='127.0.0.1', port=5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask gerekli: pip install flask")
        
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(MAIN_HTML)
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({'status': 'ok', 'price': 95000})
    
    def run(self, debug=False):
        print(f"\n{'='*50}")
        print("   BTC Bot Pro v5.0 - Dashboard")
        print(f"{'='*50}")
        print(f"\n   URL: http://{self.host}:{self.port}")
        print(f"\n   Kapatmak icin CTRL+C\n")
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_dashboard(port=5000):
    return BotDashboard(port=port)


if __name__ == '__main__':
    dashboard = BotDashboard()
    dashboard.run()
