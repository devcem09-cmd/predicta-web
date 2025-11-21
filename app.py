import os
import json
import logging
import requests
import atexit
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# --- YAPILANDIRMA ---
load_dotenv() # .env dosyasını yükle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Logging Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictaPRO")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# Veritabanı Ayarları (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictapro.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- VERİTABANI MODELİ ---
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True) # Maç Kodu
    league = db.Column(db.String(50))
    home_team = db.Column(db.String(50))
    away_team = db.Column(db.String(50))
    date = db.Column(db.DateTime)
    odds = db.Column(db.Text) # JSON formatında oranlar
    
    # Tahminler
    prob_home = db.Column(db.Float, default=0.0)
    prob_draw = db.Column(db.Float, default=0.0)
    prob_away = db.Column(db.Float, default=0.0)
    prob_over_25 = db.Column(db.Float, default=0.0)
    prob_btts = db.Column(db.Float, default=0.0)
    
    status = db.Column(db.String(20), default="Pending")

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "league": self.league,
            "home": self.home_team,
            "away": self.away_team,
            "date": self.date.strftime("%Y-%m-%d %H:%M"),
            "odds": json.loads(self.odds) if self.odds else {},
            "probs": {
                "1": round(self.prob_home * 100, 1),
                "X": round(self.prob_draw * 100, 1),
                "2": round(self.prob_away * 100, 1),
                "over": round(self.prob_over_25 * 100, 1),
                "btts": round(self.prob_btts * 100, 1)
            }
        }

# --- TAHMİN MOTORU ---
class MatchPredictor:
    def __init__(self):
        self.team_stats = {}
        self.team_list = []
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı başlatılıyor... Yol: {CSV_PATH}")
        
        if not os.path.exists(CSV_PATH):
            logger.warning(f"⚠️ UYARI: CSV Bulunamadı ({CSV_PATH}). Tahminler çalışmayacak.")
            return

        try:
            # SENİN CSV FORMATIN: home_team, away_team, home_score, away_score
            required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
            
            # CSV Okuma (Sadece gerekli sütunlar)
            df = pd.read_csv(CSV_PATH, usecols=required_cols, encoding='utf-8', on_bad_lines='skip')
            
            # Veri Tiplerini Optimize Et
            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype('int32')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype('int32')
            
            # İstatistikleri Hesapla
            self._calculate_stats(df)
            
            # Fuzzy search listesi
            self.team_list = list(self.team_stats.keys())
            
            # Belleği Temizle
            del df
            logger.info(f"✅ Veritabanı Hazır. {len(self.team_stats)} takım yüklendi.")
            
        except Exception as e:
            logger.error(f"❌ DB Hata: {e}")

    def _calculate_stats(self, df):
        if df.empty: return
        
        self.avg_home_goals = df['home_score'].mean() or 1.5
        self.avg_away_goals = df['away_score'].mean() or 1.2
        
        # Pandas GroupBy ile Hızlı Hesaplama
        home_stats = df.groupby('home_team')['home_score'].agg(['mean', 'count'])
        home_conceded = df.groupby('home_team')['away_score'].mean()
        
        away_stats = df.groupby('away_team')['away_score'].agg(['mean', 'count'])
        away_conceded = df.groupby('away_team')['home_score'].mean()
        
        all_teams = set(home_stats.index) | set(away_stats.index)
        
        for team in all_teams:
            if team not in home_stats.index or team not in away_stats.index: continue
            
            # En az 3 maç verisi olsun
            if home_stats.loc[team, 'count'] < 3 or away_stats.loc[team, 'count'] < 3: continue

            self.team_stats[team] = {
                'att_h': home_stats.loc[team, 'mean'] / self.avg_home_goals,
                'def_h': home_conceded.loc[team] / self.avg_away_goals,
                'att_a': away_stats.loc[team, 'mean'] / self.avg_away_goals,
                'def_a': away_conceded.loc[team] / self.avg_home_goals
            }

    @lru_cache(maxsize=2048)
    def find_team_cached(self, name):
        if not name or not self.team_list: return None
        # Basit temizlik
        clean_name = name.lower().replace('sk', '').replace('fk', '').replace('fc', '').strip()
        # Eşleşme bul
        match = process.extractOne(clean_name, self.team_list, scorer=fuzz.token_set_ratio, score_cutoff=60)
        return match[0] if match else None

    def predict(self, home, away):
        home_db = self.find_team_cached(home)
        away_db = self.find_team_cached(away)
        
        if not home_db or not away_db: return 0, 0, 0, 0, 0 # Veri yoksa 0 döndür
            
        hs = self.team_stats[home_db]
        as_ = self.team_stats[away_db]
        
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        # Poisson
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        p_1, p_x, p_2 = 0, 0, 0
        p_over, p_btts = 0, 0
        
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                if h > a: p_1 += p
                elif h == a: p_x += p
                else: p_2 += p
                if (h + a) > 2.5: p_over += p
                if h > 0 and a > 0: p_btts += p

        return p_1, p_x, p_2, p_over, p_btts

predictor = MatchPredictor()

# --- ARKA PLAN GÖREVİ (VERİ ÇEKME) ---
def fetch_live_data():
    with app.app_context():
        # GÜVENLİK: Token .env dosyasından okunur
