import os
import logging
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import requests
from scipy.stats import poisson
from rapidfuzz import process, fuzz, utils

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'final_unified_dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger("PredictaPRO")

NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Origin": "https://www.nesine.com"
}
NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

class MatchPredictor:
    def __init__(self):
        self.df = None
        self.team_stats = {}
        self.team_names_map = {}
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2
        self.load_database()

    def load_database(self):
        logger.info(f"📂 Veritabanı yükleniyor: {CSV_PATH}")
        
        if not os.path.exists(CSV_PATH):
            logger.error(f"❌ CSV Bulunamadı! Yol: {CSV_PATH}")
            return

        try:
            self.df = pd.read_csv(CSV_PATH, encoding='utf-8', on_bad_lines='skip')
            
            # Sütun temizliği
            self.df.columns = [c.lower().strip().replace(' ', '_').replace('hometeam', 'home_team').replace('awayteam', 'away_team').replace('fthg', 'home_score').replace('ftag', 'away_score') for c in self.df.columns]
            
            # Skor temizliği
            for col in ['home_score', 'away_score']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
            
            self._calculate_stats()
            logger.info(f"✅ DB Yüklendi: {len(self.df)} satır, {len(self.team_stats)} takım.")
            
        except Exception as e:
            logger.error(f"❌ DB Hatası: {e}")

    def _calculate_stats(self):
        if self.df is None or self.df.empty: return
        
        # Lig ortalamalarını hesapla
        if 'home_score' in self.df.columns:
            self.avg_home_goals = self.df['home_score'].mean() or 1.5
            self.avg_away_goals = self.df['away_score'].mean() or 1.2
        
        stats = {}
        h_col = 'home_team' if 'home_team' in self.df.columns else 'home'
        a_col = 'away_team' if 'away_team' in self.df.columns else 'away'
        
        if h_col not in self.df.columns: return

        teams = set(self.df[h_col].unique()) | set(self.df[a_col].unique())
        
        for team in teams:
            if pd.isna(team) or str(team).strip() == '': continue
            
            h_matches = self.df[self.df[h_col] == team]
            a_matches = self.df[self.df[a_col] == team]
            
            h_g = len(h_matches)
            a_g = len(a_matches)
            
            # Atak/Defans Güçleri (En az 3 maç yapmış olmalı)
            att_h = (h_matches['home_score'].sum() / h_g / self.avg_home_goals) if h_g > 3 else 1.0
            def_h = (h_matches['away_score'].sum() / h_g / self.avg_away_goals) if h_g > 3 else 1.0
            att_a = (a_matches['away_score'].sum() / a_g / self.avg_away_goals) if a_g > 3 else 1.0
            def_a = (a_matches['home_score'].sum() / a_g / self.avg_home_goals) if a_g > 3 else 1.0
            
            stats[team] = {
                'att_h': att_h, 'def_h': def_h,
                'att_a': att_a, 'def_a': def_a
            }
        self.team_stats = stats

    def normalize_name(self, name):
        """İsimleri standartlaştırır (Man City -> Manchester City)"""
        if not name: return ""
        n = name.lower().strip()
        n = n.replace('.', '')
        n = n.replace('-', ' ') 
        
        replacements = {
            'man city': 'manchester city',
            'man united': 'manchester united',
            'man utd': 'manchester united',
            'utd': 'united',
            'qpr': 'queens park rangers',
            'wolves': 'wolverhampton',
            "n'castle": "newcastle"
        }
        
        for k, v in replacements.items():
            if k in n: n = n.replace(k, v)
        return n

    def find_team(self, name):
        if not name or not self.team_stats: return None
        if name in self.team_names_map: return self.team_names_map[name]
        
        clean_name = self.normalize_name(name)
        db_teams = list(self.team_stats.keys())
        
        match = process.extractOne(
            clean_name, 
            db_teams, 
            scorer=fuzz.token_set_ratio, 
            score_cutoff=60
        )
        
        if match:
            self.team_names_map[name] = match[0]
            return match[0]
        return None

    def predict(self, home, away):
        """Gelişmiş Poisson Tahmini (BTTS Dahil)"""
        home_db = self.find_team(home)
        away_db = self.find_team(away)
        
        if not home_db or not away_db: return None
            
        hs = self.team_stats.get(home_db)
        as_ = self.team_stats.get(away_db)
        
        # xG Hesaplama
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        # Poisson Olasılıkları (0'dan 5 gole kadar)
        h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
        a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
        
        # Olasılıkları Topla
        prob_1, prob_x, prob_2 = 0, 0, 0
        prob_over = 0 # 2.5 Üst
        
        for h in range(6):
            for a in range(6):
                p = h_probs[h] * a_probs[a]
                
                # MS
                if h > a: prob_1 += p
                elif h == a: prob_x += p
                else: prob_2 += p
                
                # Alt/Üst
                if (h + a) > 2.5: prob_over += p
        
        # 2.5 Alt Hesabı
        prob_under = 1 - prob_over

        # KG Var (BTTS) Hesabı
        # P(Ev Gol Atar) = 1 - P(Ev 0 Gol)
        prob_home_score = 1 - poisson.pmf(0, h_xg)
        # P(Dep Gol Atar) = 1 - P(Dep 0 Gol)
        prob_away_score = 1 - poisson.pmf(0, a_xg)
        
        prob_btts_yes = prob_home_score * prob_away_score
        prob_btts_no = 1 - prob_btts_yes

        return {
            "home_team_db": home_db,
            "away_team_db": away_db,
            "stats": {
                "home_xg": round(h_xg, 2), 
                "away_xg": round(a_xg, 2)
            },
            "probs": {
                "1": round(prob_1 * 100, 1),
                "X": round(prob_x * 100, 1),
                "2": round(prob_2 * 100, 1),
                "over": round(prob_over * 100, 1),
                "under": round(prob_under * 100, 1),
                "btts_yes": round(prob_btts_yes * 100, 1),
                "btts_no": round(prob_btts_no * 100, 1)
            }
        }

predictor = MatchPredictor()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/matches/live')
def live():
    try:
        r = requests.get(NESINE_URL, headers=NESINE_HEADERS, timeout=15)
        d = r.json()
        matches = []
        
        if "sg" in d and "EA" in d["sg"]:
            for m in d["sg"]["EA"]:
                if m.get("GT") != 1: continue
                
                odds = {}
                has_ms = False
                
                for market in m.get("MA", []):
                    mtid = market.get("MTID")
                    oca = market.get("OCA", [])
                    
                    # MS (1, X, 2)
                    if mtid == 1:
                        for o in oca:
                            if o.get("N") == 1: odds["1"] = o.get("O")
                            elif o.get("N") == 2: odds["X"] = o.get("O")
                            elif o.get("N") == 3: odds["2"] = o.get("O")
                        if "1" in odds: has_ms = True
                            
                    # Alt/Üst 2.5 (İsim kontrolü ile garantiye alalım)
                    elif mtid == 450 or "2.5" in str(market.get("MN", "")):
                        if "Over/Under +2.5" not in odds: odds["Over/Under +2.5"] = {}
                        for o in oca:
                            if o.get("N") == 1: odds["Over/Under +2.5"]["Over +2.5"] = o.get("O")
                            if o.get("N") == 2: odds["Over/Under +2.5"]["Under +2.5"] = o.get("O")

                    # KG Var/Yok
                    elif mtid == 38:
                        if "Both Teams To Score" not in odds: odds["Both Teams To Score"] = {}
                        for o in oca:
                            if o.get("N") == 1: odds["Both Teams To Score"]["Yes"] = o.get("O")
                            if o.get("N") == 2: odds["Both Teams To Score"]["No"] = o.get("O")

                if has_ms:
                    # Lig İsmi Düzeltme
                    league = m.get("LN")
                    if not league or str(league).lower() == "null" or league == "":
                        league = str(m.get("LID", "Lig ID Yok"))

                    matches.append({
                        "id": str(m.get("C")),
                        "home": m.get("HN"),
                        "away": m.get("AN"),
                        "date": f"{m.get('D')} {m.get('T')}",
                        "league": league,
                        "odds": odds,
                        "prediction": predictor.predict(m.get("HN"), m.get("AN"))
                    })
        
        return jsonify({"success": True, "count": len(matches), "matches": matches})
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
