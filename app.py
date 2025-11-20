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
            
            att_h = (h_matches['home_score'].sum() / h_g / self.avg_home_goals) if h_g > 3 else 1.0
            def_h = (h_matches['away_score'].sum() / h_g / self.avg_away_goals) if h_g > 3 else 1.0
            att_a = (a_matches['away_score'].sum() / a_g / self.avg_away_goals) if a_g > 3 else 1.0
            def_a = (a_matches['home_score'].sum() / a_g / self.avg_home_goals) if a_g > 3 else 1.0
            
            form = []
            recent = pd.concat([h_matches, a_matches]).sort_index().tail(5)
            for _, r in recent.iterrows():
                try:
                    hs, as_ = r['home_score'], r['away_score']
                    is_h = r[h_col] == team
                    if hs > as_: res = 'W' if is_h else 'L'
                    elif hs < as_: res = 'L' if is_h else 'W'
                    else: res = 'D'
                    form.append(res)
                except: continue
                
            stats[team] = {
                'att_h': att_h, 'def_h': def_h,
                'att_a': att_a, 'def_a': def_a,
                'form': "".join(form)
            }
        self.team_stats = stats

    def normalize_name(self, name):
        """Takım isimlerini standartlaştırır (Man. City -> Manchester City)"""
        if not name: return ""
        n = name.lower().strip()
        n = n.replace('.', '') # Noktaları sil (Man. -> Man)
        n = n.replace('-', ' ') 
        
        # Özel değişimler (Man City sorununu çözen kısım)
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
            if k in n:
                n = n.replace(k, v)
        
        return n

    def find_team(self, name):
        if not name or not self.team_stats: return None
        
        # Cache kontrolü
        if name in self.team_names_map: return self.team_names_map[name]
        
        # Normalizasyon yap
        clean_name = self.normalize_name(name)
        db_teams = list(self.team_stats.keys())
        
        # Rapidfuzz ile ara (token_set_ratio kelime sırasına takılmaz)
        # Man City -> Manchester City eşleşmesi için token_set daha iyidir
        match = process.extractOne(
            clean_name, 
            db_teams, 
            scorer=fuzz.token_set_ratio, 
            processor=utils.default_process,
            score_cutoff=60
        )
        
        if match:
            found_name = match[0]
            score = match[1]
            logger.info(f"🔗 MATCH: {name} ({clean_name}) -> {found_name} (Skor: {score})")
            self.team_names_map[name] = found_name
            return found_name
        else:
            logger.warning(f"🚫 NO MATCH: {name} ({clean_name})")
            return None

    def predict(self, home, away):
        home_db = self.find_team(home)
        away_db = self.find_team(away)
        
        if not home_db or not away_db: return None
            
        hs = self.team_stats.get(home_db)
        as_ = self.team_stats.get(away_db)
        
        h_xg = hs['att_h'] * as_['def_a'] * self.avg_home_goals
        a_xg = as_['att_a'] * hs['def_h'] * self.avg_away_goals
        
        h_p = [poisson.pmf(i, h_xg) for i in range(6)]
        a_p = [poisson.pmf(i, a_xg) for i in range(6)]
        
        p1, px, p2, po = 0, 0, 0, 0
        for h in range(6):
            for a in range(6):
                prob = h_p[h] * a_p[a]
                if h > a: p1 += prob
                elif h == a: px += prob
                else: p2 += prob
                if (h+a) > 2.5: po += prob
                
        return {
            "home_team_db": home_db,
            "away_team_db": away_db,
            "stats": {"home_xg": round(h_xg, 2), "away_xg": round(a_xg, 2)},
            "probs": {"1": round(p1*100,1), "X": round(px*100,1), "2": round(p2*100,1), "over": round(po*100,1)}
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
                    
                    # MS
                    if mtid == 1:
                        for o in oca:
                            if o.get("N") == 1: odds["1"] = o.get("O")
                            elif o.get("N") == 2: odds["X"] = o.get("O")
                            elif o.get("N") == 3: odds["2"] = o.get("O")
                        if "1" in odds: has_ms = True
                            
                    # Alt/Üst
                    elif mtid == 450 or "2.5" in str(market.get("MN", "")):
                        if "Over/Under +2.5" not in odds: odds["Over/Under +2.5"] = {}
                        for o in oca:
                            if o.get("N") == 1: odds["Over/Under +2.5"]["Over +2.5"] = o.get("O")
                            if o.get("N") == 2: odds["Over/Under +2.5"]["Under +2.5"] = o.get("O")

                    # KG Var/Yok (MTID 38)
                    elif mtid == 38:
                        if "Both Teams To Score" not in odds: odds["Both Teams To Score"] = {}
                        for o in oca:
                            if o.get("N") == 1: odds["Both Teams To Score"]["Yes"] = o.get("O")
                            if o.get("N") == 2: odds["Both Teams To Score"]["No"] = o.get("O")

                if has_ms:
                    # Lig İsmi Fix (Null hatasını çözer)
                    league = m.get("LN")
                    if not league or str(league).lower() == "null" or league == "":
                        league = str(m.get("LID", "Lig Belirsiz"))

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
