from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import requests
import logging
from datetime import datetime
import os
import time
from functools import wraps

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nesine headers
NESINE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Authorization": "Basic RDQ3MDc4RDMtNjcwQi00OUJBLTgxNUYtM0IyMjI2MTM1MTZCOkI4MzJCQjZGLTQwMjgtNDIwNS05NjFELTg1N0QxRTZEOTk0OA==",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json",
}

NESINE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# Cache için global değişken
cached_matches = []
cache_timestamp = None
CACHE_DURATION = 300  # 5 dakika

# Rate limiting için basit tracker
request_tracker = {}

def rate_limit(max_requests=30, window=60):
    """Rate limiting decorator - dakikada 30 istek"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            
            if client_ip not in request_tracker:
                request_tracker[client_ip] = []
            
            # Eski istekleri temizle
            request_tracker[client_ip] = [
                req_time for req_time in request_tracker[client_ip]
                if now - req_time < window
            ]
            
            if len(request_tracker[client_ip]) >= max_requests:
                logger.warning(f"⚠️ Rate limit aşıldı: {client_ip}")
                return jsonify({
                    'success': False,
                    'error': 'Çok fazla istek. Lütfen biraz bekleyin.',
                    'retry_after': window
                }), 429
            
            request_tracker[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def fetch_nesine_matches(force_refresh=False):
    """Nesine'den maçları çek ve API formatına dönüştür"""
    global cached_matches, cache_timestamp
    
    # Cache kontrolü
    if not force_refresh and cached_matches and cache_timestamp:
        age = (datetime.now() - cache_timestamp).total_seconds()
        if age < CACHE_DURATION:
            logger.info(f"📦 Cache kullanılıyor (yaş: {age:.0f}s)")
            return {
                'matches': cached_matches,
                'from_cache': True,
                'cache_age': age
            }
    
    try:
        logger.info("🔄 Nesine API'den veriler çekiliyor...")
        start_time = time.time()
        
        response = requests.get(
            NESINE_URL, 
            headers=NESINE_HEADERS, 
            timeout=15,
            verify=True
        )
        response.raise_for_status()
        
        fetch_time = time.time() - start_time
        logger.info(f"⚡ Nesine yanıt süresi: {fetch_time:.2f}s")
        
        data = response.json()
        
        matches = []
        stats = {
            "total_processed": 0,
            "with_ms": 0,
            "with_ou": 0,
            "with_btts": 0,
            "complete_odds": 0,
            "skipped": 0
        }
        
        # Sporları kontrol et
        sports_data = data.get("sg", {})
        if not sports_data:
            logger.warning("⚠️ Nesine'den spor verisi gelmedi")
            return {
                'matches': cached_matches if cached_matches else [],
                'from_cache': bool(cached_matches),
                'cache_age': None
            }
        
        # Futbol maçlarını işle (EA = European Soccer)
        football_matches = sports_data.get("EA", [])
        logger.info(f"🔍 {len(football_matches)} futbol maçı bulundu")
        
        for m in football_matches:
            # Sadece futbol (GT = Game Type)
            if m.get("GT") != 1:
                stats["skipped"] += 1
                continue
            
            stats["total_processed"] += 1
            
            match_info = {
                "match_id": str(m.get("C", "")),
                "home_team": m.get("HN", ""),
                "away_team": m.get("AN", ""),
                "league_code": m.get("LC", ""),
                "league_name": m.get("LN", str(m.get("LID", ""))),
                "date": f"{m.get('D', '')}T{m.get('T', '')}:00",
                "is_live": m.get("L", False),
                "odds": {}
            }
            
            has_ms = False
            has_ou = False
            has_btts = False
            
            # Debug için tüm pazar tiplerini topla (sadece ilk maç)
            debug_markets = []
            
            # Oranları işle (MA = Market Array)
            for bahis in m.get("MA", []):
                bahis_tipi = bahis.get("MTID")  # Market Type ID
                oranlar = bahis.get("OCA", [])  # Odds Choice Array
                
                # Debug: İlk maç için tüm pazar tiplerini logla
                if stats["total_processed"] == 1:
                    debug_markets.append({
                        "MTID": bahis_tipi,
                        "market_name": bahis.get("MN", "Unknown"),
                        "odds_count": len(oranlar),
                        "odds": [{"N": o.get("N"), "O": o.get("O")} for o in oranlar[:5]]
                    })
                
                # Maç Sonucu (1, X, 2) - MTID: 1 (N değerine göre ayır)
                if bahis_tipi == 1 and len(oranlar) >= 3:
                    try:
                        home_odd = None
                        draw_odd = None
                        away_odd = None
                        
                        # N değerine göre oranları ayır
                        for oran in oranlar:
                            n_value = oran.get("N")
                            oran_degeri = float(oran.get("O", 0))
                            
                            if n_value == 1:  # N=1 → Ev Sahibi (1)
                                home_odd = oran_degeri
                            elif n_value == 2:  # N=2 → Beraberlik (X)
                                draw_odd = oran_degeri
                            elif n_value == 3:  # N=3 → Deplasman (2)
                                away_odd = oran_degeri
                        
                        # Eğer bulunamadıysa varsayılan
                        match_info["odds"]["1"] = home_odd or 2.0
                        match_info["odds"]["X"] = draw_odd or 3.2
                        match_info["odds"]["2"] = away_odd or 3.5
                        has_ms = True
                    except (ValueError, TypeError, KeyError):
                        pass
                
                # Alt/Üst 2.5 - MTID: 450 (N değerine göre ayır)
                elif bahis_tipi == 450 and len(oranlar) >= 2:
                    try:
                        over_odd = None
                        under_odd = None
                        
                        # N değerine göre oranları ayır
                        for oran in oranlar:
                            n_value = oran.get("N")
                            oran_degeri = float(oran.get("O", 0))
                            
                            if n_value == 1:  # N=1 → Üst 2.5
                                over_odd = oran_degeri
                            elif n_value == 2:  # N=2 → Alt 2.5
                                under_odd = oran_degeri
                        
                        # Eğer bulunamadıysa varsayılan
                        if over_odd is None or under_odd is None:
                            logger.warning(f"⚠️ Alt/Üst oranları eksik! Over={over_odd}, Under={under_odd}")
                            over_odd = over_odd or 1.9
                            under_odd = under_odd or 1.9
                        
                        # Mantık kontrolü (güvenlik için)
                        if over_odd > 10.0 and under_odd < 3.0:
                            logger.warning(f"⚠️ Şüpheli oranlar! Over={over_odd}, Under={under_odd}")
                        
                        match_info["odds"]["Over/Under +2.5"] = {
                            "Over +2.5": over_odd,
                            "Under +2.5": under_odd
                        }
                        has_ou = True
                        
                        # Debug log - İlk 3 maç için
                        if stats["total_processed"] <= 3:
                            logger.info(f"🎯 {match_info['home_team']} vs {match_info['away_team']}")
                            logger.info(f"   MTID {bahis_tipi}: Over={over_odd} (N=1), Under={under_odd} (N=2)")
                            
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"⚠️ Alt/Üst oran hatası: {e}")
                
                # Karşılıklı Gol (BTTS) - MTID: 38 (N değerine göre ayır)
                elif bahis_tipi == 38 and len(oranlar) >= 2:
                    try:
                        yes_odd = None
                        no_odd = None
                        
                        # N değerine göre oranları ayır
                        for oran in oranlar:
                            n_value = oran.get("N")
                            oran_degeri = float(oran.get("O", 0))
                            
                            if n_value == 1:  # N=1 → Var (Yes)
                                yes_odd = oran_degeri
                            elif n_value == 2:  # N=2 → Yok (No)
                                no_odd = oran_degeri
                        
                        # Eğer bulunamadıysa varsayılan
                        if yes_odd is None or no_odd is None:
                            yes_odd = yes_odd or 1.85
                            no_odd = no_odd or 1.95
                        
                        match_info["odds"]["Both Teams To Score"] = {
                            "Yes": yes_odd,
                            "No": no_odd
                        }
                        has_btts = True
                    except (ValueError, TypeError, KeyError):
                        pass
            
            # İlk maç için debug bilgisini logla
            if stats["total_processed"] == 1 and debug_markets:
                logger.info(f"📊 İlk maç için bulunan pazar tipleri: {match_info['home_team']} vs {match_info['away_team']}")
                for dm in debug_markets:
                    logger.info(f"  MTID {dm['MTID']}: {dm['market_name']} ({dm['odds_count']} oran)")
                    if dm['MTID'] in [1, 38, 450]:  # Sadece ilgili pazarları detaylandır
                        for odd in dm['odds']:
                            logger.info(f"    - N={odd['N']}: {odd['O']}")
            
            # Sadece en az Maç Sonucu oranı olan maçları ekle
            if has_ms:
                stats["with_ms"] += 1
                if has_ou:
                    stats["with_ou"] += 1
                if has_btts:
                    stats["with_btts"] += 1
                if has_ms and has_ou and has_btts:
                    stats["complete_odds"] += 1
                
                matches.append(match_info)
        
        # Cache'i güncelle
        cached_matches = matches
        cache_timestamp = datetime.now()
        
        process_time = time.time() - start_time
        logger.info(f"✅ Nesine'den {len(matches)} maç çekildi ({process_time:.2f}s)")
        logger.info(f"📊 İstatistikler: MS={stats['with_ms']}, OU={stats['with_ou']}, BTTS={stats['with_btts']}, TAM={stats['complete_odds']}")
        
        return {
            'matches': matches,
            'from_cache': False,
            'cache_age': 0,
            'stats': stats,
            'fetch_time': fetch_time,
            'process_time': process_time
        }
        
    except requests.Timeout:
        logger.error("⏱️ Nesine API timeout!")
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': 'timeout'
        }
    except requests.RequestException as e:
        logger.error(f"❌ Nesine API bağlantı hatası: {str(e)}")
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {str(e)}", exc_info=True)
        return {
            'matches': cached_matches if cached_matches else [],
            'from_cache': bool(cached_matches),
            'cache_age': None,
            'error': str(e)
        }

# --- ROUTES ---

@app.route('/')
def index():
    """Ana sayfayı (HTML) doğrudan dosya olarak sunar."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, 'templates', 'index.html')
        
        if not os.path.exists(template_path):
            return jsonify({
                "error": "index.html bulunamadı",
                "path": template_path
            }), 404
            
        return send_file(template_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/matches', methods=['GET'])
@app.route('/api/matches/upcoming', methods=['GET'])
@rate_limit(max_requests=30, window=60)
def get_matches():
    """Maç verilerini JSON olarak döndürür (Frontend uyumlu)"""
    try:
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        result = fetch_nesine_matches(force_refresh=force_refresh)
        matches = result['matches']
        
        return jsonify({
            "success": True,
            "count": len(matches),
            "matches": matches,
            "stats": result.get('stats', {}),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ API hatası: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "matches": [],
            "count": 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 PredictaAI API başlatılıyor (Port: {port})...")
    app.run(debug=False, host='0.0.0.0', port=port)
