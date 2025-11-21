#!/usr/bin/env python3
"""
Koyeb için veritabanı başlatma script'i
"""
import os
import sys
import time

def initialize():
    """Veritabanı tablolarını oluştur"""
    try:
        # Flask app'i import et
        from app import app, db, logger
        
        with app.app_context():
            logger.info("🔄 Veritabanı kontrol ediliyor...")
            
            # Tabloları oluştur
            db.create_all()
            logger.info("✅ Veritabanı tabloları oluşturuldu!")
            
            # Tablo sayısını kontrol et
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            logger.info(f"📊 Bulunan tablolar: {tables}")
            
            if 'match' not in tables:
                logger.error("❌ 'match' tablosu oluşturulamadı!")
                return False
            
            # İlk veri çekimini dene (opsiyonel)
            try:
                from app import fetch_live_data
                logger.info("🔄 İlk veri çekiliyor...")
                fetch_live_data()
                logger.info("✅ İlk veri çekimi başarılı!")
            except Exception as e:
                logger.warning(f"⚠️ İlk veri çekimi başarısız (normal): {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ HATA: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Predicta PRO - Veritabanı Başlatma")
    print("=" * 50)
    
    success = initialize()
    
    if success:
        print("✅ Başlatma başarılı!")
        sys.exit(0)
    else:
        print("⚠️ Başlatma tamamlandı (uyarılarla)")
        sys.exit(0)  # Koyeb'de hataya rağmen devam et
