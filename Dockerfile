# Python 3.10 sürümünü kullan
FROM python:3.10-slim

# Çalışma klasörünü ayarla
WORKDIR /app

# Gerekli kütüphaneleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını kopyala
COPY . .

# Portu dışarı aç (Koyeb genelde 8000 kullanır)
EXPOSE 8000

# Uygulamayı başlat
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
