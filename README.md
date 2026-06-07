# MATA — Multimodal Authenticity Testing and Analysis
Aplikasi Flask full stack untuk mendeteksi keaslian video berbasis OpenCV.

## Struktur Proyek

```
multimodal_authenticity_testing_and_analysis/
├── app.py                  # Flask app utama
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── analyzer.py         # Engine analisis forensik
├── templates/
│   └── index.html          # Frontend (HTML + jQuery CDN)
├── static/
│   ├── css/style.css
│   └── js/main.js          # jQuery logic
└── uploads/                # Folder sementara (auto-cleanup)
```

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Catatan:** `opencv-python-headless` digunakan untuk server (tanpa GUI). Jika ingin menjalankan di desktop dengan GUI OpenCV, ganti dengan `opencv-python`.

### 2. Jalankan Flask

```bash
python app.py
```

Akses di: **http://localhost:5000**

### 3. Produksi (opsional)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Cara Pakai

1. Buka browser ke `http://localhost:5000`
2. Drag & drop video atau klik area upload
3. Klik **RUN ANALYSIS**
4. Tunggu proses analisis selesai
5. Baca laporan forensik:
   - **AUTHENTIC** — video tampak asli (skor ≥ 50)
   - **MANIPULATED** — indikasi kuat video adalah hasil deepfake (skor < 50)

## Format yang Didukung

MP4, AVI, MOV, MKV, WEBM, FLV (maks. 500 MB)

## Catatan Penting

- Analisis bersifat **heuristik**, bukan AI model terlatih — gunakan sebagai alat bantu awal
- File video **dihapus otomatis** setelah analisis selesai
