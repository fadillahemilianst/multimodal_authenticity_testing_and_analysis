# VeriFrame — Video Authenticity Detector

Aplikasi Flask full stack untuk mendeteksi keaslian video menggunakan analisis forensik berbasis OpenCV.

## Fitur Deteksi

| Modul | Keterangan |
|---|---|
| **Metadata Integrity** | Cek FPS, resolusi, bitrate, dan konsistensi metadata |
| **Noise Analysis** | Analisis pola noise antar frame – inkonsistensi = indikasi splicing |
| **Compression Artifacts** | DCT energy analysis untuk mendeteksi re-encoding berlebihan atau AI generation |
| **Temporal Consistency** | Deteksi abrupt cut / scene jump yang mencurigakan |
| **Error Level Analysis (ELA)** | Re-kompresi ulang dan bandingkan untuk menemukan area yang diedit |
| **Copy-Move Detection** | Deteksi region yang diduplikasi dalam frame menggunakan ORB feature matching |

## Struktur Proyek

```
video-authenticity-detector/
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
   - **AUTHENTIC** — video tampak asli (skor ≥ 80)
   - **SUSPICIOUS** — ada anomali yang perlu diperhatikan (skor 55–79)
   - **MANIPULATED** — indikasi kuat video telah diedit/dipalsukan (skor < 55)

## Format yang Didukung

MP4, AVI, MOV, MKV, WEBM, FLV (maks. 500 MB)

## Catatan Penting

- Analisis bersifat **heuristik**, bukan AI model terlatih — gunakan sebagai alat bantu awal
- File video **dihapus otomatis** setelah analisis selesai
- Semakin panjang video, semakin akurat analisis temporal
