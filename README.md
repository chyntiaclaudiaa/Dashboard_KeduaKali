# Cara Menjalankan KeduaKali

## Prasyarat

Pastikan Python dan virtual environment (`.venv`) sudah tersedia pada project.

---

## 1. Menjalankan AI Service (FastAPI)

Buka **Terminal 1** lalu jalankan:

```bash
# Aktifkan virtual environment
.venv\Scripts\activate

# Masuk ke folder service
cd keduakali_ai_service

# Install dependencies
pip install -r requirements.txt

# Jalankan FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Jika berhasil, service akan berjalan pada:

```text
http://localhost:8000
```

Biarkan terminal ini tetap berjalan selama dashboard digunakan.

---

## 2. Menjalankan Dashboard Streamlit

Buka **Terminal 2** pada root project lalu jalankan:

```bash
# Aktifkan virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Jalankan dashboard
streamlit run dashboard.py
```

Dashboard akan tersedia pada:

```text
http://localhost:8501
```

---

## Struktur Menjalankan Aplikasi

### Terminal 1

```bash
.venv\Scripts\activate
cd keduakali_ai_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Catatan

* Pastikan FastAPI Service berjalan terlebih dahulu sebelum membuka dashboard.
* Dashboard akan melakukan request ke API untuk proses AI Prediction.
* Jika terjadi error koneksi, periksa apakah service FastAPI pada port `8000` masih aktif.
