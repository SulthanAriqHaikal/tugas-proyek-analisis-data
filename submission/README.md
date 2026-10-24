# Beijing Air Quality Data Analysis Dashboard 🌍

Proyek akhir analisis data dan pengembangan dashboard interaktif menggunakan **Air Quality Dataset** (12 Stasiun Pemantauan Udara Beijing periode Maret 2013 – Februari 2017).

## Struktur Proyek
- `dashboard/dashboard.py`: Kode utama aplikasi dashboard interaktif berbasis Streamlit.
- `dashboard/main_data.csv`: Dataset bersih yang telah melalui tahap *Data Wrangling* dan siap divisualisasikan.
- `data/`: Folder berisi 12 berkas CSV mentah stasiun kualitas udara Beijing.
- `notebook.ipynb`: Jupyter Notebook yang memuat seluruh tahapan analisis (*SMART Business Questions, Data Wrangling, EDA, Visualization & Explanatory Analysis, Advanced Analysis: Binning & Geospatial,* hingga *Conclusion & Recommendation*).
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
- `url.txt`: Tautan menuju dashboard yang telah di-*deploy* ke Streamlit Community Cloud.

## Cara Menjalankan Dashboard di Lokal

### 1. Persiapan Environment (Menggunakan Conda atau Virtualenv)
```bash
conda create --name air-quality-ds python=3.10
conda activate air-quality-ds
pip install -r requirements.txt