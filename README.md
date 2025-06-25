# 🎨 WarnaYOLO: Real-time Color Detection with YOLOv8 and Streamlit

Selamat datang di repositori WarnaYOLO! Proyek ini mendemonstrasikan implementasi deteksi warna secara real-time menggunakan model YOLOv8 (varian medium) yang dilatih dengan dataset warna dari KaggleHub, dan diintegrasikan ke dalam aplikasi web interaktif menggunakan Streamlit.

## 🌟 Fitur Utama

* **Deteksi Warna Real-time:** Mampu mengidentifikasi warna objek secara langsung melalui webcam.
* **Model Efisien:** Menggunakan YOLOv8n (nano), varian terkecil dan tercepat dari YOLOv8, cocok untuk aplikasi real-time.
* **Aplikasi Web Interaktif:** Antarmuka pengguna berbasis Streamlit untuk kontrol yang mudah.
* **Dataset Kustom:** Dilatih menggunakan dataset warna khusus dari KaggleHub.
* **Deployment Cloud-Ready:** Siap untuk di-deploy di platform seperti Streamlit Community Cloud.

## 📁 Struktur Repositori
```
warnayolo/
├── app.py                   # Kode utama aplikasi Streamlit
├── requirements.txt         # Daftar dependensi Python
├── packages.txt             # Dependensi sistem operasi untuk deployment
├── best.pt                  # Model YOLOv8 terbaik hasil training
├── .streamlit/
│   └── config.toml          # Konfigurasi Streamlit (contoh: pengaturan versi Python)
├── notebooks/
│   └── Deteksi_Warna_YOLOv8m.ipynb  # Notebook untuk pelatihan model dan analisis data
```

## 🛠️ Persiapan Lingkungan

Sebelum menjalankan proyek ini, pastikan Anda memiliki Python 3.10+ terinstal.

1.  **Clone Repositori:**
    ```bash
    git clone [https://github.com/darkswan12/warnayolo.git](https://github.com/darkswan12/warnayolo.git)
    cd warnayolo
    ```

2.  **Buat Virtual Environment (opsional):**
    ```bash
    python -m venv venv_warnayolo
    # Untuk Linux/macOS
    source venv_warnayolo/bin/activate
    # Untuk Windows
    .\venv_warnayolo\Scripts\activate.bat
    ```

3.  **Instal Dependensi Python:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Tutorial Lengkap Cara Menjalankan

Proyek ini melibatkan dua fase utama: **Persiapan Data & Pelatihan Model** (biasanya di Jupyter Notebook seperti Google Colab) dan **Menjalankan Aplikasi Deteksi Real-time** (menggunakan Streamlit secara lokal atau di-deploy).

### Fase 1: Persiapan Data dan Pelatihan Model (Direkomendasikan di Google Colab)

Karena pelatihan model membutuhkan GPU, Google Colab adalah pilihan yang sangat baik.

1.  **Buka Jupyter Notebook:**
    * Buka Google Colab (`colab.research.google.com`).
    * Unggah notebook `Deteksi_Warna_YOLOv8.ipynb` dari folder `notebooks/` di repositori ini.

2.  **Atur Runtime ke GPU:**
    * Di Colab, klik `Runtime` > `Change runtime type`.
    * Pilih `T4 GPU` atau `V100 GPU` (jika tersedia) sebagai `Hardware accelerator`.

3.  **Jalankan Cell-by-Cell:**
    Ikuti instruksi di dalam notebook `Deteksi_Warna_YOLOv8.ipynb` dan jalankan setiap cell secara berurutan:
    * **Instalasi Library:** Menginstal `ultralytics`, `opencv-python-headless`, `kagglehub`, `scikit-learn`, dll.
    * **Pengunduhan dan Penyiapan Dataset:**
        * Dataset `adikurniawan/color-dataset-for-color-recognition` akan diunduh.
        * **PENTING:** Kode dalam notebook akan secara otomatis mengatur ulang struktur dataset dari folder-folder per warna (`training_dataset/red/`, `training_dataset/blue/`) ke format yang diharapkan YOLOv8 (`yolov8_formatted_dataset/images/train/`, `yolov8_formatted_dataset/labels/train/`). Ini termasuk membuat *dummy bounding box* (`class_id 0.5 0.5 1.0 1.0`) untuk setiap gambar karena dataset asli tidak menyediakan anotasi.
        * Dataset akan dibagi menjadi set training dan validation (80% train, 20% val).
    * **Pelatihan Model YOLOv8:** Model `yolov8n.pt` akan dilatih selama 30 *epoch* (atau lebih, jika Anda memodifikasi).
        * Output pelatihan (termasuk model `best.pt`, metrik, dan grafik) akan disimpan di folder `runs/detect/trainX/` (misalnya `runs/detect/train3/`).

4.  **Unduh Model `best.pt`:**
    Setelah pelatihan selesai di Colab, navigasikan ke folder `runs/detect/trainX/weights/` (sesuai dengan sesi training terakhir Anda) di panel file Colab. Unduh file `best.pt`.
    * **Saran:** Pindahkan file `best.pt` yang diunduh ini ke folder `warnayolo/runs/detect/train3/weights/` di repositori lokal Anda agar sesuai dengan jalur default di `app.py`. Jika Anda menggunakan folder `train4` atau yang lain, ubah juga jalur di `app.py`.

### Fase 2: Menjalankan Aplikasi Deteksi Real-time dengan Streamlit

#### Opsi A: Menjalankan Secara Lokal (Disarankan untuk Pengembangan)

1.  **Pastikan Model `best.pt` Sudah Ada:**
    Pastikan Anda telah mengunduh `best.pt` dari Colab dan meletakkannya di jalur yang benar di repositori lokal Anda, misalnya: `warnayolo/best.pt`.

2.  **Buka Terminal/Command Prompt:**
    Navigasi ke *root directory* proyek Anda (`warnayolo/`) di terminal atau Command Prompt yang sama di mana virtual environment Anda aktif.

3.  **Jalankan Aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Aplikasi Streamlit akan terbuka di browser web default Anda (biasanya di `http://localhost:8501`).

#### Opsi B: Deployment ke Streamlit Community Cloud (Akses via Web)

Ini memungkinkan aplikasi Anda diakses melalui URL publik dari mana saja.

1.  **Siapkan Repositori GitHub:**
    * Pastikan seluruh kode proyek Anda (`app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, dan folder `best.pt`) sudah di-commit dan di-push ke repositori GitHub Anda (misalnya `https://github.com/darkswan12/warnayolo`).

2.  **Konfigurasi File Deployment:**
    * **`requirements.txt`**: Pastikan berisi semua dependensi Python (sudah diatur di fase persiapan).
        ```
        streamlit
        opencv-python-headless
        numpy
        ultralytics
        streamlit-webrtc
        ```
    * **`packages.txt`**: Buat file ini di *root directory* repositori GitHub Anda. Ini memberitahu Streamlit Cloud untuk menginstal dependensi sistem operasi yang diperlukan oleh OpenCV.
        ```
        libgl1-mesa-glx
        ```
    * **`.streamlit/config.toml`**: Buat folder `.streamlit/` di *root directory* dan di dalamnya buat file `config.toml`. Ini memaksa Streamlit Cloud menggunakan versi Python yang stabil.
        ```toml
        [compatibility]
        pythonVersion = 3.10 # Atau 3.11, versi stabil yang umum didukung
        ```

3.  **Deploy Aplikasi:**
    * Pergi ke [Streamlit Community Cloud](https://share.streamlit.io/).
    * Login dengan akun GitHub Anda.
    * Klik `New app` (atau `Deploy an app`).
    * Pilih repositori GitHub Anda (`darkswan12/warnayolo`).
    * Pilih `main` sebagai branch (atau branch lain jika Anda menggunakan branch berbeda).
    * Pastikan `app.py` dipilih sebagai *Main file path*.
    * Klik `Deploy!`.

## ⚠️ Troubleshooting Umum

* **`FileNotFoundError: No images found...` selama training:**
    * Pastikan Anda sudah menjalankan dan menyelesaikan langkah "Mengatur Ulang Struktur Dataset Sesuai Kebutuhan YOLOv8" di notebook. Dataset asli perlu diformat ulang ke `images/train`, `images/val`, dan `labels/train`, `labels/val`.

* **`libGL.so.1: cannot open shared object file` saat deploy:**
    * Pastikan file `packages.txt` ada di *root directory* repositori GitHub Anda dan berisi `libgl1-mesa-glx`.

* **`StreamlitDuplicateElementKey` atau `RuntimeError: no running event loop`:**
    * Pastikan Anda telah mengganti logika kontrol kamera lama dengan implementasi `streamlit-webrtc` seperti yang ada di `app.py` terbaru.
    * Pastikan file `.streamlit/config.toml` ada dan menentukan `pythonVersion = 3.10` atau `3.11`.
    * Pastikan `streamlit-webrtc==0.45.0` ada di `requirements.txt`.

* **Koneksi Kamera "Taking Longer Than Expected" saat deploy:**
    * Ini masalah WebRTC. Pastikan `rtc_configuration` di `app.py` Anda berisi daftar `iceServers` yang lengkap seperti yang disarankan.

* **Video Muncul tapi Tidak Ada Deteksi:**
    * **Jalur Model Salah:** Pastikan `model_path` di `app.py` (misal: `'runs/detect/train3/weights/best.pt'`) benar-benar sesuai dengan lokasi `best.pt` di repositori GitHub Anda saat di-deploy. Jika model tidak ditemukan atau dimuat, akan ada error di log deployment.
    * **Confidence Threshold Terlalu Tinggi:** Di aplikasi Streamlit, coba geser slider "Confidence Threshold" ke nilai yang sangat rendah (misalnya 0.1 atau 0.05). Jika deteksi muncul, berarti model berhasil tetapi confidence-nya rendah.
    * **Performa Model:** Jika deteksi masih tidak muncul bahkan dengan threshold rendah, model mungkin belum terlatih dengan cukup baik. Kembali ke fase pelatihan:
        * Latih model untuk lebih banyak *epoch*.
        * Pertimbangkan menggunakan varian YOLOv8 yang lebih besar (`yolov8s.pt` atau `yolov8m.pt`) jika sumber daya komputasi memungkinkan.
        * Periksa kembali kualitas dataset dan labeling (misalnya, jika gambar warna solid tidak benar-benar memenuhi seluruh frame atau ada mislabeling).