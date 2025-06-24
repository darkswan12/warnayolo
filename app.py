import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- 1. Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Real-time Color Detector",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Real-time Color Detector with YOLOv8")
st.write("Deteksi warna secara real-time menggunakan model YOLOv8 yang telah dilatih.")

# --- Inisialisasi session_state untuk mengelola status kamera ---
# Ini penting agar status 'run_camera' tidak hilang saat Streamlit melakukan re-run
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# --- 2. Muat Model YOLOv8 ---
@st.cache_resource # Cache resource untuk menghindari model dimuat berulang kali
def load_yolo_model():
    model_path = 'best.pt' # Pastikan jalur ini benar
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di jalur: {model_path}. Harap pastikan model 'best.pt' berada di '{model_path}'")
        st.stop()
    
    model = YOLO(model_path)
    return model

model = load_yolo_model()

# --- 3. Konfigurasi Deteksi (Optional: Tambahkan slider untuk user) ---
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, 0.05)

# --- 4. Kontrol Kamera ---
st.subheader("Live Camera Feed")

# Tombol untuk memulai/menghentikan deteksi
# Kita menggunakan st.session_state untuk menyimpan statusnya
col1, col2 = st.columns(2)

with col1:
    start_button = st.button("Mulai Deteksi Kamera")
with col2:
    stop_button = st.button("Hentikan Deteksi Kamera")

if start_button:
    st.session_state.run_camera = True
elif stop_button:
    st.session_state.run_camera = False

frame_placeholder = st.empty() # Placeholder untuk menampilkan frame video

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Gagal mengakses kamera. Pastikan kamera terhubung dan tidak sedang digunakan oleh aplikasi lain.")
        st.stop()

    st.info("Kamera dimulai. Klik 'Hentikan Deteksi Kamera' untuk mengakhiri.")

    while st.session_state.run_camera: # Loop akan berjalan selama st.session_state.run_camera True
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca frame dari kamera. Menghentikan deteksi.")
            st.session_state.run_camera = False # Set state ke False untuk menghentikan loop
            break

        # Lakukan Inferensi
        results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)

        # Gambar Bounding Boxes dan Label
        annotated_frame = results[0].plot()

        # Tampilkan frame yang sudah dianotasi di Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # Perbaikan deprecation warning: use_container_width
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True) 

    cap.release()
    st.info("Kamera dihentikan.")
else:
    st.info("Tekan 'Mulai Deteksi Kamera' untuk memulai deteksi real-time dari webcam Anda.")