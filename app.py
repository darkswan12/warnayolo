import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase # Import ini

# --- Bagian 1, 2, 3 (Konfigurasi Aplikasi, Muat Model, Konfigurasi Deteksi) tetap sama ---
st.set_page_config(
    page_title="Real-time Color Detector",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Real-time Color Detector with YOLOv8")
st.write("Deteksi warna secara real-time menggunakan model YOLOv8 yang telah dilatih.")

# Inisialisasi session_state
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di jalur: {model_path}. Harap pastikan model 'best.pt' berada di '{model_path}'")
        st.stop()
    model = YOLO(model_path)
    return model

model = load_yolo_model()

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, 0.05)

# --- 4. Stream Kamera Real-time dengan streamlit-webrtc ---
st.subheader("Live Camera Feed")

# Definisi Video Transformer Class
class VideoProcessor(VideoTransformerBase):
    def __init__(self, model_instance, conf_thresh, iou_thresh):
        self.model = model_instance
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def transform(self, frame):
        # Frame yang diterima adalah av.VideoFrame, perlu diubah ke numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Lakukan Inferensi
        # Pastikan model.predict menerima frame BGR
        results = self.model.predict(img, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)

        # Gambar Bounding Boxes dan Label
        annotated_frame = results[0].plot() # YOLOv8 .plot() menghasilkan BGR numpy array

        return annotated_frame # Kembalikan numpy array BGR

# Panggil webrtc_streamer
# Key ini harus unik jika Anda memiliki lebih dari satu webrtc_streamer
webrtc_streamer(
    key="color_detector_camera",
    video_processor_factory=lambda: VideoProcessor(model, confidence_threshold, iou_threshold),
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}, # Hanya aktifkan video, nonaktifkan audio
    async_transform=True # Mengizinkan pemrosesan frame asinkron
)

st.info("Aplikasi siap. Izinkan akses kamera di browser Anda.")

# Bagian tombol Mulai/Hentikan tidak lagi relevan dengan webrtc_streamer secara langsung
# karena webrtc_streamer punya tombol start/stop sendiri di UI.
# Namun, Anda bisa tetap menampilkan slider conf/iou di atasnya.