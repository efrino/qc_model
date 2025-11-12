# -------------------------------------------------------------------
# KODE API DETEKSI CACAT (main.py)
# -------------------------------------------------------------------
import uvicorn              # Server untuk menjalankan API
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io                   # Untuk membaca file upload
# from transformers import AutoImageProcessor, DetrForObjectDetection

# # --- 1. Muat Model Saat Startup (Hanya Sekali) ---
# # Ini adalah bagian terpentING. Model dimuat saat API pertama kali
# # dijalankan, bukan setiap kali ada request. Ini menghemat waktu.

# print("Memuat model... (Ini mungkin perlu beberapa detik)")

# MODEL_PATH = "model-deteksi-cacat-roboflow-final" # Path ke folder model Anda
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Muat processor dan model
# image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
# model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
# model.to(DEVICE) # Pindahkan model ke GPU jika ada

# # Ambil mapping label untuk respon JSON
# id2label = model.config.id2label

# print(f"Model berhasil dimuat dan berjalan di device: {DEVICE}")
import os
from transformers import AutoImageProcessor, DetrForObjectDetection

# --- 1. Muat Model Saat Startup ---

# Ganti ini dengan NAMA_USER/NAMA_REPO Anda di Hugging Face
# Contoh: "efrino/deteksi-cacat-stamping"
MODEL_ID = "efrino/deteksi-cacat-stamping" # <-- GANTI INI DENGAN REPO ANDA (Contoh)

# ... (kode cache_dir_onrender sudah benar) ...
cache_dir_path = CACHE_DIR_ONRENDER if os.path.exists("/var/data") else None

print(f"Memuat model {MODEL_ID}...")
print(f"Menyimpan cache di: {cache_dir_path or 'default'}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=cache_dir_path
)
model = DetrForObjectDetection.from_pretrained(
    MODEL_ID,
    cache_dir=cache_dir_path
)
model.to(DEVICE)

# -----------------------------------------------------------------
# V  PERBAIKAN: TAMBAHKAN BARIS INI V
# -----------------------------------------------------------------
# Ambil mapping label dari konfigurasi model yang baru dimuat
id2label = model.config.id2label
# -----------------------------------------------------------------
# ^  PERBAIKAN: TAMBAHKAN BARIS DI ATAS ^
# -----------------------------------------------------------------

print(f"Model berhasil dimuat dan berjalan di device: {DEVICE}")
# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="API Deteksi Cacat Stamping",
    description="API untuk mendeteksi cacat pada komponen stamping (NEU-DET)",
    version="1.0.0"
)

# --- 2. Definisikan Endpoint API ---

@app.get("/")
def read_root():
    """Endpoint 'Health Check' untuk memastikan API berjalan."""
    return {"status": "OK", "message": "API Deteksi Cacat Aktif!"}


@app.post("/deteksi/")
async def deteksi_cacat(file: UploadFile = File(...)):
    """
    Endpoint utama untuk mendeteksi cacat.
    
    Upload gambar (JPG/PNG) dan API akan mengembalikan 
    daftar cacat yang terdeteksi (label, skor, dan kotak).
    """
    
    # 1. Baca gambar dari file upload
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Lakukan Inferensi (SAMA SEPERTI DI COLAB)
    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Post-processing
    target_sizes = torch.tensor([image.size[::-1]]) # (height, width)
    results = image_processor.post_process_object_detection(
        outputs, 
        threshold=0.5, # Ambil deteksi di atas 50% confidence
        target_sizes=target_sizes
    )[0]

    # 4. Format Output sebagai JSON yang Rapi
    detections = []
    detections.extend(
        {
            "label": id2label[label_id.item()],
            "score": round(score.item(), 3),
            "box": [
                round(i, 2) for i in box.tolist()
            ],  # [xmin, ymin, xmax, ymax]
        }
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        )
    )
    return {
        "filename": file.filename,
        "detections": detections
    }

# --- 3. Perintah untuk Menjalankan (dari Terminal) ---
# (Blok ini hanya untuk referensi, jangan di-uncomment)
# if __name__ == "__main__":
#     # Gunakan perintah ini di terminal Anda:
#     # uvicorn main:app --reload
#     pass