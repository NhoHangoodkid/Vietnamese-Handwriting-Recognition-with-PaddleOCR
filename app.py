import time
import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# Import custom src modules
from src.config import AppConfig
from src.preprocessor import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.utils import export_to_json

# Streamlit Page Configuration
st.set_page_config(
    page_title="Nhận Dạng Chữ Viết Tay Tiếng Việt",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4facfe 100%);
        padding: 22px 26px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.12);
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2rem;
        color: #ffffff;
    }
    
    .main-header p {
        margin: 5px 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    
    .canvas-border {
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        padding: 10px;
        background: #ffffff;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize OCR Engine
config = AppConfig()
engine = OCREngine(config)

# Header
st.markdown("""
<div class="main-header">
    <h1>Nhận Dạng Chữ Viết Tay Tiếng Việt</h1>
    <p>Sử dụng mô hình PaddleOCR chuyên biệt cho tiếng Việt</p>
</div>
""", unsafe_allow_html=True)

# Select Input Mode
st.markdown("Chọn phương thức nhập dữ liệu:")
input_mode = st.radio(
    "Chọn phương thức:",
    ["Bảng vẽ viết tay trực tiếp (Canvas)", "Tải file ảnh từ máy tính (Upload)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ================= 1. MODE: CANVAS DRAWING =================
if input_mode == "Bảng vẽ viết tay trực tiếp (Canvas)":
    col_cv1, col_cv2 = st.columns([1.1, 0.9], gap="large")
    
    with col_cv1:
        st.markdown("Khung Vẽ Chữ Viết Tay")
        
        # Tool options for drawing
        tool_col1, tool_col2 = st.columns(2)
        with tool_col1:
            stroke_width = st.slider("Độ dày nét bút (Pixel):", min_value=4, max_value=25, value=10)
        with tool_col2:
            stroke_color = st.color_picker("Màu nét vẽ:", "#000000")
        
        # Streamlit Canvas
        try:
            from streamlit_drawable_canvas import st_canvas
            st.markdown('<div class="canvas-border">', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#ffffff",
                height=260,
                width=550,
                drawing_mode="freedraw",
                key="main_paint_canvas",
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Dùng chuột hoặc ngón tay / bút cảm ứng để viết chữ tiếng Việt vào ô trắng phía trên.")
        except ImportError:
            st.error("Chưa cài đặt thư viện bảng vẽ `streamlit-drawable-canvas`. Vui lòng chạy lệnh: `pip install streamlit-drawable-canvas`")
            canvas_result = None

    with col_cv2:
        st.markdown("Kết Quả Nhận Diện")
        
        if canvas_result is not None and canvas_result.image_data is not None:
            alpha_channel = canvas_result.image_data[:, :, 3]
            is_empty = np.all(alpha_channel == 0)
            
            if st.button("Nhận diện chữ vừa vẽ", type="primary", use_container_width=True, key="btn_run_canvas"):
                if is_empty:
                    st.warning("Khung vẽ đang trống. Hãy vẽ chữ tiếng Việt trước khi bấm nhận diện.")
                else:
                    canvas_rgba = canvas_result.image_data.astype('uint8')
                    drawn_pil = Image.fromarray(canvas_rgba, 'RGBA')
                    
                    # White background paste
                    bg_img = Image.new("RGB", drawn_pil.size, (255, 255, 255))
                    bg_img.paste(drawn_pil, mask=drawn_pil.split()[3])
                    
                    # Auto crop strokes
                    cropped_drawing = ImagePreprocessor.crop_to_content(bg_img, padding=20)
                    
                    with st.spinner("Đang phân tích nét vẽ"):
                        start_time = time.time()
                        result = engine.recognize(cropped_drawing)
                        elapsed = time.time() - start_time
                        
                    texts = result.get("texts", [])
                    scores = result.get("scores", [])
                    
                    if texts:
                        full_text = " ".join(texts)
                        avg_score = float(np.mean(scores)) if scores else 1.0
                        
                        st.markdown(f"""
                        <div style="background: #f8fafc; border: 2px solid #3b82f6; border-radius: 12px; padding: 18px; margin: 15px 0;">
                            <div style="font-size: 13px; color: #475569; font-weight: 600; margin-bottom: 6px;">Văn bản nhận diện được:</div>
                            <div style="font-size: 28px; font-weight: 700; color: #0f172a; word-break: break-word;">{full_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Độ tin cậy (Confidence)", f"{avg_score:.1%}")
                        m2.metric("Thời gian xử lý", f"{elapsed:.2f}s")
                        
                        st.download_button(
                            "📥 Tải kết quả (.txt)",
                            data=full_text,
                            file_name="ket_qua_ve_tay.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning("Chưa nhận diện được chữ. Bạn hãy thử tăng độ dày nét bút lên 10-14px và viết rõ nét hơn.")
        else:
            st.info("Hãy viết chữ vào ô bảng vẽ bên trái rồi bấm nút **Nhận diện chữ vừa vẽ**.")


# ================= 2. MODE: IMAGE UPLOAD =================
else:
    col_up1, col_up2 = st.columns([1.1, 0.9], gap="large")
    
    with col_up1:
        st.markdown("Tải Tệp Ảnh Lên")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh chữ viết tay (PNG, JPG, JPEG):", 
            type=["png", "jpg", "jpeg"],
            key="file_uploader_single"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Ảnh tải lên: {uploaded_file.name}", use_container_width=True)

    with col_up2:
        st.markdown("Kết quả")
        if uploaded_file is not None:
            if st.button("Nhận diện ngay", type="primary", use_container_width=True, key="btn_run_upload"):
                with st.spinner("Đang phân tích hình ảnh..."):
                    # Auto crop text content from uploaded image
                    cropped_img = ImagePreprocessor.crop_to_content(image, padding=15)
                    
                    start_time = time.time()
                    result = engine.recognize(cropped_img)
                    elapsed = time.time() - start_time
                
                texts = result.get("texts", [])
                scores = result.get("scores", [])
                
                if texts:
                    full_text = " ".join(texts)
                    avg_score = float(np.mean(scores)) if scores else 1.0
                    
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 2px solid #3b82f6; border-radius: 12px; padding: 18px; margin: 15px 0;">
                        <div style="font-size: 13px; color: #475569; font-weight: 600; margin-bottom: 6px;">Văn bản nhận diện được:</div>
                        <div style="font-size: 28px; font-weight: 700; color: #0f172a; word-break: break-word;">{full_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Độ tin cậy (Confidence)", f"{avg_score:.1%}")
                    m2.metric("Thời gian xử lý", f"{elapsed:.2f}s")
                    
                    st.download_button(
                        "📥 Tải file kết quả (.txt)",
                        data=full_text,
                        file_name="ket_qua_ocr.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.warning("Không nhận diện được chữ trong ảnh hoặc chưa tìm thấy kết quả.")
        else:
            st.info("Hãy chọn một tệp ảnh ở bên trái rồi bấm **Nhận diện ngay**.")
