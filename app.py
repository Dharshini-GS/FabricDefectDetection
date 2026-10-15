"""
Fabric Defect Detection Web Application
Model: E-EAF-YOLO11n (Efficient Channel Attention + Adaptive Feature Fusion)
Optimized for AWS EC2/ECS CPU Deployment
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Ensure local directory is in Python path for eaf_modules
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import custom architecture module and register hooks BEFORE loading model
try:
    import eaf_modules
    eaf_modules.register()
except Exception as e:
    st.error(f"Failed to load eaf_modules: {e}")

from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fabric Defect Detection | E-EAF-YOLO11n",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background-color: #DCFCE7; color: #166534; }
    .badge-defect { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loader (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_detection_model(weights_path: str):
    """Load and cache the trained YOLO model checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")
    model = YOLO(weights_path)
    return model


MODEL_PATH = str(CURRENT_DIR / "best.pt")

try:
    model = load_detection_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_err = str(e)


# -----------------------------------------------------------------------------
# Sidebar: Settings & Model Specs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fabric.png", width=64)
    st.title("Settings & Specs")

    st.subheader("Model Configuration")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.35,  # Optimal F1 threshold identified during validation
        step=0.05,
        help="Minimum confidence score for defect detection (recommended: 0.35)."
    )

    iou_threshold = st.slider(
        "IoU / NMS Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression."
    )

    device_option = st.selectbox(
        "Compute Device",
        options=["cpu", "0"],
        index=0,
        format_func=lambda x: "CPU (Optimized for Web/AWS)" if x == "cpu" else "GPU (CUDA Device 0)",
        help="Select CPU for lightweight AWS deployment without GPU overhead."
    )

    line_thickness = st.slider("Bounding Box Thickness", 1, 6, 2)

    st.markdown("---")
    st.subheader("Architecture Info")
    st.markdown("""
    - **Model:** `E-EAF-YOLO11n`
    - **Attention:** ECA (Efficient Channel Attention)
    - **Fusion:** AFF (Adaptive Feature Fusion)
    - **Parameters:** ~2.73M
    - **Model Size:** 5.6 MB
    - **Trained Classes:** 7 defects
    - **mAP@50:** 84.4% (Val) | 82.8% (Test)
    """)

    st.markdown("---")
    st.caption("Ready for AWS EC2/ECS Deployment")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧵 Fabric Defect Inspection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-time automated fabric quality inspection powered by custom E-EAF-YOLO11n architecture.</div>',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"⚠️ Could not load model from `{MODEL_PATH}`: {load_err}")
    st.stop()


# -----------------------------------------------------------------------------
# Input Source Selection
# -----------------------------------------------------------------------------
tab_upload, tab_samples, tab_cam = st.tabs(["📤 Upload Image", "🖼️ Sample Fabric Images", "📷 Live Camera"])

input_image = None
image_source_name = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a fabric image for defect inspection...",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_source_name = uploaded_file.name

with tab_samples:
    samples_dir = CURRENT_DIR / "samples"
    sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    if sample_files:
        sample_names = [f.name for f in sample_files]
        selected_sample_name = st.selectbox("Select a test fabric sample:", sample_names)
        if st.button("Load Selected Sample"):
            selected_path = samples_dir / selected_sample_name
            input_image = Image.open(selected_path).convert("RGB")
            image_source_name = selected_sample_name
            st.session_state["active_sample"] = selected_sample_name
        elif "active_sample" in st.session_state:
            input_image = Image.open(samples_dir / st.session_state["active_sample"]).convert("RGB")
            image_source_name = st.session_state["active_sample"]
    else:
        st.info("No sample images found in `samples/` directory.")

with tab_cam:
    cam_file = st.camera_input("Capture live fabric photo")
    if cam_file is not None:
        input_image = Image.open(cam_file).convert("RGB")
        image_source_name = "Camera Capture"


# -----------------------------------------------------------------------------
# Inference and Results
# -----------------------------------------------------------------------------
if input_image is not None:
    st.markdown("---")

    # Run inference with timing
    img_array = np.array(input_image)
    start_time = time.time()
    
    with st.spinner("Analyzing fabric for defects..."):
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device_option,
            line_width=line_thickness,
            verbose=False
        )
    
    elapsed_ms = (time.time() - start_time) * 1000.0
    res = results[0]

    # Parse detections
    boxes = res.boxes
    num_detections = len(boxes)
    annotated_bgr = res.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_pil = Image.fromarray(annotated_rgb)

    # Class distribution
    class_names = model.names
    detection_data = []

    if num_detections > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detection_data.append({
                "Defect Class": cls_name,
                "Confidence": f"{conf * 100:.1f}%",
                "Confidence_Raw": conf,
                "Bounding Box [x1, y1, x2, y2]": str(xyxy)
            })

    # Summary Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Defects Detected</div>
            <div class="metric-val">{num_detections}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        status_html = (
            '<span class="status-badge badge-defect">DEFECT DETECTED</span>'
            if num_detections > 0
            else '<span class="status-badge badge-success">QUALITY PASSED</span>'
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Quality Status</div>
            <div style="margin-top: 10px;">{status_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Inference Latency</div>
            <div class="metric-val">{elapsed_ms:.1f} <span style="font-size:1rem;color:#64748B;">ms</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        primary_cls = detection_data[0]["Defect Class"] if num_detections > 0 else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Primary Defect</div>
            <div class="metric-val" style="font-size:1.3rem; margin-top:6px;">{primary_cls}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Comparison Columns
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Original Fabric Image")
        st.image(input_image, use_container_width=True, caption=f"Source: {image_source_name}")

    with col_right:
        st.subheader("Defect Localization")
        st.image(annotated_pil, use_container_width=True, caption=f"Detections at {conf_threshold*100:.0f}% confidence")

    # Detailed Table & Breakdown
    if num_detections > 0:
        st.subheader("Defect Analysis Breakdown")
        df_dets = pd.DataFrame(detection_data)
        
        col_t1, col_t2 = st.columns([3, 2])
        with col_t1:
            st.dataframe(
                df_dets[["Defect Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"]],
                use_container_width=True
            )
        with col_t2:
            counts = df_dets["Defect Class"].value_counts()
            st.bar_chart(counts)

    # Download Button
    st.markdown("---")
    import io
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Annotated Inspection Image",
        data=byte_im,
        file_name=f"inspected_{image_source_name or 'fabric'}.jpg",
        mime="image/jpeg"
    )

else:
    # Landing Guide
    st.info("👆 Please upload a fabric image, select a sample from the tabs above, or use your webcam to begin automated inspection.")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown("### 1. Upload Fabric")
        st.write("Upload high-resolution textile fabric swatches or capture directly with your camera.")
    with col_g2:
        st.markdown("### 2. Neural Detection")
        st.write("E-EAF-YOLO11n extracts multi-scale features and channel attention to locate subtle defects.")
    with col_g3:
        st.markdown("### 3. Quality Report")
        st.write("Inspect localized bounding boxes, confidence ratings, and defect type categorization.")
