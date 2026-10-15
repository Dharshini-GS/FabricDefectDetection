# 🧵 Fabric Defect Detection Web Application

Automated, real-time fabric defect inspection system powered by a custom **E-EAF-YOLO11n** architecture (incorporating Efficient Channel Attention and Adaptive Feature Fusion) and packaged into an intuitive **Streamlit** dashboard.

Optimized for lightweight, cost-effective **CPU deployment on AWS** (EC2, ECS, or App Runner).

---

## 📌 Architecture Highlights

- **Base Model:** YOLO11n
- **Custom Enhancements:**
  - **ECA (Efficient Channel Attention):** Captures local cross-channel interactions without dimensionality reduction.
  - **AFF (Adaptive Feature Fusion):** Dynamic, multi-scale feature fusion for improved defect localization.
- **Model Size:** ~5.6 MB (2.73M Parameters)
- **Validation Metrics:**
  - **mAP@50:** 84.4%
  - **mAP@50-95:** 52.4%
  - **CPU Latency:** ~150–170 ms per image

---

## 🎯 Defect Categories

The model detects and localizes 7 fabric defect categories:
1. `baekra`
2. `color issues`
3. `contamination`
4. `cut`
5. `gray stitch`
6. `selvet`
7. `stain`

---

## 📂 Project Structure

```text
fabric-defect-web/
├── app.py                # Streamlit web application
├── best.pt               # Trained E-EAF-YOLO11n checkpoint
├── eaf_modules.py        # Custom ECA & AFF architectural modules and hooks
├── requirements.txt      # Python dependencies for deployment
├── samples/              # Sample fabric images for testing
└── README.md             # Project documentation
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/Dharshini-GS/FabricDefectDetection.git
cd FabricDefectDetection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ☁️ Deployment on AWS (EC2 / Ubuntu)

1. Launch an EC2 instance (e.g. `t3.medium` or `t3.large`).
2. Connect to the instance and install Python & Git:
   ```bash
   sudo apt update && sudo apt install -y python3-pip git
   ```
3. Clone and install:
   ```bash
   git clone https://github.com/Dharshini-GS/FabricDefectDetection.git
   cd FabricDefectDetection
   pip3 install -r requirements.txt
   ```
4. Run as a background service:
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
   ```
5. Ensure your EC2 Security Group allows inbound traffic on port `8501`.
