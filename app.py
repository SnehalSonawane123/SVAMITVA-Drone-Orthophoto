import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
st.set_page_config(page_title="SVAMITVA AI Feature Extraction", layout="wide", page_icon="🛰️")
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
@st.cache_resource
def load_model():
    try:
        import cv2
        from ultralytics import YOLO
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        model = YOLO('yolov8n.pt')
        return model, cv2
    except Exception as e:
        return None, str(e)
def detect_objects(image, model, cv2, confidence):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    results = model(img_array, conf=confidence/100)
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    detections = results[0].boxes
    buildings = 0
    vehicles = 0
    trees = 0
    roads = 0
    water = 0
    people = 0
    class_names = results[0].names
    for box in detections:
        cls = int(box.cls[0])
        class_name = class_names[cls].lower()
        if any(x in class_name for x in ['person', 'people']):
            people += 1
        elif any(x in class_name for x in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            vehicles += 1
        elif any(x in class_name for x in ['tree', 'plant', 'potted']):
            trees += 1
        elif 'road' in class_name or 'street' in class_name:
            roads += 1
        elif 'water' in class_name or 'lake' in class_name or 'river' in class_name:
            water += 1
    total_objects = len(detections)
    results_dict = {
        'total_objects': total_objects,
        'buildings': buildings,
        'vehicles': vehicles,
        'trees': trees,
        'roads': roads,
        'water': water,
        'people': people,
        'detections': detections,
        'class_names': class_names
    }
    return annotated_img, results_dict
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛰️ SVAMITVA AI Feature Extraction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748b;'>Real-time Object Detection for Drone Imagery</h3>", unsafe_allow_html=True)
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Results", "💾 Export"])
with tab1:
    st.header("📤 Upload Drone Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader("Image Info")
            width, height = image.size
            st.metric("Width", f"{width} px")
            st.metric("Height", f"{height} px")
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("Size", f"{file_size:.2f} MB")
        st.divider()
        st.subheader("🎯 Detection Settings")
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold (%)", 10, 100, 50)
        with col2:
            use_gpu = st.checkbox("Use GPU", value=False)
        if st.button("🚀 Detect Objects", type="primary", use_container_width=True):
            result = load_model()
            if result[0] is None:
                st.error(f"Model loading failed: {result[1]}")
                st.info("Please ensure packages.txt file exists with libgl1 and libglib2.0-0")
            else:
                model, cv2 = result
                with st.spinner("🔄 Detecting objects..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    try:
                        annotated_img, results = detect_objects(image, model, cv2, confidence_threshold)
                        st.session_state.annotated_image = annotated_img
                        st.session_state.results = results
                        st.session_state.processed = True
                        st.success(f"✅ Detected {results['total_objects']} objects!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
with tab2:
    st.header("📊 Detection Results")
    if not st.session_state.processed:
        st.warning("⚠️ Please upload and process an image first")
    else:
        results = st.session_state.results
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total", results['total_objects'])
        col2.metric("People", results['people'])
        col3.metric("Vehicles", results['vehicles'])
        col4.metric("Buildings", results['buildings'])
        col5.metric("Trees", results['trees'])
        col6.metric("Water", results['water'])
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.uploaded_image, use_container_width=True)
        with col2:
            st.subheader("Detected Objects")
            st.image(st.session_state.annotated_image, use_container_width=True)
        st.divider()
        st.subheader("Detected Objects List")
        detection_data = []
        for i, box in enumerate(results['detections']):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results['class_names'][cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detection_data.append({
                'ID': i+1,
                'Class': class_name,
                'Confidence': f"{conf*100:.2f}%",
                'BBox': f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            })
        if detection_data:
            df = pd.DataFrame(detection_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
with tab3:
    st.header("💾 Export Results")
    if not st.session_state.processed:
        st.warning("⚠️ No results to export")
    else:
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        if st.button("📥 Download Results", type="primary"):
            results = st.session_state.results
            detection_data = []
            for i, box in enumerate(results['detections']):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results['class_names'][cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detection_data.append({
                    'ID': i+1,
                    'Class': class_name,
                    'Confidence': conf,
                    'X1': int(x1),
                    'Y1': int(y1),
                    'X2': int(x2),
                    'Y2': int(y2)
                })
            df = pd.DataFrame(detection_data)
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "detections.csv", "text/csv")
            elif export_format == "JSON":
                json_str = df.to_json(orient='records')
                st.download_button("Download JSON", json_str, "detections.json", "application/json")
            elif export_format == "Excel":
                from io import BytesIO
                buffer = BytesIO()
                df.to_excel(buffer, index=False)
                st.download_button("Download Excel", buffer.getvalue(), "detections.xlsx", "application/vnd.ms-excel")
        if st.button("📥 Download Annotated Image"):
            from io import BytesIO
            img = Image.fromarray(st.session_state.annotated_image)
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download Image", buf.getvalue(), "annotated_image.png", "image/png")
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 SVAMITVA AI System | Ministry of Panchayati Raj</p>", unsafe_allow_html=True)
