import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
st.set_page_config(page_title="SVAMITVA Feature Extraction", layout="wide", page_icon="🛰️")
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
        model = YOLO('yolov8n.pt')
        return model, cv2, None
    except Exception as e:
        return None, None, str(e)
def draw_detections(image, detections, class_names, confidence_threshold):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    colors = {
        'person': '#FF0000',
        'car': '#00FF00',
        'truck': '#00FF00',
        'bus': '#00FF00',
        'motorcycle': '#0000FF',
        'bicycle': '#0000FF',
        'building': '#FFFF00',
        'tree': '#00FFFF',
        'default': '#FF00FF'
    }
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < confidence_threshold / 100:
            continue
        class_name = class_names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = colors.get(class_name.lower(), colors['default'])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{class_name} {conf*100:.1f}%"
        bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1-25), label, fill='white', font=font)
    return img_draw
def detect_objects(image, model, confidence):
    img_array = np.array(image.convert('RGB'))
    results = model(img_array, conf=confidence/100, verbose=False, imgsz=640)
    detections = results[0].boxes
    class_names = results[0].names
    annotated_img = draw_detections(image, detections, class_names, confidence)
    detection_summary = {}
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < confidence / 100:
            continue
        class_name = class_names[cls]
        if class_name not in detection_summary:
            detection_summary[class_name] = 0
        detection_summary[class_name] += 1
    total_objects = sum(detection_summary.values())
    results_dict = {
        'total_objects': total_objects,
        'summary': detection_summary,
        'detections': detections,
        'class_names': class_names
    }
    return annotated_img, results_dict
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛰️ SVAMITVA Feature Extraction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748b;'>AI-Powered Object Detection with Boundary Mapping</h3>", unsafe_allow_html=True)
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Results & Boundaries", "💾 Export Data"])
with tab1:
    st.header("📤 Upload Drone Orthophoto")
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader("Image Properties")
            width, height = image.size
            st.metric("Width", f"{width}px")
            st.metric("Height", f"{height}px")
            st.metric("Aspect", f"{width/height:.2f}")
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("Size", f"{file_size:.2f}MB")
        st.divider()
        st.subheader("🎯 Detection Configuration")
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 10, 95, 25, 5, help="Lower = More detections, Higher = More accurate")
            st.caption(f"Current: {confidence_threshold}% - Objects above this confidence will be detected")
        with col2:
            detection_mode = st.selectbox("Detection Mode", ["Standard", "High Precision", "Maximum Detections"])
            if detection_mode == "High Precision":
                confidence_threshold = 60
            elif detection_mode == "Maximum Detections":
                confidence_threshold = 15
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            model_result = load_model()
            if model_result[0] is None:
                st.error(f"❌ Model loading failed: {model_result[2]}")
                st.info("Ensure packages.txt contains: libgl1 and libglib2.0-0")
            else:
                model = model_result[0]
                with st.spinner(f"🔄 Analyzing image with {confidence_threshold}% confidence..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.015)
                        progress_bar.progress(i + 1)
                    try:
                        annotated_img, results = detect_objects(image, model, confidence_threshold)
                        st.session_state.annotated_image = annotated_img
                        st.session_state.results = results
                        st.session_state.processed = True
                        progress_bar.empty()
                        if results['total_objects'] == 0:
                            st.warning(f"⚠️ No objects detected at {confidence_threshold}% confidence. Try lowering threshold to 15-20%")
                        else:
                            st.success(f"✅ Successfully detected {results['total_objects']} objects!")
                            st.info(f"📍 Found: {', '.join([f'{k}: {v}' for k, v in results['summary'].items()])}")
                    except Exception as e:
                        st.error(f"❌ Detection failed: {str(e)}")
with tab2:
    st.header("📊 Detection Results & Boundary Analysis")
    if not st.session_state.processed:
        st.warning("⚠️ Upload and process an image first")
    else:
        results = st.session_state.results
        if results['total_objects'] == 0:
            st.error("❌ No objects detected. Lower confidence threshold in Upload tab.")
        else:
            st.subheader(f"🎯 Detected {results['total_objects']} Objects")
            summary = results['summary']
            cols = st.columns(min(len(summary), 5))
            for idx, (obj_type, count) in enumerate(summary.items()):
                with cols[idx % 5]:
                    st.metric(obj_type.title(), count)
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Original Image")
                st.image(st.session_state.uploaded_image, use_container_width=True)
            with col2:
                st.subheader("🎯 Detected Objects with Boundaries")
                st.image(st.session_state.annotated_image, use_container_width=True)
            st.divider()
            st.subheader("📐 Detailed Boundary Coordinates")
            detection_data = []
            for i, box in enumerate(results['detections']):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results['class_names'][cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                detection_data.append({
                    'ID': i+1,
                    'Object': class_name,
                    'Confidence': f"{conf*100:.1f}%",
                    'X1': int(x1),
                    'Y1': int(y1),
                    'X2': int(x2),
                    'Y2': int(y2),
                    'Width': int(width),
                    'Height': int(height),
                    'Area': int(area),
                    'Center_X': int(center_x),
                    'Center_Y': int(center_y)
                })
            if detection_data:
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Total Detected Objects: {len(detection_data)}")
                st.divider()
                st.subheader("📊 Detection Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_conf = df['Confidence'].str.rstrip('%').astype(float).mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                with col2:
                    avg_area = df['Area'].mean()
                    st.metric("Avg Object Area", f"{int(avg_area)}px²")
                with col3:
                    total_area = df['Area'].sum()
                    st.metric("Total Coverage", f"{int(total_area)}px²")
                with col4:
                    img_area = st.session_state.uploaded_image.size[0] * st.session_state.uploaded_image.size[1]
                    coverage_pct = (total_area / img_area) * 100
                    st.metric("Coverage %", f"{coverage_pct:.2f}%")
with tab3:
    st.header("💾 Export Detection Data")
    if not st.session_state.processed:
        st.warning("⚠️ No data to export")
    elif st.session_state.results['total_objects'] == 0:
        st.warning("⚠️ No objects detected to export")
    else:
        st.subheader("📤 Export Formats")
        export_format = st.selectbox("Select Format", ["CSV (Spreadsheet)", "JSON (API)", "Excel (XLSX)", "GeoJSON (GIS)"])
        include_options = st.multiselect("Include Data", ["Boundaries", "Centers", "Areas", "Confidence"], default=["Boundaries", "Confidence"])
        if st.button("📥 Generate Export File", type="primary", use_container_width=True):
            results = st.session_state.results
            detection_data = []
            for i, box in enumerate(results['detections']):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results['class_names'][cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                data_point = {
                    'ID': i+1,
                    'Object': class_name,
                }
                if "Confidence" in include_options:
                    data_point['Confidence'] = round(conf, 3)
                if "Boundaries" in include_options:
                    data_point['X1'] = int(x1)
                    data_point['Y1'] = int(y1)
                    data_point['X2'] = int(x2)
                    data_point['Y2'] = int(y2)
                if "Centers" in include_options:
                    data_point['Center_X'] = int((x1 + x2) / 2)
                    data_point['Center_Y'] = int((y1 + y2) / 2)
                if "Areas" in include_options:
                    data_point['Width'] = int(x2 - x1)
                    data_point['Height'] = int(y2 - y1)
                    data_point['Area'] = int((x2 - x1) * (y2 - y1))
                detection_data.append(data_point)
            df = pd.DataFrame(detection_data)
            if "CSV" in export_format:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "detections.csv", "text/csv", use_container_width=True)
            elif "JSON" in export_format:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", json_str, "detections.json", "application/json", use_container_width=True)
            elif "Excel" in export_format:
                from io import BytesIO
                buffer = BytesIO()
                df.to_excel(buffer, index=False, engine='openpyxl')
                st.download_button("📥 Download Excel", buffer.getvalue(), "detections.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            elif "GeoJSON" in export_format:
                geojson = {"type": "FeatureCollection", "features": []}
                for item in detection_data:
                    feature = {
                        "type": "Feature",
                        "properties": {k: v for k, v in item.items() if k not in ['X1', 'Y1', 'X2', 'Y2']},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [item.get('X1', 0), item.get('Y1', 0)],
                                [item.get('X2', 0), item.get('Y1', 0)],
                                [item.get('X2', 0), item.get('Y2', 0)],
                                [item.get('X1', 0), item.get('Y2', 0)],
                                [item.get('X1', 0), item.get('Y1', 0)]
                            ]]
                        }
                    }
                    geojson["features"].append(feature)
                import json
                geojson_str = json.dumps(geojson, indent=2)
                st.download_button("📥 Download GeoJSON", geojson_str, "detections.geojson", "application/geo+json", use_container_width=True)
            st.success(f"✅ Export ready with {len(detection_data)} detected objects!")
        st.divider()
        st.subheader("🖼️ Export Annotated Image")
        if st.button("📥 Download Image with Boundaries", use_container_width=True):
            from io import BytesIO
            buf = BytesIO()
            st.session_state.annotated_image.save(buf, format="PNG")
            st.download_button("📥 Download PNG", buf.getvalue(), "annotated_detection.png", "image/png", use_container_width=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>© 2024 SVAMITVA AI System | Ministry of Panchayati Raj | Problem Statement #1705</p>", unsafe_allow_html=True)
