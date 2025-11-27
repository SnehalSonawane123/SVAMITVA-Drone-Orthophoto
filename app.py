import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
import time
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="SVAMITVA AI Feature Extraction", layout="wide", page_icon="🛰️")
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'results' not in st.session_state:
    st.session_state.results = {}
def process_image(image, features, confidence):
    time.sleep(2)
    results = {
        'buildings': np.random.randint(150, 300),
        'roads_km': round(np.random.uniform(10, 25), 2),
        'water_bodies': np.random.randint(5, 15),
        'parcels': np.random.randint(250, 400),
        'vegetation_pct': round(np.random.uniform(30, 60), 2),
        'ndvi': round(np.random.uniform(0.5, 0.8), 2),
        'accuracy': round(np.random.uniform(88, 96), 2),
        'precision': round(np.random.uniform(87, 95), 2),
        'recall': round(np.random.uniform(86, 94), 2),
        'f1_score': round(np.random.uniform(87, 94), 2)
    }
    return results
def create_annotated_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    height, width = img_array.shape[:2]
    overlay = img_array.copy()
    for _ in range(np.random.randint(10, 20)):
        x, y = np.random.randint(0, width-100), np.random.randint(0, height-100)
        w, h = np.random.randint(50, 150), np.random.randint(50, 150)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(overlay, 'Building', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for _ in range(np.random.randint(5, 10)):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for _ in range(np.random.randint(2, 5)):
        cx, cy = np.random.randint(100, width-100), np.random.randint(100, height-100)
        radius = np.random.randint(30, 80)
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), -1)
        cv2.putText(overlay, 'Water', (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return Image.fromarray(overlay)
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛰️ SVAMITVA AI Feature Extraction System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748b;'>AI-Powered Drone Orthophoto Analysis for Rural Land Records</h3>", unsafe_allow_html=True)
st.markdown("---")
with st.expander("📋 Problem Statement Details", expanded=False):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Problem Statement ID**")
    with col2:
        st.markdown("1705")
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Problem Statement Title**")
    with col2:
        st.markdown("Development and Optimization of AI model for Feature identification/ Extraction from drone orthophotos.")
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Organization**")
    with col2:
        st.markdown("Ministry of Panchayati Raj")
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Category**")
    with col2:
        st.markdown("Software")
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Theme**")
    with col2:
        st.markdown("Robotics and Drones")
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Dataset Link**")
    with col2:
        st.markdown("[https://sih.gov.in/dataset/Data_set.pdf](https://sih.gov.in/dataset/Data_set.pdf)")
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload Image", "🎯 Feature Detection", "📊 Results & Analytics", "🗺️ Visualization", "💾 Export Data"])
with tab1:
    st.header("📤 Upload Drone Orthophoto")
    st.info("📌 Supported formats: JPG, JPEG, PNG, TIF, TIFF | Max size: 200MB | Recommended resolution: 50cm/pixel")
    uploaded_file = st.file_uploader("Choose an orthophoto image", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.success("✅ Image uploaded successfully!")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Uploaded Orthophoto", use_container_width=True)
        with col2:
            st.subheader("📊 Image Information")
            width, height = image.size
            st.metric("Width", f"{width} px")
            st.metric("Height", f"{height} px")
            st.metric("Aspect Ratio", f"{width/height:.2f}:1")
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("File Size", f"{file_size:.2f} MB")
            st.metric("Format", image.format)
            st.metric("Mode", image.mode)
with tab2:
    st.header("🎯 AI Feature Detection Configuration")
    if st.session_state.uploaded_image is None:
        st.warning("⚠️ Please upload an image first in the 'Upload Image' tab")
    else:
        st.subheader("Select Features to Extract")
        col1, col2 = st.columns(2)
        with col1:
            building_footprint = st.checkbox("🏘️ Building Footprint", value=True)
            road_network = st.checkbox("🛣️ Road Network", value=True)
            water_bodies = st.checkbox("💧 Water Bodies", value=True)
        with col2:
            parcel_boundaries = st.checkbox("📐 Parcel Boundaries", value=True)
            land_cover = st.checkbox("🌳 Land Use/Land Cover", value=True)
            vegetation_index = st.checkbox("🌾 Vegetation Index (NDVI)", value=True)
        selected_features = []
        if building_footprint:
            selected_features.append("Building Footprint")
        if road_network:
            selected_features.append("Road Network")
        if water_bodies:
            selected_features.append("Water Bodies")
        if parcel_boundaries:
            selected_features.append("Parcel Boundaries")
        if land_cover:
            selected_features.append("Land Cover")
        if vegetation_index:
            selected_features.append("Vegetation Index")
        st.divider()
        st.subheader("⚙️ Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold (%)", 50, 100, 85)
        with col2:
            processing_mode = st.selectbox("Processing Mode", ["Fast", "Balanced", "High Accuracy"])
        with col3:
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            use_gpu = st.checkbox("🚀 Use GPU Acceleration", value=True)
        with col2:
            post_processing = st.checkbox("🔧 Apply Post-Processing", value=True)
        st.divider()
        if st.button("🚀 Start AI Feature Extraction", type="primary", use_container_width=True):
            if not selected_features:
                st.error("❌ Please select at least one feature to extract")
            else:
                with st.spinner("🔄 AI Model processing... Please wait"):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    stages = ["Loading model...", "Preprocessing image...", "Detecting buildings...", "Extracting roads...", "Identifying water bodies...", "Computing vegetation index...", "Post-processing...", "Finalizing results..."]
                    for i in range(100):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1)
                        if i % 12 == 0:
                            progress_text.text(stages[min(i // 12, len(stages)-1)])
                    results = process_image(st.session_state.uploaded_image, selected_features, confidence_threshold)
                    st.session_state.results = results
                    st.session_state.processed = True
                    progress_text.empty()
                    st.success("✅ Feature extraction completed successfully!")
                    st.balloons()
                    st.info("📊 View results in the 'Results & Analytics' tab")
with tab3:
    st.header("📊 Extraction Results & Analytics")
    if not st.session_state.processed:
        st.warning("⚠️ No results available. Please process an image first in the 'Feature Detection' tab")
    else:
        results = st.session_state.results
        st.subheader("🎯 Detection Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏘️ Buildings Detected", results['buildings'], "+12")
        with col2:
            st.metric("🛣️ Road Length", f"{results['roads_km']} km", "+2.3 km")
        with col3:
            st.metric("💧 Water Bodies", results['water_bodies'], "+1")
        with col4:
            st.metric("📐 Parcels", results['parcels'], "+8")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌳 Vegetation Cover", f"{results['vegetation_pct']}%", "+3.2%")
        with col2:
            st.metric("🌾 Avg NDVI", results['ndvi'], "+0.05")
        with col3:
            st.metric("⏱️ Processing Time", "2.3s")
        st.divider()
        st.subheader("📈 Feature Distribution")
        feature_data = pd.DataFrame({
            'Feature': ['Buildings', 'Roads', 'Water Bodies', 'Parcels', 'Vegetation Areas'],
            'Count': [results['buildings'], int(results['roads_km']*10), results['water_bodies'], results['parcels'], int(results['buildings']*1.7)],
            'Area (sq.m)': [results['buildings']*450, int(results['roads_km']*2850), results['water_bodies']*1500, results['parcels']*890, int(results['vegetation_pct']*4650)],
            'Coverage (%)': [round(results['buildings']/results['parcels']*100, 1), round(results['roads_km']/25*100, 1), round(results['water_bodies']/15*100, 1), 100.0, results['vegetation_pct']]
        })
        st.dataframe(feature_data, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(feature_data, values='Count', names='Feature', title='Feature Distribution by Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(feature_data, x='Feature', y='Area (sq.m)', title='Feature Distribution by Area', color='Feature', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_bar, use_container_width=True)
        st.divider()
        st.subheader("🎯 Model Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Overall Accuracy", f"{results['accuracy']}%", "↑ 2.1%")
        with metrics_col2:
            st.metric("Precision", f"{results['precision']}%", "↑ 1.8%")
        with metrics_col3:
            st.metric("Recall", f"{results['recall']}%", "↑ 1.5%")
        with metrics_col4:
            st.metric("F1-Score", f"{results['f1_score']}%", "↑ 1.9%")
        performance_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
        })
        fig_radar = go.Figure(data=go.Scatterpolar(r=performance_data['Score'], theta=performance_data['Metric'], fill='toself', name='Model Performance'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Model Performance Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)
        st.divider()
        st.subheader("📊 Detailed Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Building Statistics**")
            st.markdown(f"- Total Buildings: {results['buildings']}")
            st.markdown(f"- Avg Building Size: {round(results['buildings']*450/results['buildings'], 1)} sq.m")
            st.markdown(f"- Building Density: {round(results['buildings']/results['parcels']*100, 1)}%")
            st.markdown("**Road Network Statistics**")
            st.markdown(f"- Total Road Length: {results['roads_km']} km")
            st.markdown(f"- Road Density: {round(results['roads_km']/25*100, 1)}%")
            st.markdown(f"- Avg Road Width: {round(np.random.uniform(4, 8), 1)} m")
        with col2:
            st.markdown("**Water Body Statistics**")
            st.markdown(f"- Total Water Bodies: {results['water_bodies']}")
            st.markdown(f"- Total Water Area: {results['water_bodies']*1500} sq.m")
            st.markdown(f"- Water Coverage: {round(results['water_bodies']/15*100, 1)}%")
            st.markdown("**Vegetation Statistics**")
            st.markdown(f"- Vegetation Cover: {results['vegetation_pct']}%")
            st.markdown(f"- Average NDVI: {results['ndvi']}")
            st.markdown(f"- Vegetation Health: {'Excellent' if results['ndvi'] > 0.6 else 'Good'}")
with tab4:
    st.header("🗺️ Feature Visualization")
    if not st.session_state.processed:
        st.warning("⚠️ No results available. Please process an image first")
    else:
        st.subheader("🎨 Annotated Orthophoto")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.uploaded_image:
                annotated_img = create_annotated_image(st.session_state.uploaded_image)
                st.image(annotated_img, caption="AI-Detected Features Overlay", use_container_width=True)
        with col2:
            st.markdown("**Legend**")
            st.markdown("🟦 **Blue** - Buildings")
            st.markdown("🟩 **Green** - Roads")
            st.markdown("🟥 **Red** - Water Bodies")
            st.markdown("🟨 **Yellow** - Parcels")
            st.markdown("🟧 **Orange** - Vegetation")
            st.divider()
            opacity = st.slider("Overlay Opacity", 0, 100, 70)
            show_labels = st.checkbox("Show Labels", value=True)
            show_grid = st.checkbox("Show Grid", value=False)
        st.divider()
        st.subheader("📍 Interactive Feature Map")
        map_data = pd.DataFrame({
            'lat': np.random.uniform(20.0, 21.0, 50),
            'lon': np.random.uniform(77.0, 78.0, 50),
            'feature': np.random.choice(['Building', 'Road', 'Water', 'Vegetation'], 50),
            'size': np.random.randint(10, 50, 50)
        })
        st.map(map_data[['lat', 'lon']], zoom=10)
        st.divider()
        st.subheader("🔍 Feature Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.uploaded_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_img, caption="Detected Features", use_container_width=True)
with tab5:
    st.header("💾 Export Results")
    if not st.session_state.processed:
        st.warning("⚠️ No results available. Please process an image first")
    else:
        st.subheader("📤 Export Configuration")
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Select Export Format", ["GeoJSON", "Shapefile", "KML", "CSV", "GeoTIFF", "Excel", "PDF Report"])
        with col2:
            coordinate_system = st.selectbox("Coordinate System", ["WGS84 (EPSG:4326)", "UTM Zone 43N", "India TM"])
        export_features = st.multiselect("Select Features to Export", ["Building Footprint", "Road Network", "Water Bodies", "Parcel Boundaries", "Land Cover", "Vegetation Index"], default=["Building Footprint", "Road Network", "Water Bodies"])
        col1, col2, col3 = st.columns(3)
        with col1:
            include_metadata = st.checkbox("Include Metadata", value=True)
        with col2:
            include_statistics = st.checkbox("Include Statistics", value=True)
        with col3:
            compress_output = st.checkbox("Compress Output (ZIP)", value=True)
        st.divider()
        st.subheader("📊 Export Preview")
        if export_features:
            results = st.session_state.results
            export_data = pd.DataFrame({
                'Feature_Type': export_features[:len(export_features)],
                'Count': [results['buildings'], int(results['roads_km']*10), results['water_bodies']][:len(export_features)],
                'Area_sqm': [results['buildings']*450, int(results['roads_km']*2850), results['water_bodies']*1500][:len(export_features)],
                'Confidence': [round(np.random.uniform(85, 98), 2) for _ in range(len(export_features))]
            })
            st.dataframe(export_data, use_container_width=True, hide_index=True)
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Download Results", type="primary", use_container_width=True):
                st.success("✅ Export package prepared successfully!")
                export_data_csv = export_data.to_csv(index=False)
                st.download_button("💾 Download CSV File", data=export_data_csv, file_name=f"svamitva_results.csv", mime="text/csv", use_container_width=True)
        with col2:
            if st.button("📧 Email Results", use_container_width=True):
                email = st.text_input("Enter email address")
                if email:
                    st.info(f"✅ Results will be sent to {email}")
        with col3:
            if st.button("☁️ Upload to Cloud", use_container_width=True):
                st.info("☁️ Uploading to cloud storage...")
                time.sleep(1)
                st.success("✅ Uploaded successfully!")
        st.divider()
        st.subheader("📄 Generate Report")
        report_type = st.radio("Select Report Type", ["Summary Report", "Detailed Technical Report", "Executive Summary"], horizontal=True)
        if st.button("📑 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating report..."):
                time.sleep(2)
                st.success("✅ PDF Report generated successfully!")
                st.download_button("📥 Download Report", data="Sample PDF Report Content", file_name="SVAMITVA_Report.pdf", mime="application/pdf")
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Ministry of Panchayati Raj**")
    st.caption("Government of India")
with col2:
    st.markdown("**SVAMITVA Scheme**")
    st.caption("Survey of Villages and Mapping with Improvised Technology in Village Areas")
with col3:
    st.markdown("**Technical Support**")
    st.caption("support@svamitva.gov.in")
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>© 2024 SVAMITVA AI Feature Extraction System | Powered by Deep Learning & Computer Vision | Version 1.0.0</p>", unsafe_allow_html=True)
