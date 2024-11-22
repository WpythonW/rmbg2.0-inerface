import streamlit as st
from PIL import Image

SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']

class ImageCache:
    def __init__(self):
        self.images = {}
        self.angles = {}
        
    def add_image(self, key, image):
        self.images[key] = image
        self.angles[key] = 0
        
    def rotate(self, key, direction):
        if key not in self.images:
            return None
        delta = 90 if direction == 'cw' else -90
        self.angles[key] = (self.angles[key] + delta) % 360
        return self.get_image(key)
    
    def get_image(self, key):
        if key not in self.images:
            return None
        if self.angles[key] == 0:
            return self.images[key]
        return self.images[key].rotate(self.angles[key], expand=True)
    
    def get_angle(self, key):
        return self.angles.get(key, 0)
    
    def clear(self):
        self.images.clear()
        self.angles.clear()

def init_session_state():
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 2
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.5
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'image_cache' not in st.session_state:
        st.session_state.image_cache = ImageCache()
    # Добавляем временные значения
    if 'temp_batch_size' not in st.session_state:
        st.session_state.temp_batch_size = st.session_state.batch_size
    if 'temp_threshold' not in st.session_state:
        st.session_state.temp_threshold = st.session_state.threshold

def show_settings():
    with st.sidebar:
        st.header("Settings")
        
        def update_values():
            st.session_state.batch_size = st.session_state.new_batch_size
            st.session_state.threshold = st.session_state.new_threshold
        
        with st.form(key='settings_form'):
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=4,
                value=st.session_state.batch_size,
                key="new_batch_size"
            )
            st.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.01,
                format="%.2f",
                key="new_threshold"
            )
            
            st.form_submit_button("Apply Settings", on_click=update_values)

def process_images(images, model, batch_size, threshold):
    all_processed = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        processed = model.process_images(
            [img for img in batch],  # Передаем сами изображения без углов
            [0] * len(batch),        # Углы уже учтены в изображениях
            threshold
        )
        all_processed.extend(processed)
    return all_processed

def show_image_controls(uploaded_files):
    st.subheader("Preview and Rotation Controls")
    cache = st.session_state.image_cache
    
    for idx, file in enumerate(uploaded_files):
        if file.name not in cache.images:
            cache.add_image(file.name, Image.open(file))
        
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col1:
            if st.button("⟳", key=f"ccw_{idx}", use_container_width=True):
                cache.rotate(file.name, 'ccw')
                st.rerun()
        with col2:
            img = cache.get_image(file.name)
            st.image(img, use_container_width=True)
        with col3:
            if st.button("⟲", key=f"cw_{idx}", use_container_width=True):
                cache.rotate(file.name, 'cw')
                st.rerun()
        st.markdown("---")

def show_results(processed_results):
    st.subheader("Results")
    for idx, (original, processed) in enumerate(processed_results):
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, use_container_width=True)
        with col2:
            st.image(processed, use_container_width=True)

def show_batch_processor_tab():
    uploaded_files = st.file_uploader(
        "Choose images", 
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="file_uploader_batch"
    )
    
    if not uploaded_files:
        return
        
    curr_files = {f.name for f in uploaded_files}
    if 'prev_files' not in st.session_state or curr_files != st.session_state.prev_files:
        st.session_state.image_cache.clear()
        st.session_state.prev_files = curr_files
        st.session_state.processed_results = None
    
    show_image_controls(uploaded_files)
    
    if not st.button("Process Images", type="primary"):
        return
        
    with st.spinner("Processing..."):
        cache = st.session_state.image_cache
        # Получаем уже повернутые изображения
        images = [cache.get_image(f.name) for f in uploaded_files]
        
        processed = process_images(
            images,
            st.session_state.model,
            st.session_state.batch_size,
            st.session_state.threshold
        )
        
        st.session_state.processed_results = list(zip(images, processed))
        
    if st.session_state.processed_results:
        show_results(st.session_state.processed_results)

def show_comparison_tab():
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=SUPPORTED_FORMATS,
        key="comparison_uploader"
    )
    
    if not uploaded_file:
        return
        
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("RMBG 1.4")
        if 'model_v1' not in st.session_state:
            from model_classes import BackgroundRemoverV1
            st.session_state.model_v1 = BackgroundRemoverV1()
        result_v1 = st.session_state.model_v1.process_images([image], [0], st.session_state.threshold)[0]
        st.image(result_v1, use_container_width=True)
        
    with col3:
        st.subheader("RMBG 2.0")
        result_v2 = st.session_state.model.process_images([image], [0], st.session_state.threshold)[0]
        st.image(result_v2, use_container_width=True)