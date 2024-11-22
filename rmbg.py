import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import io
from collections import defaultdict

class BackgroundRemover:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Using device: {self.device}")
        self.model = self._load_model()
        self.transform = self._setup_transform()
        
    def _load_model(self):
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                'briaai/RMBG-2.0', 
                trust_remote_code=True
            )
            if self.device == 'cuda':
                torch.set_float32_matmul_precision('high')
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise
        
    def _setup_transform(self):
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def process_images(self, images, rotations):
        try:
            processed_images = []
            for img, angle in zip(images, rotations):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if angle != 0:
                    img = img.rotate(angle, expand=True)
                processed_images.append(img)
            
            input_tensors = []
            for img in processed_images:
                input_tensors.append(self.transform(img))
            
            input_tensors = torch.stack(input_tensors).to(self.device)
            
            with torch.no_grad():
                preds = self.model(input_tensors)[-1].sigmoid().cpu()
                
            results = []
            for img, pred in zip(processed_images, preds):
                pred_pil = transforms.ToPILImage()(pred.squeeze())
                mask = pred_pil.resize(img.size)
                img_rgba = img.convert('RGBA')
                img_rgba.putalpha(mask)
                results.append(img_rgba)
                
            return results
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            raise

class ImageCache:
    def __init__(self):
        self.images = {}
        self.angles = defaultdict(int)
    
    def add_image(self, key, image):
        if key not in self.images:
            self.images[key] = {0: image}
            self.angles[key] = 0
    
    def rotate(self, key, direction):
        if key not in self.images:
            return
        
        delta = 90 if direction == 'cw' else -90
        self.angles[key] = (self.angles[key] + delta) % 360
        
        angle = self.angles[key]
        if angle not in self.images[key]:
            orig = self.images[key][0]
            self.images[key][angle] = orig.rotate(angle, expand=True)
        
        return self.get_image(key)
    
    def get_image(self, key):
        if key not in self.images:
            return None
        return self.images[key][self.angles[key]]
    
    def get_angle(self, key):
        return self.angles[key]
    
    def clear(self):
        self.images.clear()
        self.angles.clear()

def update_batch_size():
    st.session_state.batch_size = st.session_state.new_batch_size

def main():
    st.title("Batch Background Remover")
    
    # Инициализация session_state
    if 'model' not in st.session_state:
        st.session_state.model = BackgroundRemover()
    if 'image_cache' not in st.session_state:
        st.session_state.image_cache = ImageCache()
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 2
    if 'new_batch_size' not in st.session_state:
        st.session_state.new_batch_size = st.session_state.batch_size
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'download_clicks' not in st.session_state:
        st.session_state.download_clicks = set()
        
    with st.sidebar:
        st.header("Settings")
        st.number_input(
            "Batch size",
            min_value=1,
            max_value=4,
            value=st.session_state.batch_size,
            key='new_batch_size',
            on_change=update_batch_size
        )
    
    uploaded_files = st.file_uploader(
        "Choose images", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        curr_files = {f.name for f in uploaded_files}
        if 'prev_files' not in st.session_state or curr_files != st.session_state.prev_files:
            st.session_state.image_cache.clear()
            st.session_state.prev_files = curr_files
            st.session_state.processed_results = None
            st.session_state.download_clicks.clear()
        
        st.subheader("Preview and Rotation Controls")
        
        for idx, file in enumerate(uploaded_files):
            if file.name not in st.session_state.image_cache.images:
                st.session_state.image_cache.add_image(file.name, Image.open(file))
            
            col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
            
            with col1:
                if st.button("⟳ Поворот\nвправо", key=f"ccw_{idx}", use_container_width=True):
                    st.session_state.image_cache.rotate(file.name, 'ccw')
                    st.rerun()
            
            with col2:
                img = st.session_state.image_cache.get_image(file.name)
                angle = st.session_state.image_cache.get_angle(file.name)
                st.image(img, caption=f"Image {idx+1} (Rotated {angle}°)", use_container_width=True)
            
            with col3:
                if st.button("⟲ Поворот\nвлево", key=f"cw_{idx}", use_container_width=True):
                    st.session_state.image_cache.rotate(file.name, 'cw')
                    st.rerun()
            
            st.markdown("---")
        
        if st.button("Process Images", type="primary"):
            with st.spinner("Processing..."):
                images = []
                rotations = []
                for file in uploaded_files:
                    images.append(st.session_state.image_cache.images[file.name][0])
                    rotations.append(st.session_state.image_cache.get_angle(file.name))
                
                all_processed = []
                for i in range(0, len(images), st.session_state.batch_size):
                    batch_images = images[i:i + st.session_state.batch_size]
                    batch_rotations = rotations[i:i + st.session_state.batch_size]
                    processed_batch = st.session_state.model.process_images(
                        batch_images,
                        batch_rotations
                    )
                    all_processed.extend(processed_batch)
                
                st.session_state.processed_results = list(zip(uploaded_files, all_processed, rotations))
                st.session_state.download_clicks.clear()
                st.rerun()
        
        if st.session_state.processed_results:
            st.subheader("Results")
            for idx, (file, proc, rotation) in enumerate(st.session_state.processed_results):
                result_key = f"{file.name}_{rotation}"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    orig_rotated = st.session_state.image_cache.get_image(file.name)
                    st.image(
                        orig_rotated,
                        caption=f"Original {idx+1} (Rotated {rotation}°)",
                        use_container_width=True
                    )
                    
                with col2:
                    st.image(
                        proc, 
                        caption=f"Processed {idx+1}",
                        use_container_width=True
                    )
                    
                    if result_key not in st.session_state.download_clicks:
                        buf = io.BytesIO()
                        proc.save(buf, format='PNG')
                        st.session_state.download_clicks.add(result_key)
                        
                        st.download_button(
                            label=f"Download image {idx+1}",
                            data=buf.getvalue(),
                            file_name=f"processed_{idx+1}.png",
                            mime="image/png",
                            type="primary",
                            key=f"download_{result_key}"
                        )

if __name__ == "__main__":
    main()