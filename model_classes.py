import torch, io
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, pipeline
import streamlit as st
import numpy as np
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
        
    def process_images(self, images, rotations, threshold=0.5):
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
                # Применяем пороговое значение
                preds = (preds > threshold).float()
                
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

class BackgroundRemoverV1:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = pipeline("image-segmentation", 
                            model="briaai/RMBG-1.4", 
                            trust_remote_code=True,
                            device=0 if self.device == 'cuda' else -1)
        
    def process_images(self, images, rotations, threshold=0.5):
        try:
            results = []
            for img, angle in zip(images, rotations):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if angle != 0:
                    img = img.rotate(angle, expand=True)
                
                # Обрабатываем напрямую PIL Image
                mask = self.model(img, return_mask=True)
                # Применяем пороговое значение к маске
                mask = Image.fromarray((np.array(mask) > threshold * 255).astype(np.uint8) * 255)
                
                # Создаем RGBA изображение
                img_rgba = img.convert('RGBA')
                img_rgba.putalpha(mask)
                results.append(img_rgba)
                
            return results
        except Exception as e:
            st.error(f"Error during processing with rembg v1: {str(e)}")
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