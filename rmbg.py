import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import gradio as gr

class BackgroundRemover:
    def __init__(self, model_name='briaai/RMBG-2.0', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        
        self.image_size = (1024, 1024)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def remove_background(self, image_path, preview=False):
        # Загрузка и подготовка изображения
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(image_path).convert('RGB')
            
        # Преобразование изображения
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Получение маски
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred)
        mask = mask.resize(image.size)
        
        # Применение маски
        image.putalpha(mask)
        
        if preview:
            # Создаем превью с шахматным фоном для прозрачности
            preview_img = Image.new('RGBA', image.size, (255, 255, 255, 0))
            preview_img.paste(image, (0, 0), image)
            return preview_img
        
        return image

def create_gradio_app():
    bg_remover = BackgroundRemover()
    
    def process_image(input_image, preview_mode):
        try:
            result = bg_remover.remove_background(input_image, preview=preview_mode)
            return result, None
        except Exception as e:
            return None, str(e)
    
    # Создание улучшенного интерфейса
    with gr.Blocks(title="AI Background Remover") as iface:
        gr.Markdown("## 🎨 AI Background Remover")
        gr.Markdown("Загрузите изображение, чтобы удалить фон")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Исходное изображение")
                preview_checkbox = gr.Checkbox(label="Показать превью с прозрачностью", value=True)
                
            with gr.Column():
                output_image = gr.Image(type="pil", label="Результат")
                error_output = gr.Textbox(label="Ошибки", visible=False)
        
        with gr.Row():
            clear_btn = gr.Button("Очистить")
            submit_btn = gr.Button("Обработать", variant="primary")
            
        # Обработка событий
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, preview_checkbox],
            outputs=[output_image, error_output]
        )
        
        clear_btn.click(
            lambda: [None, None, None],
            outputs=[input_image, output_image, error_output]
        )
        
    return iface

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()