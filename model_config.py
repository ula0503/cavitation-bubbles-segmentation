# model_config.py
import os


class ModelConfig:
    def __init__(self):
        self.segmentation_model = r"C:\Users\Admin\Desktop\cavitation_bubbles_segmentation\models\segmentation_model.pt"

    def check_models(self):
        if os.path.exists(self.segmentation_model):
            print(f"Модель найдена: {self.segmentation_model}")
            return True
        else:
            print(f"Модель не найдена: {self.segmentation_model}")
            return False


model_config = ModelConfig()
