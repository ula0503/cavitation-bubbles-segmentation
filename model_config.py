"""Model configuration module for bubble segmentation."""

import os
from typing import Optional


class ModelConfig:
    """Configuration manager for segmentation models."""

    def __init__(self, model_path: str) -> None:
        """Initialize model configuration with specific model path.

        Args:
            model_path: Path to model weights file.
        """
        self.segmentation_model = model_path

    def check_models(self) -> bool:
        """Check if model files exist.

        Returns:
            True if model exists, False otherwise.
        """
        if os.path.exists(self.segmentation_model):
            print(f"Model found: {self.segmentation_model}")
            return True
        else:
            print(f"Model not found: {self.segmentation_model}")
            return False


# Create the default configuration instance
model_config = ModelConfig(
    r"C:\Users\Admin\Desktop\cavitation_bubbles_segmentation\models\best2.pt"
)
