import torch
import torch.nn as nn


class OpenCLIPWrapper(nn.Module):
    """
    Wrapper to convert OpenCLIP model output to CLIP-style output
    
    OpenCLIP output: (image_features, text_features, logit_scale)
    CLIP output: (logits_per_image, logits_per_text)
    """
    
    def __init__(self, openclip_model):
        """
        Args:
            openclip_model: Model created with open_clip.create_model()
        """
        super().__init__()
        self.model = openclip_model
        
    def forward(self, images, text_tokens):
        """
        Args:
            images: Image tensor [batch_size, channels, height, width]
            text_tokens: Text tokens [batch_size, seq_len]
            
        Returns:
            logits_per_image: [image_batch, text_batch]
            logits_per_text: [text_batch, image_batch]
        """
        # Inference with OpenCLIP style
        output = self.model(images, text_tokens)
        if len(output) == 4:
            # Format: (image_features, text_features, logit_scale, logit_bias)
            image_features, text_features, logit_scale, logit_bias = output
        elif len(output) == 3:
            # Standard format: (image_features, text_features, logit_scale)
            image_features, text_features, logit_scale = output
            logit_bias = None
        
        # Convert to CLIP style: dot product of features scaled by logit_scale
        logits_per_image = logit_scale * image_features @ text_features.T

        # Add bias if present
        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias

        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def encode_image(self, images):
        """Encode images only"""
        return self.model.encode_image(images)
    
    def encode_text(self, text_tokens):
        """Encode text only"""
        return self.model.encode_text(text_tokens)
    
    def get_logit_scale(self):
        """Get logit scale value"""
        return self.model.logit_scale.exp()
    
    @property
    def logit_scale(self):
        """Access to raw logit scale parameter (before exp)"""
        return self.model.logit_scale