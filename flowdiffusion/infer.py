import torch
from PIL import Image
from torchvision import transforms
import imageio
import numpy as np
from flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from flowdiffusion.unet import UnetBridge as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from ema_pytorch import EMA
import os

CHECKPOINT_PATH = "/home/cobot/testing/avdc/results/bridge/model-188000.pt"

class InferAVDC:
    def __init__(self, checkpoint_path, sample_steps=100, target_size=(48, 64)):
        self.checkpoint_path = checkpoint_path
        self.sample_steps = sample_steps
        self.target_size = target_size
        
        # Init Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Initialize the model and tokenizer
        self.unet = Unet().to(self.device)
        pretrained_model = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Initialize the diffusion model
        self.diffusion = GoalGaussianDiffusion(
            channels=3*(7-1),  # Assuming sample_per_seq=7
            model=self.unet,
            image_size=self.target_size,
            timesteps=100,
            sampling_timesteps=self.sample_steps,
            loss_type='l2',
            objective='pred_v',
            beta_schedule='cosine',
            min_snr_loss_weight=True,
        ).to(self.device)
        if self.accelerator.is_main_process:
            self.ema = EMA(self.diffusion, beta = 0.999, update_every = 10)
            self.ema.to(self.device)

        # Prepare the model with the accelerator
        self.diffusion, self.text_encoder = self.accelerator.prepare(self.diffusion, self.text_encoder)

        # Load the checkpoint
        self.load_checkpoint()

    def load_checkpoint(self):
        class_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(self.checkpoint_path):
            checkpoint_path = self.checkpoint_path
        else:
            checkpoint_path = os.path.join(class_dir, self.checkpoint_path )
        
        if os.path.exists(checkpoint_path):
            data = torch.load(checkpoint_path, map_location=self.device)
            model = self.accelerator.unwrap_model(self.diffusion)
            model.load_state_dict(data['model'])
            self.ema.load_state_dict(data["ema"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def preprocess_image(self, image):
        image = image.convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
        image = transform(image).to(self.device)
        return image.unsqueeze(0)

    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    def generate(self, image, text, output_gif_path):
        image = self.preprocess_image(image)
        batch_size = 1
        device = self.device
        with torch.inference_mode():
            task_embeds = self.encode_batch_text(text)
            output = self.ema.ema_model.sample(image.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=0).cpu()
        output = output[0].reshape(-1, 3, *self.target_size)
        output = torch.cat([image.cpu(), output], dim=0)
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')

        imageio.mimsave(output_gif_path, output, duration=200, loop=1000)
        print(f'Generated {output_gif_path}')
        return output_gif_path

# Usage
if __name__ == "__main__":
    inference_model = InferAVDC(checkpoint_path=CHECKPOINT_PATH)  # Example checkpoint number
    image = Image.open("sun.jpg")
    inference_model.generate(
        image=image,  # Example image path
        text="Place the object in the bin",  # Example task description
        output_gif_path="../datasets/kitting_dataset/sample_output.gif"  # Output GIF path
    )
