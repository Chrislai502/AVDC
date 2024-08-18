import torch
from PIL import Image
from torchvision import transforms
import imageio
import numpy as np
from goal_diffusion import GoalGaussianDiffusion
from unet import UnetBridge as Unet
from transformers import CLIPTextModel, CLIPTokenizer
import os

class InferAVDC:
    def __init__(self, checkpoint_num, sample_steps=100, target_size=(48, 64)):
        self.checkpoint_path = checkpoint_num
        self.sample_steps = sample_steps
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model and tokenizer
        self.unet = Unet().to(self.device)
        pretrained_model = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Initialize the diffusion model
        self.diffusion = GoalGaussianDiffusion(
            channels=3*(7-1),  # Assuming sample_per_seq=7 based on your code
            model=self.unet,
            image_size=self.target_size,
            timesteps=100,
            sampling_timesteps=self.sample_steps,
            loss_type='l2',
            objective='pred_v',
            beta_schedule='cosine',
            min_snr_loss_weight=True,
        ).to(self.device)

        # Load the checkpoint
        self.load_checkpoint()

    def load_checkpoint(self):
        class_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(self.checkpoint_path):
            checkpoint_path = self.checkpoint_path
        else:
            checkpoint_path = os.path.join(class_dir, self.checkpoint_path )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def preprocess_image(self, image):
        # image = Image.open(image_path).convert("RGB")
        # Turn image into PIL Image
        
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
        image = transform(image).to(self.device)
        return image.unsqueeze(0)

    def generate(self, image, text, output_gif_path):
        image = self.preprocess_image(image)
        batch_size = 1
        output = self.diffusion.sample(image, [text], batch_size).cpu()
        output = output[0].reshape(-1, 3, *self.target_size)
        output = torch.cat([image.cpu(), output], dim=0)
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')

        # Resize for better visualization (320x240)
        # output = [np.array(Image.fromarray(frame).resize((320, 240))) for frame in output]
        imageio.mimsave(output_gif_path, output, duration=200, loop=1000)
        print(f'Generated {output_gif_path}')

# Usage
if __name__ == "__main__":
    inference_model = InferAVDC(checkpoint_path="../results/bridge/kitting/model.pth")  # Example checkpoint number
    inference_model.generate(
        image_path="../datasets/kitting_dataset/sample_image.png",  # Example image path
        text="Place the object in the bin",  # Example task description
        output_gif_path="../datasets/kitting_dataset/sample_output.gif"  # Output GIF path
    )
