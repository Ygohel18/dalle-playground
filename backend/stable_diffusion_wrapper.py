from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import requests
import torch
from PIL import Image
from io import BytesIO

class StableDiffusionWrapper:
    def __init__(self, model = "stabilityai/stable-diffusion-2-1-base") -> None:
        self.repo_id = model

        # repo_id = "stabilityai/stable-diffusion-2"
        # repo_id = "stabilityai/stable-diffusion-2-depth"
        # pipe = DiffusionPipeline.from_pretrained(
        #     repo_id, revision="fp16",
        #     torch_dtype=torch.float16
        # )

        if (self.model_id != "stabilityai/stable-diffusion-x4-upscaler"):
            pipe = DiffusionPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config)
            self.pipe = pipe.to("cuda")

    
    def enhance_image(self, text_prompt: str, img: str, steps: int = 5):
        self.repo_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = DiffusionPipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

        response = requests.get(img)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))

        upscaled_image = self.pipe(prompt=text_prompt, image=low_res_img, num_inference_steps=steps).images[0]
        return upscaled_image


    def generate_images(self, text_prompt: str, num_images: int, steps: int = 5):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=steps).images
        return images
