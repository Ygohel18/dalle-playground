from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

class StableDiffusionWrapper:
    def __init__(self, model = "stabilityai/stable-diffusion-2") -> None:
        repo_id = model
        # repo_id = "stabilityai/stable-diffusion-2"
        # repo_id = "stabilityai/stable-diffusion-2-depth"
        # pipe = DiffusionPipeline.from_pretrained(
        #     repo_id, revision="fp16",
        #     torch_dtype=torch.float16
        # )

        pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

            
    def generate_images(self, text_prompt: str, num_images: int, steps: int = 20):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=steps).images
        return images
