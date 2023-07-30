from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token_path = Path("token.txt")
token = token_path.read_text().strip()

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    #revision="fp16",
    ##
    torch_dtype=torch.float32,
    use_auth_token=token,
)

##
pipe.to("cpu")

prompt = "smoking tiger in ancient times"


#image = pipe(prompt)["sample"][0]


def obtain_image(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    ##
    generator = None if seed is None else torch.Generator("cpu").manual_seed(seed)
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image


image = obtain_image(prompt, num_inference_steps=10, seed=1024)
image.save( "smokingtiger.png")
Image.show(image)
