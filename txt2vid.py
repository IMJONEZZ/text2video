import diffusers
import transformers
print(diffusers.__version__, transformers.__version__)
# try:
#     import xformers
#     print(xformers.__version__)
# except:
#     print("xformers not installed correctly")
    
import torch
print("torch imported")
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video
print("diffusers imported")
from IPython.display import HTML
from base64 import b64encode
import cv2
from PIL import Image
import numpy as np
print("cv2, PIL, and numpy imported")

import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
print("all imported")

pipe = DiffusionPipeline.from_pretrained("./diffusion_pipeline/", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
print("pipeline created")

controlpipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./controlnet_pipeline", torch_dtype=torch.float16
)
controlpipe.scheduler = UniPCMultistepScheduler.from_config(controlpipe.scheduler.config)
controlpipe.enable_model_cpu_offload()
#controlpipe.enable_xformers_memory_efficient_attention()
print("controlnet pipeline created")

prompt = 'Spiderman chatting with a llama, best quality, extremely detailed'
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
video_duration_seconds = 3
num_frames = video_duration_seconds * 10
video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_frames=num_frames).frames
video_path = export_to_video(video_frames)
print("video created")

def display_video(video):
    fig = plt.figure(figsize=(4.2,4.2))  #Display size specification
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=100, repeat_delay=1000)

    plt.close()
    return anime
video = imageio.mimread(video_path)  #Loading video
print("video loaded")
print(video)

for i in range(len(video_frames)):
    image = np.array(video_frames[i])

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    #canny_image.show()
    
    output = controlpipe(
        prompt,
        canny_image,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device='cpu'),
        num_inference_steps=20,
    )
    video_frames[i] = output
print("controlnet video created")
    
video_path = export_to_video(video_frames)
video = imageio.mimread(video_path)  #Loading video
