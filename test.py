import time

import tomesd
start = time.time()

# MODEL LOADING

from diffusers import AutoPipelineForText2Image,AutoencoderKL,AutoencoderTiny
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" , torch_dtype=torch.float16, variant="fp16")

# pipe.upcast_vae()
# pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.unet.to(memory_format=torch.channels_last)

# pipe.enable_xformers_memory_efficient_attention()


pipe.to("cuda")
# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

# tomesd.apply_patch(pipe, ratio=0.5)

load = time.time()

# INIT GENERATION

image = pipe(prompt="comic style car chase", num_inference_steps=1, guidance_scale=0.0).images[0]
image.save(f"test.jpg")

initial_generation = time.time()


# BATCH GENERATION

prompts = [ "A cinematic shot of a baby racoon wearing an intricate italian priest robe." ,
            "A person celebrating with a finished puzzle" ,
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A majestic lion jumping from a big stone at night" ]

'''
# Batch Inference via pipeline
image_arr = pipe(prompt=prompts, num_inference_steps=1, guidance_scale=0.0).images

for i in range(len(image_arr)):
    image_arr[i].save(f"test_{i}.jpg")
'''

for i in range(len(prompts)):
    image = pipe(prompt=prompts[i], num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save(f"test_{i}.jpg")

final_generation = time.time()

total_time = final_generation - start
calculated_load_time = load - start
calculated_init_gen_time = initial_generation - load
calculated_generation_per_img = (final_generation - initial_generation) / len(prompts)


print(f"Total Time : {total_time}")
print(f"Load Time : {calculated_load_time}")
print(f"Init Gen Time : {calculated_init_gen_time}")
print(f"Avg Gen Time : {calculated_generation_per_img}")