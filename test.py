from diffusers import AutoPipelineForText2Image,AutoencoderTiny
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" , torch_dtype=torch.float16, variant="fp16")
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe.to("cuda")

prompts = [ "A cinematic shot of a baby racoon wearing an intricate italian priest robe." ,
            "A person celebrating with a finished puzzle" ,
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A majestic lion jumping from a big stone at night" , 
            "style of Shawn Coss, iridescent, by Takato Yamamoto, (pointed, Harlem Renaissance Art but extremely beautiful), (intricate details, masterpiece, best quality) , in the style of nicola samori, High Fashion, dynamic, dramatic, haute couture, elegant, ornate clothing, High Fashion, looking at viewer",
            "cinematic photo of a jucy hamburger in a plate, 1970's kitchen, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            "ethical dilemma",
            "two friend playing in the snow, sunset, dusk, wide shot,highly detailed, best quality"]

for i in range(len(prompts)):
    image = pipe(prompt=prompts[i], num_inference_steps=2, guidance_scale=0.0).images[0]
    image.save(f"test_{i}.jpg")