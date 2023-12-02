# SDXL Turbo Optimization Experiment

ComfyUI and other UI powered full backend systems were super slow and I wanted to optimize it for better efficiency and real time performane for a local running LLM based ppt generator application I was building .

I wrote a custom script to benchmark different stages of the image generation while trying out different configurations and this is just me writing down my finding for future reference primarily

My Specs and Configuration :
Ryzen 7 5800X | 32 GB RAM | RTX 3060 Ti
Windows 11 WSL Ubuntu , python 3.10

Comfy UI SDXL Turbo Generation Speed averages at 2.5 second per image (without prompt caching)

My Test Criterias distinguishes Total Time , Load Time for any modules , Init Gen Time , Avg Gen Time. For Avg Time calculation I test with 4 prompts

## Basic Diffuser Performance :

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 19.3036789894104        | 17.443750381469727      | 17.60261082649231       |
| Load Time     | 6.236544132232666       | 4.532968759536743       | 4.691980600357056       |
| Init Gen Time | 2.950467348098755       | 2.9440150260925293      | 3.0313751697540283      |
| Avg Gen Time  | 2.529166877269745       | 2.4916916489601135      | 2.4698137640953064      |

The default example script from sdxl turbo repo was used here

## Batch Prompt Processing :

## VAE 16fp Optimization :

