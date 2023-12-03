# SDXL Turbo Optimization Experiment

ComfyUI and other UI powered full backend systems were super slow and I wanted to optimize it for better efficiency and real time performane for a local running LLM based ppt generator application I was building .

I wrote a custom script to benchmark different stages of the image generation while trying out different configurations and this is just me writing down my finding for future reference primarily

My Specs and Configuration :
Ryzen 7 5800X | 32 GB RAM | RTX 3060 Ti
Windows 11 WSL Ubuntu , python 3.10

Comfy UI SDXL Turbo Generation Speed averages at 2.5 second per image (without prompt caching)

My Test Criterias distinguishes Total Time , Load Time for any modules , Init Gen Time , Avg Gen Time.
For Avg Time calculation I test with 1 + 4 prompts.
I am not optimizing for memory footprint , just performance.

## Basic Diffuser Performance :

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 19.3036789894104        | 17.443750381469727      | 17.60261082649231       |
| Load Time     | 6.236544132232666       | 4.532968759536743       | 4.691980600357056       |
| Init Gen Time | 2.950467348098755       | 2.9440150260925293      | 3.0313751697540283      |
| Avg Gen Time  | 2.529166877269745       | 2.4916916489601135      | 2.4698137640953064      |

The default example script from sdxl turbo repo was used here , with multiple prompts being iterated

## Batch Prompt Processing :

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 24.790883779525757      | 23.44549536705017       | 28.178327798843384      |
| Load Time     | 4.713327169418335       | 4.722451210021973       | 4.594852685928345       |
| Init Gen Time | 2.940718412399292       | 2.895763397216797       | 3.0071566104888916      |
| Avg Gen Time  | 4.2842095494270325      | 3.9568201899528503      | 5.144079625606537       |

Instead of iterating through each prompt , passed the entire list for batch processing to the pipeline
Its suprising to see a decline in performance while using the built in batch processing methond than iterating the pipe

## Upcase VAE Precision

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 17.625147342681885      | 23.426799774169922      | 17.705934762954712      |
| Load Time     | 5.011475086212158       | 4.612784147262573       | 4.5227577686309814      |
| Init Gen Time | 2.9415533542633057      | 4.21022629737854        | 3.078413963317871       |
| Avg Gen Time  | 2.4180297255516052      | 3.650947332382202       | 2.526190757751465       |

Decrease in performance compared to base diffuser performance
source : https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo#speed-up-sdxl-turbo-even-more

## VAE 16fp Optimization :

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 11.680602312088013      | 13.030585050582886      | 11.378620147705078      |
| Load Time     | 5.208057641983032       | 5.137163877487183       | 4.957686901092529       |
| Init Gen Time | 1.538787603378296       | 1.9612927436828613      | 1.539346694946289       |
| Avg Gen Time  | 1.2334392666816711      | 1.4830321073532104      | 1.220396637916565       |

Gives Promising Speed (2x) with no noticeable quality loss
source : https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo#speed-up-sdxl-turbo-even-more
```
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
```

## VAE : Tiny Auto Encoder

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 6.502416133880615       | 5.839867115020752       | 6.068450450897217       |
| Load Time     | 5.318273067474365       | 4.63173770904541        | 4.82595682144165        |
| Init Gen Time | 0.6518356800079346      | 0.664395809173584       | 0.704963207244873       |
| Avg Gen Time  | 0.13307684659957886     | 0.13593339920043945     | 0.13438260555267334     |

Incredible Speed , Its said to be quality reducing but I couldnt notice
source : https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md#tiny-autoencoder
```
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
```

> Doing all subsequent tests with Tiny Auto Encoder since the vae is pretty much optimized and we will not need to swap that out , for all following performance comparisons use the Tiny Auto Encoder benchmark as reference point now

## Compile UNet

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 75.41857695579529       | 65.76266884803772       | 64.25783467292786       |
| Load Time     | 5.129103422164917       | 7.217745304107666       | 5.239185810089111       |
| Init Gen Time | 68.40448021888733       | 57.36990666389465       | 57.89517068862915       |
| Avg Gen Time  | 0.4712483286857605      | 0.2937542200088501      | 0.2808695435523987      |

Massive Decrease in init generation and 2x slower in avg gen time
source : https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo#speed-up-sdxl-turbo-even-more
```
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

## CPU Offloading

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 17.635058403015137      | 16.957004070281982      | 16.976733684539795      |
| Load Time     | 7.488234281539917       | 9.042913675308228       | 7.669692039489746       |
| Init Gen Time | 2.28349232673645        | 2.0885674953460693      | 2.0610623359680176      |
| Avg Gen Time  | 1.9658329486846924      | 1.4563807249069214      | 1.8114948272705078      |

Decrease in performance
source : https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md#model-cpu-offloading

General rule of thumb that if you can fit the model in memory , offloading is only going to cost you performance

## VAE Slicing

| Metric        | Run 1                   | Run 2                   | Run 3                   |
|---------------|-------------------------|-------------------------|-------------------------|
| Total Time    | 7.269654989242554       | 6.002238035202026       | 6.339429616928101       |
| Load Time     | 5.831358909606934       | 4.8166186809539795      | 4.8016180992126465      |
| Init Gen Time | 0.6312739849090576      | 0.6541228294372559      | 0.6929543018341064      |
| Avg Gen Time  | 0.20175552368164062     | 0.13287413120269775     | 0.21121430397033691     |

No Impact or Slight Decrease in performance , hard to gauge.
source : https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md#slicing

## MORE TESTING CONCLUSIONS

I tried all the optimizations listed in https://huggingface.co/docs/diffusers/optimization/opt_overview
Including Xformers , Token Merging , Offloading . But none of them beats the base Tiny Encoder Optimization Benchmarks.
I think it indicates some bottleneck deep within python thats causing the performance issue now and I have hit a bottleneck in what I could optimize,if anyone else wants to try out the optimizations in a different language like Rust I think that would be the way forward

## Additional Notes :
- I couldnt try out https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md#caching-computations since i couldnt figure out how to get the tokenizers and encoders for the model
- Couldnt try out tracing UNet https://huggingface.co/docs/diffusers/optimization/memory#tracing