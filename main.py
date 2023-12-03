import streamlit as st
from diffusers import AutoPipelineForText2Image,AutoencoderTiny
import torch
import time

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" , torch_dtype=torch.float16, variant="fp16")
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
    pipe.to("cuda")

    return pipe

pipe = load_model()

st.title("Image Generator")
    

with st.form("my_form"):
    text_input = st.text_input("What do you want to see")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    start = time.time()
    image = pipe(prompt=text_input, num_inference_steps=1, guidance_scale=0.0).images[0]
    stop = time.time()
    st.write(f"processing time : {stop-start}")
    st.image(image)