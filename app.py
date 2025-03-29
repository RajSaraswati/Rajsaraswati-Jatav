# Copyright 2025 Raj Saraswati
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = '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'  # Replace with your token

import torch
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

# Memory Management (साधारण वर्जन)
class MemoryManager:
    def load_models_to_gpu(self, models):
        if not isinstance(models, list):
            models = [models]
        for m in models:
            m.to('cuda')
            print(f"Loaded to GPU: {m.__class__.__name__}")

    def unload_all_models(self, models):
        if not isinstance(models, list):
            models = [models]
        for m in models:
            m.to('cpu')
            print(f"Unloaded to CPU: {m.__class__.__name__}")
        torch.cuda.empty_cache()

memory_management = MemoryManager()

# SDXL मॉडल
sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16)
unet = UNet2DConditionModel.from_pretrained(sdxl_name, subfolder="unet", torch_dtype=torch.float16)
memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# LLM मॉडल
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, token=HF_TOKEN)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, token=HF_TOKEN)
memory_management.unload_all_models(llm_model)

# Canvas सिमुलेशन (साधारण)
class Canvas:
    def process(self):
        return {"initial_latent": np.zeros((90, 90, 3), dtype=np.float32)}

omost_canvas = Canvas()

# चैट और इमेज जेनरेशन फंक्शन
def chat_fn(message: str, history: list, temperature: float, top_p: float, max_new_tokens: int):
    conversation = [{"role": "system", "content": "You are an AI to generate images."}]
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)
    input_ids = llm_tokenizer.apply_chat_template(conversation, return_tensors="pt").to('cuda')
    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
    if temperature == 0:
        generate_kwargs['do_sample'] = False
    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

def generate_image(message: str):
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2, vae, unet])
    pipe = StableDiffusionXLImg2ImgPipeline(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, unet=unet, tokenizer=tokenizer, tokenizer_2=tokenizer_2)
    initial_latent = torch.from_numpy(omost_canvas.process()["initial_latent"]).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255.0
    output = pipe(prompt=message, image=initial_latent, strength=0.7, num_inference_steps=25, guidance_scale=5.0)
    output.images[0].save("blog_visual.png")
    return "blog_visual.png"

# Gradio इंटरफेस
with gr.Blocks() as demo:
    gr.Markdown("### मेरा ब्लॉग - इमेज जेनरेटर")
    chatbot = gr.Chatbot(label="चैटबॉट")
    with gr.Row():
        msg = gr.Textbox(label="अपना मैसेज डालें")
        submit = gr.Button("भेजें")
    img_output = gr.Image(label="जेनरेटेड इमेज")
    submit.click(chat_fn, [msg, chatbot], chatbot).then(generate_image, msg, img_output)

if __name__ == "__main__":
    demo.launch()