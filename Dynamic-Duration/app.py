import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import random
import uuid
from typing import Union, List, Optional
import numpy as np
import time
import zipfile

# Description for the app
DESCRIPTION = """## Qwen Image Hpc/."""

# Helper functions
def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# Load Qwen/Qwen-Image pipeline
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_qwen = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=dtype).to(device)

# Aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

def get_duration(
    prompt: str,
    negative_prompt: str,
    use_negative_prompt: bool,
    seed: int,
    width: int,
    height: int,
    guidance_scale: float,
    randomize_seed: bool,
    num_inference_steps: int,
    num_images: int,
    zip_images: bool,
    progress=gr.Progress(track_tqdm=True),
):
    step_duration = 0.6
    step_duration *= (width * height) / (1024 * 1024)
    per_image_duration = 5 + num_inference_steps * step_duration
    return 10 + num_images * per_image_duration

# Generation function for Qwen/Qwen-Image
def generate_qwen(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.0,
    randomize_seed: bool = False,
    num_inference_steps: int = 50,
    num_images: int = 1,
    zip_images: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device).manual_seed(seed)
    
    start_time = time.time()
    
    images = pipe_qwen(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
    ).images
    
    end_time = time.time()
    duration = end_time - start_time
    
    image_paths = [save_image(img) for img in images]
    zip_path = None
    if zip_images:
        zip_name = str(uuid.uuid4()) + ".zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for i, img_path in enumerate(image_paths):
                zipf.write(img_path, arcname=f"Img_{i}.png")
        zip_path = zip_name
    
    return image_paths, seed, f"{duration:.2f}", zip_path

# Wrapper function to handle UI logic
@spaces.GPU(duration=get_duration)
def generate(
    prompt: str,
    negative_prompt: str,
    use_negative_prompt: bool,
    seed: int,
    width: int,
    height: int,
    guidance_scale: float,
    randomize_seed: bool,
    num_inference_steps: int,
    num_images: int,
    zip_images: bool,
    progress=gr.Progress(track_tqdm=True),
):
    final_negative_prompt = negative_prompt if use_negative_prompt else ""
    return generate_qwen(
        prompt=prompt,
        negative_prompt=final_negative_prompt,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        randomize_seed=randomize_seed,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        zip_images=zip_images,
        progress=progress,
    )

# Examples
examples = [
    "A decadent slice of layered chocolate cake on a ceramic plate with a drizzle of chocolate syrup and powdered sugar dusted on top. photographed from a slightly low angle with high resolution, natural soft lighting, rich contrast, shallow depth of field, and professional color grading to highlight the dessert’s textures --ar 85:128 --v 6.0 --style raw",
    "Realistic still life photography style: A single, fresh apple, resting on a clean, soft-textured surface. The apple is slightly off-center, softly backlit to highlight its natural gloss and subtle color gradients—deep crimson red blending into light golden hues. Fine details such as small blemishes, dew drops, and a few light highlights enhance its lifelike appearance. A shallow depth of field gently blurs the neutral background, drawing full attention to the apple. Hyper-detailed 8K resolution, studio lighting, photorealistic render, emphasizing texture and form.",
    "一幅精致细腻的工笔画，画面中心是一株蓬勃生长的红色牡丹，花朵繁茂，既有盛开的硕大花瓣，也有含苞待放的花蕾，层次丰富，色彩艳丽而不失典雅。牡丹枝叶舒展，叶片浓绿饱满，脉络清晰可见，与红花相映成趣。一只蓝紫色蝴蝶仿佛被画中花朵吸引，停驻在画面中央的一朵盛开牡丹上，流连忘返，蝶翼轻展，细节逼真，仿佛随时会随风飞舞。整幅画作笔触工整严谨，色彩浓郁鲜明，展现出中国传统工笔画的精妙与神韵，画面充满生机与灵动之感。",
    "A young girl wearing school uniform stands in a classroom, writing on a chalkboard. The text Introducing Qwen-Image, a foundational image generation model that excels in complex text rendering and precise image editing appears in neat white chalk at the center of the blackboard. Soft natural light filters through windows, casting gentle shadows. The scene is rendered in a realistic photography style with fine details, shallow depth of field, and warm tones. The girl's focused expression and chalk dust in the air add dynamism. Background elements include desks and educational posters, subtly blurred to emphasize the central action. Ultra-detailed 32K resolution, DSLR-quality, soft bokeh effect, documentary-style composition",
    "手绘风格的水循环示意图，整体画面呈现出一幅生动形象的水循环过程图解。画面中央是一片起伏的山脉和山谷，山谷中流淌着一条清澈的河流，河流最终汇入一片广阔的海洋。山体和陆地上绘制有绿色植被。画面下方为地下水层，用蓝色渐变色块表现，与地表水形成层次分明的空间关系。太阳位于画面右上角，促使地表水蒸发，用上升的曲线箭头表示蒸发过程。云朵漂浮在空中，由白色棉絮状绘制而成，部分云层厚重，表示水汽凝结成雨，用向下箭头连接表示降雨过程。雨水以蓝色线条和点状符号表示，从云中落下，补充河流与地下水。整幅图以卡通手绘风格呈现，线条柔和，色彩明亮，标注清晰。背景为浅黄色纸张质感，带有轻微的手绘纹理。",
    "A capybara wearing a suit, holding a sign, that reads Hello World"
]

css = '''
.gradio-container {
    max-width: 590px !important;
    margin: 0 auto !important;
}
h1 {
    text-align: center;
}
footer {
    visibility: hidden;
}
'''

# Gradio interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            max_lines=1,
            placeholder="Enter your prompt",
            container=False,
        )
        run_button = gr.Button("Run", scale=0, variant="primary")
    result = gr.Gallery(label="Result", columns=1, show_label=False, preview=True)
    
    with gr.Row():
        aspect_ratio = gr.Dropdown(
            label="Aspect Ratio",
            choices=list(aspect_ratios.keys()),
            value="1:1",
        )
    with gr.Accordion("Additional Options", open=False):
        use_negative_prompt = gr.Checkbox(
            label="Use negative prompt",
            value=True,
            visible=True
        )
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            value="text, watermark, copyright, blurry, low resolution",
            visible=True,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=64,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=64,
                value=1024,
            )
        guidance_scale = gr.Slider(
            label="Guidance Scale",
            minimum=0.0,
            maximum=20.0,
            step=0.1,
            value=4.0,
        )
        num_inference_steps = gr.Slider(
            label="Number of inference steps",
            minimum=1,
            maximum=100,
            step=1,
            value=50,
        )
        num_images = gr.Slider(
            label="Number of images",
            minimum=1,
            maximum=5,
            step=1,
            value=1,
        )
        zip_images = gr.Checkbox(label="Zip generated images", value=False)
        
        gr.Markdown("### Output Information")
        seed_display = gr.Textbox(label="Seed used", interactive=False)
        generation_time = gr.Textbox(label="Generation time (seconds)", interactive=False)
        zip_file = gr.File(label="Download ZIP")

    # Update aspect ratio
    def set_dimensions(ar):
        w, h = aspect_ratios[ar]
        return gr.update(value=w), gr.update(value=h)
    
    aspect_ratio.change(
        fn=set_dimensions,
        inputs=aspect_ratio,
        outputs=[width, height]
    )

    # Negative prompt visibility
    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt
    )

    # Run button and prompt submit
    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            num_inference_steps,
            num_images,
            zip_images,
        ],
        outputs=[result, seed_display, generation_time, zip_file],
        api_name="run",
    )

    # Examples
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed_display, generation_time, zip_file],
        fn=generate,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)