import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def load_model():
    """Loads the Stable Diffusion model once globally for faster execution."""
    print("✔ Loading optimized model on CUDA with FP16...")

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # ✅ Use FP16 for Speed
        safety_checker=None
    )

    if torch.cuda.is_available():
        pipe.to("cuda")
        torch.backends.cudnn.benchmark = True  # ✅ Optimized CUDA Performance
        pipe.enable_attention_slicing()  # ✅ Reduces VRAM Usage
        pipe.enable_vae_slicing()  # ✅ Faster Latent Space Processing
        
        try:
            pipe.enable_xformers_memory_efficient_attention()  # ✅ Works on newer GPUs
        except:
            print("⚠ xFormers not supported, skipping optimization.")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

# ✅ Load model once at startup (prevents reloading)
pipe = load_model()

def generate_image(prompt, num_steps=20, guidance_scale=7.5):
    """Generates an image and returns the file path and time taken."""
    if not prompt.strip():
        return None, 0

    print(f"🎨 Generating image for: '{prompt}' with {num_steps} steps...")

    import time
    start_time = time.time()

    try:
        image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]
        output_path = "generated_image.png"
        image.save(output_path)

        time_taken = round(time.time() - start_time, 2)
        print(f"✅ Image saved at {output_path} in {time_taken} sec")

        return output_path, time_taken

    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return None, 0
