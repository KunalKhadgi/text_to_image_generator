import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def load_model():
    """Loads Stable Diffusion model with CPU mode (better for MX450 users)."""
    print("✔ Loading optimized model on CPU...")

    model_id = "CompVis/stable-diffusion-v1-4"  # ✅ Use a lighter model

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # ✅ Keep FP32 for CPU stability
        safety_checker=None
    )

    pipe.to("cpu")  # ✅ Force CPU Mode (MX450 is too slow for GPU inference)

    return pipe

# ✅ Load model once
pipe = load_model()

def generate_image(prompt, num_steps=10, guidance_scale=7.0):
    """Generates an image using CPU and returns the file path and time taken."""
    if not prompt.strip():
        return None, 0

    print(f"🎨 Generating image for: '{prompt}' with {num_steps} steps on CPU...")

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
