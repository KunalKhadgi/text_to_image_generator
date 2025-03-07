import streamlit as st
import asyncio
from cpu_mode import generate_image
import os

st.title("🖼️ AI Image Generator (GPU Optimized)")

# ✅ Fix Streamlit Async Issue
def ensure_event_loop(): 
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

ensure_event_loop()  # ✅ Ensure event loop before Streamlit starts

prompt = st.text_input("Enter your prompt:")
num_steps = st.slider("Number of Inference Steps:", 10, 50, 20)  # ✅ Lowered to 20 for Speed
guidance_scale = st.slider("Guidance Scale:", 1.0, 20.0, 7.5, 0.5)  # ✅ Default = 7.5

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt!")
    else:
        st.info("⏳ Generating image...")

        # ✅ Generate Image
        with st.spinner("Processing..."):
            image_path, time_taken = generate_image(prompt, num_steps, guidance_scale)

        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=f"Generated in {time_taken} sec", use_column_width=True)
            st.success(f"✅ Done! Total Time: {time_taken} sec")
        else:
            st.error("❌ Image generation failed!")
