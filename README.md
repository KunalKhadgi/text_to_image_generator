# text_to_image_generator
Takes text prompt as a input and generates a image using `runwayml/stable-diffusion-v1-5` model (Diffusion model)


# AI Text-to-Image Generator

This project is a **Stable Diffusion-based text-to-image generator** with a **Streamlit frontend** and an **optimized backend** using PyTorch and Hugging Face Diffusers.

## 📌 Features
- **Generate images from text prompts** using Stable Diffusion.
- **Choose inference steps & guidance scale** for better results.
- **Supports both CPU and GPU execution** (automatically selects based on hardware).
- **Optimized for low VRAM GPUs** (MX450, RTX 3050, etc.).
- **Streamlit UI for easy interaction**.

---

## 🚀 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/text-to-image-generator.git
cd text-to-image-generator
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Backend & Frontend**
#### **Run Streamlit Frontend**
```bash
streamlit run frontend.py
```

#### **Run Backend Separately (Optional Debugging)**
```bash
python img_gen.py
```

---

## 🛠 Usage
1. **Enter a text prompt** in the Streamlit UI.
2. **Adjust the number of inference steps** (default: 20, lower for speed, higher for quality).
3. **Adjust the guidance scale** (default: 7.5, lower for diversity, higher for accuracy).
4. **Click 'Generate Image'** and wait for processing.
5. **View & save the generated image**.

---

## ⚙️ Configuration & Optimization
### **Change Execution Mode (GPU vs CPU)**
- The script **automatically detects GPU availability**.
- To force CPU execution (if GPU is slow):
  ```python
  generate_image(prompt, num_steps=15, device="cpu")
  ```

### **Adjust Model for Low VRAM GPUs**
- Change model to **Stable Diffusion 1.4** (lighter version):
  ```python
  model_id = "CompVis/stable-diffusion-v1-4"
  ```
- Reduce `num_steps` to **10-15** for faster generation.

---

## 📜 Requirements
- **Python 3.8+**
- **NVIDIA GPU with CUDA (Recommended)**
- **Minimum 8GB RAM (16GB Recommended for CPU mode)**
- **Libraries:** See `requirements.txt`

---

## 🛠 Troubleshooting
| **Issue** > **Solution** |
|-----------|-------------|
| Slow generation (10+ min) > Reduce `num_steps` to 10-15, use `CompVis/stable-diffusion-v1-4` |
| GPU crashes due to VRAM > Use CPU mode (`device='cpu'`), enable `enable_attention_slicing()` |
| Black images generated > Increase `num_steps`, try different `guidance_scale` values |
| `xformers` error > Remove `xformers` from `requirements.txt` and install manually |

---


## 📜 License
This project is open-source under the Appache License.

