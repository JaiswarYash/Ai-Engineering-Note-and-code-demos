# 🎨 Multimodal AI Studio

A comprehensive Streamlit web application that lets you interact with AI across multiple modalities: **text, image, and audio**.

## ✨ Features

### 1. **💬 Text → Text (Chat)**
- Chat with powerful LLMs via Groq API
- Choose from multiple Groq models (e.g., Llama 3.3 70B, Mixtral 8x7B)
- Maintain conversation history within a session
- Real-time responses with loading indicators

### 2. **🎨 Text → Image (Generation)**
- Generate high-quality images from text descriptions
- Powered by Stable Diffusion XL via Hugging Face
- Adjustable inference steps for quality/speed trade-off
- Download generated images as PNG files

### 3. **🎵 Text → Audio (Text-to-Speech)**
- Convert text to natural-sounding speech
- Multiple voice options (Kore, Phoebe, Charon, Fenrir, Aoede)
- Powered by Google Gemini TTS API
- Audio player and download functionality

### 4. **📸 Image → Text (Vision Analysis)**
- Upload images for AI analysis
- Multiple analysis types: detailed description, caption, objects, scene analysis, custom
- Choose between Google Gemini or Groq Llama Vision models
- Side-by-side display of image and analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- API keys from:
  - **Groq**: https://console.groq.com/keys
  - **Hugging Face**: https://huggingface.co/settings/tokens
  - **Google Gemini**: https://aistudio.google.com/apikey

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd assignment_multimodel_application
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys:
     ```
     GROQ_API_KEY=your_key_here
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     GEMINI_API_KEY=your_key_here
     ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   The app will open automatically in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### API Key Setup
- Set your API keys in the `.env` file (copy from `.env.example`)
- The app loads keys from environment variables on startup

### Text → Text Mode
- Select a model from the dropdown
- Type your message in the input field
- Click "Send" or press Enter
- View your conversation history in real-time

### Text → Image Mode
- Write a detailed prompt describing the image you want
- Adjust inference steps (20-50) for quality/speed trade-off
- Click "Generate Image"
- Download the result as PNG

### Text → Audio Mode
- Select a voice from the dropdown
- Enter the text to convert
- Click "Convert to Audio"
- Listen in the player or download as WAV

### Image → Text Mode
- Select a vision model (Gemini recommended)
- Upload an image (JPG, PNG, GIF, WebP)
- Choose an analysis type or provide custom prompt
- Click "Analyze Image"
- View the analysis result

## 🛠️ Configuration

### Available Models

**Groq LLMs:**
- `llama-3.3-70b-versatile` (latest, recommended)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`
- `llama-3.1-8b-instant` (fastest)

**Groq Vision Models:**
- `llama-3.2-11b-vision-preview`
- `llama-3.2-90b-vision-preview`

**Image Generation:**
- `stabilityai/stable-diffusion-xl-base-1.0`

**Text-to-Speech:**
- Multiple voices: Kore, Phoebe, Charon, Fenrir, Aoede

**Vision Analysis:**
- Google Gemini 2.5 Flash (recommended)
- Groq Llama Vision models

## 🔒 Security Notes

- **Never** commit `.env` file with real keys to version control
- Use `.env.example` as a template for team members
- API keys are only loaded from environment variables for security

## ⚠️ Error Handling

The app includes comprehensive error handling for:
- **Rate limits**: Friendly message with retry guidance
- **Invalid API keys**: Clear error messages with troubleshooting steps
- **Network timeouts**: Graceful error messages
- **Invalid inputs**: Form validation and helpful prompts

## 📊 Performance Tips

- **Text-to-Image**: Inference steps 20-30 for fast generation, 40-50 for higher quality
- **LLM responses**: Adjust temperature (0 = deterministic, 1 = creative) if needed
- **Large files**: Images over 5MB may take longer to upload/process

## 🐛 Troubleshooting

### "API key not found" error
- Ensure your `.env` file is in the project directory
- Check that environment variable names match exactly (case-sensitive)
- Copy `.env.example` to `.env` and fill in your keys

### "Rate limit exceeded"
- Wait a moment before making the next request
- Rate limits vary by API provider
- Check your account on the provider's dashboard for usage

### "Connection timeout"
- Check your internet connection
- Verify the API provider's status page
- Try again in a few moments

### Image generation very slow
- Lower the inference steps slider
- Try a simpler, shorter prompt

## 📝 Project Structure

```
assignment_multimodel_application/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .env                  # Your actual API keys (not in git)
├── pyproject.toml        # Project configuration
├── README.md             # Main project README
├── README_STREAMLIT_APP.md  # Streamlit app documentation
└── notebook/
    └── multimodel.ipynb  # Jupyter notebook experiments
```

## 🎓 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Groq API Docs**: https://console.groq.com/docs
- **Hugging Face Hub**: https://huggingface.co/docs/hub
- **Google Gemini API**: https://ai.google.dev

## 🤝 Contributing

Feel free to extend this app with:
- More AI models
- Additional output formats
- Advanced UI customizations
- Batch processing capabilities
- Model comparison features

## 📄 License

This project is open source and available under the MIT License.

---

**Happy experimenting! 🚀**
