# Multimodal Streamlit App

This folder contains a Streamlit web app that accepts text, image, and audio inputs and returns AI-generated outputs.

## Setup

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Copy the example environment file and add your API keys:
   ```powershell
   copy .env.example .env
   ```

3. Add your API keys to `.env`:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`

## Run the app

```powershell
streamlit run streamlit_app.py
```

## Features

- Text prompts using Gemini or Groq
- Text-to-speech conversion using Gemini or Hugging Face
- Image captioning via Hugging Face inference
- Audio transcription via Hugging Face Whisper

## Notes

- If no supported API keys are available, the app will display status hints in the sidebar.
- Image and audio features require `HUGGINGFACEHUB_API_TOKEN` and `genimi_api_key`.
