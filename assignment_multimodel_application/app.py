"""
Multimodal AI Studio - A Streamlit web app for interacting with multiple AI models
Supports: Text-to-Text, Text-to-Image, Text-to-Audio, Image-to-Text

To run this app:
    streamlit run app.py

Required API Keys (set in .env file):
    - GROQ_API_KEY: Get from https://console.groq.com/keys
    - HUGGINGFACEHUB_API_TOKEN: Get from https://huggingface.co/settings/tokens
    - GEMINI_API_KEY: Get from https://aistudio.google.com/apikey
"""

import streamlit as st
import os
from dotenv import load_dotenv
import base64
import io
from typing import Optional
import wave

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# IMPORTS FOR AI APIs
# ============================================================================
from groq import Groq
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from PIL import Image
import requests


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Multimodal AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .mode-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #0066cc;
    }
    .mode-description {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
        background-color: #f0f0f0;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize session state variables for conversation history and UI state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = "llama-3.3-70b-versatile"
    if "vision_model" not in st.session_state:
        st.session_state.vision_model = "gemini"


initialize_session_state()


# ============================================================================
# API CLIENT SETUP
# ============================================================================
def get_groq_client(api_key: str) -> Groq:
    """Initialize and return Groq API client."""
    return Groq(api_key=api_key)


def get_gemini_client(api_key: str):
    """Initialize and return Google Gemini API client."""
    return genai.Client(api_key=api_key)


def get_hf_client(api_token: str) -> InferenceClient:
    """Initialize and return Hugging Face Inference client."""
    return InferenceClient(
        provider="nscale",
        api_key=api_token,
    )


# ============================================================================
# GROQ LLM MODELS AVAILABLE
# ============================================================================
GROQ_LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant"
]

GROQ_VISION_MODELS = [
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]


# ============================================================================
# TEXT-TO-TEXT HANDLER
# ============================================================================
def handle_text_to_text(groq_api_key: str, model: str):
    """
    Chat-style interface using Groq LLM.
    Maintains conversation history in session state.
    """
    st.markdown('<p class="mode-header">💬 Text → Text Chat</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mode-description">Chat with a powerful LLM. Your conversation history is maintained during this session.</p>',
        unsafe_allow_html=True
    )
    
    # Model selector
    st.session_state.groq_model = st.selectbox(
        "Select LLM Model:",
        GROQ_LLM_MODELS,
        index=GROQ_LLM_MODELS.index(st.session_state.groq_model) if st.session_state.groq_model in GROQ_LLM_MODELS else 0,
        key="llm_model_select"
    )
    
    # Display conversation history
    st.divider()
    st.subheader("Conversation History")
    
    # Create a scrollable container for chat history
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.write(f"**You:** {msg['content']}")
            else:
                st.write(f"**Assistant:** {msg['content']}")
    
    st.divider()
    
    # Input section
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input("Your message:", placeholder="Ask me anything...", key="text_input")
    with col2:
        send_button = st.button("Send", key="send_button")
    
    # Process user input
    if send_button and user_input:
        try:
            with st.spinner("🤖 Thinking..."):
                # Initialize Groq client
                client = get_groq_client(groq_api_key)
                
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Prepare messages for API (keeping last 10 messages for context)
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]
                ]
                
                # Call Groq API
                response = client.chat.completions.create(
                    model=st.session_state.groq_model,
                    messages=api_messages,
                    max_tokens=1024,
                    temperature=0.7
                )
                
                assistant_response = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                st.success("✅ Response received!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if "rate_limit" in str(e).lower():
                st.warning("⏳ Rate limit exceeded. Please wait a moment and try again.")
            elif "api_key" in str(e).lower() or "401" in str(e):
                st.error("🔑 Invalid API key. Please check your GROQ_API_KEY in `.env`.")


# ============================================================================
# TEXT-TO-IMAGE HANDLER
# ============================================================================
def handle_text_to_image(hf_api_token: str):
    """
    Generate images from text prompts using Hugging Face Stable Diffusion.
    """
    st.markdown('<p class="mode-header">🎨 Text → Image Generation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mode-description">Describe an image in detail, and let AI create it for you using Stable Diffusion XL.</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Prompt input
    prompt = st.text_area(
        "📝 Describe the image you want to generate:",
        placeholder="e.g., A serene mountain landscape at sunset with snow peaks, cinematic lighting, photorealistic...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        generate_button = st.button("🚀 Generate Image", key="generate_image_btn")
    with col2:
        num_steps = st.slider("Inference Steps (higher = better quality, slower):", 20, 50, 30)
    
    if generate_button and prompt:
        try:
            with st.spinner("🎨 Generating image... This may take 30-60 seconds..."):
                # Initialize Hugging Face client
                client = get_hf_client(hf_api_token)
                
                # Generate image
                image = client.text_to_image(
                    prompt=prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                    num_inference_steps=num_steps,
                    height=768,
                    width=768
                )
                
                st.success("✅ Image generated successfully!")
                
                # Display image
                st.image(image, caption="Generated Image", use_container_width=True)
                
                # Download button
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                st.download_button(
                    label="⬇️ Download Image",
                    data=img_byte_arr.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if "rate_limit" in str(e).lower():
                st.warning("⏳ Rate limit exceeded. Please wait before generating another image.")
            elif "401" in str(e) or "Unauthorized" in str(e):
                st.error("🔑 Invalid Hugging Face API token. Please check your HUGGINGFACEHUB_API_TOKEN in `.env`.")


# ============================================================================
# TEXT-TO-AUDIO HANDLER
# ============================================================================
def handle_text_to_audio(gemini_api_key: str):
    """
    Convert text to speech using Google Gemini API.
    """
    st.markdown('<p class="mode-header">🎵 Text → Audio (Text-to-Speech)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mode-description">Convert your text into natural-sounding speech with multiple voice options.</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Voice selection
    voice_options = ["Kore", "Phoebe", "Charon", "Fenrir", "Aoede"]
    selected_voice = st.selectbox("🎤 Select Voice:", voice_options, index=0)
    
    # Text input
    text_input = st.text_area(
        "📝 Enter text to convert to speech:",
        placeholder="Type or paste the text you want to convert to speech...",
        height=100
    )
    
    convert_button = st.button("🔊 Convert to Audio", key="convert_audio_btn")
    
    if convert_button and text_input:
        try:
            with st.spinner("🎵 Converting text to speech..."):
                # Initialize Gemini client
                client = get_gemini_client(gemini_api_key)
                
                # Generate audio
                response = client.models.generate_content(
                    model="gemini-2.5-flash-exp",
                    contents=text_input,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=selected_voice,
                                )
                            )
                        ),
                    )
                )
                
                st.success("✅ Audio generated successfully!")
                
                # Extract audio data
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                
                # Display audio player
                st.audio(audio_data, format="audio/wav")
                
                # Download button
                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_data,
                    file_name="generated_audio.wav",
                    mime="audio/wav"
                )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if "rate_limit" in str(e).lower():
                st.warning("⏳ Rate limit exceeded. Please wait before generating another audio.")
            elif "401" in str(e) or "Unauthorized" in str(e):
                st.error("🔑 Invalid Gemini API key. Please check your GEMINI_API_KEY in `.env`.")


# ============================================================================
# IMAGE-TO-TEXT HANDLER
# ============================================================================
def handle_image_to_text(groq_api_key: str, gemini_api_key: str, use_gemini: bool):
    """
    Generate captions/descriptions from images using either Groq (Llama Vision) or Gemini.
    """
    st.markdown('<p class="mode-header">📸 Image → Text (Vision Analysis)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mode-description">Upload an image and let AI generate a detailed description or analysis of it.</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Model selection
    model_choice = st.radio(
        "🤖 Select Vision Model:",
        ("Google Gemini (Recommended)", "Groq Llama Vision"),
        horizontal=True,
        index=0 if use_gemini else 1
    )
    use_gemini_model = model_choice == "Google Gemini (Recommended)"
    
    # Image uploader
    uploaded_file = st.file_uploader("📁 Upload an image:", type=["jpg", "jpeg", "png", "gif", "webp"])
    
    # Analysis type
    analysis_type = st.selectbox(
        "📋 Analysis Type:",
        ["Detailed Description", "Brief Caption", "Objects & Details", "Scene Analysis", "Custom Prompt"]
    )
    
    custom_prompt = None
    if analysis_type == "Custom Prompt":
        custom_prompt = st.text_area("Enter your custom prompt for the image:")
    
    analyze_button = st.button("🔍 Analyze Image", key="analyze_image_btn")
    
    if uploaded_file and analyze_button:
        try:
            # Read and display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Prepare analysis prompt
            prompts = {
                "Detailed Description": "Provide a comprehensive and detailed description of this image. Include all visible elements, colors, composition, mood, and any notable details.",
                "Brief Caption": "Provide a brief one-sentence caption for this image.",
                "Objects & Details": "List all objects, people, animals, and important details visible in this image, organized by category.",
                "Scene Analysis": "Analyze the scene in this image. Describe the setting, lighting, perspective, and overall atmosphere.",
            }
            
            prompt = custom_prompt if custom_prompt else prompts.get(analysis_type, prompts["Detailed Description"])
            
            with st.spinner("🔍 Analyzing image..."):
                if use_gemini_model:
                    # Use Gemini for image analysis
                    client = get_gemini_client(gemini_api_key)
                    
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_data = base64.standard_b64encode(img_byte_arr.read()).decode("utf-8")
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-exp",
                        contents=[
                            types.Part.from_uri(
                                mime_type="image/png",
                                uri=f"data:image/png;base64,{image_data}"
                            ),
                            prompt
                        ]
                    )
                    
                    analysis_result = response.text
                else:
                    # Use Groq Llama Vision for image analysis
                    client = get_groq_client(groq_api_key)
                    
                    # Convert PIL image to base64
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_data = base64.standard_b64encode(img_byte_arr.read()).decode("utf-8")
                    
                    response = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": image_data,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ],
                            }
                        ],
                        max_tokens=1024,
                    )
                    
                    analysis_result = response.choices[0].message.content
                
                st.success("✅ Analysis complete!")
                
                # Display result
                with col2:
                    st.subheader("📝 Analysis Result")
                    st.write(analysis_result)
                    
                    # Copy to clipboard button
                    st.text_area(
                        "Copy result:",
                        value=analysis_result,
                        height=100,
                        disabled=True
                    )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if "rate_limit" in str(e).lower():
                st.warning("⏳ Rate limit exceeded. Please wait before analyzing another image.")
            elif "401" in str(e) or "Unauthorized" in str(e) or "Invalid API" in str(e):
                st.error("🔑 Invalid API key. Please check your API keys in `.env`.")
            else:
                st.error(f"Failed to analyze image. Please try again. Error: {str(e)}")


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
def setup_sidebar():
    """Setup sidebar with mode selection and settings."""
    st.sidebar.title("⚙️ Settings")
    
    # Load API keys from environment variables only
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    st.sidebar.divider()
    
    # Mode selection
    st.sidebar.subheader("🎯 Select Mode")
    mode = st.sidebar.radio(
        "Choose interaction mode:",
        [
            "💬 Text → Text",
            "🎨 Text → Image",
            "🎵 Text → Audio",
            "📸 Image → Text"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.success("✅ Conversation history cleared!")
        st.rerun()
    
    st.sidebar.divider()
    
    # Info section
    st.sidebar.subheader("ℹ️ How to Use")
    st.sidebar.info(
        """
        1. **Set API Keys**: Add your API keys to `.env` file
        2. **Select Mode**: Choose an interaction mode from the sidebar
        3. **Follow Prompts**: Each mode has specific inputs and outputs
        4. **Download Results**: Save generated images or audio files
        
        ### Quick Links:
        - [Groq Console](https://console.groq.com/keys)
        - [HuggingFace Tokens](https://huggingface.co/settings/tokens)
        - [Google AI Studio](https://aistudio.google.com/apikey)
        """
    )
    
    return mode, groq_api_key, hf_api_token, gemini_api_key


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application entry point."""
    st.title("🎨 Multimodal AI Studio")
    st.markdown(
        "Interact with AI across **text, image, and audio**. Powered by Groq, Google Gemini, and Hugging Face.",
        help="A complete multimodal AI experience"
    )
    
    # Setup sidebar and get configuration
    mode, groq_api_key, hf_api_token, gemini_api_key = setup_sidebar()
    
    # Check if API keys are set in environment
    if not all([groq_api_key, hf_api_token, gemini_api_key]):
        st.error("⚠️ Missing API keys. Please set these environment variables in your `.env` file:")
        st.code("""
GROQ_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
        """)
        st.info("See `.env.example` for detailed instructions on how to get each API key.")
        return
    
    st.divider()
    
    # Route to appropriate handler based on mode
    if mode == "💬 Text → Text":
        handle_text_to_text(groq_api_key, st.session_state.groq_model)
    elif mode == "🎨 Text → Image":
        handle_text_to_image(hf_api_token)
    elif mode == "🎵 Text → Audio":
        handle_text_to_audio(gemini_api_key)
    elif mode == "📸 Image → Text":
        handle_image_to_text(groq_api_key, gemini_api_key, use_gemini=True)


if __name__ == "__main__":
    main()