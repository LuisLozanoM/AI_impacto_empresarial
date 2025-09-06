import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Set page config
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide"
)

# Language code mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'pl': 'Polish',
    'tr': 'Turkish',
    'th': 'Thai',
    'vi': 'Vietnamese'
}

# Cache the models to avoid reloading
@st.cache_resource
def load_models():
    """Load the translation model"""
    try:
        # Load the text generation model for translation
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def detect_language(text, tokenizer, model):
    """Detect the language of the input text using the Qwen model"""
    try:
        if not text.strip():
            return "Unknown"
        
        # Create a language detection prompt
        prompt = f"""Identify the language of the following text. Respond with only the language name (like English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Dutch, etc.):

Text: {text[:200]}
Language:"""
        
        # Generate language detection
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_text = tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Clean up the output
        detected_language = generated_text.strip().split('\n')[0].strip()
        
        # Remove any artifacts and get the language name
        detected_language = re.sub(r'^(Language:|Text:)', '', detected_language).strip()
        
        # Validate the detected language against known languages
        language_lower = detected_language.lower()
        for code, name in LANGUAGE_NAMES.items():
            if name.lower() in language_lower or language_lower in name.lower():
                return name
        
        # If we found a reasonable language name, return it
        if detected_language and len(detected_language) > 2 and detected_language.isalpha():
            return detected_language.title()
        
        return "Unknown"
        
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return "Unknown"

def translate_to_french(text, source_language, tokenizer, model):
    """Translate text from any language to French using few-shot learning"""
    try:
        # Create language-specific examples based on detected language
        if 'English' in source_language:
            examples = """English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: Thank you very much.
French: Merci beaucoup.

English: Where is the train station?
French: Où est la gare?

English: I would like to order food.
French: Je voudrais commander de la nourriture."""
            lang_prefix = "English"
        elif 'Spanish' in source_language:
            examples = """Spanish: Hola, ¿cómo estás?
French: Bonjour, comment allez-vous?

Spanish: Muchas gracias.
French: Merci beaucoup.

Spanish: ¿Dónde está la estación de tren?
French: Où est la gare?

Spanish: Me gustaría pedir comida.
French: Je voudrais commander de la nourriture."""
            lang_prefix = "Spanish"
        elif 'German' in source_language:
            examples = """German: Hallo, wie geht es dir?
French: Bonjour, comment allez-vous?

German: Vielen Dank.
French: Merci beaucoup.

German: Wo ist der Bahnhof?
French: Où est la gare?

German: Ich möchte Essen bestellen.
French: Je voudrais commander de la nourriture."""
            lang_prefix = "German"
        else:
            # Default to treating as English for unknown languages
            examples = """Text: Hello, how are you?
French: Bonjour, comment allez-vous?

Text: Thank you very much.
French: Merci beaucoup.

Text: Where is the train station?
French: Où est la gare?

Text: I would like to order food.
French: Je voudrais commander de la nourriture."""
            lang_prefix = "Text"
        
        # Create the translation prompt
        prompt = f"""Translate to French:

{examples}

{lang_prefix}: {text}
French:"""
        
        # Generate translation
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_text = tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Clean up the output
        french_translation = generated_text.strip().split('\n')[0].strip()
        
        # Remove any remaining artifacts
        french_translation = re.sub(r'^(French:|Text:|English:|Spanish:|German:)', '', french_translation).strip()
        
        return french_translation if french_translation else "Translation not available"
        
    except Exception as e:
        st.error(f"Error in translation: {e}")
        return "Translation failed"

def main():
    st.title("🌍 AI Language Translator")
    st.markdown("### Detect language and translate to French")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models..."):
        tokenizer, model = load_models()
    
    if tokenizer is None or model is None:
        st.error("Failed to load translation models. Please check your environment.")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        # Text input
        input_text = st.text_area(
            "Enter text in any language:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        # Translate button
        translate_button = st.button("🔄 Translate to French", type="primary")
    
    with col2:
        st.subheader("🎯 Results")
        
        if translate_button and input_text.strip():
            with st.spinner("Detecting language and translating..."):
                # Detect language
                detected_language = detect_language(input_text, tokenizer, model)
                
                # Translate to French
                french_translation = translate_to_french(input_text, detected_language, tokenizer, model)
                
                # Display results
                st.success("Translation completed!")
                
                # Show detected language
                st.markdown(f"**Detected Language:** `{detected_language}`")
                
                # Show original text
                st.markdown("**Original Text:**")
                st.info(input_text)
                
                # Show translation
                st.markdown("**French Translation:**")
                st.success(french_translation)
                
                # Copy button simulation
                st.markdown("---")
                st.code(french_translation, language="text")
        
        elif translate_button and not input_text.strip():
            st.warning("Please enter some text to translate.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Powered by Qwen2-0.5B for Language Detection & Translation | Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar with examples
    with st.sidebar:
        st.header("📚 Example Texts")
        st.markdown("Try these examples:")
        
        examples = [
            ("English", "Hello, how are you today?"),
            ("Spanish", "Hola, ¿cómo estás hoy?"),
            ("German", "Hallo, wie geht es dir heute?"),
            ("Italian", "Ciao, come stai oggi?"),
            ("Portuguese", "Olá, como você está hoje?")
        ]
        
        for lang, text in examples:
            if st.button(f"{lang}: {text[:20]}...", key=f"example_{lang}"):
                st.session_state.example_text = text
        
        # Additional info
        st.markdown("---")
        st.markdown("**Supported Languages:**")
        st.markdown("✅ English, Spanish, German, Italian, Portuguese, French, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Dutch, and many more!")
        
        st.markdown("**Features:**")
        st.markdown("• AI-powered language detection")
        st.markdown("• AI-powered translation")
        st.markdown("• Few-shot learning approach")
        st.markdown("• Real-time processing")

if __name__ == "__main__":
    main()
