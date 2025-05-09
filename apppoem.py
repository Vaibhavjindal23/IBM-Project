import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import uuid
import traceback

# Set up the Streamlit page
st.set_page_config(
    page_title="AI Poetry Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Remove horizontal spacing and center content */
    .stApp {
        max-width: 100%;
        padding: 0 1.5rem;
        margin: 0 auto;
        font-size: 20px; /* Increased global font size from 18px */
        font-family: 'Georgia', 'Garamond', serif;
        line-height: 1.6;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #8e44ad, #3498db);
        padding: 30px 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: white;
        font-size: 26px; /* Larger header font size */
        font-weight: 700;
        text-align: center;
        font-family: 'Georgia', serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    /* Poetry container improvements */
    .poetry-container {
        background-color: #f9f7f1; /* Parchment-like color */
        border-radius: 12px;
        padding: 40px 45px;
        border: 1px solid #e0d8c0;
        margin-top: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
    }

    .poetry-text {
        font-family: 'Garamond', 'Georgia', serif;
        font-size: 24px; /* Larger font for poems */
        line-height: 1.9;
        color: #2c3e50; /* Dark text for readability */
        white-space: pre-wrap;
        text-align: center;
        letter-spacing: 0.02em;
        font-weight: 500;
    }

    /* Style badges */
    .style-badge {
        background-color: #e8daef;
        color: #8e44ad;
        border-radius: 15px;
        padding: 6px 14px;
        margin-right: 10px;
        font-size: 1.05em;
        font-weight: 600;
        font-family: 'Georgia', serif;
        display: inline-block;
        box-shadow: 0 1px 5px rgba(142, 68, 173, 0.2);
    }

    .theme-badge {
        background-color: #d1e7f0;
        color: #3498db;
        border-radius: 15px;
        padding: 6px 14px;
        margin-right: 10px;
        font-size: 1.05em;
        font-weight: 600;
        font-family: 'Georgia', serif;
        display: inline-block;
        box-shadow: 0 1px 5px rgba(52, 152, 219, 0.2);
    }

    /* Input fields styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #f9f7f1;
        border: 1.8px solid #e0d8c0;
        color: #2c3e50;
        font-size: 18px;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: 'Georgia', serif;
        transition: border-color 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #8e44ad;
        outline: none;
        box-shadow: 0 0 8px rgba(142, 68, 173, 0.5);
    }

    /* Button styling */
    .stButton > button {
        background-color: #8e44ad;
        color: white;
        border-radius: 12px;
        padding: 8px 22px;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        font-family: 'Georgia', serif;
        cursor: pointer;
        transition: all 0.35s ease;
        box-shadow: 0 4px 12px rgba(142, 68, 173, 0.4);
        width: 100%;
        max-width: 350px;
        margin-top: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stButton > button:hover {
        background-color: #9b59b6;
        box-shadow: 0 6px 18px rgba(155, 89, 182, 0.6);
        transform: translateY(-2px);
    }

    /* Label improvements */
    label {
        font-size: 1.15rem;
        color: #2c3e50;
        font-family: 'Georgia', serif;
        font-weight: 600;
    }

    /* Radio button styling */
    .stRadio > div {
        margin-top: 8px;
        padding: 14px;
        border-radius: 12px;
        background-color: #2c3e50;
        font-family: 'Georgia', serif;
        font-weight: 600;
        font-size: 18px;
        color: white;
    }

    /* Sidebar adjustments */
    .css-1d391kg {
        font-family: 'Georgia', serif !important;
        font-size: 18px !important;
    }

    /* Footer improvements */
    footer {
        margin-top: 40px;
        padding: 15px 0;
        font-size: 1.25rem;
        font-family: 'Georgia', serif;
        color: #2c3e50;
        text-align: center;
        letter-spacing: 0.03em;
        font-weight: 500;
        opacity: 0.75;
    }
</style>
""", unsafe_allow_html=True)


# Title and description in a custom header
st.markdown("""
<div class="header-container">
    <h1>AI Poetry Generator</h1>
    <p>Create beautiful poems with HuggingFace language models</p>
</div>
""", unsafe_allow_html=True)

# Function to initialize HuggingFace LLM
def initialize_huggingface_llm(api_key, model_name, temperature=0.7, max_length=None):
    """Initialize the HuggingFace LLM"""
    # Set max_length based on poem length if not specified
    if max_length is None:
        max_length = 1000  # Default
    
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        max_new_tokens=max_length,
        provider="hyperbolic"
    )
    return llm

# Function to create poetry prompt
def setup_poetry_prompt(theme, emotion, style, form, length):
    """Create a prompt for poetry generation based on user parameters"""
    template = """
    You are a skilled poet with expertise in many poetic forms and traditions. Write a beautiful, 
    evocative poem based on the theme, emotion, style, form, and length provided by the user.
    
    Guidelines:
    - Create vivid imagery and metaphors that resonate with the theme and emotion
    - Use language, rhythm, and structure appropriate to the requested poetic form
    - Maintain consistency in tone and voice throughout the poem
    - Balance clarity of meaning with artistic expression
    - Follow the traditional constraints of the specified poetic form when applicable
    - Match the requested length (short ~10 lines, medium ~20 lines, long ~30 lines)
    
    Please write a {length} poem with the following elements:
    
    Theme: {theme}
    Emotion/Mood: {emotion}
    Style: {style}
    Form: {form}
    
    Make it evocative, beautiful, and thoughtful. Ensure the poem is complete and has proper structure.
    Return only the poem without introductions or explanations.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["theme", "emotion", "style", "form", "length"]
    )
    
    return prompt

# Function to generate poem
def generate_poem(api_key, theme, emotion, style, form, length, temperature, model_name):
    """Generate a poem using the HuggingFace LLM"""
    # Define length tokens
    length_tokens = {
        "short": 500,   # ~10 lines
        "medium": 800,  # ~20 lines
        "long": 1200    # ~30 lines
    }
    
    max_tokens = length_tokens.get(length, 800)
    
    # Initialize the LLM
    llm = initialize_huggingface_llm(
        api_key,
        model_name,
        temperature,
        max_tokens
    )
    
    # Create and format the prompt
    prompt = setup_poetry_prompt(theme, emotion, style, form, length)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the poem
    result = llm_chain.invoke({
        "theme": theme,
        "emotion": emotion,
        "style": style,
        "form": form,
        "length": length
    })
    
    # Extract the poem text
    if isinstance(result, dict) and "text" in result:
        poem_text = result["text"]
    else:
        poem_text = str(result)
    
    return poem_text

# Function to save poem
def save_poem(poem_text, title="My Poem"):
    """Save poem to a file and provide download link"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(poem_text)
        temp_path = temp_file.name
    
    # Read the file for download
    with open(temp_path, 'rb') as f:
        st.download_button(
            label="Download Poem",
            data=f,
            file_name=f"{title.replace(' ', '_')}.txt",
            mime="text/plain"
        )

# Session state initialization
if 'generated_poem' not in st.session_state:
    st.session_state['generated_poem'] = None
if 'poem_metadata' not in st.session_state:
    st.session_state['poem_metadata'] = None

# Sidebar for API configuration
st.sidebar.header("Configuration")

# API key input
api_key = st.sidebar.text_input("HuggingFace API Token", type="password")
if api_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Model selection
model_options = {
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Nous-Hermes-2-Mixtral-8x7B-DPO": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "DeepSeek-Prover-V2-671B": "deepseek-ai/DeepSeek-V3-0324",
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_name = model_options[selected_model]

# Generation parameters
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                             help="Higher values (0.8+) produce more creative, varied poems. Lower values (0.2-0.5) create more predictable, structured poems.")

# Provider selection
provider_options = ["auto", "hf-inference"]
selected_provider = st.sidebar.selectbox("Select Provider", provider_options, index=1)

# Poetry parameters section
st.header("Poetry Parameters")

# Create columns for more compact layout
col1, col2 = st.columns(2)

with col1:
    theme = st.text_input("Theme", 
                        placeholder="e.g., nature, love, mortality, dreams",
                        help="The central subject or concept of your poem")

    emotion = st.text_input("Emotion/Mood", 
                          placeholder="e.g., melancholy, joy, wonder, longing",
                          help="The emotional tone or atmosphere of your poem")

with col2:
    # Poetry style selection
    style_options = [
        "Romantic", "Modern", "Victorian", "Minimalist", 
        "Confessional", "Imagist", "Beat", "Lyrical", "Gothic", "Surrealist"
    ]
    style = st.selectbox("Poetic Style", style_options, index=0,
                       help="The literary tradition or movement that influences the poem's language and themes")

    # Poetry form selection
    form_options = [
        "Free Verse", "Sonnet", "Haiku", "Villanelle", "Ballad", 
        "Limerick", "Ode", "Ghazal", "Tanka", "Sestina"
    ]
    form = st.selectbox("Poetic Form", form_options, index=0,
                      help="The structural pattern or traditional form of the poem")

# Length selection with radio buttons
length = st.radio("Poem Length", 
                ["short", "medium", "long"],
                index=1,
                horizontal=True,
                help="Short ~10 lines, Medium ~20 lines, Long ~30 lines")

# Generate button
generate_button = st.button("Generate Poem", type="primary", use_container_width=True)

# Display previous poem if available
if st.session_state['generated_poem'] and not generate_button:
    st.markdown("### Your Generated Poem")
    
    meta = st.session_state['poem_metadata']
    
    # Display metadata
    meta_col1, meta_col2 = st.columns(2)
    
    with meta_col1:
        st.markdown(f"**Theme:** <span class='theme-badge'>{meta['theme']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Emotion:** {meta['emotion']}")
        st.markdown(f"**Length:** {meta['length']}")
    
    with meta_col2:
        st.markdown(f"**Style:** <span class='style-badge'>{meta['style']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Form:** {meta['form']}")
    
    # Display poem in a styled container
    st.markdown("<div class='poetry-container'><div class='poetry-text'>" + 
                st.session_state['generated_poem'].replace('\n', '<br>') + 
                "</div></div>", unsafe_allow_html=True)
    
    # Save poem option
    poem_title = st.text_input("Poem Title", value="My Poem")
    save_poem(st.session_state['generated_poem'], poem_title)

# Generate poem when button is clicked
if generate_button:
    # Validate inputs
    if not theme:
        theme = "nature and the passing of time"
    if not emotion:
        emotion = "contemplative"
    
    if not api_key:
        st.error("Please enter your HuggingFace API token in the sidebar.")
    else:
        with st.spinner(f"Crafting your {length} {form} poem..."):
            try:
                # Generate poem
                poem = generate_poem(
                    api_key,
                    theme,
                    emotion,
                    style,
                    form,
                    length,
                    temperature,
                    model_name
                )
                
                # Store in session state
                st.session_state['generated_poem'] = poem
                st.session_state['poem_metadata'] = {
                    'theme': theme,
                    'emotion': emotion,
                    'style': style,
                    'form': form,
                    'length': length
                }
                
                # Force a rerun to display the poem
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating poem: {str(e)}")
                st.info("Try adjusting your parameters or checking your API key.")
                st.code(traceback.format_exc(), language='python')

# Information section
with st.expander("About the Poetry Generator"):
    st.markdown("""
    ### How It Works
    
    This app uses HuggingFace language models to generate beautiful poetry based on your input parameters:
    
    - **Theme**: The central subject or concept of your poem
    - **Emotion/Mood**: The emotional tone or atmosphere of your poem
    - **Style**: The literary tradition or movement influencing the language
    - **Form**: The structural pattern or traditional form of the poem
    - **Length**: How long the poem should be
    
    ### Length Options
    
    - **Short**: Approximately 10 lines
    - **Medium**: Approximately 20 lines
    - **Long**: Approximately 30 lines
    
    ### Poetry Forms
    
    - **Free Verse**: Poetry without regular rhythm or rhyme patterns
    - **Sonnet**: 14-line poem with specific rhyme schemes (often Shakespearean or Petrarchan)
    - **Haiku**: Japanese form with 3 lines (5-7-5 syllables)
    - **Villanelle**: 19-line poem with repeating lines and two rhymes
    - **Ballad**: Narrative poetry with a simple, singable rhythm and rhyme scheme
    - **Limerick**: Five-line humorous poem with AABBA rhyme scheme
    - **Ode**: Formal address to a subject with complex stanza forms
    - **Ghazal**: Persian form with couplets and recurring rhymes
    - **Tanka**: Japanese 5-line form (5-7-5-7-7 syllables)
    - **Sestina**: Complex form with six stanzas of six lines and specific word patterns
    """)

with st.expander("Tips for Better Poetry"):
    st.markdown("""
    ### Creating More Compelling Poetry
    
    1. **Select complementary themes and emotions**: For example, pair "autumn" with "nostalgia" or "ocean" with "wonder".
    
    2. **Match form to content**: Some themes work better with certain forms:
       - Haiku for nature observations
       - Sonnets for love or philosophical questions
       - Free verse for contemporary or complex emotions
       - Ballads for narrative or historical themes
    
    3. **Use temperature wisely**:
       - Higher (0.8-1.0): More creative, surprising language
       - Medium (0.5-0.7): Balanced creativity and coherence
       - Lower (0.2-0.4): More consistent, traditional poetry
    
    4. **Experiment with styles**:
       - Romantic for emotional, nature-based poems
       - Modern for experimental, fragmented imagery
       - Minimalist for stark, essential observations
       - Confessional for personal, intimate reflections
    
    5. **Be specific**: Instead of just "nature" try "autumn forest at dusk" or "coastal tidepools"
    """)

with st.expander("About the Models"):
    st.markdown("""
    ### HuggingFace Endpoint Models
    These models run on HuggingFace's servers via API calls. You need an API token to use them.
    
    - **Mistral-7B-Instruct-v0.3**: Good general-purpose model for poetry.
    - **Mixtral-8x7B-Instruct-v0.1**: Excellent for varied styles and complex forms.
    - **Mistral-Nemo-Instruct-2407**: Strong emotional understanding and imagery.
    - **Qwen2.5-7B-Instruct**: Great for concise, descriptive poetry.
    - **Nous-Hermes-2-Mixtral-8x7B-DPO**: Produces creative, nuanced verse with excellent form adherence.
    
    ### Parameters
    - **Temperature**: Controls creativity level. Higher values (0.8+) create more varied, surprising language.
    - **Provider**: The inference provider to use. "hf-inference" is recommended for most users.
    """)

with st.expander("Requirements"):
    st.code("""
    pip install streamlit
    pip install langchain-huggingface
    pip install huggingface_hub
    pip install langchain
    """)

with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    
    1. **API key errors**: Make sure your HuggingFace API token has the necessary permissions.
    
    2. **Model access**: Some models require explicit approval on HuggingFace before you can use them via API.
    
    3. **Rate limits**: Free API tokens have usage limitations. Consider upgrading if you hit rate limits.
    
    4. **Form adherence**: Complex forms like villanelles or sestinas may not be perfectly executed by the models.
    
    5. **Syllable counts**: While models try to follow syllabic forms like haiku, they may not always count syllables perfectly.
    """)

# Footer
st.markdown("""
<footer>
Built with Streamlit, LangChain, and HuggingFace 🤗 • Created for poetic expression
</footer>
""", unsafe_allow_html=True)
