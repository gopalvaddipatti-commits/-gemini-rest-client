import re
import requests
import streamlit as st
import google.generativeai as genai

# Page Config
st.set_page_config(
    page_title="API REST Client Assistant", page_icon="🤖", layout="centered"
)

st.title("API REST Client Assistant 🤖")
st.write(
    "Ami apnar API documentation link pore, apnake authentications (API Key/Client"
    " ID) niye question korbo ebong REST client setup kore debo!"
)

# Sidebar for Gemini API Key
with st.sidebar:
  st.header("⚙️ Configuration")
  gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
  st.markdown(
      "[Get your free Gemini API"
      " Key](https://aistudio.google.com/app/apikey)"
  )

# Check if API key is provided
if not gemini_api_key:
  st.warning(
      "Onugroho kore age sidebar theke apnar Gemini API Key-ti din."
  )
  st.stop()

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# System Instructions for Gemini
system_instruction = """
You are an expert AI API Assistant and REST Client Builder. 
Your task is to help developers set up REST clients (like VS Code .http format or cURL) based on API documentation links they provide.

Rules:
1. When the user provides a documentation link, analyze the documentation content provided in the context.
2. Identify what authentication method is required (e.g., API Key, Client ID, Client Secret, Bearer Token) and the main endpoints.
3. Conversationally ask the user to provide the required credentials (API Key, Client ID, etc.). Do not assume keys; ask the user for them.
4. Once the user provides the credentials, generate a complete, ready-to-use REST client request (in VS Code .http format or cURL) using their actual credentials.
5. Be helpful, concise, and developer-friendly. Reply in Bengali or English based on how the user speaks.
"""

# Initialize Gemini Model with System Instruction (Updated model name)
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro", system_instruction=system_instruction
)

# Initialize Chat Session in Streamlit Session State
if "chat_session" not in st.session_state:
  st.session_state.chat_session = model.start_chat(history=[])

# Initialize Chat History for UI Display
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])


# Helper function to fetch documentation from URL using Jina Reader API
def fetch_docs_from_url(text):
  # Find URL in text
  urls = re.findall(
      r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
      text,
  )
  if urls:
    target_url = urls[0]
    # Using Jina AI Reader to convert website docs to clean markdown
    jina_url = f"https://r.jina.ai/{target_url}"
    try:
      res = requests.get(jina_url)
      if res.status_code == 200:
        return (
            target_url,
            f"\n\n[System Auto-Fetched Documentation from {target_url}]:\n"
            + res.text[:12000],
        )
    except Exception as e:
      return target_url, f"\n\n[Error fetching URL content: {e}]"
  return None, ""


# User Input Box
if prompt := st.chat_input("Documentation link din ba kono API niye kotha bolun..."):
  # Display user message
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Check if user gave a link and fetch it automatically
  url, fetched_content = fetch_docs_from_url(prompt)
  full_prompt = prompt + fetched_content

  # Send message to Gemini chat session
  with st.chat_message("assistant"):
    with st.spinner("AI documentation analyse korche..."):
      try:
        response = st.session_state.chat_session.send_message(full_prompt)
        ai_response = response.text
        st.markdown(ai_response)
        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response}
        )
      except Exception as e:
        error_msg = f"Kono somoshya hoyeche: {e}"
        st.error(error_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
system_instruction = """
You are an expert AI API Assistant and REST Client Builder. 
Your task is to help developers set up REST clients (like VS Code .http format or cURL) based on API documentation links they provide.

Rules:
1. When the user provides a documentation link, analyze the documentation content provided in the context.
2. Identify what authentication method is required (e.g., API Key, Client ID, Client Secret, Bearer Token) and the main endpoints.
3. Conversationally ask the user to provide the required credentials (API Key, Client ID, etc.). Do not assume keys; ask the user for them.
4. Once the user provides the credentials, generate a complete, ready-to-use REST client request (in VS Code .http format or cURL) using their actual credentials.
5. Be helpful, concise, and developer-friendly. Reply in Bengali or English based on how the user speaks.
"""

# Initialize Gemini Model with System Instruction
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", system_instruction=system_instruction
)

# Initialize Chat Session in Streamlit Session State
if "chat_session" not in st.session_state:
  st.session_state.chat_session = model.start_chat(history=[])

# Initialize Chat History for UI Display
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])


# Helper function to fetch documentation from URL using Jina Reader API
def fetch_docs_from_url(text):
  # Find URL in text
  urls = re.findall(
      r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
      text,
  )
  if urls:
    target_url = urls[0]
    # Using Jina AI Reader to convert website docs to clean markdown
    jina_url = f"https://r.jina.ai/{target_url}"
    try:
      res = requests.get(jina_url)
      if res.status_code == 200:
        return (
            target_url,
            f"\n\n[System Auto-Fetched Documentation from {target_url}]:\n"
            + res.text[:12000],
        )
    except Exception as e:
      return target_url, f"\n\n[Error fetching URL content: {e}]"
  return None, ""


# User Input Box
if prompt := st.chat_input("Documentation link din ba kono API niye kotha bolun..."):
  # Display user message
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Check if user gave a link and fetch it automatically
  url, fetched_content = fetch_docs_from_url(prompt)
  full_prompt = prompt + fetched_content

  # Send message to Gemini chat session
  with st.chat_message("assistant"):
    with st.spinner("AI documentation analyse korche..."):
      try:
        response = st.session_state.chat_session.send_message(full_prompt)
        ai_response = response.text
        st.markdown(ai_response)
        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response}
        )
      except Exception as e:
        error_msg = f"Kono somoshya hoyeche: {e}"
        st.error(error_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
