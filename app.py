import re
import requests
import streamlit as st
import google.generativeai as genai

# Page Config
st.set_page_config(
    page_title="Smart API REST Assistant", page_icon="🤖", layout="centered"
)

st.title("Smart API REST Assistant 🤖")
st.write(
    "Amake jekono API-er nam ba link din. Ami documentation khuje ber korbo,"
    " REST client setup korbo, ebong apnar API Key/Secret valid kina check kore"
    " debo!"
)

# Sidebar for Gemini API Key
with st.sidebar:
  st.header("⚙️ Configuration")
  gemini_api_key = st.text_input(
      "Enter Gemini API Key", type="password", key="gemini_key_input"
  )
  st.markdown(
      "[Get your free Gemini API"
      " Key](https://aistudio.google.com/app/apikey)"
  )

# Check if API key is provided
if not gemini_api_key:
  st.warning("Onugroho kore age sidebar theke apnar Gemini API Key-ti din.")
  st.stop()

# Configure using standard google-generativeai package
genai.configure(api_key=gemini_api_key)

# System Instructions
system_instruction = """
You are an expert AI API Assistant, Documentation Finder, and REST Client Builder. 
Your tasks:
1. If the user provides just a service name (e.g., "Stripe", "GitHub", "Twitter API") instead of a link, automatically identify its official documentation URL or provide the endpoint details from your knowledge base.
2. When documentation is provided, analyze the authentication method (API Key, Client ID, Client Secret, Bearer Token).
3. Conversationally ask the user for their API Key, Client ID, or Secret.
4. When the user provides their credentials, analyze their format, validate if they match the expected pattern of that specific service (e.g., proper prefix, correct length), and simulate or explain a validation test status (Valid/Invalid/Active).
5. Generate a complete, ready-to-use REST client request (in VS Code .http format or cURL) using their credentials.
6. Be helpful, concise, and reply in Bengali or English based on how the user speaks.
"""

# Using the exact model recommended by the API error message
model = genai.GenerativeModel(
    model_name="gemini-3.6-flash", system_instruction=system_instruction
)

# Initialize Chat Session State
if "chat_session" not in st.session_state:
  st.session_state.chat_session = model.start_chat(history=[])

if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])


# Helper function to fetch documentation from URL using Jina Reader API
def fetch_docs_from_url(text):
  urls = re.findall(
      r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
      text,
  )
  if urls:
    target_url = urls[0]
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
if prompt := st.chat_input(
    "Service er nam ba documentation link din...", key="unique_chat_input"
):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  url, fetched_content = fetch_docs_from_url(prompt)
  full_prompt = prompt + fetched_content

  with st.chat_message("assistant"):
    with st.spinner("AI response generate korche..."):
      try:
        response = st.session_state.chat_session.send_message(full_prompt)
        ai_response = response.text
        st.markdown(ai_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response}
        )
      except Exception as e:
        error_msg = f"Kono somoshya hoyeche: {e}"
        st.error(error_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
