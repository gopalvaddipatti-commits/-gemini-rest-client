import streamlit as st
from groq import Groq
import requests
import json
import io
import sys

st.set_page_config(page_title="Autonomous API Testing Agent", page_icon="🤖", layout="centered")
st.title("🤖 Autonomous REST API Testing Agent")

# Groq API Key সেটআপ
groq_api_key = ""
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.sidebar.text_input("Groq API Key দিন:", type="password")

if not groq_api_key:
    st.warning("অ্যাপটি ব্যবহার করতে অনুগ্রহ করে সাইডবারে আপনার Groq API Key দিন।")
    st.stop()

# ফিক্সড এবং নিশ্চিত চ্যাট মডেল (যাতে ক্লাসিফিকেশন মডেলের এরর আর না আসে)
ACTIVE_MODEL = "llama-3.1-8b-instant"

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Groq কানেকশন তৈরি করতে সমস্যা হয়েছে: {e}")
    st.stop()

# চ্যাট হিস্ট্রি ইনিশিয়ালাইজ করা
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "হ্যালো! আমি আপনার Autonomous API Tester। আপনি কোন সার্ভিসের API টেস্ট করতে চান? (যেমন: GitHub ইত্যাদি এবং আপনার ক্রেডেনশিয়াল বা টোকেন দিন)"}
    ]

# চ্যাট হিস্ট্রি স্ক্রিনে দেখানো
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ইউজার ইনপুট
if user_input := st.chat_input("এখানে আপনার মেসেজ লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI রেসপন্স এবং API এক্সিকিউশন
    with st.chat_message("assistant"):
        with st.spinner("AI চ্যাট প্রসেস করছে এবং API টেস্ট কোড তৈরি করছে..."):
            try:
                system_prompt = (
                    "You are an expert Autonomous API Testing Agent. "
                    "Converse with the user step-by-step to get the service name and required credentials/headers. "
                    "Once you have enough information to test the API, write a clean, executable Python script "
                    "using the 'requests' library to make the API call. "
                    "You MUST wrap the Python executable code inside a standard markdown code block using ```python and ```. "
                    "Inside the code, print the HTTP Status Code and the response text/JSON clearly using print statements. "
                    "If you still need info (like API key or endpoint), just ask the user conversationally without code."
                )

                # Groq চ্যাট মডেলের জন্য সঠিক মেসেজ লিস্ট তৈরি (সিস্টেম প্রম্পট + ইউজার মেসেজগুলো)
                api_messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    # অ্যাসিস্ট্যান্টের প্রথম ওয়েলকাম মেসেজটি এপিআই কলের বাইরে রাখা হলো যাতে সিকোয়েন্স ঠিক থাকে
                    if msg["role"] == "assistant" and msg["content"].startswith("হ্যালো! আমি আপনার Autonomous"):
                        continue
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

                response = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=api_messages,
                    temperature=0.2,
                )
                
                ai_reply = response.choices[0].message.content
                st.markdown(ai_reply)
                
                # চেক করা যে AI কোড জেনারেট করেছে কিনা API টেস্ট করার জন্য
                if "```python" in ai_reply:
                    try:
                        code_start = ai_reply.find("```python") + 9
                        code_end = ai_reply.find("```", code_start)
                        python_code = ai_reply[code_start:code_end].strip()
                        
                        st.info("🔄 ব্যাকগ্রাউন্ডে লাইভ API রিকোয়েস্ট পাঠানো হচ্ছে...")
                        
                        # পাইথন কোড সেফলি রান করা এবং আউটপুট ক্যাপচার করা
                        old_stdout = sys.stdout
                        new_stdout = io.StringIO()
                        sys.stdout = new_stdout
                        
                        exec(python_code, {"requests": requests, "json": json})
                        
                        sys.stdout = old_stdout
                        execution_output = new_stdout.getvalue()
                        
                        # রেজাল্ট চ্যাটে দেখানো
                        st.markdown("### 📊 API Execution Result:")
                        st.code(execution_output, language="text")
                        
                        ai_reply += f"\n\n### Execution Output:\n```text\n{execution_output}\n```"
                        
                    except Exception as exec_err:
                        sys.stdout = sys.stdout if 'old_stdout' not in locals() else old_stdout
                        st.error(f"API কোড রান করার সময় এরর হয়েছে: {exec_err}")
                
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"এরর হয়েছে: {e}")
