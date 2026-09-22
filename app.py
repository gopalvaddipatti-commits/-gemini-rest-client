import streamlit as st
from groq import Groq
import json

st.set_page_config(page_title="AI REST API Tester", page_icon="🤖", layout="wide")

st.title("🤖 Groq AI Autonomous REST API Tester")
st.write("সার্ভিসের নাম দিন এবং কী করতে চান তা লিখুন। Groq (Llama 3) ব্যাকগ্রাউন্ডে কোড জেনারেট করে API টেস্ট করবে!")

# ১. API Key ম্যানেজমেন্ট (Streamlit Secrets অথবা ইউজার ইনপুট)
groq_api_key = ""
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.sidebar.text_input("আপনার Groq API Key দিন:", type="password")

# মডেল সিলেকশন ড্রপডাউন
model_choice = st.sidebar.selectbox(
    "মডেল সিলেক্ট করুন:",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
)

# ইউজার ইনপুট ফিল্ডস
col1, col2 = st.columns(2)
with col1:
    service_name = st.text_input("সার্ভিস বা সাইটের নাম (যেমন: GitHub, Spotify):", value="GitHub")
with col2:
    user_task = st.text_input("টাস্ক বা রিকোয়েস্টের বিবরণ (যেমন: Get user profile of 'octocat'):", value="Get user profile of octocat")

auth_token = st.text_input("অথেন্টিকেশন টোকেন বা API Key (যদি লাগে):", type="password")

if st.button("Run AI API Test"):
    if not groq_api_key:
        st.error("দয়া করে আপনার Groq API Key প্রদান করুন (Sidebar অথবা Streamlit Secrets-এ)।")
    elif not service_name or not user_task:
        st.warning("দয়া করে সার্ভিস নাম এবং টাস্ক ফিল্ড পূরণ করুন।")
    else:
        with st.spinner(f"Groq ({model_choice}) কাজ করছে... কোড তৈরি করা হচ্ছে..."):
            try:
                # Groq ক্লায়েন্ট ইনিশিয়ালাইজ করা
                client = Groq(api_key=groq_api_key)
                
                # প্রম্পট তৈরি যা মডেলকে পাইথন কোড লিখতে নির্দেশ দেবে
                prompt = f"""
                You are an expert AI Python Developer and API Tester.
                The user wants to test an API for the service: '{service_name}'.
                Task description: '{user_task}'
                Auth Token/Key provided: '{auth_token}' (Include this in headers if provided).
                
                Write a complete, executable Python script using the 'requests' library to perform this API call. 
                Ensure you print the HTTP Status Code and the JSON response clearly using python print statements.
                Return ONLY the executable Python code inside a markdown code block (```python ... ```). Do not add extra text outside the code block.
                """
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_choice,
                    temperature=0.1,
                )
                
                ai_output = chat_completion.choices[0].message.content
                
                st.success("AI কোড জেনারেট সম্পন্ন হয়েছে!")
                
                # জেনারেট হওয়া কোড স্ক্রিনে দেখানো
                st.markdown("### 📝 AI Generated Python Code:")
                st.markdown(ai_output)
                
                # ভবিষ্যতে এখানে আপনি কোড রান করার অংশ (exec বা subprocess) যুক্ত করতে পারেন
                
            except Exception as e:
                st.error(e)
