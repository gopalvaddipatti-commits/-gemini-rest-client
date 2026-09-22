import streamlit as st
from groq import Groq
import requests
import json

st.set_page_config(page_title="Conversational API Tester", page_icon="🤖")
st.title("🤖 AI API Testing Assistant")

# Groq API Key সেটআপ (সাইডবার বা সিক্রেট থেকে)
groq_api_key = ""
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.sidebar.text_input("Groq API Key দিন:", type="password")

if not groq_api_key:
    st.warning("চালু করার জন্য দয়া করে সাইডবারে আপনার Groq API Key দিন।")
    st.stop()

client = Groq(api_key=groq_api_key)

# চ্যাট হিস্ট্রি সেভ করার জন্য সেশন স্টেট ইনিশিয়ালাইজ করা
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "হ্যালো! আপনি কোন ওয়েবসাইটের বা সার্ভিসের REST API টেস্ট করতে চান? শুধু সেটির নাম বা ডকসের লিংক দিন।"}
    ]

# আগের সব চ্যাট মেসেজ স্ক্রিনে দেখানো
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ইউজারের ইনপুট নেওয়ার চ্যাট বক্স
if user_input := st.chat_input("এখানে আপনার মেসেজ লিখুন..."):
    # ইউজারের মেসেজ চ্যাটে যোগ করা
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI রেসপন্স জেনারেট করা
    with st.chat_message("assistant"):
        with st.spinner("ভেবে দেখছি..."):
            try:
                # Groq মডেলকে কল করে চ্যাট কনটেক্সট পাঠানো
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful AI API Testing Assistant. Your job is to converse with the user step-by-step. First, ask for the service name. Once they give it, analyze what credentials (Client ID, Secret, API Key) or endpoint details are needed, and ask the user for them conversationally. Guide them like an interactive assistant."
                        }
                    ] + st.session_state.messages,
                    temperature=0.3,
                )
                
                ai_reply = response.choices.message.content
                st.markdown(ai_reply)
                
                # এআই-এর রিপ্লাই হিস্টরিতে সেভ করা
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"এরর হয়েছে: {e}")
