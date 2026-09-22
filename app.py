import streamlit as st
from groq import Groq
from pymongo import MongoClient

st.set_page_config(page_title="AI API Tester with Database", page_icon="🤖", layout="centered")
st.title("🤖 AI API Testing Assistant (with MongoDB)")

# ১. MongoDB কানেকশন সেটআপ (Streamlit Secrets অথবা সাইডবার থেকে)
mongo_uri = ""
try:
    mongo_uri = st.secrets["MONGO_URI"]
except:
    mongo_uri = st.sidebar.text_input("MongoDB URI দিন:", type="password")

# ২. Groq API Key সেটআপ
groq_api_key = ""
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.sidebar.text_input("Groq API Key দিন:", type="password")

if not mongo_uri or not groq_api_key:
    st.warning("চালু করার জন্য দয়া করে সাইডবারে আপনার MongoDB URI এবং Groq API Key দিন।")
    st.stop()

try:
    client_db = MongoClient(mongo_uri)
    db = client_db["ai_api_tester"]
    chats_collection = db["chat_history"]
except Exception as e:
    st.error(f"ডাটাবেস কানেকশনে সমস্যা হয়েছে: {e}")
    st.stop()

client_groq = Groq(api_key=groq_api_key)

# ৩. ডাটাবেস থেকে আগের চ্যাট হিস্ট্রি লোড করা
if "messages" not in st.session_state:
    st.session_state.messages = []
    # ডাটাবেস থেকে আগের চ্যাটগুলো ফেচ করা
    saved_chats = chats_collection.find().sort("_id", 1)
    for chat in saved_chats:
        st.session_state.messages.append({"role": chat["role"], "content": chat["content"]})
    
    # যদি একদম নতুন হয়, তবে ওয়েলকাম মেসেজ দেওয়া
    if not st.session_state.messages:
        initial_msg = "হ্যালো! আপনি কোন ওয়েবসাইটের বা সার্ভিসের REST API টেস্ট করতে চান? শুধু সেটির নাম দিন।"
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        chats_collection.insert_one({"role": "assistant", "content": initial_msg})

# ৪. চ্যাট মেসেজগুলো স্ক্রিনে দেখানো
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ৫. ইউজারের ইনপুট নেওয়া এবং সেভ করা
if user_input := st.chat_input("এখানে আপনার মেসেজ লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    chats_collection.insert_one({"role": "user", "content": user_input}) # MongoDB তে সেভ
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # ৬. AI রেসপন্স জেনারেট করা
    with st.chat_message("assistant"):
        with st.spinner("ভেবে দেখছি... মডেল কাজ করছে..."):
            try:
                response = client_groq.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are a helpful AI API Testing Assistant. "
                                "Your job is to converse with the user step-by-step. "
                                "First, ask for the service name. Once they give it, "
                                "analyze what credentials (Client ID, Secret, API Key) or endpoint details are needed, "
                                "and ask the user for them conversationally. Guide them like an interactive assistant."
                            )
                        }
                    ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    temperature=0.3,
                )
                
                ai_reply = response.choices[0].message.content
                st.markdown(ai_reply)
                
                # AI এর রিপ্লাই স্ক্রিনে দেখানো এবং MongoDB তে সেভ করা
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                chats_collection.insert_one({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"এরর হয়েছে: {e}")
