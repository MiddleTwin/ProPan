import time
import streamlit as st

@st.cache(ttl=30*60)
def get_image(prompt, seed=1):
    time.sleep(5)
    return "ai_pan.png"
