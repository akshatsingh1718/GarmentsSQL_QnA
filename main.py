import streamlit as st
from utils import get_chain

st.title("Akshat Garments Questioning")

question = st.text_input("Question")

if question:
    chain = get_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
