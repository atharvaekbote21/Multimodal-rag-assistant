import streamlit as st
from rag import ask_rag

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖"
)

st.title("🤖 Multimodal RAG Assistant")
st.write("Ask questions that are related to the document you have uploaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask something about the report...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            result = ask_rag(question)

            answer = result["answer"]

            st.write(answer)

            if result.get("image_paths"):
                st.write("### Retrieved Visuals")

                for image_path in result["image_paths"]:
                    st.image(image_path)

        except Exception as e:
            st.error(f"Error: {e}")