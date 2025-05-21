import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="FINUA", layout="centered")
st.title("🚙 Emobi AIチャット - FINUA")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_store", embedding_function=embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = load_chain()

user_input = st.chat_input("質問をどうぞ")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        result = qa_chain.invoke({"query": user_input})
        st.markdown(result["result"])
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": result["result"]})
