import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

# Load env (local only)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Firebase config from Streamlit secrets
firebase_config = json.loads(st.secrets["FIREBASE_CONFIG"])

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Firebase Admin SDK
if not firebase_admin._apps:
    firebase_creds = json.loads(st.secrets["FIREBASE_ADMIN"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# UI config
st.set_page_config(page_title="PDF Student Assistant", layout="centered", page_icon="🔍")

# Styling
st.markdown("""
<style>
.main { background-color: #f5f6fa; }
.stButton button {
    background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px;
}
.message {
    border-radius: 8px; padding: 10px; margin: 10px 0;
}
.user-msg { text-align: right; color: black; }
.bot-msg { text-align: left; color: #2c3e50; }
.chat-container {
    max-height: 450px; overflow-y: auto; padding: 10px; margin-bottom: 70px;
}
.input-bar {
    position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f9f9f9;
    padding: 10px 20px; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); z-index: 999;
}
.input-bar input {
    width: 80%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 16px;
}
.input-bar button {
    padding: 10px 20px; border-radius: 8px; background-color: #4CAF50; color: white;
    font-size: 16px; border: none;
}
.user-info {
    position: fixed; top: 10px; right: 20px; background-color: #f0f2f6;
    padding: 8px 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-size: 14px; z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "answer is not available in the context". Do not guess.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db_vector.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Bot", response["output_text"]))
    save_chat_to_firebase(st.session_state.user['localId'], st.session_state.get("pdf_name", "General"))

def summarize_document_and_respond(text):
    prompt = PromptTemplate(input_variables=["text"], template="Summarize this for students:\n\n{text}\n\nSummary:")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text=text)
    st.session_state.chat_history.append(("You", "Summarize the document"))
    st.session_state.chat_history.append(("Bot", summary))
    save_chat_to_firebase(st.session_state.user['localId'], st.session_state.get("pdf_name", "General"))

def extract_concepts_and_respond(text):
    prompt = PromptTemplate(input_variables=["text"], template="List key topics from:\n\n{text}\n\nConcepts:")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    chain = LLMChain(llm=llm, prompt=prompt)
    concepts = chain.run(text=text)
    st.session_state.chat_history.append(("You", "List key concepts"))
    st.session_state.chat_history.append(("Bot", concepts))
    save_chat_to_firebase(st.session_state.user['localId'], st.session_state.get("pdf_name", "General"))

def save_chat_to_firebase(user_id, pdf_name="General"):
    ref = db.collection("chats").document(user_id).collection("pdfs").document(pdf_name)
    ref.set({
        "chat_history": [{"speaker": s, "message": m} for s, m in st.session_state.chat_history],
        "timestamp": firestore.SERVER_TIMESTAMP
    })

def generate_pdf(chat_history):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    text_obj = p.beginText(40, 750)
    text_obj.setFont("Helvetica", 12)
    for speaker, message in chat_history:
        for line in f"{speaker}: {message}".split("\n"):
            text_obj.textLine(line)
        text_obj.textLine("")
    p.drawText(text_obj)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def show_login():
    st.title(" Login to PDF Assistant")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = user
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    if st.button("Register"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success(" Registered. Now log in.")
        except Exception as e:
            st.error(f"Registration failed: {e}")

def view_past_chats():
    if st.button(" Back"):
        st.session_state['viewing_past_chats'] = False
        st.experimental_rerun()

    user_id = st.session_state.user['localId']
    chats_ref = db.collection("chats").document(user_id).collection("pdfs")
    docs = chats_ref.stream()

    for doc in docs:
        st.markdown(f"###  {doc.id}")
        for item in doc.to_dict().get("chat_history", []):
            st.markdown(f"**{item['speaker']}:** {item['message']}")
        st.markdown("---")

def main():
    if "user" not in st.session_state:
        show_login()
        return

    user_email = st.session_state.user.get("email", "Unknown User")
    st.markdown(f"<div class='user-info'> {user_email}</div>", unsafe_allow_html=True)

    st.sidebar.title(" Menu")
    st.sidebar.markdown(f" Logged in as: **{user_email}**")
    if st.sidebar.button(" Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.sidebar.button(" View Past Chats"):
        view_past_chats()
        return

    st.title(" Chat with PDF – Gemini AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    pdf_docs = st.file_uploader(" Upload PDF Files", accept_multiple_files=True)

    if st.button(" Submit PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.session_state.raw_text = raw_text
                st.session_state.pdf_name = pdf_docs[0].name
                st.success(" PDFs processed!")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.session_state.raw_text and st.button(" Summarize"):
            summarize_document_and_respond(st.session_state.raw_text)
    with col2:
        if st.session_state.raw_text and st.button(" Key Concepts"):
            extract_concepts_and_respond(st.session_state.raw_text)
    with col3:
        if st.button(" Clear All"):
            st.session_state.chat_history.clear()
            st.success(" Cleared!")

    chat_html = "<div class='chat-container'>"
    for speaker, message in st.session_state.chat_history:
        align = "user-msg" if speaker == "You" else "bot-msg"
        chat_html += f"<div class='message {align}'><strong>{speaker}:</strong> {message}</div>"
    chat_html += "<div id='bottom-scroll'></div></div>"
    st.markdown(chat_html, unsafe_allow_html=True)
    st.markdown("""
        <script>
            var objDiv = document.querySelector('.chat-container');
            if(objDiv) objDiv.scrollTop = objDiv.scrollHeight;
        </script>
    """, unsafe_allow_html=True)

    with st.form("user_input_form", clear_on_submit=True):
        st.markdown("<div class='input-bar'>", unsafe_allow_html=True)
        user_question = st.text_input(" Ask a question from the PDF content:", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)
        if submitted and user_question.strip():
            user_input(user_question)

    if st.session_state.chat_history:
        pdf = generate_pdf(st.session_state.chat_history)
        st.download_button(" Download Chat", data=pdf, file_name="chat_history.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
