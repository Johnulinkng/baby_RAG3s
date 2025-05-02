import streamlit as st
import asyncio
from agent import main as agent_main
import sys
from io import StringIO
import contextlib
import os
from math_mcp_embeddings import process_documents
import requests
from urllib.parse import urlparse
import hashlib
import tempfile

st.set_page_config(
    page_title="NanAI - your baby's AI nanny",
    page_icon="👶",
    layout="wide"
)

st.title("👶 NanAI - your baby's AI nanny")
st.markdown("""
This assistant can help you with various baby care questions and concerns. 
Ask anything about baby care, safety, development, product information, or parenting advice.
""")

# Add file upload functionality
st.markdown("### Add to my knowledge base")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a document to add to my knowledge base", type=['txt', 'pdf', 'docx'])
    if uploaded_file is not None:
        # Save the file
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, "documents", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the new document
        process_documents()
        
        st.success("Thanks for providing the file! It is now permanently part of my memory and I am ready to answer your questions based on it.")

with col2:
    url = st.text_input("Or enter a URL to add to my knowledge base")
    if url:
        try:
            # Download and process HTML content
            response = requests.get(url)
            response.raise_for_status()
            
            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)+".html"
            if not filename:
                filename = f"webpage_{hashlib.md5(url.encode()).hexdigest()[:8]}.html"
            
            # Save the HTML content
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, "documents", filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Process the new document
            process_documents()
            
            st.success("Thanks for providing the URL! The content is now part of my memory and I am ready to answer your questions based on it.")
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")

st.markdown("---")  # Add a separator line

# Add example questions as buttons in the main window
st.markdown("### Example Questions")
example_questions = [
    "What is the weight limit for baby bath tub sling?",
    "What should I do in case of labour pain?",
    "My baby has a fever, what should I do?",
    "What is the ideal temperature for baby to sleep in celsius?"
]

    # "When do I switch baby from infant car seat to booster seat?",
    # "My baby is 7.2 kg, how should I use the baby bath tub?"

# Create columns for the example questions
cols = st.columns(2)
for i, question in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(question, key=question, use_container_width=True):
            # Add user message to chat history without displaying it
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get the response
            response = asyncio.run(agent_main(question))
            
            # Add assistant response to chat history without displaying it
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to show the new messages in the chat
            st.rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Get the final answer directly from the agent
        response = asyncio.run(agent_main(prompt)).strip('[]')
        
        message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 