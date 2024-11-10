import os
import streamlit as st
import time
from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Set page configuration
st.set_page_config(
    page_title="Spotify Google Store Review Analysis Bot",
    page_icon="🎵",
    layout="wide"
)

# Initialize models and vectorstore
@st.cache_resource
def initialize_chain():
    """Initialize the RAG chain and its components"""
    # Instantiate embeddings model
    embeddings_model = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY, 
        model='text-embedding-ada-002', 
        max_retries=100, 
        chunk_size=16, 
        show_progress_bar=False
    )
    
    # Instantiate chat model
    chat_model = ChatOpenAI(
        api_key=OPENAI_API_KEY, 
        temperature=0.5, 
        model='gpt-4o-mini'
    )
    
    # Load chroma from disk
    vectorstore = Chroma(
        persist_directory="chroma_data", 
        embedding_function=embeddings_model
    )
    
    # Set up the vectorstore retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Define the review template
    review_template_str = """You are an advanced Q&A assistant specialized in analyzing user reviews for a music streaming application
    similar to Spotify. Your role is to extract meaningful and actionable insights from a large dataset of 3.4 million Google Store reviews.
    These insights should help management understand what users like, dislike, compare, and suggest about the application.

    Your responses should be accurate, clear, and concise, directly addressing the management's inquiries. Ensure your answers cover the
    following aspects:
    1. Positive aspects and features users appreciate most.
    2. Comparisons made with competitor apps (e.g., Pandora).
    3. Common complaints and areas of dissatisfaction among users.
    4. Emerging trends or patterns that could influence product strategy.

    In addition to your answers, provide a brief confidence score (out of 10) for each response to indicate the answer's reliability.

    Example responses to management questions might look like:
    - "What are the features users appreciate most?"
      "Users often praise the intuitive UI design, extensive variety of music options, and seamless listening experience."

    - "Which platforms do users compare us with most?"
      "Users frequently compare the application with Pandora, especially regarding feature availability and user experience."

    {context}
    """

    # Create prompt templates
    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template_str,
        )
    )

    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )

    messages = [review_system_prompt, review_human_prompt]

    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    # Create RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the RAG chain
rag_chain = initialize_chain()

def generate_answer(message, history):
    """Generate answer using the RAG chain"""
    return rag_chain.invoke(message)['answer']


def initialize_ui():
    """Initialize the UI components"""
    st.title("🎵 Spotify Google Store Review Analysis Bot")
    st.markdown("""
    Welcome to the Music App Review Analysis Assistant! Ask questions about user reviews and get insights
    about what users like, dislike, and compare about the music streaming application.
    """)
    
    # Add a sidebar with example questions
    st.sidebar.title("Example Questions")
    st.sidebar.markdown("""
    Try asking questions like:
    - What are the most common complaints about the app?
    - What features do users love the most?
    - How does our app compare to competitors?
    - What are users saying about the premium subscription?
    """)

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    initialize_ui()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the spotify reviews..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            cursor = "|"
            with st.spinner("Answering..."):
                response = generate_answer(prompt, st.session_state.messages)
                
                # Type out each word with a blinking cursor
                for word in response.split():
                    full_response += word + " "
                    message_placeholder.markdown(full_response + cursor)
                    time.sleep(0.1)  # Adjust the delay as needed for typing speed
                
                # Clear the cursor after the response is fully typed
                message_placeholder.markdown(full_response.strip())
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()