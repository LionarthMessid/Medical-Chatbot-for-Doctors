import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load and cache the FAISS vector store"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Load the LLM with enhanced configuration for detailed responses"""
    model_kwargs = {
        "temperature": 0.7,  # Increased for more creative/detailed responses
        "timeout": 10000,    # Increased timeout for longer responses
        "do_sample": True,   # Enable sampling for more varied responses
        "max_new_tokens": 568,  # Significantly increased for longer answers
        "top_p": 0.9,        # Nucleus sampling for better quality
        "top_k": 50,         # Top-k sampling
        "repetition_penalty": 1.1,  # Prevent repetition
    }
    
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        **model_kwargs
    )
    
    # Wrap with ChatHuggingFace for better compatibility
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def clean_markdown(text):
    # Remove triple backticks
    text = text.replace('```', '')
    # Replace <br> and <br/> with newlines
    text = text.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    # Remove leading spaces from each line
    lines = text.split('\n')
    cleaned_lines = [line.lstrip() for line in lines]
    return '\n'.join(cleaned_lines)

def main():
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Chatbot")
    st.markdown("Ask questions about medical topics based on The Gale Encyclopedia of Medicine")

    # Initialize session state for message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Enhanced prompt template for detailed responses
        CUSTOM_PROMPT_TEMPLATE = """
        You are a knowledgeable medical assistant with access to The Gale Encyclopedia of Medicine. 
        Provide comprehensive, detailed, and well-structured answers based on the given context.
        
        Instructions:
        1. Provide thorough explanations with multiple aspects of the topic
        2. Include relevant medical terminology with clear explanations
        3. Structure your response with clear sections when appropriate
        4. Include causes, symptoms, treatments, and complications if relevant
        5. Mention any important medical considerations or warnings
        6. If the context covers multiple related topics, discuss them comprehensively
        7. if asked for full detailed answer, provide it.Unless asked, provide brief answers.
        8. If you don't know something specific, clearly state what you don't know
        9. Don't make up information not in the context
        10. Aim for detailed, educational responses that would be helpful to someone seeking medical information

        Context Information:
        {context}

        Question: {question}

        Please provide a comprehensive and detailed response:
        """
        
        # Use a model that supports longer, more detailed responses
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Changed to a more reliable model
        HF_TOKEN = os.environ.get("HF_TOKEN")

        # Show loading spinner
        with st.spinner("Searching medical knowledge base and generating detailed response..."):
            try:
                # Load vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the medical knowledge base. Please check if the vector store exists.")
                    return

                # Create QA chain with enhanced retrieval
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(
                        search_kwargs={
                            'k': 5,  # Increased from 3 to get more context
                            'fetch_k': 10,  # Fetch more candidates before filtering
                        }
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                # Get response
                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format the response with additional information
                result_to_show = result
                
                # Add source information if available
                if source_documents:
                    result_to_show += "\n\n---\n**References:**\n"
                    for i, doc in enumerate(source_documents[:5], 1):
                        # Get a short section summary (first sentence or 80 chars)
                        section = doc.metadata.get('section', None)
                        if not section:
                            section = doc.page_content.strip().split('\n')[0][:80]
                        page = doc.metadata.get('page', None)
                        source = doc.metadata.get('source', 'Medical Encyclopedia')
                        # Format as a formal citation
                        if page is not None:
                            result_to_show += f"[{i}] {section}, in {source}, p. {page}.\n"
                        else:
                            result_to_show += f"[{i}] {section}, in {source}.\n"

                # Clean up the answer to avoid code block rendering
                result_to_show = clean_markdown(result_to_show)

                # Display assistant response
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This enhanced medical chatbot provides:
        - **Detailed Responses**: Comprehensive answers with multiple aspects
        - **Medical Terminology**: Clear explanations of complex terms
        - **Structured Information**: Organized sections for easy reading
        - **Multiple Sources**: Uses up to 5 relevant sources per query
        - **Knowledge Base**: The Gale Encyclopedia of Medicine
        
        
        Ask questions about medical conditions, treatments, and procedures for detailed explanations!
        """)
      
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()