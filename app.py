import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="CSV Data Analysis Agent", layout="wide")

# Initialize session state for the agent
if 'csv_agent' not in st.session_state:
    st.session_state.csv_agent = None

def initialize_agent():
    """Initialize the CSV agent with Groq LLM"""
    try:
        # Get API key from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # Set the API key for the session
        os.environ["GROQ_API_KEY"] = api_key
            
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        # Create CSV agent
        csv_agent = create_csv_agent(
            llm,
            "data.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )
        
        return csv_agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def main():
    st.title("📊 Vikas Group Demo")
    
    # Initialize agent on startup
    if st.session_state.csv_agent is None:
        st.session_state.csv_agent = initialize_agent()
    
    # Display data preview
    try:
        df = pd.read_csv("data.csv")
        st.subheader("Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return
    
    # Preset question buttons
    st.subheader("Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("How many individual purchased sedan?"):
            if st.session_state.csv_agent:
                with st.spinner("Analyzing..."):
                    response = st.session_state.csv_agent("How many individual purchased sedan?")
                st.success(response)
            else:
                st.error("Failed to initialize the agent. Please check if GROQ_API_KEY is set correctly in .env file")
    
    with col2:
        if st.button("What is the price of sedan in Gurgaon?"):
            if st.session_state.csv_agent:
                with st.spinner("Analyzing..."):
                    response = st.session_state.csv_agent("What is the price of sedan in Gurgaon?")
                st.success(response)
            else:
                st.error("Failed to initialize the agent. Please check if GROQ_API_KEY is set correctly in .env file")
    
    with col3:
        if st.button("Total price of cars purchased by corporates?"):
            if st.session_state.csv_agent:
                with st.spinner("Analyzing..."):
                    response = st.session_state.csv_agent("what is the total price value of cars purchased by corporates?")
                st.success(response)
            else:
                st.error("Failed to initialize the agent. Please check if GROQ_API_KEY is set correctly in .env file")
    
    # Custom question input
    st.subheader("Ask Your Own Question")
    custom_question = st.text_input("Enter your question about the data:")
    if custom_question:
        if st.session_state.csv_agent:
            with st.spinner("Analyzing..."):
                response = st.session_state.csv_agent(custom_question)
            st.success(response)
        else:
            st.error("Failed to initialize the agent. Please check if GROQ_API_KEY is set correctly in .env file")

if __name__ == "__main__":
    main()
