import streamlit as st
import os
import pypdf

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# --- UI Configuration ---
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG System")
st.write("An autonomous agent that selects tools, researches information, and synthesizes answers.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration ⚙️")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key (Optional)", type="password")
    
    model_id = st.selectbox("OpenRouter Model", [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-pro-1.5"
    ])
    
    st.markdown("### Upload Documents")
    uploaded_pdf = st.file_uploader("Upload PDF (Injected as context)", type=["pdf"])
    
    st.markdown("### Available Tools")
    st.markdown("- 🌐 **Tavily:** Real-time web search\n- 📚 **Wikipedia:** General knowledge\n- 🎓 **arXiv:** Academic papers")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you research today?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper Function: PDF Extraction ---
def extract_pdf_context(file):
    try:
        reader = pypdf.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        # Limit to avoid token overflow issues, adjusting depending on model capability
        return text[:15000] 
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & Agent Execution ---
if prompt := st.chat_input("Ask a question (e.g., 'What are the latest AI models?')..."):
    if not openrouter_api_key:
        st.error("⚠️ Please provide an OpenRouter API key in the sidebar.")
        st.stop()
        
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Initialize Tools
        tools = []
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            tools.append(TavilySearchResults(max_results=3, description="Use this to search the web for current events, news, or latest information."))
        else:
            st.info("ℹ️ Tavily API key missing. Running without Web Search tool.")
            
        tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), description="Use this to search Wikipedia for definitions, history, and encyclopedic knowledge."))
        tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(), description="Use this to search for academic and scientific research papers."))

        # 2. Setup OpenRouter LLM using LangChain's OpenAI wrapper
        llm = ChatOpenAI(
            model=model_id,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )

        # 3. Setup System Prompt & PDF Context
        sys_msg = """You are an intelligent, autonomous research agent. 
        You have access to tools for web searches, Wikipedia, and arXiv papers.
        Instructions:
        - Autonomously decide whether to use tools or answer directly.
        - You can execute multiple tools sequentially if your first search doesn't return enough context.
        - If answering simple logic, math, or greeting questions, do not use tools.
        - ALWAYS append source URLs or Wikipedia/arXiv reference identifiers at the end of your response when citing tool-retrieved information.
        """
        
        if uploaded_pdf:
            pdf_text = extract_pdf_context(uploaded_pdf)
            sys_msg += f"\n\nThe user has uploaded a PDF. Here is its extracted text context to help answer questions:\n<pdf_context>\n{pdf_text}\n</pdf_context>"

        # 4. Construct Agent
        agent = create_agent(llm, tools=tools, system_prompt=sys_msg)

        # 5. Execute Agent with UI visibility
        with st.status("🧠 Agent actively thinking & routing...", expanded=True) as status:
            try:
                response = agent.invoke(
                    {"messages": [*st.session_state.chat_history, HumanMessage(content=prompt)]},
                    config={"recursion_limit": 12},
                )
                
                # Render tool execution traces seamlessly
                tool_calls = [
                    tool_call
                    for message in response["messages"]
                    for tool_call in getattr(message, "tool_calls", [])
                ]
                tool_results = {
                    message.tool_call_id: message
                    for message in response["messages"]
                    if isinstance(message, ToolMessage)
                }
                if not tool_calls:
                    st.write("✅ Answered directly by LLM (No tools called).")
                else:
                    for tool_call in tool_calls:
                        observation = tool_results.get(tool_call["id"])
                        result_text = observation.content if observation else "No result returned."
                        with st.expander(f"🛠️ Tool Executed: `{tool_call['name']}`", expanded=False):
                            st.markdown(f"**Query parameters:** `{tool_call['args']}`")
                            st.markdown(f"**Retrieved data:**\n```\n{result_text}\n```")
                            
                status.update(label="Response generated successfully!", state="complete", expanded=False)
                
                final_answer = response["messages"][-1].content
                
                # Save into memory
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                # Display output
                st.markdown(final_answer)

            except Exception as e:
                error_msg = f"An error occurred during agent execution: {str(e)}"
                status.update(label="Error", state="error", expanded=False)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})