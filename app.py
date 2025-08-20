# app.py
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
import operator

# --- LangChain and LangGraph Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient, models

# Load environment variables from the .env file
load_dotenv()

# --- API Key Configuration ---
try:
    # Removed GOOGLE_API_KEY since we're using a local embedding model
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()
    if not OPENWEATHERMAP_API_KEY:
        st.error("OPENWEATHERMAP_API_KEY not found. Please set it in your .env file.")
        st.stop()
    if not QDRANT_URL:
        st.error("QDRANT_URL not found. Please set it in your .env file.")
        st.stop()
    if not QDRANT_API_KEY:
        st.error("QDRANT_API_KEY not found. Please set it in your .env file.")
        st.stop()

except Exception as e:
    st.error(f"Error loading API keys: {e}")
    st.stop()


# --- Initialize Session State for Document Processing ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_document' not in st.session_state:
    st.session_state.processed_document = False


# --- Global Instances for Performance and Async Compatibility ---
# Creating these objects at the top-level avoids re-creation on every rerun
# and resolves the RuntimeError: no current event loop issue.
# Switched to HuggingFaceEmbeddings to avoid async conflicts with Streamlit.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# No need to instantiate the client here, Qdrant.from_documents will do it internally
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "weather-rag-app"


# --- UI for PDF Upload and Processing ---
st.title("LangChain Agent with Weather & PDF RAG")

st.markdown("---")
st.header("Upload a PDF to power the agent's knowledge.")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.processed_document:
    
    with st.spinner("Processing PDF..."):
        # Use tempfile to save the uploaded file to a temporary location
        # so PyPDFLoader can access it via a file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            # Load and split the document
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create the vector store with Qdrant
            st.session_state.vector_store = Qdrant.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=collection_name,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            st.session_state.processed_document = True
            st.success("PDF processed and stored in Qdrant successfully!")
        
        finally:
            # Ensure the temporary file is deleted after processing
            os.remove(temp_path)

# --- Define the Agent and Tools ---

# 1. Define the tools the agent can use
@tool
def get_weather(location: str):
    """
    Get the current weather information for a specified location.
    Input should be a location string (e.g. "London,GB").
    """
    api_wrapper = OpenWeatherMapQueryRun()
    return api_wrapper.run(location)

@tool
def get_answer_from_pdf(query: str):
    """
    Use this tool to answer questions about the uploaded PDF document.
    Input should be a question string.
    """
    if not st.session_state.vector_store:
        return "PDF document is not loaded. Please upload a PDF first."
    
    # Create a retriever from the vector store
    retriever = st.session_state.vector_store.as_retriever()
    retrieved_docs = retriever.invoke(query)
    
    # Create a chain to combine documents and answer the question
    # This part uses the Groq model for text generation and summarization
    rag_chain = create_stuff_documents_chain(
        llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192"),
        prompt=ChatPromptTemplate.from_template(
            """
            Answer the user's question based on the following context:
            
            {context}
            
            Question: {input}
            """
        )
    )
    return rag_chain.invoke({"input": query, "context": retrieved_docs})

# Define the list of available tools
tools = [get_weather, get_answer_from_pdf]

# 2. Set up the LLM with tool calling capabilities
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# 3. Create the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant. You have access to a weather tool and a PDF RAG tool. "
         "If the user asks for the weather, use the `get_weather` tool. "
         "If the user asks a question about the uploaded document, use the `get_answer_from_pdf` tool. "
         "Do not use any tools for general conversational questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Create the agent executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Define the LangGraph state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 6. Define the graph nodes
def call_model(state):
    messages = state["messages"]
    response = agent_executor.invoke({"input": messages[-1].content, "chat_history": messages[:-1]})
    return {"messages": [AIMessage(content=response["output"])]}

def call_tool(state):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    # Execute tool calls
    tool_outputs = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # This is a basic way to execute the tool based on its name
        if tool_name == "get_weather":
            output = get_weather.invoke(tool_args["location"])
        elif tool_name == "get_answer_from_pdf":
            output = get_answer_from_pdf.invoke(tool_args["query"])
        else:
            output = f"Unknown tool: {tool_name}"
        
        tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

    return {"messages": tool_outputs}

# 7. Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool_use", call_tool)

# Define edges to control flow
workflow.add_edge(START, "agent")
workflow.add_edge("tool_use", "agent")

# Define conditional logic for the agent node
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_use"
    return END

workflow.add_conditional_edges("agent", should_continue)

# Compile the graph
app = workflow.compile()

st.markdown("---")
st.header("Chat with the Agent")

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Hello! I can tell you the weather or answer questions about a PDF. What can I help you with?")]

# Display messages from session state
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Get user input
if prompt := st.chat_input("Enter a message..."):
    # Update chat history
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("Thinking..."):
        # The app should ideally be written to use the compiled graph (`app`),
        # not the agent_executor, for a full agentic loop.
        # However, for a simple Streamlit app, we can simplify the invocation
        response = agent_executor.invoke({
            "input": prompt,
            "chat_history": st.session_state.messages
        })
        
        assistant_response = response["output"]

        # Display the response
        st.chat_message("assistant").write(assistant_response)
        st.session_state.messages.append(AIMessage(content=assistant_response))
