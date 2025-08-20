# Streamlit Weather & PDF RAG Agent
This project is a powerful and interactive Streamlit application that combines a conversational AI agent with a Retrieval-Augmented Generation (RAG) system. The application can perform two key functions:

Retrieve current weather data for any location worldwide using the OpenWeatherMap API.

Answer questions from a PDF document that the user uploads, using a RAG system powered by Qdrant.

The core of the application is built using LangChain and LangGraph, which provide a robust framework for creating intelligent, tool-using agents. The application leverages Groq for fast and efficient language model inference.

# Features
Intelligent Agent: The AI agent automatically decides whether to use a tool to get weather information or to retrieve information from the uploaded PDF.

Weather Tool: Fetches real-time weather conditions using the OpenWeatherMap API.

PDF RAG: Processes a user-uploaded PDF, splits it into manageable chunks, and stores the embeddings in a Qdrant vector database for efficient retrieval.

Conversation History: Maintains chat history within the Streamlit session for a seamless user experience.

Efficient LLM: Uses Groq's high-speed API for low-latency responses.

# Prerequisites
Before running this application, you need to have the following:

A Groq API Key

An OpenWeatherMap API Key

A Qdrant Cloud Account or self-hosted instance.

A Qdrant API Key and URL

# Setup and Installation
1. Clone the repository
Bash
git clone https://github.com/your-username/streamlit-weather-rag.git
cd streamlit-weather-rag
2. Set up a virtual environment
It's a best practice to use a virtual environment to manage dependencies.

Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

2. Install dependencies
3. Install all the required libraries from the requirements.txt file.

Bash

pip install -r requirements.txt

4. Configure API keys
   
5. Create a .env file in the root of your project directory and add your API keys.

# Code snippet

GROQ_API_KEY="your_groq_api_key"

OPENWEATHERMAP_API_KEY="your_openweathermap_api_key"

QDRANT_URL="your_qdrant_url"

QDRANT_API_KEY="your_qdrant_api_key"

How to Run the App
Once you have completed the setup, you can run the Streamlit application from your terminal.

Bash

streamlit run app.py

This will start a local web server and open the app in your browser.

# Deployment
This app is designed to be easily deployed to Streamlit Community Cloud. For deployment, you will need to:

Push your code (including app.py and requirements.txt) to a public GitHub repository.

Create a .streamlit/secrets.toml file in your repository to securely store your API keys.

Ini, TOML

GROQ_API_KEY = "your_groq_api_key"

OPENWEATHERMAP_API_KEY = "your_openweathermap_api_key"

QDRANT_URL = "your_qdrant_url"

QDRANT_API_KEY = "your_qdrant_api_key"

Deploy the app from your GitHub repository using the Streamlit Community Cloud dashboard.
