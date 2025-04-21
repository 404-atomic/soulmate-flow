# LangGraph Chat Demo with Streamlit

This project demonstrates a simple conversational flow using LangGraph, LangChain, and OpenAI, with a Streamlit frontend.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

You can run the application in two ways:

### 1. Run the Streamlit interface:
```
streamlit run streamlit_app.py
```
This will open a web interface where you can interact with the conversation flow.

### 2. Run the console version:
```
python main.py
```
This will run the conversation flow in the console with step-by-step prompts.

## How It Works

The application uses LangGraph to create a directed graph of conversation nodes:
- Node 1: Says "hello"
- Node 2: Says "my name is kenz"
- Node 3: Asks "what is my name"

In the Streamlit interface, you can progress through the conversation by clicking "Run Next Step".
