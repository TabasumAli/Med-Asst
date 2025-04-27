# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# import cohere
# from langchain.llms import Cohere
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Cohere

# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# embeddings = download_hugging_face_embeddings()

# index_name = "medicalchatbot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# llm = Cohere(temperature=0.4, max_tokens=500)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # Initialize conversation history
# conversation_history = []  # List to store the conversation history

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     global conversation_history  # Declare to use the global variable
#     msg = request.form["msg"]
#     input_message = msg
#     print("Input Message:", input_message)  # Log the input message

#     # Append the user's message to the conversation history
#     conversation_history.append(f"User: {input_message}")

#     # Combine the conversation history into the input for the model
#     combined_input = "\n".join(conversation_history) + "\nAssistant:"

#     try:
#         response = rag_chain.invoke({"input": combined_input})
#         answer = response.get("answer", "No answer provided.")
#         print("Response:", answer)  # Log the response

#         # Append the assistant's answer to the conversation history
#         conversation_history.append(f"Assistant: {answer}")

#         return str(answer)
#     except Exception as e:
#         print("Error during API call:", str(e))  # Log the error
#         return "An error occurred while processing your request.", 500


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)


from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import openai
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up AIML API key
AIML_API_KEY = os.getenv('AIML_API_KEY')
openai.api_key = AIML_API_KEY
openai.api_base = "https://api.aimlapi.com/v1"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define system prompt
system_prompt = "You are a helpful assistant."

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Initialize conversation history
conversation_history = []

# Define function to get response from AIML API
def get_aiml_response(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return "An error occurred while processing your request."

# Define routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    global conversation_history
    msg = request.form["msg"]
    input_message = msg
    print("Input Message:", input_message)

    conversation_history.append(f"User: {input_message}")
    combined_input = "\n".join(conversation_history) + "\nAssistant:"

    messages = [{"role": "user", "content": combined_input}]
    assistant_reply = get_aiml_response(messages)

    print("Response:", assistant_reply)

    conversation_history.append(f"Assistant: {assistant_reply}")

    return str(assistant_reply)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
