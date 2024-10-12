# Import necessary libraries
from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
import os
from langchain_together.chat_models import ChatTogether
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Set Streamlit page configuration with title and layout
st.set_page_config(page_title='Sudharsan\'s Personal AI Assistant', layout='wide')

# Load the model if not already loaded in cache_resource
@st.cache_resource()
def load_model():
    return HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

embedding_model = load_model()

# Load the vectordb from local storage
new_db = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)

# Retriever
retriever = new_db.as_retriever(search_kwargs={"k": 7, "hnsw:space": "cosine"})

# Set api key
load_dotenv("D:\FarmwiseAI\Reddit\.env")

# Load the API key from the environment and set it in the environment 
api_key = '6b0637401d057af5d54b59bdfa0cb55d1b4c35115b3403d27cbed6720b453744'
os.environ["TOGETHER_API_KEY"] = api_key

# Language Model
llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0, max_tokens= 2000,model_kwargs={"top_p": 0.5})

# Prompt for condensing the chat history 
condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
# Chain for condensing the question
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

# System Prompt for the AI Assistant
qa_system_prompt = """You are an AI assistant designed to provide comprehensive information about Sudharsan. Your primary role is to help users understand various aspects of Sudharsan's life, work, interests, achievements, etc. based on the context provided and the question asked by them. Below is a detailed guide on how to interact and provide valuable assistance:

User Profile:
Name: Sudharsan
Profession: SQL Developer
Experience: 10 months working with SQL databases and ETL workflows to build applications such as a relational database for insurance claim scenarios and an automated data retrieval system.
Interests: Tech, Algorithm design and optimization, health, fitness, longevity, self-help podcasts, and working out.

Interaction Guidelines:
* Be Clear and Informative: Provide detailed and accurate information about Sudharsan. Avoid overly technical jargon unless necessary.
* Be Supportive and Encouraging: Encourage users to learn more about Sudharsan's interests and achievements. Offer positive reinforcement.
* Be Detailed and Comprehensive: Provide in-depth information about Sudharsan's work, interests, and achievements. Include relevant examples and details.
* Make sure to only provide information about Sudharsan from the parts of the context that is relevant to the question asked.
* If you do not have enough information to answer a question, politely inform the user and offer to provide more details on a different topic and NEVER try to come up with a answer to which you do not have enough information to provide.

Example Dialogue:
User: Can you tell me about Sudharsan's professional background?

AI Assistant: Sudharsan is a SQL Developer with 9 months of experience designing and optimizing relational databases for complex insurance claim scenarios. He has implemented ETL workflows using SQL Server Integration Services (SSIS), streamlining data extraction, transformation, and loading processes. Additionally, he has developed and refined SQL queries and stored procedures, focusing on optimization to enhance efficiency and automate routine database tasks. Sudharsan is passionate about leveraging innovative technologies to drive results in his work.

Example Dialogue where the AI does not have enough information:

User: What is Sudharsan's Mother's name?
AI Assistant: I'm sorry, but I do not have that information. Would you like to know more about Sudharsan's professional background or interests?

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)

def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

# Initialize chat history
chat_history = []

# Streamlit app setup
st.title("Sudharsan's Personal AI Assistant")

# Function to handle message sending
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Process user message
        ai_msg = rag_chain.invoke({"question": user_input, "chat_history": st.session_state['chat_history']})
        st.session_state['chat_history'].extend([HumanMessage(content=user_input), ai_msg])
        # Print chat history for debugging
        print("Updated chat history:", st.session_state['chat_history'])


        # Clear the input box
        st.session_state.user_input = ""

# Use columns for better layout - col 1 for chat history and col 2 for user input
col1, col2 = st.columns([3, 1])

# Container for chat history
# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Container for chat history
with col1:
    with st.container():
        for message in st.session_state['chat_history']:
            if isinstance(message, HumanMessage):
                st.markdown(f"""**USER** 🧑‍💻:  
                            {message.content}""")
                st.write('------------')
            else:
                st.markdown(f"""**AI** 🤖:  
                            {message.content}""")
                st.write('------------')

# Container for user input
with col2:
    with st.container():
        # Text input for user message
        user_input = st.text_area("Your message:", key="user_input", args=(), height=400)

        # Send button
        send_btn = st.button("Send", on_click=send_message)
        
# CSS for making the first column scrollable
css = '''
<style>
    /*section.main > div {
        /* padding-bottom: 1rem; */
    }
    [data-testid="column"] > div > div > div > div > div {
        overflow: auto;
        height: 80vh;
    }
    .element-container {
        margin-bottom: 0rem !important;
    }
    .stTextArea {
        margin-bottom: 1rem !important;
    }*/
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# JavaScript to manage the scroll position
scroll_js = """
<script>
function maintainScrollPosition(){
    var container = document.querySelector("[data-testid='column']>div>div>div>div>div");
    var scrollPosition = localStorage.getItem('scrollPosition');
    if (scrollPosition) {
        container.scrollTop = scrollPosition;
    }
    container.addEventListener('scroll', function() {
        localStorage.setItem('scrollPosition', container.scrollTop);
    });
}
// Call the function when the page loads
setTimeout(maintainScrollPosition, 400);
</script>
"""

# Render the JavaScript to manage the scroll position
st.markdown(scroll_js, unsafe_allow_html=True)