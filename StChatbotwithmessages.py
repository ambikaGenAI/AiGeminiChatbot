from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage

import streamlit as st
from streamlit_chat import message

import os
os.environ["GOOGLE_API_KEY"]=st.secrets["GOOGLE_API_KEY"]

google_flash_model=ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

if 'messages' not in st.session_state.keys():
    st.session_state.messages=[{'role' : 'Assistant','content': 'How May i help you today?'}]

system_message="""You are an AICHATBOT . you will answer only AI related queries.For NON AI Related Queries you will politely say 'I cannot answer NON AI related Queries'"""

st.set_page_config(page_title="Google Gemini Flash Chatbot",
                   page_icon="🦖")
st.header("Generative AI Conversational Chatbot")
st.subheader("Simple Chatbot answer AI Related Queries")

conversation=ConversationChain(llm=google_flash_model,memory=st.session_state.buffer_memory)
conversation.memory.chat_memory.add_message(SystemMessage(content=system_message))

if Getprompt := st.chat_input("Your Question"):
    st.session_state.messages.append({"role": "user", "content": Getprompt})
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message['content'])

if st.session_state.messages[-1]["role"] != "Assistant":
    with st.chat_message("Assistant"):
        with st.spinner("Thinking.........."):
            response=conversation.predict(input=Getprompt)
            st.write(response)
            message={ "role":"Assistant","content": response}
            st.session_state.messages.append(message)

