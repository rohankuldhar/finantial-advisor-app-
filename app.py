import streamlit as st
from dotenv import load_dotenv

import openai
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate

load_dotenv()

def main():

    st.header("Financial Advisor Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    with st.form('chat_input_form'):
        query = st.text_input( "Ask your query:", placeholder= "Ask your query:", label_visibility='collapsed')
        submit = st.form_submit_button("Submit")


    if st.button("New Chat"):
        st.session_state.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if submit:

        st.session_state.messages.append({"role": "human", "content": query})
        llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        template= """

        Do not generate user responses on your own and avoid repeating questions.

        Yor are a helpful financial advisor chatbot designed to assist/help user with their financial needs. You can provide personalized advice on 
        managing finances, loan repayment strategies, investment options, retirement planning, and other financial queries. 
        You are here to help user make informed decisions and achieve their financial goals. 
        {chat_history}
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template="{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        messagee_place_holder = MessagesPlaceholder(variable_name="chat_history")

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,messagee_place_holder, human_message_prompt])

        chain = ConversationChain(llm=llm, prompt=chat_prompt, memory = memory)
        
        for chat in st.session_state.chat_history:
            memory.save_context({"input": chat["input"]}, {"output": chat["output"]})

        query
    

        with get_openai_callback() as cb:
            response = chain.run(query)
            print(cb)

        response

        st.session_state.messages.append({"role": "ai", "content": response})
        st.session_state.chat_history.append({"input": query, "output": response})

if __name__ == '__main__':
    main()
