# financial advisor chatbot with gpt cache

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
import time

from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

load_dotenv()

def main():

    st.header("Financial Advisor Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    start_time = time.time()

    with st.form('chat_input_form'):
        query = st.text_input( "Ask your query:", placeholder= "Ask your query:", label_visibility='collapsed')
        submit = st.form_submit_button("Submit")
    if st.button("New Chat"):
        st.session_state.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if submit:

        st.session_state.messages.append({"role": "human", "content": query})
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        template= """

        Do not generate user responses on your own and avoid repeating questions.

        Yor are a helpful financial advisor chatbot designed to assist/help user with their financial needs.
        You can provide personalized advice on 
        managing finances, loan repayment strategies, investment options, retirement planning, and other financial queries. 
        You are here to help user make informed decisions and achieve their financial goals. 

        The major goal for you is to have conversation with the user about whatever financial concern or 
        needs the user has and if the user gets satisfied with the financial requirements of the user based 
        on the conversation, you should connect the user with a human representative. So basically you as a financial
        advisor for a bank and your goal is to sell some sort of financial product of the bank (loan, insurance, health plan, etc)
        to the user.
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

        with get_openai_callback() as cb:
            response = chain.run(query)
            print(cb)

        st.session_state.messages.append({"role": "ai", "content": response})
        st.session_state.chat_history.append({"input": query, "output": response})


        st.write("Time consuming: {:.2f}s".format(time.time() - start_time))
        for i, message in enumerate(st.session_state.chat_history):
            
                st.write('🧑- ',message['input'])
                st.write('🤖- ',message['output'])
        

if __name__ == '__main__':
    main()
