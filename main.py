import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import playsound

filename = "abc.mp3"
DB_FAISS_PATH = 'db/db_faiss'

def retrieval_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff', 
        retriever=db.as_retriever(search_kwargs={'k': 2}), 
        return_source_documents=True
    )

def load_llm():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token='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',
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    return retrieval_qa_chain(llm, db)

def final_result(query):
    qa = qa_bot()
    response = qa({'query': query})
    helpful_answer = response['result'].split("Helpful Answer:")[1].strip()
    return helpful_answer

def generate_audio(text):
    tts = gTTS(text, lang='ta')
    tts.save(filename)
    playsound.playsound(filename)

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Sankara Netralaya's Eye Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res['result'].split("Helpful Answer:")[1].strip()
    translated_answer = GoogleTranslator(source = 'auto', target='ta').translate(answer)
    generate_audio(translated_answer)
    await cl.Message(content=translated_answer, elements =  [
        cl.Audio(name = 'audiofile.mp3', path = "./abc.mp3", display="inline")
    ]).send()
    os.remove(filename)
 



