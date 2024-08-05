import streamlit as st
import boto3
from loguru import logger
import os

from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.memory import StreamlitChatMessageHistory

# AWS 프로파일 설정
os.environ["AWS_PROFILE"] = "igenip"


# 로깅 설정
logger.add("app.log", rotation="500 MB")

# Bedrock 클라이언트 생성 함수
def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        profile_name="igenip"
    )

def main():
    st.set_page_config(page_title="genip")

    st.title("스포츠법률상담 챗봇")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        
    if uploaded_files:
        with st.spinner(""):
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            logger.info("Vectorstore created successfully")
            
            conversation_chain = get_conversation_chain(vectorstore)
            
            if conversation_chain:
                logger.info("Conversation chain created successfully")
                st.session_state.conversation = conversation_chain
                st.session_state.processComplete = True
                st.success("Files processed successfully!")
            else:
                logger.error("Failed to create conversation chain")
                st.error("Failed to process files. Please check the logs and try again.")

    #시작 메세지
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    #메세지 칸
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #이전 대화 내용 기억하기 위해 
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    # 쿼리가 있다면 위에 session staet 메세지에 user와 쿼리를 넣음
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    if st.session_state.conversation:
                        chain = st.session_state.conversation
                        result = chain({"question": query})
                        st.session_state.chat_history = result['chat_history']
                        response = result['answer']
                        source_documents = result['source_documents']

                        st.markdown(response)
                        with st.expander("참고 문서 확인"):
                            for i, doc in enumerate(source_documents[:3]):
                                st.markdown(f"문서 {i+1}: {doc.metadata['source']}", help=doc.page_content)
                    else:
                        response = st.session_state.llm.predict(query)
                        st.markdown(response)

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    logger.error(f"Error during chat: {str(e)}")
                    st.error("죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요.")

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        try:
            if '.pdf' in doc.name:
                loader = PyPDFLoader(file_name)
                documents = loader.load_and_split()
            elif '.docx' in doc.name:
                loader = Docx2txtLoader(file_name)
                documents = loader.load_and_split()
            elif '.pptx' in doc.name:
                loader = UnstructuredPowerPointLoader(file_name)
                documents = loader.load_and_split()
            doc_list.extend(documents)
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            st.error(f"파일 {file_name} 처리 중 오류가 발생했습니다.")
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        bedrock_client = get_bedrock_client()
        embeddings = BedrockEmbeddings(client=bedrock_client)
        vectordb = FAISS.from_documents(text_chunks, embeddings)
        return vectordb
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        st.error("벡터 저장소 생성 중 오류가 발생했습니다.")
        return None

def get_llm():
    try:
        bedrock_client = get_bedrock_client()
        return BedrockChat(
            client=bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "temperature": 0,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"]
            }
        )
    except Exception as e:
        logger.error(f"Error creating LLM: {str(e)}")
        st.error(f"LLM 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )

        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        st.error("대화 체인 생성 중 오류가 발생했습니다.")
        return None

if __name__ == '__main__':
    main()
