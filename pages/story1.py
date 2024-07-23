import time
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackHandler

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = "OPENAI_API_KEY"


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write("llm ended!")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="파도와 망각",
    page_icon="📄"
)

llm = ChatOpenAI(
    temperature=0.5
)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state['messages'].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state['messages']:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

st.title("파도와 망각")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

file_path = "./story/파도와_망각.pdf"  # 로컬 파일 경로 지정

if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    name = st.text_input("당신의 이름을 입력해주세요.", key='name_input')

    age = st.number_input("당신은 몇살인가요?", min_value=0, max_value=100)

    job = st.text_input("직업은 무엇인가요?", key='job_input')

    face = st.text_input("외모는 어떤가요?", key='face_input')

    personality = st.text_input("성격은 어떤가요?", key='personality_input')

    special = st.text_input("특기가 무엇인가요?", key='special_input')

    if st.button("입력 완료"):
        if name and age and job and face and personality and special:
            st.session_state.step = 2
            st.session_state["name"] = name
            st.session_state["age"] = age
            st.session_state["job"] = job
            st.session_state["face"] = face
            st.session_state["personality"] = personality
            st.session_state["special"] = special
        else:
            st.warning("모든 필드를 입력해주세요.")

elif st.session_state.step == 2:
    int_stat = st.number_input('캐릭터의 지능을 0에서 10사이의 숫자로 매긴다면 몇인가요?',min_value=0, max_value=10)

    mp = st.number_input('캐릭터의 마력을 0에서 10사이의 숫자로 매긴다면 몇인가요?',min_value=0, max_value=10)

    sight = st.number_input('캐릭터의 관찰력을 0에서 10사이의 숫자로 매긴다면 몇인가요?',min_value=0, max_value=10)

    dex = st.number_input('캐릭터의 민첩을 0에서 10사이의 숫자로 매긴다면 몇인가요?',min_value=0, max_value=10)

    if st.button("입력 완료"):
        if int_stat and mp and sight and dex:
            st.session_state.step = 3
            st.session_state["int_stat"] = int_stat
            st.session_state["mp"] = mp
            st.session_state["sight"] = sight
            st.session_state["dex"] = dex
        else:
            st.warning("모든 필드를 입력해주세요.")

elif st.session_state.step == 3:
    st.session_state.character_sheet = f"""
    이름 : {st.session_state["name"]}\n
    나이 : {st.session_state["age"]}\n
    직업 : {st.session_state["job"]}\n
    외모 : {st.session_state["face"]}\n
    성격 : {st.session_state["personality"]}\n
    특기 : {st.session_state["special"]}\n
    """

    st.session_state.stat_sheet = f"""
    정신력 : 10\n
    지능 : {st.session_state["int_stat"]}\n
    이성 : 10\n
    마력 : {st.session_state["mp"]}\n
    관찰력 : {st.session_state["sight"]}\n
    민첩 : {st.session_state["dex"]}\n
    """

    st.subheader("캐릭터 시트:")
    st.write(st.session_state.character_sheet)

    st.write("\n\n")

    st.subheader("능력치 시트:")
    st.write(st.session_state.stat_sheet)
    if st.button("확인 완료"):
        st.session_state.step = 4

elif st.session_state.step == 4:
    st.markdown(
        """
        세상이 멸망한 지도 벌써 10여 년이 흘렀습니다.
        우리는 함께 멸망한 세상을 여행하고 있습니다.
        기나긴 여행도 이제 막바지에 다다랐습니다.
    
        영국에서 가장 아름답다는, 새하얀 절벽과 바다를 이웃한 세븐시스터즈.
        당신은 그곳에서 지울 수 없는 위화감을 느낍니다.
    
        우리는 분명히 ◼ ◼◼◼ ◼◼◼◼….
    
        파도 소리가 귓가를 스칩니다. 머릿속이 혼란하게 흔들립니다.
        …우리가 왜 이 여행을 하고 있었죠?
        """
    )

    query = """
         Act as a Narrator of a text based adventure game. Your task is to describe the environment and supporting characters. Use direct speech when support characters are speaking. There is a Player controlling the actions and speech of their player character (PC). You may never act or speak for the player character. The game proceeds in turns between the Narrator describing the situation and the player saying what the player character is doing. When speaking about the player character, use second-person point of view. Your output should be expertly written, as if written by a best selling author. 무조건 한글로 말하세요.
    
         kpc는 플레이어가 스토리를 잘 진행할 수 있도록 게임 내에서 내레이터가 조종하여 이끌어주는 캐릭터입니다. 과한 개입은 불가합니다.
         PC는 당신으로 수정하여 출력하라
    
         D는 다이스(Dice)의 약자입니다. 3D6의 경우, 1부터 6까지의 숫자가 적힌 6면체 주사위를 3회 굴리면 된다고 이해하시면 되겠습니다. CoC에서 탐사자의 특성치(스탯)를 정할 때 굴리게 될 주사위입니다. 
         마찬가지로 2D6은 6면체 주사위를 2회 굴리면 됩니다. CoC에서 탐사자의 특성치(스탯)를 정할 때 굴리게 될 주사위입니다.
         D100은 CoC에서 특성치, 기능 등을 판정할 때, 즉 어떤 행동의 성공/실패 여부를 판정할 때 주로 사용하는 주사위입니다. 100면체 주사위를 1회 굴리는 것입니다.
         주사위의 결과가 기준치 이하면 성공, 기준치를 초과하면 실패입니다. 기본은 이렇고, 서로 대항해야 하는 상황에서는 성공 수준을 비교합니다.
         펌블(대실패) : 주사위 값으로 행동(특성치나 기능 판정)의 성공·실패 여부를 판정하는 CoC에서, 96~100 혹은 100의 결과값은 대실패로 처리합니다. 그냥 실패보다 훨씬 더 나쁜 결과를 낳으며, 기본적으로 효과를 바로 적용하고, 강행(재시도)를 할 수 없습니다. 대실패로 벌어지는 일은 수호자가 결정합니다. 
         이성 0/1D3 상실이라는 말은 이성 판정을 해서 성공 시 / 앞의 0만큼의 이성을, 실패 시 / 뒤의 1D3(주사위를 굴립니다)만큼의 이성을 줄이라는 뜻입니다.
         정신력이나 이성 판정을 할 때 특별한 판정 규칙이 없다면 1D100주사위를 굴려서 50초과면 성공, 50이하면 실패로 판정하라
         판정을 굴릴 때 주사위를 굴리고 그에따른 주사위 결과도 출력하라. 판정 결과 성공하면 정신력이나 이성을 감소시키지않고, 실패하면 규칙에 따라 현재 스탯에서 정신력이나 이성을 감소시켜라
    
         판정 결과 출력 예시 : 
    
         정신력 판정 - 1D100
         결과 - 52 성공!
    
         지능 판정 - 1D10
         결과 - 5 실패!
    
         이성 판정 - 1/1D2
         결과 - 1 성공!
    
         마지막 부분에 항상 현재 정신력과 이성을 출력하라.
         출력할 때 정신력과 이성만 출력하라.
    
         현재 스탯 출력 예시 : 
    
         현재 상태
         정신력 - 10
         이성 - 10
    
         현재 상태
         정신력 - 7
         이성 - 3
    
         ----------
    
        {setting_info}
         """

    query += st.session_state.character_sheet
    query = query + "\n" + st.session_state.stat_sheet

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         query
         ),
        ("human", "{question}")
    ])

    if file_path:
        retriever = embed_file(file_path)

        send_message("""
        당신은 머리가 어지러워지면서 동시에 바닷속으로 빠지는 듯한 감각을 느낍니다...
        발밑이 크게 흔들리더니 딛고 있던 것이 사라집니다.
        무언가 당신의 발목을 꽉 붙듭니다. 강한 인력 같은 힘이 발목을 쥐어짜듯 휘감아 당깁니다.
        아래로 쑥 빨려들어가는 느낌과 함께 어둑한 물이 온몸을 덮칩니다.
        어느덧 머리끝까지 잠긴 물 속에서 이상한 물체가 눈에 들어오기 시작합니다.
        다리, 몸통, …옷? 꼭 사람의 신체 같은 그것에서 거품이 오르고 있습니다.
        저것이 사람이고, 거품이 올라오고 있다면 아직 살아 있다는 뜻일 텐데….
        """, "ai", save=False)
        paint_history()
        message = st.chat_input("다음 행동을 입력하세요...")
        if message:
            send_message(message, "human")
            chain = {"setting_info": retriever, "question": RunnablePassthrough()} | prompt | llm

            response = chain.invoke(message)
            send_message(response.content, "ai")
    else:
        st.session_state["messages"] = []