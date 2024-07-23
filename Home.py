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

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
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


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Act as a Narrator of a text based adventure game. Your task is to describe the environment and supporting characters. Use direct speech when support characters are speaking. There is a Player controlling the actions and speech of their player character (PC). You may never act or speak for the player character. The game proceeds in turns between the Narrator describing the situation and the player saying what the player character is doing. When speaking about the player character, use second-person point of view. Your output should be expertly written, as if written by a best selling author. 무조건 한글로 말하세요.

     kpc는 플레이어가 스토리를 잘 진행할 수 있도록 게임 내에서 내레이터가 조종하여 이끌어주는 캐릭터입니다. 과한 개입은 불가합니다.

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

     현재 스탯 출력 예시 : 

     현재 스탯
     정신력 - 10
     이성 - 10

     현재 스탯
     정신력 - 7
     이성 - 3

     ----------

    {setting_info}

    이름: 에이든 스파크
    나이: 27세
    직업: 대학생
    외모: 검은 곱슬머리, 갈색 눈, 시력이 나빠 안경을 착용
    성격: 호기심 많고 모험을 좋아하며, 약간 서툴지만 열정적
    특기: 기계 조작 및 제작

    정신력 : 10
    지능 : 7
    이성 : 10
    마력 : 0
    관찰력 : 7
    민첩 : 2

    키퍼 정보 : PC는 플레이 시작과 동시에 망각의 순간을 겪습니다. 바닷속으로 빠지는 듯한 감각입니다.
    이후 PC가 바닷속으로 진입할 때마다 비슷한 상황을 연출해 주시면 유용합니다.
    발밑이 크게 흔들리더니 딛고 있던 것이 사라집니다.
    무언가 PC의 발목을 꽉 붙듭니다. 강한 인력 같은 힘이 발목을 쥐어짜듯 휘감아 당깁니다.
    아래로 쑥 빨려들어가는 느낌과 함께 어둑한 물이 온몸을 덮칩니다.
    어느덧 머리끝까지 잠긴 물 속에서 이상한 물체가 눈에 들어오기 시작합니다.
    다리, 몸통, …옷? 꼭 사람의 신체 같은 그것에서 거품이 오르고 있습니다.
    저것이 사람이고, 거품이 올라오고 있다면 아직 살아 있다는 뜻일 텐데….
    PC는 [정신력] 판정합니다.
    성공 : 흐릿한 현실감, 전혀 막혀 오지 않는 숨, 지나치게 비현실적인 풍경입니다. PC는 지금
    이 광경이 꿈이라는 사실을 자각합니다. 하지만 눈앞에 보이는 기괴한 풍경은 소름이 오를 만큼
    끔찍스러운 기분을 선사합니다.
    실패 : 정신이 흐려지더니 가물거리기 시작합니다. 발목이 끊어질 듯이 아파 오며 발목이
    끊어질 듯한 격통이 느껴집니다.
    기분 나쁜 느낌을 미처 떨쳐 버리지 못하고 비명을 지를 뻔했던 순간, PC는 눈을 뜹니다.
    노을빛 하늘이 보입니다. 물이라고는 처음부터 존재하지 않았던 것처럼 잠잠합니다. 아무래도 더럽게
    기분 나쁜 꿈이었던 모양이죠.
    노을진 하늘과 고운 자갈밭, 파도 치는 소리, 그리고 익숙한 기척.
    귀에 익은 목소리와 함께 PC는 눈을 뜹니다.
    키퍼 정보 : KPC는 PC가 단기 기억상실증을 동반하는 광기에 빠져 있다는 점을 알고 있습니다.
    PC를 안심시키는 대화를 진행해 주세요. PC가 어느 정도 안정을 찾았다면 다음으로
    진행됩니다.

     """
     ),
    ("human", "{question}")
])

st.title("파도와 망각")

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

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["txt", "pdf", "docx"])

if file:
    retriever = embed_file(file)

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
        chain = {"setting_info":retriever,"question" : RunnablePassthrough()} | prompt | llm

        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []