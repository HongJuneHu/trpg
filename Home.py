import streamlit as st

#페이지 config
st.set_page_config(
    page_title="Trpg master",
)

#페이지 타이틀
st.title("Trpg master")

#기본적으로 페이지에 출력할 내용
st.markdown(
    """
    추가 예정
    """
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

#메시지를 입력하는 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )

message = st.chat_input("행동을 입력하세요...")

if message:
    send_message(message, "human")
    send_message(f"입력한 행동: {message}", "Game master")
