import streamlit as st
from pages import analytics, modeling


# --- Настройки страницы ---
st.set_page_config(
    page_title="Предсказание зарплаты",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/behzod33/ds_basic_final/blob/master/README.md",
        "About": """Github проекта: 
                    https://github.com/behzod33/ds_basic_final/"""
    }
)



def show():
    st.title("Главная страница")
    st.write("Добро пожаловать на главную страницу!")


if __name__ == "__main__":
    show()