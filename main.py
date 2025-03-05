import streamlit as st
from pages import analytics, modeling

def show():
    st.title("Главная страница")
    st.write("Добро пожаловать на главную страницу!")


if __name__ == "__main__":
    show()