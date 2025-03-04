import streamlit as st
from pages import analytics, modeling

def show():
    st.title("Главная страница")
    st.write("Добро пожаловать на главную страницу!")
    st.image("plots/Влияние_удалённой_работы_на_зарплату.png", caption="Случайное изображение")


if __name__ == "__main__":
    show()