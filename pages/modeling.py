import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.title("Страница моделирования")
    st.write("Здесь будут отображаться графики и данные.")

    # Генерируем случайные данные
    df = pd.DataFrame({
        "Дата": pd.date_range(start="2025-01-01", periods=30),
        "Значение": np.random.randint(50, 150, size=30)
    })

    # Отображаем таблицу
    st.dataframe(df)

    # Создаем график
    fig, ax = plt.subplots()
    ax.plot(df["Дата"], df["Значение"], marker='o', linestyle='-')
    ax.set_title("График значений по датам")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение")

    # Отображаем график
    st.pyplot(fig)


if __name__ == "__main__":
    show()