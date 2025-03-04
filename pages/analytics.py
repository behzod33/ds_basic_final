import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


# Расширяем ширину страницы
st.set_page_config(
    page_title="Аналитика данных",
    layout="centered", # centered wide
    initial_sidebar_state="expanded",
    menu_items={  
        "Get Help": "https://github.com/behzod33/ds_basic_final/blob/master/README.md",
        "About": """Github проекта: 
                    https://github.com/behzod33/ds_basic_final/"""
    }
)


# Функция для загрузки и объединения данных с кешированием
@st.cache_data
def load_data(remove_duplicates=True):
    # Загружаем данные
    df1 = pd.read_csv("datasets/ds_salaries.csv")
    df2 = pd.read_csv("datasets/salaries.csv")
    df3 = pd.read_csv("datasets/ds_salary_2024.csv")
    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    if remove_duplicates:
        df = df.drop_duplicates()

    job_titles_counts = df["job_title"].value_counts().to_dict()

    total_count = sum(job_titles_counts.values())
    threshold = total_count * 0.005

    valid_job_titles = {k for k, v in job_titles_counts.items() if v >= threshold}

    df = df[df["job_title"].isin(valid_job_titles)]

    df["job_title"] = df["job_title"].apply(lambda x: "Machine Learning Engineer" if x == "ML Engineer" 
                                        else "Data Scientist" if x == "Data Science" else x)

    return df


# Функция для построения графика распределения зарплат с возможностью включения KDE
def plot_salary_distribution(df, show_kde=False):
    st.subheader("📊 Распределение зарплат в долларах США")

    # Вычисляем смещение (skewness)
    skewness = df["salary_in_usd"].skew()
    st.write(f"**Смещение ЗП от среднего: {skewness:.2f}**")

    # Создаем гистограмму
    fig = px.histogram(
        df, 
        x="salary_in_usd", 
        nbins=30, 
        title="Распределение зарплат в долларах",
        labels={"salary_in_usd": "Зарплата в долларах США"},
        opacity=0.6,
        histnorm="probability density" if show_kde else None,  # Только если KDE включен
    )

    # Если пользователь включил KDE, добавляем линию плотности
    if show_kde:
        x_values = np.linspace(df["salary_in_usd"].min(), df["salary_in_usd"].max(), 1000)
        kde = gaussian_kde(df["salary_in_usd"].dropna())  # Удаляем NaN перед расчетом KDE
        y_values = kde(x_values)

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="KDE (Плотность)",
            line=dict(color="red", width=2)
        ))

    # Настраиваем оси
    fig.update_layout(
        bargap=0.05,
        xaxis_title="Зарплата в долларах США",
        yaxis_title="Плотность" if show_kde else "Частота",
        legend_title="Легенда"
    )

    # Отображаем график в Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Функция для построения boxplot
def plot_experience_salary(df, palette):
    st.subheader("📊 Зарплата по уровню опыта")

    # Создаем интерактивный boxplot
    fig = px.box(
        df, 
        x="experience_level", 
        y="salary_in_usd", 
        color="experience_level", 
        title="Зарплата по уровню опыта",
        labels={"experience_level": "Уровень опыта", "salary_in_usd": "Зарплата в долларах США"},
        color_discrete_sequence=px.colors.qualitative.__dict__[palette]  # Динамическая палитра
    )

    # Настроим оси
    fig.update_layout(
        xaxis_title="Уровень опыта",
        yaxis_title="Зарплата в долларах США",
        boxmode="group",
        showlegend=False
    )

    # Отображаем график в Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_top_jobs(df, top_n=20, palette="Viridis"):
    st.subheader(f"📊 Топ-{top_n} самых популярных должностей")

    # Получаем топ профессий
    top_job_titles = df["job_title"].value_counts().nlargest(top_n)

    # Создаем интерактивный bar chart
    fig = px.bar(
        x=top_job_titles.values,
        y=top_job_titles.index,
        orientation="h",
        title=f"Частота самых популярных должностей (Топ {top_n})",
        labels={"x": "Частота", "y": "Должности"},
        color=top_job_titles.values,  # Цвет зависит от частоты
        color_continuous_scale=palette
    )

    # Улучшаем стиль графика
    fig.update_layout(
        xaxis_title="Частота",
        yaxis_title="Должности",
        coloraxis_showscale=False,  # Убираем легенду цвета
        bargap=0.3
    )

    # Отображаем график в Streamlit
    st.plotly_chart(fig, use_container_width=True)


def show():
    st.title("Страница аналитики")
    st.write("Здесь будут отображаться графики и данные.")

    # Чекбокс для выбора удаления дубликатов
    remove_duplicates = st.checkbox("Удалять дубликаты?", value=True)

    # Загружаем данные с учетом выбора пользователя
    df = load_data(remove_duplicates)

    # Выбор столбцов
    selected_columns = st.multiselect("Выберите столбцы для отображения", df.columns.tolist(), default=df.columns.tolist())

    # Выбор диапазона строк
    min_index, max_index = st.slider("Выберите диапазон строк", 0, len(df) - 1, (0, min(100, len(df) - 1)))

    # Фильтруем данные по выбранным столбцам и строкам
    filtered_df = df.loc[min_index:max_index, selected_columns]

    # Отображаем отфильтрованные данные
    st.subheader("🔍 Отфильтрованные данные")
    st.dataframe(filtered_df, height=500, width=1200)  # Увеличенная ширина и высота таблицы

    # Разделитель
    st.markdown("---")

    # **Просмотр уникальных значений по выбранному столбцу**
    st.subheader("Просмотр уникальных значений в столбце")

    column_for_unique_values = st.selectbox("🔎 Выберите столбец", df.columns.tolist())

    if column_for_unique_values:
        unique_values = df[column_for_unique_values].dropna().unique()
        st.write(f"**Уникальные значения в `{column_for_unique_values}`:**")

        # Отображение уникальных значений в более широком формате
        unique_df = pd.DataFrame(unique_values, columns=["Уникальные значения"])
        st.dataframe(unique_df, height=250, width=300)  
    
    show_kde = st.checkbox("📈 Включить KDE (Плотность распределения)", value=False)

    # Вызываем функцию построения графика
    plot_salary_distribution(df, show_kde)

     # Выбор цветовой палитры
    palette_options = ["Set1", "Set2", "Set3", "Pastel", "Dark2"]
    selected_palette = st.selectbox("🎨 Выберите цветовую палитру", palette_options, index=1)

    # Построение графика
    plot_experience_salary(df, selected_palette)

    # Выбор количества топ профессий
    top_n = st.slider("🔢 Количество профессий в рейтинге", min_value=5, max_value=20, value=20)

    # Выбор цветовой палитры
    palette_options = ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo"]
    selected_palette = st.selectbox("🎨 Выберите цветовую палитру", palette_options, index=0)

    # Построение графика
    plot_top_jobs(df, top_n, selected_palette)

if __name__ == "__main__":
    show()
