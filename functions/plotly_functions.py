import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde

# --- Загрузка данных ---
@st.cache_data
def load_data(remove_duplicates=True):
    """Загружает и объединяет данные из CSV-файлов."""
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
    df["job_title"] = df["job_title"].apply(
        lambda x: "Machine Learning Engineer" if x == "ML Engineer" 
        else "Data Scientist" if x == "Data Science" else x
    )
    return df


def plot_salary_distribution(df, show_kde=False):
    """Гистограмма распределения зарплат."""
    st.subheader("Распределение зарплат в долларах США")
    skewness = df["salary_in_usd"].skew()
    st.write(f"**Смещение ЗП от среднего: {skewness:.2f}**")

    fig = px.histogram(
        df, x="salary_in_usd", nbins=30, title="Распределение зарплат",
        labels={"salary_in_usd": "Зарплата в долларах США"}, opacity=0.6,
        histnorm="probability density" if show_kde else None
    )

    if show_kde:
        x_values = np.linspace(df["salary_in_usd"].min(), df["salary_in_usd"].max(), 1000)
        kde = gaussian_kde(df["salary_in_usd"].dropna())
        y_values = kde(x_values)
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="KDE", 
                                 line=dict(color="red", width=2)))

    fig.update_layout(bargap=0.05, xaxis_title="Зарплата в долларах США",
                      yaxis_title="Плотность" if show_kde else "Частота")
    st.plotly_chart(fig, use_container_width=True)


def plot_experience_salary(df, palette):
    """Boxplot зарплат по уровням опыта."""
    st.subheader("Зарплата по уровню опыта")
    fig = px.box(
        df, x="experience_level", y="salary_in_usd", color="experience_level",
        title="Зарплата по уровню опыта",
        labels={"experience_level": "Уровень опыта", "salary_in_usd": "Зарплата в долларах США"},
        color_discrete_sequence=px.colors.qualitative.__dict__[palette]
    )
    fig.update_layout(xaxis_title="Уровень опыта", yaxis_title="Зарплата в долларах США",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_top_jobs(df, top_n=20, palette="Viridis"):
    """Горизонтальный bar chart топ-N профессий."""
    st.subheader(f"Топ-{top_n} самых популярных должностей")
    top_job_titles = df["job_title"].value_counts().nlargest(top_n)
    fig = px.bar(
        x=top_job_titles.values, y=top_job_titles.index, orientation="h",
        title=f"Топ {top_n} должностей",
        labels={"x": "Частота", "y": "Должности"}, color=top_job_titles.values,
        color_continuous_scale=palette
    )
    fig.update_layout(xaxis_title="Частота", yaxis_title="Должности", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_salary_experience(df, selected_palette):
    """Группированный bar chart зарплат по годам и опыту."""
    # Группировка данных по уровню опыта и году, вычисление средней зарплаты
    exp_salary = df.groupby(['experience_level', 'work_year'])['salary_in_usd'].mean().round().reset_index()
    
    # Порядок уровней опыта
    exp_order = ["EN", "MI", "SE", "EX"]
    
    # Словарь палитр с указанием их типа (discrete — качественная, continuous — последовательная)
    palette_options = {
        "Viridis": (px.colors.sequential.Viridis, "continuous"),
        "Plasma": (px.colors.sequential.Plasma, "continuous")
    }
    
    # Получение выбранной палитры и её типа
    selected_colors, palette_type = palette_options[selected_palette]
    
    # Построение графика в зависимости от типа палитры
    if palette_type == "discrete":
        fig = px.bar(
            exp_salary, 
            x="experience_level", 
            y="salary_in_usd", 
            color="work_year", 
            barmode="group",
            title="Зарплата по годам и опыту",
            labels={"experience_level": "Уровень опыта", "salary_in_usd": "Средняя ЗП (USD)", "work_year": "Год"},
            text_auto=".0f", 
            category_orders={"experience_level": exp_order},
            color_discrete_sequence=selected_colors
        )
    else:
        fig = px.bar(
            exp_salary, 
            x="experience_level", 
            y="salary_in_usd", 
            color="work_year", 
            barmode="group",
            title="Зарплата по годам и опыту",
            labels={"experience_level": "Уровень опыта", "salary_in_usd": "Средняя ЗП (USD)", "work_year": "Год"},
            text_auto=".0f", 
            category_orders={"experience_level": exp_order},
            color_continuous_scale=selected_colors
        )
    
    # Настройка отображения текста и осей
    fig.update_traces(textfont_size=12, textposition="outside")
    fig.update_layout(xaxis_title="Уровень опыта", yaxis_title="Средняя ЗП (USD)", legend_title="Год")
    
    # Отображение графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(df, selected_palette):
    """Тепловая карта корреляционной матрицы."""
    st.subheader("Матрица корреляции")
    numeric_cols = df.select_dtypes("number").drop(columns="salary", errors='ignore').columns.sort_values().tolist()
    if "salary_in_usd" in numeric_cols:
        numeric_cols.remove("salary_in_usd")
        numeric_cols.append("salary_in_usd")
    cor_matrix = df[numeric_cols].corr()
    cor_matrix = cor_matrix[numeric_cols].loc[numeric_cols][:-1]
    palette_options = {"Viridis": "viridis", "Plasma": "plasma"}
    selected_colorscale = palette_options[selected_palette]
    fig = ff.create_annotated_heatmap(
        z=cor_matrix.values, x=cor_matrix.columns.tolist(), y=cor_matrix.index.tolist(),
        colorscale=selected_colorscale, annotation_text=cor_matrix.round(2).values, showscale=True
    )
    fig.update_layout(title="Матрица корреляции", width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)