from functions.plotly_functions import *

# --- Настройки страницы ---
st.set_page_config(
    page_title="Аналитика данных",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/behzod33/ds_basic_final/blob/master/README.md",
        "About": """Github проекта: 
                    https://github.com/behzod33/ds_basic_final/"""
    }
)



# --- Основной интерфейс ---
def show():
    """Отображает интерфейс и все графики."""
    st.title("Аналитика данных")

    # Боковая панель для элементов управления
    with st.sidebar:
        st.header("Настройки")

        # Чекбокс для удаления дубликатов
        remove_duplicates = st.checkbox("Удалять дубликаты?", value=True)

        # Загрузка данных
        df = load_data(remove_duplicates)

        # Мультиселект для выбора столбцов
        selected_columns = st.multiselect("Выберите столбцы", df.columns.tolist(), default=df.columns.tolist())

        # Слайдер для выбора диапазона строк
        min_index, max_index = st.slider("Диапазон строк", 0, len(df) - 1, (0, min(100, len(df) - 1)))

        # Чекбокс для отображения KDE
        show_kde = st.checkbox("Показать KDE", value=False)

        # Слайдер для выбора топ-N профессий
        top_n = st.slider("Топ-N профессий", 5, 20, 20)

        # Выбор палитр для всех графиков
        st.subheader("Палитры для графиков")
        palette_experience = st.selectbox("Палитра для опыта", ["Set1", "Set2", "Set3"], index=0)
        palette_top_jobs = st.selectbox("Палитра для топ профессий", ["Viridis", "Plasma", "Inferno"], index=0)
        palette_salary_experience = st.selectbox("Палитра для зарплат по годам", ["Viridis", "Plasma"], index=0)
        palette_correlation = st.selectbox("Палитра для корреляции", ["Viridis", "Plasma"], index=0)

    # Основное окно: отфильтрованные данные и графики
    st.subheader("Отфильтрованные данные")
    filtered_df = df.loc[min_index:max_index, selected_columns]
    st.dataframe(filtered_df)

    st.markdown("---")

    st.subheader("Графики")

    # График 1: Распределение зарплат
    plot_salary_distribution(df, show_kde)

    # График 2: Зарплата по опыту
    plot_experience_salary(df, palette_experience)

    # График 3: Топ профессий
    plot_top_jobs(df, top_n, palette_top_jobs)

    # График 4: Зарплата по годам и опыту
    plot_salary_experience(df, palette_salary_experience)

    # График 5: Матрица корреляции
    plot_correlation_matrix(df, palette_correlation)

# --- Запуск ---
if __name__ == "__main__":
    show()