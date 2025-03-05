import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Предсказание зарплаты", layout="wide")

@st.cache_data
def load_models(save_path="saved_models"):
    """Загружает все сохраненные модели (каждая - Pipeline)"""
    models = {}
    try:
        models["Linear Regression"] = joblib.load(os.path.join(save_path, "Linear_Regression.pkl"))
        models["Random Forest"] = joblib.load(os.path.join(save_path, "Random_Forest.pkl"))
        models["CatBoost"] = joblib.load(os.path.join(save_path, "CatBoost.pkl"))
    except FileNotFoundError as e:
        st.error(f"Не удалось найти модели в папке {save_path}: {e}")
        return None
    return models

def main():
    st.title("✨ Предсказание зарплаты в Data Science (Pipeline)")
    st.write("Введите данные для предсказания зарплаты (USD)")

    models = load_models()
    if models is None:
        st.stop()

    # Боковая панель для ввода данных
    with st.sidebar:
        st.header("⚙️ Ввод данных")
        work_year = st.selectbox("Год работы", [2020, 2021, 2022, 2023, 2024], index=3)
        experience_level = st.selectbox("Уровень опыта", ["EN", "MI", "SE", "EX"], index=1)
        employment_type = st.selectbox("Тип занятости", ["FT", "CT", "FL", "PT"], index=0)
        job_title = st.text_input("Должность", "Data Scientist")
        salary_currency = st.selectbox("Валюта зарплаты", ["USD", "EUR", "INR"])
        employee_residence = st.text_input("Страна сотрудника", "US")
        remote_ratio = st.slider("Удалённо (%)", 0, 100, 0, 50)
        company_location = st.text_input("Локация компании", "US")
        company_size = st.selectbox("Размер компании", ["S", "M", "L"], index=2)

        if st.button("Предсказать"):
            # Собираем данные в DataFrame (как в обучении)
            input_data = pd.DataFrame([{
                "work_year": work_year,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "job_title": job_title,
                "salary_currency": salary_currency,
                "employee_residence": employee_residence,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": company_size
            }])

            predictions = {}
            for model_name, model in models.items():
                try:
                    # Так как у нас модель - это уже Pipeline, просто .predict(input_data)
                    pred = model.predict(input_data)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    st.warning(f"Ошибка предсказания для {model_name}: {e}")

            if predictions:
                # Усредняем
                avg_pred = np.mean(list(predictions.values()))

                st.subheader("Результаты:")
                for model_name, val in predictions.items():
                    st.write(f"**{model_name}:** ${val:,.2f}")

                st.write(f"**Среднее:** ${avg_pred:,.2f}")

                # Визуализация
                pred_df = pd.DataFrame({
                    'Модель': list(predictions.keys()),
                    'Предсказание (USD)': list(predictions.values())
                })
                fig = px.bar(pred_df, x='Модель', y='Предсказание (USD)',
                             color='Модель', title="Предсказания моделей",
                             labels={'Предсказание (USD)': 'USD'})
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
