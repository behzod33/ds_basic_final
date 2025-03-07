import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from functions.model_utils import *

def main():
    st.title("Предсказание зарплаты (3 модели)")

    # Загружаем модели
    models, error = load_models()
    if models is None:
        st.error(error)
        return

    # Показываем веса моделей
    st.markdown("### Веса моделей (на основе R² Test, нормализованные до 100%)")
    weights_df = pd.DataFrame({
        "Модель": list(model_weights.keys()),
        "Вес (%)": [f"{weight:.2f}" for weight in model_weights.values()]
    }).style.background_gradient(cmap='Blues')
    st.table(weights_df)

    # Получаем данные для справочников
    reverse_country_dict, country_dict, experience_level_dict, employment_type_dict, company_size_dict, salary_currency_dict = get_category_data()

    # Боковая панель ввода
    with st.sidebar:
        st.header("Ввод данных для предсказания")
        with st.form("input_form"):
            company_location_code = st.selectbox("Локация компании (ввод кода)", list(reverse_country_dict.keys()), index=0)
            company_location = reverse_country_dict[company_location_code]
            st.write(f"Вы выбрали: {company_location} ({company_location_code})")

            employee_residence_code = st.selectbox("Страна сотрудника (ввод кода)", list(reverse_country_dict.keys()), index=0)
            employee_residence = reverse_country_dict[employee_residence_code]
            st.write(f"Вы выбрали: {employee_residence} ({employee_residence_code})")
           
            work_year = st.selectbox("Год работы", [2020, 2021, 2022, 2023, 2024], index=3)
            experience_level = st.selectbox("Уровень опыта", list(experience_level_dict.keys()), index=1)
            employment_type = st.selectbox("Тип занятости", list(employment_type_dict.keys()), index=0)
            job_title = st.selectbox("Должность", ["Data Scientist", "Machine Learning Engineer", "Data Engineer"], index=0)
            salary_currency = st.selectbox("Валюта зарплаты", list(salary_currency_dict.keys()), index=1)
            remote_ratio = st.slider("Удалённо (%)", 0, 100, 50)
            company_size = st.selectbox("Размер компании", list(company_size_dict.keys()), index=2)

            submit = st.form_submit_button("Предсказать")

    # Центральная часть: предсказания и визуализация
    if submit:
        # Формируем DataFrame
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

        # Предсказания от каждой модели
        predictions = {}
        for model_name, model in models.items():
            try:
                pred_val = model.predict(input_data)[0]
                predictions[model_name] = pred_val
            except Exception as e:
                st.warning(f"Ошибка предсказания для {model_name}: {e}")

        if predictions:
            # 1) Обычное среднее
            avg_pred = np.mean(list(predictions.values()))

            # 2) Взвешенное среднее (с весами, нормализованными до 100%)
            valid_models = [m for m in predictions if m in model_weights and model_weights[m] > 0]
            if not valid_models:
                weighted_pred = avg_pred
            else:
                sum_w = sum(model_weights[m] for m in valid_models)
                weighted_sum = sum(predictions[m] * (model_weights[m] / 100) for m in valid_models)  # Нормализация весов
                weighted_pred = weighted_sum

            # Центральная колонка для предсказаний
            with st.container():
                st.subheader("Результаты предсказаний")
                col1, col2 = st.columns(2)
                
                with col1:
                    for model_name, val in predictions.items():
                        # Выводим вес в процентах для прозрачности
                        weight = model_weights[model_name]
                        if model_name in ["Random Forest", "CatBoost"]:
                            st.markdown(f"**{model_name} (вес {weight:.2f}%):** ${val:,.2f} USD", unsafe_allow_html=True)
                        else:
                            st.write(f"**{model_name} (вес {weight:.2f}%):** ${val:,.2f} USD")
                
                with col2:
                    st.write(f"**Обычное среднее:** ${avg_pred:,.2f} USD")
                    st.write(f"**Взвешенное среднее:** ${weighted_pred:,.2f} USD")

                # Визуализация с увеличенным акцентом на CatBoost и Random Forest
                pred_df = pd.DataFrame({
                    'Модель': list(predictions.keys()),
                    'Предсказание (USD)': list(predictions.values()),
                    'Вес (%)': [model_weights[m] for m in predictions.keys()]
                })
                fig = px.bar(pred_df, x='Модель', y='Предсказание (USD)', 
                             color='Вес (%)', title="Предсказания трёх моделей",
                             labels={'Предсказание (USD)': 'Зарплата (USD)', 'Вес (%)': 'Вес модели (%)'},
                             color_continuous_scale='Viridis')
                fig.update_layout(showlegend=True, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Пояснение к расчету взвешенного среднего
                st.markdown("### Как рассчитано взвешенное среднее?")
                st.write("""
                Взвешенное среднее вычисляется как:
                \n`Взвешенное среднее = (Предсказание₁ × Вес₁ + Предсказание₂ × Вес₂ + Предсказание₃ × Вес₃) / 100`,
                где веса нормализованы до 100% на основе R² Test каждой модели.
                """)

if __name__ == "__main__":
    main()
