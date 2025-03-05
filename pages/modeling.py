import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

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

# МЕТРИКИ ДЛЯ ВЕСОВ (R2 Test из таблицы, нормализованные до 100%)
r2_test_values = {
    "Linear Regression": 33.88,
    "Random Forest": 36.40,
    "CatBoost": 36.63
}

# Нормализация весов до 100% на основе R2 Test
total_r2 = sum(r2_test_values.values())
model_weights = {model: (score / total_r2) * 100 for model, score in r2_test_values.items()}

@st.cache_data
def load_models(save_path="saved_models"):
    """Загружает три сохраненные модели (каждая - Pipeline)."""
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
    st.title("Предсказание зарплаты (3 модели)")

    # Показываем веса моделей с улучшенным стилем
    st.markdown("### Веса моделей (на основе R² Test, нормализованные до 100%)")
    weights_df = pd.DataFrame({
        "Модель": list(model_weights.keys()),
        "Вес (%)": [f"{weight:.2f}" for weight in model_weights.values()]
    }).style.background_gradient(cmap='Blues')
    st.table(weights_df)

    # Загружаем модели
    models = load_models()
    if models is None:
        st.stop()

    # === Справочники (упорядоченные списки) ===
    work_year_options = [2020, 2021, 2022, 2023, 2024]
    experience_level_options = ["EN", "MI", "SE", "EX"]
    employment_type_options = ["FT", "CT", "FL", "PT"]
    company_size_options = ["S", "M", "L"]
    
    job_title_options = sorted([
        "Principal Data Scientist", "Machine Learning Engineer", "Data Scientist",
        "Applied Scientist", "Data Analyst", "Data Modeler", "Research Engineer",
        "Analytics Engineer", "Business Intelligence Engineer", "Data Strategist",
        "Data Engineer", "Computer Vision Engineer", "Data Quality Analyst",
        "Data Architect", "AI Developer", "Research Scientist",
        "Data Analytics Manager", "Business Data Analyst", "Applied Data Scientist",
        "Head of Data", "Data Science Manager", "Data Manager",
        "Machine Learning Researcher", "Big Data Engineer", "Data Specialist",
        "Director of Data Science", "Machine Learning Scientist", "MLOps Engineer",
        "AI Scientist", "Applied Machine Learning Scientist", "Lead Data Scientist",
        "Cloud Database Engineer", "Data Infrastructure Engineer",
        "Data Operations Engineer", "BI Developer", "Data Science Lead",
        "BI Analyst", "Data Science Consultant",
        "Machine Learning Infrastructure Engineer", "BI Data Analyst",
        "Head of Data Science", "Insight Analyst",
        "Machine Learning Software Engineer", "Data Analytics Lead", "Data Lead",
        "Data Science Engineer", "NLP Engineer", "Data Management Specialist",
        "Data Operations Analyst", "ETL Developer", "AI Architect",
        "Business Intelligence", "Data Integration Engineer",
        "Business Intelligence Analyst", "Research Analyst",
        "Business Intelligence Developer", "Data Product Manager", "AI Engineer",
        "Business Intelligence Manager", "Data Developer", "Prompt Engineer",
        "Robotics Engineer", "Data Management Analyst",
        "Data Integration Specialist", "Data Science Practitioner",
        "Data Visualization Specialist", "Decision Scientist"
    ])
    salary_currency_options = sorted([
        'EUR', 'USD', 'INR', 'HKD', 'CHF', 'GBP', 'AUD', 'SGD', 'CAD', 'ILS', 'BRL', 'THB',
        'PLN', 'HUF', 'CZK', 'JPY', 'MXN', 'TRY', 'CLP', 'DKK', 'PHP', 'NOK', 'ZAR'
    ])
    employee_residence_options = sorted([
        'ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'PT', 'NL', 'CH', 'CF', 'AU', 'FI',
        'UA', 'IE', 'IL', 'AT', 'CO', 'SE', 'SI', 'MX', 'FR', 'UZ', 'BR', 'TH', 'GH', 'HR',
        'PL', 'KW', 'VN', 'CY', 'AR', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'IT', 'MA',
        'LT', 'IR', 'HU', 'CN', 'CZ', 'CR', 'TR', 'CL', 'PR', 'BO', 'PH', 'DO', 'BE', 'SG',
        'EG', 'ID', 'AE', 'MY', 'JP', 'EE', 'PK', 'TN', 'RU', 'DZ', 'BG', 'JE', 'RS', 'DK',
        'MD', 'LU', 'MT', 'OM', 'NZ', 'ZA', 'LB', 'UG', 'KR', 'QA', 'AD', 'EC', 'PE', 'NO'
    ])
    company_location_options = sorted([
        'ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'NL', 'CH', 'CF', 'FI', 'UA', 'IE',
        'IL', 'CO', 'SE', 'SI', 'MX', 'FR', 'BR', 'PT', 'RU', 'TH', 'GH', 'HR', 'VN', 'EE',
        'AM', 'BA', 'KE', 'GR', 'MK', 'SG', 'LV', 'RO', 'PK', 'IT', 'MA', 'PL', 'AR', 'LT',
        'AU', 'IR', 'HU', 'CZ', 'TR', 'PR', 'AS', 'BO', 'PH', 'BE', 'ID', 'AT', 'EG', 'AE',
        'LU', 'MY', 'JP', 'DZ', 'CN', 'CL', 'DK', 'MD', 'MT', 'OM', 'NZ', 'ZA', 'LB', 'GI',
        'KR', 'QA', 'AD', 'EC', 'NO'
    ])

    # Боковая панель ввода
    with st.sidebar:
        st.header("Ввод данных для предсказания")
        with st.form("input_form"):
            work_year = st.selectbox("Год работы", work_year_options, index=3)
            experience_level = st.selectbox("Уровень опыта", experience_level_options, index=1)
            employment_type = st.selectbox("Тип занятости", employment_type_options, index=0)
            job_title = st.selectbox("Должность", job_title_options, index=job_title_options.index("Data Scientist"))
            salary_currency = st.selectbox("Валюта зарплаты", salary_currency_options, index=salary_currency_options.index("USD"))
            employee_residence = st.selectbox("Страна сотрудника", employee_residence_options, index=employee_residence_options.index("US"))
            remote_ratio = st.slider("Удалённо (%)", 0, 100, 0, 10)
            company_location = st.selectbox("Локация компании", company_location_options, index=company_location_options.index("US"))
            company_size = st.selectbox("Размер компании", company_size_options, index=2)

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