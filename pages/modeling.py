
import plotly.express as px
from functions.model_utils import *

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
    
    job_title_options = get_job_title_options()

    # Создаем словари
    country_dict = get_country_data()
    experience_level_dict = get_experience_level_data()
    employment_type_dict = get_employment_type_data()
    company_size_dict = get_company_size_data()
    salary_currency_dict = get_salary_currency_data()

    # Обратные словари: key=русское название, value=код
    reverse_country_dict = {v: k for k, v in country_dict.items()}
    reverse_experience_level_dict = {v: k for k, v in experience_level_dict.items()}
    reverse_employment_type_dict = {v: k for k, v in employment_type_dict.items()}
    reverse_company_size_dict = {v: k for k, v in company_size_dict.items()}
    reverse_salary_currency_dict = {v: k for k, v in salary_currency_dict.items()}

    # Боковая панель ввода
    with st.sidebar:
        st.header("Ввод данных для предсказания")
        with st.form("input_form"):
            # Локация компании
            company_location_code = st.selectbox(
                "Локация компании", 
                list(reverse_country_dict.keys()), 
                index=list(reverse_country_dict.values()).index("US")  # по умолчанию = "US"
            )
            company_location = reverse_country_dict[company_location_code]

            # Страна сотрудника
            employee_residence_code = st.selectbox(
                "Страна сотрудника", 
                list(reverse_country_dict.keys()), 
                index=list(reverse_country_dict.values()).index("US")  # по умолчанию = "US"
            )
            employee_residence = reverse_country_dict[employee_residence_code]
           
            work_year = st.selectbox("Год работы", work_year_options, index=4)  # 2024 по умолчанию

            # Уровень опыта (пример: по умолчанию "Средний")
            default_experience_label = "Средний"
            experience_level_label = st.selectbox(
                "Уровень опыта", 
                list(reverse_experience_level_dict.keys()), 
                index=list(reverse_experience_level_dict.keys()).index(default_experience_label)
            )
            # Получаем код, например "MI"
            experience_level = reverse_experience_level_dict[experience_level_label]

            # Тип занятости (пример: по умолчанию "Полная занятость")
            default_employment_label = "Полная занятость"
            employment_type_label = st.selectbox(
                "Тип занятости", 
                list(reverse_employment_type_dict.keys()), 
                index=list(reverse_employment_type_dict.keys()).index(default_employment_label)
            )
            employment_type = reverse_employment_type_dict[employment_type_label]

            # Должность
            job_title = st.selectbox("Должность", job_title_options, index=job_title_options.index("Data Scientist"))

            # Валюта (пример: по умолчанию "USD")
            default_currency_label = "Доллар США"
            salary_currency_label = st.selectbox(
                "Валюта зарплаты", 
                list(reverse_salary_currency_dict.keys()), 
                index=list(reverse_salary_currency_dict.keys()).index(default_currency_label)
            )
            salary_currency = reverse_salary_currency_dict[salary_currency_label]

            # Удалённо %
            remote_ratio = st.slider("Удалённо (%)", 0, 100, 0, 50)

            # Размер компании (пример: по умолчанию "Большая компания")
            default_company_size_label = "Большая компания"
            company_size_label = st.selectbox(
                "Размер компании", 
                list(reverse_company_size_dict.keys()), 
                index=list(reverse_company_size_dict.keys()).index(default_company_size_label)
            )
            company_size = reverse_company_size_dict[company_size_label]

            submit = st.form_submit_button("Предсказать")

    # Центральная часть: предсказания и визуализация
    if submit:
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
                weighted_sum = sum(predictions[m] * (model_weights[m] / 100) for m in valid_models)
                weighted_pred = weighted_sum

            with st.container():
                st.subheader("Результаты предсказаний")
                col1, col2 = st.columns(2)
                
                # Вывод результатов по моделям
                with col1:
                    for model_name, val in predictions.items():
                        weight = model_weights.get(model_name, 0)
                        if model_name in ["Random Forest", "CatBoost"]:
                            st.markdown(f"**{model_name} (вес {weight:.2f}%):** ${val:,.2f} USD", unsafe_allow_html=True)
                        else:
                            st.write(f"**{model_name} (вес {weight:.2f}%):** ${val:,.2f} USD")
                
                # Средние значения
                with col2:
                    st.write(f"**Обычное среднее:** ${avg_pred:,.2f} USD")
                    st.write(f"**Взвешенное среднее:** ${weighted_pred:,.2f} USD")

                # Визуализация
                pred_df = pd.DataFrame({
                    'Модель': list(predictions.keys()),
                    'Предсказание (USD)': list(predictions.values()),
                    'Вес (%)': [model_weights[m] for m in predictions.keys()]
                })
                fig = px.bar(
                    pred_df, 
                    x='Модель', 
                    y='Предсказание (USD)', 
                    color='Вес (%)', 
                    title="Предсказания трёх моделей",
                    labels={'Предсказание (USD)': 'Зарплата (USD)', 'Вес (%)': 'Вес модели (%)'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=True, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Пояснение к расчету взвешенного среднего
                st.markdown("### Как рассчитано взвешенное среднее?")
                st.write(
                    """
                    Взвешенное среднее вычисляется как:
                    \n`Взвешенное среднее = (Предсказание_1 × Вес_1 + Предсказание_2 × Вес_2 + Предсказание_3 × Вес_3) / 100`, 
                    где веса нормализованы до 100% на основе R^2 Test каждой модели. 
                    """
                )


if __name__ == "__main__":
    main()
