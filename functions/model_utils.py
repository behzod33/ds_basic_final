import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np


# МЕТРИКИ ДЛЯ ВЕСОВ (R2 Test из таблицы, нормализованные до 100%)
r2_test_values = {
    "Linear Regression": 2,
    "Random Forest": 3.5,
    "CatBoost": 4.5
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


def get_country_data():
    country_data = [
        ("AU", "Австралия"),
        ("AT", "Австрия"),
        ("AZ", "Азербайджан"),
        ("AL", "Албания"),
        ("DZ", "Алжир"),
        ("AI", "Ангилья о. (GB)"),
        ("AO", "Ангола"),
        ("AD", "Андорра"),
        ("AQ", "Антарктика"),
        ("AG", "Антигуа и Барбуда"),
        ("AN", "Антильские о‐ва (NL)"),
        ("AR", "Аргентина"),
        ("AM", "Армения"),
        ("AW", "Аруба"),
        ("AF", "Афганистан"),
        ("BS", "Багамы"),
        ("BD", "Бангладеш"),
        ("BB", "Барбадос"),
        ("BH", "Бахрейн"),
        ("BY", "Беларусь"),
        ("BZ", "Белиз"),
        ("BE", "Бельгия"),
        ("BJ", "Бенин"),
        ("BM", "Бермуды"),
        ("BV", "Бове о. (NO)"),
        ("BG", "Болгария"),
        ("BO", "Боливия"),
        ("BA", "Босния и Герцеговина"),
        ("BW", "Ботсвана"),
        ("BR", "Бразилия"),
        ("BN", "Бруней Дарассалам"),
        ("BF", "Буркина‐Фасо"),
        ("BI", "Бурунди"),
        ("BT", "Бутан"),
        ("VU", "Вануату"),
        ("VA", "Ватикан"),
        ("GB", "Великобритания"),
        ("HU", "Венгрия"),
        ("VE", "Венесуэла"),
        ("VG", "Виргинские о‐ва (GB)"),
        ("VI", "Виргинские о‐ва (US)"),
        ("AS", "Восточное Самоа (US)"),
        ("TP", "Восточный Тимор"),
        ("VN", "Вьетнам"),
        ("GA", "Габон"),
        ("HT", "Гаити"),
        ("GY", "Гайана"),
        ("GM", "Гамбия"),
        ("GH", "Гана"),
        ("GP", "Гваделупа"),
        ("GT", "Гватемала"),
        ("GN", "Гвинея"),
        ("GW", "Гвинея‐Бисау"),
        ("DE", "Германия"),
        ("GI", "Гибралтар"),
        ("HN", "Гондурас"),
        ("HK", "Гонконг (CN)"),
        ("GD", "Гренада"),
        ("GL", "Гренландия (DK)"),
        ("GR", "Греция"),
        ("GE", "Грузия"),
        ("GU", "Гуам"),
        ("DK", "Дания"),
        ("CD", "Демократическая Республика Конго"),
        ("DJ", "Джибути"),
        ("DM", "Доминика"),
        ("DO", "Доминиканская Республика"),
        ("EG", "Египет"),
        ("ZM", "Замбия"),
        ("EH", "Западная Сахара"),
        ("ZW", "Зимбабве"),
        ("IL", "Израиль"),
        ("IN", "Индия"),
        ("ID", "Индонезия"),
        ("JO", "Иордания"),
        ("IQ", "Ирак"),
        ("IR", "Иран"),
        ("IE", "Ирландия"),
        ("IS", "Исландия"),
        ("ES", "Испания"),
        ("IT", "Италия"),
        ("YE", "Йемен"),
        ("CV", "Кабо‐Верде"),
        ("KZ", "Казахстан"),
        ("KY", "Каймановы о‐ва (GB)"),
        ("KH", "Камбоджа"),
        ("CM", "Камерун"),
        ("CA", "Канада"),
        ("QA", "Катар"),
        ("KE", "Кения"),
        ("CY", "Кипр"),
        ("KG", "Киргизстан"),
        ("KI", "Кирибати"),
        ("CN", "Китай"),
        ("CC", "Кокосовые (Киилинг) о‐ва (AU)"),
        ("CO", "Колумбия"),
        ("KM", "Коморские о‐ва"),
        ("CG", "Конго"),
        ("CR", "Коста‐Рика"),
        ("CI", "Кот‐д'Ивуар"),
        ("CU", "Куба"),
        ("KW", "Кувейт"),
        ("CK", "Кука о‐ва (NZ)"),
        ("LA", "Лаос"),
        ("LV", "Латвия"),
        ("LS", "Лесото"),
        ("LR", "Либерия"),
        ("LB", "Ливан"),
        ("LY", "Ливия"),
        ("LT", "Литва"),
        ("LI", "Лихтенштейн"),
        ("LU", "Люксембург"),
        ("MU", "Маврикий"),
        ("MR", "Мавритания"),
        ("MG", "Мадагаскар"),
        ("YT", "Майотта о. (KM)"),
        ("MO", "Макао (PT)"),
        ("MK", "Македония"),
        ("MW", "Малави"),
        ("MY", "Малайзия"),
        ("ML", "Мали"),
        ("MV", "Мальдивы"),
        ("MT", "Мальта"),
        ("MA", "Марокко"),
        ("MQ", "Мартиника"),
        ("MH", "Маршалловы о‐ва"),
        ("MX", "Мексика"),
        ("FM", "Микронезия (US)"),
        ("MZ", "Мозамбик"),
        ("MD", "Молдова"),
        ("MC", "Монако"),
        ("MN", "Монголия"),
        ("MS", "Монсеррат о. (GB)"),
        ("MM", "Мьянма"),
        ("NA", "Намибия"),
        ("NR", "Науру"),
        ("NP", "Непал"),
        ("NE", "Нигер"),
        ("NG", "Нигерия"),
        ("NL", "Нидерланды"),
        ("NI", "Никарагуа"),
        ("NU", "Ниуэ о. (NZ)"),
        ("NZ", "Новая Зеландия"),
        ("NC", "Новая Каледония о. (FR)"),
        ("NO", "Норвегия"),
        ("NF", "Норфолк о. (AU)"),
        ("AE", "Объединенные Арабские Эмираты"),
        ("OM", "Оман"),
        ("PK", "Пакистан"),
        ("PW", "Палау (US)"),
        ("PS", "Палестинская автономия"),
        ("PA", "Панама"),
        ("PG", "Папуа‐Новая Гвинея"),
        ("PY", "Парагвай"),
        ("PE", "Перу"),
        ("PN", "Питкэрн о‐ва (GB)"),
        ("PL", "Польша"),
        ("PT", "Португалия"),
        ("PR", "Пуэрто‐Рико (US)"),
        ("RE", "Реюньон о. (FR)"),
        ("CX", "Рождества о. (AU)"),
        ("RU", "Россия"),
        ("RW", "Руанда"),
        ("RO", "Румыния"),
        ("SV", "Сальвадор"),
        ("WS", "Самоа"),
        ("SM", "Сан Марино"),
        ("ST", "Сан‐Томе и Принсипи"),
        ("SA", "Саудовская Аравия"),
        ("SZ", "Свазиленд"),
        ("SJ", "Свалбард и Ян Мейен о‐ва (NO)"),
        ("SH", "Святой Елены о. (GB)"),
        ("KP", "Северная Корея (КНДР)"),
        ("MP", "Северные Марианские"),
        ("SC", "Сейшелы"),
        ("VC", "Сен‐Винсент и Гренадины"),
        ("PM", "Сен‐Пьер и Микелон (FR)"),
        ("SN", "Сенегал"),
        ("KN", "Сент‐Кристофер и Невис"),
        ("LC", "Сент‐Люсия"),
        ("SG", "Сингапур"),
        ("SY", "Сирия"),
        ("SK", "Словакия"),
        ("SI", "Словения"),
        ("US", "Соединенные Штаты Америки"),
        ("SB", "Соломоновы о‐ва"),
        ("SO", "Сомали"),
        ("SD", "Судан"),
        ("SR", "Суринам"),
        ("SL", "Сьерра‐Леоне"),
        ("TJ", "Таджикистан"),
        ("TH", "Таиланд"),
        ("TW", "Тайвань"),
        ("TZ", "Танзания"),
        ("TC", "Теркс и Кайкос о‐ва (GB)"),
        ("TG", "Того"),
        ("TK", "Токелау о‐ва (NZ)"),
        ("TO", "Тонга"),
        ("TT", "Тринидад и Тобаго"),
        ("TV", "Тувалу"),
        ("TN", "Тунис"),
        ("TM", "Туркменистан"),
        ("TR", "Турция"),
        ("UG", "Уганда"),
        ("UZ", "Узбекистан"),
        ("UA", "Украина"),
        ("WF", "Уоллис и Футунао‐ва (FR)"),
        ("UY", "Уругвай"),
        ("FO", "Фарерские о‐ва (DK)"),
        ("FJ", "Фиджи"),
        ("PH", "Филиппины"),
        ("FI", "Финляндия"),
        ("FK", "Фолклендские (Мальвинские) о‐ва (GB/AR)"),
        ("FR", "Франция"),
        ("GF", "Французская Гвиана (FR)"),
        ("PF", "Французская Полинезия"),
        ("HM", "Херд и Макдональд о‐ва (AU)"),
        ("HR", "Хорватия"),
        ("CF", "Центрально‐африканская Республика"),
        ("TD", "Чад"),
        ("CZ", "Чехия"),
        ("CL", "Чили"),
        ("CH", "Швейцария"),
        ("SE", "Швеция"),
        ("LK", "Шри‐Ланка"),
        ("EC", "Эквадор"),
        ("GQ", "Экваториальная Гвинея"),
        ("ER", "Эритрия"),
        ("EE", "Эстония"),
        ("ET", "Эфиопия"),
        ("YU", "Югославия"),
        ("ZA", "Южная Африка"),
        ("GS", "Южная Георгия и Южные Сандвичевы о‐ва"),
        ("KR", "Южная Корея (Республика Корея)"),
        ("JM", "Ямайка"),
        ("JP", "Япония"),
        ("TF", "Французские южные территории (FR)"),
        ("IO", "Британская территория Индийского океана (GB)"),
        ("UM", "Соединенные Штаты Америки Внешние малые острова (US)")
    ]
    
    return dict(country_data)


def get_experience_level_data():
    experience_level_data = [
        ("EN", "Начальный"),
        ("MI", "Средний"),
        ("SE", "Старший"),
        ("EX", "Эксперт")
    ]

    return dict(experience_level_data)

def get_employment_type_data():
    employment_type_data = [
        ("FT", "Полная занятость"),
        ("CT", "Контрактная занятость"),
        ("FL", "Фриланс"),
        ("PT", "Частичная занятость")
    ]

    return dict(employment_type_data)


def get_company_size_data():
    company_size_data = [
        ("S", "Маленькая компания"),
        ("M", "Средняя компания"),
        ("L", "Большая компания")
    ]
    
    return dict(company_size_data)


def get_salary_currency_data():
    salary_currency_data = [
        ('EUR', 'Евро'),
        ('USD', 'Доллар США'),
        ('INR', 'Индийская рупия'),
        ('HKD', 'Гонконгский доллар'),
        ('CHF', 'Швейцарский франк'),
        ('GBP', 'Британский фунт'),
        ('AUD', 'Австралийский доллар'),
        ('SGD', 'Сингапурский доллар'),
        ('CAD', 'Канадский доллар'),
        ('ILS', 'Израильский шекель'),
        ('BRL', 'Бразильский реал'),
        ('THB', 'Таиландский бат'),
        ('PLN', 'Польский злотый'),
        ('HUF', 'Венгерский форинт'),
        ('CZK', 'Чешская крона'),
        ('JPY', 'Японская иена'),
        ('MXN', 'Мексиканское песо'),
        ('TRY', 'Турецкая лира'),
        ('CLP', 'Чилийское песо'),
        ('DKK', 'Датская крона'),
        ('PHP', 'Филиппинское песо'),
        ('NOK', 'Норвежская крона'),
        ('ZAR', 'Южноафриканский рэнд')
    ]
    return dict(salary_currency_data)

def get_job_title_options():
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

    return job_title_options