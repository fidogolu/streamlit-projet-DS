import streamlit as st
from utils.data_processing import process_data, try_read_csv

# Étape 1 : Upload du fichier
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
if uploaded_file:
    # Charger les données brutes
    data_raw = try_read_csv(uploaded_file)
    st.write("Données brutes :", data_raw.head())

    # Étape 2 : Orchestration des transformations
    processed_data = process_data(data_raw)

    # Option pour afficher ou masquer les graphiques
    show_plots = st.checkbox("Afficher les graphiques", value=True)

    # Sélection de l'étape à afficher
    step = st.selectbox(
        "Choisissez une étape à afficher :",
        [
            "Suppression des doublons",
            "Gestion des valeurs manquantes",
            "Colonnes catégoriques",
            "Nettoyage des colonnes",
            "Discrétisation de 'annee_construction'",
            "Suppression des colonnes inutiles",
            "Conversion en booléen",
            "Données finales"
        ]
    )

    # Affichage des données et des graphiques selon l'étape choisie
    if step == "Suppression des doublons":
        st.write(processed_data["data_no_duplicates"].head())
    elif step == "Gestion des valeurs manquantes":
        st.write(processed_data["data_no_missing"].head())
        if show_plots:
            st.pyplot(processed_data["plot_missing_values"])
    elif step == "Colonnes catégoriques":
        st.write("Colonnes catégoriques identifiées :", processed_data["categorical_columns"])
        if show_plots:
            selected_column = st.selectbox(
                "Choisissez une colonne catégorique à afficher :",
                processed_data["categorical_columns"]
            )
            st.pyplot(processed_data["plot_categorical_distributions"][selected_column])
    elif step == "Nettoyage des colonnes":
        st.write(processed_data["data_cleaned"].head())
    elif step == "Discrétisation de 'annee_construction'":
        st.write(processed_data["data_discretized"].head())
    elif step == "Suppression des colonnes inutiles":
        st.write(processed_data["data_reduced"].head())
    elif step == "Conversion en booléen":
        st.write(processed_data["data_boolean"].head())
    elif step == "Données finales":
        st.write(processed_data["data_final"].head())
        if show_plots:
            st.pyplot(processed_data["plot_correlation_matrix"])

    