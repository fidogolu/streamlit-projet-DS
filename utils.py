import streamlit as st
import os
import pandas as pd
import numpy as np
# import glob

import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from shapely.geometry import Point
import matplotlib.patches as mpatches
import geopandas as gpd

import time
from datetime import datetime

import joblib

#############################################################################################################

# Transformation logarithmique
def log_transform(data, column,):
    data[column] = np.log1p(data[column])
    return data
    
#############################################################################################################

@st.cache_data
def load_df_sales_clean_ST():
    # Define the path to the CSV file
    folder_path_C = './data/processed/Sales'
    input_file = os.path.join(folder_path_C, 'df_sales_clean_ST.csv')

    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file, sep=';', index_col='date', parse_dates=['date'], on_bad_lines='skip', low_memory=False)

    # Calculate 50% of the total number of rows
    sample_size = int(0.5 * len(data))

    # Generate a random sample
    df_sales = data.sample(n=sample_size, random_state=1)  # random_state for reproducibility
    df_sales.dropna(inplace=True)
    df_sales = df_sales.drop_duplicates()
    
    return df_sales

#############################################################################################################

def make_evolution_mensuelle_plots(Train_pour_graph, Train_pour_graph_cp):

    # Global plot
    fig_mensuel_glob = px.line(
        Train_pour_graph,
        x="date",
        y="prix_m2_vente",
        title="Évolution mensuelle du prix moyen au m²",
        labels={"date": "Date", "prix_m2_vente": "Prix moyen (€ / m²)"},
    )

    fig_mensuel_glob.update_traces(mode="lines+markers")
    fig_mensuel_glob.update_layout(
        title_x=0.1,
        title_y=0.95,
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Prix moyen (€ / m²)",
        hovermode="x unified",
    )

    fig_mensuel_glob.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig_mensuel_glob.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

    st.plotly_chart(fig_mensuel_glob)

    # Department plot with dropdown
    fig_mensuel = px.line(
        Train_pour_graph_cp,
        x="date",
        y="prix_m2_vente",
        color="departement",
        title="Évolution mensuelle du prix moyen au m² par département",
        labels={
            "date": "Date",
            "prix_m2_vente": "Prix moyen (€ / m²)",
            "departement": "Département",
        },
    )

    fig_mensuel.update_traces(mode="lines+markers")
    fig_mensuel.update_layout(
        title_x=0.1,
        title_y=0.95,
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Prix moyen (€ / m²)",
        legend_title_text="Département",
        hovermode="x unified",
    )

    fig_mensuel.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig_mensuel.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

    # Add dropdown menus to filter by department
    departement = Train_pour_graph_cp["departement"].unique()

    departement_buttons = [
        dict(
            label=str(cp),
            method="update",
            args=[
                {"visible": [cp == c for c in Train_pour_graph_cp["departement"]]},
                {"title": f"Évolution mensuelle pour le département {cp}"},
            ],
        )
        for cp in departement
    ]

    fig_mensuel.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=departement_buttons,
                direction="down",
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.1,
                yanchor="top",
                pad={"r": 10, "t": 10},
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )
        ]
    )

    return fig_mensuel
    
#############################################################################################################

def kmean_par_coude(df_cluster_input, number):
    features = [
        "prix_m2_mean",
        "prix_m2_std",
        "prix_m2_max",
        "prix_m2_min",
        "tc_am_reg",
        "prix_m2_cv",
    ]
    columns = ["codePostal_recons"] + features

    X = df_cluster_input[columns].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna()

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    # Déterminer le bon nombre de clusters (méthode du coude)
    inertias = []

    K_range = range(2, number)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Création du graphique avec Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(K_range, inertias, marker="o")
    ax.set_title("Méthode du coude")
    ax.set_xlabel("Nombre de clusters")
    ax.set_ylabel("Inertie intra-cluster")
    ax.grid(True)

    return fig

#############################################################################################################

@st.cache_data
def distrib_cluster(df_cluster_input):

    features = [
        "prix_m2_mean",
        "prix_m2_std",
        "prix_m2_max",
        "prix_m2_min",
        "tc_am_reg",
        "prix_m2_cv",
    ]
    columns = ["codePostal_recons"] + features

    X = df_cluster_input[columns].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna()

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    kmeans = KMeans(n_clusters=4, random_state=0)
    X["cluster"] = kmeans.fit_predict(X_scaled)

    # Créer le pair plot
    sns.pairplot(X, vars=features, hue="cluster", palette="tab10",height=1.6)
    # plt.suptitle("Distribution des indicateurs par cluster", y=1.02)

    # Affichage du pair plot dans Streamlit
    st.pyplot(plt)

#############################################################################################################

# Plot autocorrelation
def plot_autocorrelation(data):
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.plotting.autocorrelation_plot(data, ax=ax)
    st.pyplot(fig)

#############################################################################################################

# Function to generate and display ACF and PACF plots
def plot_acf_pacf(data):
    max_lags = len(data) // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    plot_acf(data, lags=max_lags, ax=ax1)
    plot_pacf(data, lags=max_lags, ax=ax2)
    st.pyplot(fig)

#############################################################################################################

     
def plot_explicatifs_par_cluster(geo_df):
    # Boxplots explicatifs par cluster
    features = ["prix_m2_mean", "prix_m2_std", "tc_am_reg", "prix_m2_cv"]

    for feature in features:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x="cluster", y=feature, data=geo_df, palette="tab10")
        plt.title(f"{feature} par cluster")
        plt.tight_layout()

        st.pyplot(plt)

#############################################################################################################

@st.cache_data
def make_correlation_matrix(train_periodique_q12):

    var_targ = [
        "prix_m2_vente", 
        "taux_rendement_n7", 
        "taux", 
        "loyer_m2_median_n7",
        "y_geo", 
        "x_geo", 
        "z_geo", 
        "dpeL", 
        "nb_pieces"
        ]

    # Calculer la matrice de corrélation
    correlation_matrix = train_periodique_q12[var_targ].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation des variables avec la cible')
    st.pyplot(plt)

    #features = ["taux_rendement_n7", 
    # 'taux', 
    # "loyer_m2_median_n7",
    # "y_geo", 
    # "x_geo", 
    # "z_geo", 
    # "dpeL", 
    # "nb_pieces", 
    # 'IPS_primaire',
    # 'rental_yield_pct',
    #]

    features = [
        "taux_rendement_n7", 
        'taux', 
        "loyer_m2_median_n7",
        "y_geo", 
        "x_geo", 
        "z_geo", 
        "dpeL", 
        "nb_pieces"
        ]

    #Définir les variables explicatives (qu’on souhaite décaler d’un mois)

    #features_lag = [
    # "taux_rendement_n7",  
    # "loyer_m2_median_n7", 
    # "nb_pieces", 
    # "IPS_primaire", 
    # "rental_yield_pct"
    #]

    # features_lag = ["taux_rendement_n7",  "loyer_m2_median_n7", "nb_pieces", ]

    # # Appliquer le lag 1 à chaque variable (valeur du mois précédent)
    # for col in features_lag:
    #     train_periodique_q12[col] = train_periodique_q12[col].shift(1)


    # # Supprimer les lignes avec valeurs manquantes (au début, à cause du lag)
    # train_periodique_q12 = train_periodique_q12.dropna(subset= features + ["prix_m2_vente"])

    # correlation_matrix = train_periodique_q12[var_targ].corr()

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Matrice de corrélation des variables avec un lag d\'un mois')
    

    # st.pyplot(plt)

#############################################################################################################

@st.cache_data
def seasonal_decompose_for_clusters(train_periodique_q12):

    # features = ["taux_rendement_n7", 'taux', "loyer_m2_median_n7","y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces"]
    # train_periodique_q12 = train_periodique_q12.dropna(subset= features + ["prix_m2_vente"])

    for cluster in sorted(train_periodique_q12["cluster"].dropna().unique()):
        
        # Extraire la série de prix par cluster
        df_cluster = train_periodique_q12[train_periodique_q12["cluster"] == cluster]
        y = df_cluster["prix_m2_vente"]
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()

        # Décomposition additive
        data_add = seasonal_decompose(y, model="additive", period=12)
        # plt.figure(figsize=(6, 6))
        fig = data_add.plot()
        fig.set_size_inches((18, 10), forward=True)
        fig.suptitle(f"Décomposition additive – Cluster {cluster}", fontsize=25)
        fig.tight_layout()
        st.pyplot(plt)


        # Décomposition multiplicative
        data_mult = seasonal_decompose(y, model="multiplicative", period=12)
        # plt.figure(figsize=(6, 6))
        fig = data_mult.plot()
        fig.set_size_inches((15, 10), forward=True)
        fig.suptitle(f"Décomposition multiplicative – Cluster {cluster}", fontsize=25)
        fig.tight_layout()
        st.pyplot(plt)

#############################################################################################################

@st.cache_data
def draw_clusters_cvs(train_periodique_q12):
    for cluster in sorted(train_periodique_q12["cluster"].dropna().unique()):

        # Filtrer les données du cluster
        df_cluster = train_periodique_q12[train_periodique_q12["cluster"] == cluster]

        # Extraire la série
        y = df_cluster["prix_m2_vente"]
        y.index = pd.DatetimeIndex(df_cluster.index).to_period("M").to_timestamp()

        # Décomposition multiplicative

        data_mult = seasonal_decompose(y, model="multiplicative", period=12)

        # Correction : log(y) - composante saisonnière => puis exp()
        cvs = y / data_mult.seasonal
        y_corrige = np.exp(cvs)

        # Affichage
        plt.figure(figsize=(10, 4))
        plt.plot(np.exp(y), label="Série originale")
        plt.plot(y_corrige, label="Corrigée saisonnalité", linestyle="--")
        plt.title(f"Série originale vs corrigée – Cluster {cluster}")
        plt.xlabel("Date")
        plt.ylabel("Prix/m²")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

#############################################################################################################

@st.cache_data
def differeciate_cluster(train_periodique_q12):

    # cluster 0
    clusters_st_0 = pd.DataFrame()
    df_cluster_0 = train_periodique_q12[train_periodique_q12["cluster"] == 0]
   
    y = df_cluster_0["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )
    # Concaténation
    clusters_st_0 = pd.concat([clusters_st_0, y_diff_order_1], axis=0)
    clusters_st_0["cluster"] = 0

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 0")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    test_stationarity(clusters_st_0["diff_order_1"], window=12)

    # cluster 1
    clusters_st_1 = pd.DataFrame()
    df_cluster_1 = train_periodique_q12[train_periodique_q12["cluster"] == 1]
    
    y = df_cluster_1["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()
    y_diff_order_2 = y_diff_order_1.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")
    y_diff_order_2 = y_diff_order_2.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)
    y_diff_order_2.rename(columns={"prix_m2_vente": "diff_order_2"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )
    y_diff_order_2.index = (
        pd.DatetimeIndex(y_diff_order_2.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_1 = pd.concat([y_diff_order_1, y_diff_order_2], axis=1)
    clusters_st_1["cluster"] = 1

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(121)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 1")
    plt.grid(True)

    plt.subplot(122)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title(f"Différenciation ordre 2 – Cluster 1")
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

    test_stationarity(clusters_st_1["diff_order_2"], window=12)

    # cluster 2
    clusters_st_2 = pd.DataFrame()
    df_cluster_2 = train_periodique_q12[train_periodique_q12["cluster"] == 2]

    y = df_cluster_2["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_2 = pd.concat([clusters_st_2, y_diff_order_1], axis=0)
    clusters_st_2["cluster"] = 2

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 2")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    
    test_stationarity(clusters_st_2["diff_order_1"], window=12)

    # cluster 3
    clusters_st_3 = pd.DataFrame()
    df_cluster_3 = train_periodique_q12[train_periodique_q12["cluster"] == 3]
    
    y = df_cluster_3["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_3 = pd.concat([clusters_st_3, y_diff_order_1], axis=0)
    clusters_st_3["cluster"] = 3

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 3")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    
    test_stationarity(clusters_st_3["diff_order_1"], window=12)

    return clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3


#############################################################################################################

def test_stationarity(timeseries, window=12):
    # Calculer la moyenne mobile et l'écart type mobile
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Tracer la série temporelle, la moyenne mobile et l'écart type mobile
    # plt.figure(figsize=(20, 6))
    # plt.plot(timeseries, color="blue", label="Série originale")
    # plt.plot(rolmean, color="red", label="Moyenne mobile")
    # plt.plot(rolstd, color="black", label="Écart type mobile")
    # plt.legend(loc="best")
    # plt.title("Moyenne mobile et écart type mobile")
    # plt.grid(True)
    # st.pyplot(plt)

    # Effectuer le test ADF
    result = adfuller(timeseries.dropna())
    st.write("Résultats du test ADF:")
    st.write("Statistique ADF:", result[0])
    st.write("p-value:", result[1])
    # st.write("Valeurs critiques:")
    # for key, value in result[4].items():
    #     st.write(f"\t{key}: {value}")

    # Interprétation des résultats
    if result[1] < 0.05:
        st.write("La série est stationnaire (p-value < 0.05)")
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)
    else:
        st.write("La série n'est pas stationnaire (p-value >= 0.05)")
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

#############################################################################################################

def plot_acf_pacf(clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3):

    cluster = 0
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_0[clusters_st_0["cluster"] == cluster]

    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)



    series = clusters_st_1["diff_order_2"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    plot_acf(series, lags=len(series) // 2, ax=ax1, title="ACF diff_order_2 – Cluster 1")
    plot_pacf(series, lags=len(series) // 2, ax=ax2, title="PACF diff_order_2 – Cluster 1")
    st.pyplot(plt)



    cluster = 2
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_2[clusters_st_2["cluster"] == cluster]
    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)


    cluster = 3
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_3[clusters_st_3["cluster"] == cluster]
    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)

#############################################################################################################

def simulate_grid_search_cluster_0():
    total_combinations = 4464
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 3.43it/s]")

        time.sleep(0.0001)  # Simulate some processing time

    # Print the best parameters found
    st.text("--- SARIMAX - Cluster 0 ---")
    st.text("Meilleure combinaison d'exogènes : ('z_geo', 'x_geo')")
    st.text("Meilleur ordre (p,d,q) : (0, 1, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -244.53759005266392")

    cluster_0_model = joblib.load('./models/best_sarimax_cluster0.joblib')
    st.write('### Résumé du modèle SARIMAX pour le cluster 0')
    st.write(cluster_0_model.summary())

    fig = cluster_0_model.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)
#############################################################################################################

def simulate_grid_search_cluster_1():
    total_combinations = 4464
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 3.43it/s]")
        time.sleep(0.0001)  # Simulate some processing time

    # Print the best parameters found
    st.text("--- SARIMAX - Cluster 1 ---")
    st.text("Meilleure combinaison d'exogènes : ('x_geo',)")
    st.text("Meilleur ordre (p,d,q) : (0, 2, 2)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -193.9435592356376")

    cluster_1_model = joblib.load('./models/best_sarimax_cluster1.joblib')
    st.write('### Résumé du modèle SARIMAX pour le cluster 1')
    st.write(cluster_1_model.summary())

    fig = cluster_1_model.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)

#############################################################################################################

def simulate_grid_search_cluster_2():
    total_combinations = 4464
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 3.43it/s]")
        time.sleep(0.0001)  # Simulate some processing time

    # Print the best parameters found
    st.text("--- SARIMAX - Cluster 2 ---")
    st.text("Meilleure combinaison d'exogènes : ('z_geo', 'x_geo')")
    st.text("Meilleur ordre (p,d,q) : (0, 1, 2)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -221.15676688592735")

    cluster_2_model = joblib.load('./models/best_sarimax_cluster2.joblib')
    st.write('### Résumé du modèle SARIMAX pour le cluster 2')
    st.write(cluster_2_model.summary())

    fig = cluster_2_model.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)
#############################################################################################################

def simulate_grid_search_cluster_3():
    total_combinations = 4464
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 3.43it/s]")
        time.sleep(0.0001)  # Simulate some processing time

    # Print the best parameters found
    st.text("--- SARIMAX - Cluster 3 ---")
    st.text("Meilleure combinaison d'exogènes : ('loyer_m2_median_n7', 'x_geo')")
    st.text("Meilleur ordre (p,d,q) : (1, 1, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -181.32407221862943")
    
    cluster_3_model = joblib.load('./models/best_sarimax_cluster3.joblib')
    st.write('### Résumé du modèle SARIMAX pour le cluster 3')
    st.write(cluster_3_model.summary())

    fig = cluster_3_model.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)
#############################################################################################################




# def my_function():
    # # Define the pattern to match the split files
    # file_pattern = 'data/raw/Sales/merged_sales_data_part_*.csv'
    
    # # Use glob to get a list of all split files
    # split_files = glob.glob(file_pattern)

    # # Sort the list of files to ensure they are in the correct order
    # split_files.sort()

    # # Initialize an empty list to store DataFrames
    # dataframes = []

    # # Read each split file and append its DataFrame to the list
    # for file in split_files:
    #     df = pd.read_csv(file)
    #     dataframes.append(df)

    # # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dataframes, ignore_index=True)

    # # Define the output file path for the combined CSV file
    # output_file_path = 'data/raw/Sales/merged_sales_data2.csv'

    # # Save the combined DataFrame to a CSV file
    # combined_df.to_csv(output_file_path, index=False)

    # st.write(f"Combined file saved to {output_file_path}")

