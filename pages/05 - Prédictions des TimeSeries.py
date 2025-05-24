import streamlit as st
import pandas as pd

from utils import make_sarimax_prediction

#############################################################################################################
# Load the dataset
#############################################################################################################

train_periodique_q12 = pd.read_csv('./data/processed/Sales/train_periodique_q12.csv', sep=';' ,index_col="date", parse_dates=["date"])
test_periodique_q12 = pd.read_csv('./data/processed/Sales/test_periodique_q12.csv', sep=';' ,index_col="date", parse_dates=["date"])

train_periodique_q12.sort_index(inplace=True)
test_periodique_q12.sort_index(inplace=True)    

#############################################################################################################
# Prédictions pour SARIMAX
#############################################################################################################

st.title("Prédictions des time series avec SARIMAX")

st.write('### Prédictions pour le cluster 0')
make_sarimax_prediction(train_periodique_q12, test_periodique_q12,cluster_number=0,order=(1, 0, 0), seasonal_order=(0, 1, 0, 24))

st.write('### Prédictions pour le cluster 1')
make_sarimax_prediction(train_periodique_q12, test_periodique_q12,cluster_number=1,order=(2, 2, 2), seasonal_order=(0, 2, 0, 14))

st.write('### Prédictions pour le cluster 2')
make_sarimax_prediction(train_periodique_q12, test_periodique_q12,cluster_number=2,order=(1, 1, 1), seasonal_order=(0, 1, 0, 36))

st.write('### Prédictions pour le cluster 3')
make_sarimax_prediction(train_periodique_q12, test_periodique_q12,cluster_number=3,order=(0, 1, 0), seasonal_order=(0, 1, 0, 36))

#############################################################################################################
# Conclusions
#############################################################################################################

st.write('## Conclusions')

st.write("""
    Certaines variables exogènes utilisées (notamment taux_rendement_n7, loyer_m2_median_n7, rental_yield_pct)
    sont en réalité structurellement endogènes. Leur inclusion dans un cadre prédictif est acceptable uniquement
    si elles sont lagguées et connues au moment t-1. Leur définition exacte doit être clarifiée auprès des équipes métier.
    """)

st.write("Les 4 clusters atteignent des MAPE très raisonnables, même dans un contexte mensuel immobilier complexe.")