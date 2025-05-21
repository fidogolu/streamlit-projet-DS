import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import make_correlation_matrix,seasonal_decompose_for_clusters, draw_clusters_cvs, differeciate_cluster
from utils import test_stationarity,plot_acf_pacf

#############################################################################################################
# Title
#############################################################################################################

st.title("Modélisation des séries temporelles")

#############################################################################################################
# Load the dataset
#############################################################################################################

train_periodique_q12 = pd.read_csv('./data/processed/Sales/train_periodique_q12.csv', sep=';' ,index_col="date", parse_dates=["date"])
test_periodique_q12 = pd.read_csv('./data/processed/Sales/test_periodique_q12.csv', sep=';' ,index_col="date", parse_dates=["date"])

train_periodique_q12.sort_index(inplace=True)
test_periodique_q12.sort_index(inplace=True)    

st.write('Données d\'entrainement')
st.dataframe(train_periodique_q12.sort_index().head())

# st.write('Données de test')
# st.dataframe(test_periodique_q12.sort_index().head())

#############################################################################################################
# Corelation matrix
#############################################################################################################
st.write('### Matrice de corrélation des variables avec la cible')

make_correlation_matrix(train_periodique_q12)

#############################################################################################################
# Determine if multiplicative or additive series
#############################################################################################################

st.write('## Modélisation - SARIMAX')
st.write('Choisir le modèle qui donne les résidus les plus stationnaires')

seasonal_decompose_for_clusters(train_periodique_q12)

st.write('### Les times series corrigées des variations saisonnières (CVS)')

#############################################################################################################
# Draw CVS series
#############################################################################################################

draw_clusters_cvs(train_periodique_q12)

#############################################################################################################
# Stationarization
#############################################################################################################

st.write('### Différenciation et stationnarisation')

clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3 = differeciate_cluster(train_periodique_q12)

#############################################################################################################
# ACF et PACF
#############################################################################################################

st.write('### ACF et PACF')
st.write("""Pour déterminer les ordres \( p \), \( q \), \( P \), et \( Q \) d'un modèle ARIMA saisonnier (SARIMA), les fonctions d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) sont des outils essentiels.""")

plot_acf_pacf(clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3)