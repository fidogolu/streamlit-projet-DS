import streamlit as st
import pandas as pd

from utils import make_correlation_matrix,seasonal_decompose_for_clusters, draw_clusters_cvs, differeciate_cluster
from utils import plot_acf_pacf
from utils import grid_search_cluster_0, grid_search_cluster_1, grid_search_cluster_2, grid_search_cluster_3

#############################################################################################################
# Title
#############################################################################################################

st.title("Modélisation des séries temporelles")

#############################################################################################################
# Load the dataset
#############################################################################################################

train_periodique_q12 = pd.read_csv('./data/processed/Sales/train_periodique_q12.csv', sep=';' ,index_col="date", parse_dates=["date"])

train_periodique_q12 = train_periodique_q12.loc[:, ~train_periodique_q12.columns.str.contains('^Unnamed')]
train_periodique_q12.sort_index(inplace=True)

st.write('Données d\'entrainement')

config = {
    "_index": st.column_config.DateColumn("Date", format="MMM YYYY"),
}

st.dataframe(train_periodique_q12.sort_index().head(),column_config=config)

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

#############################################################################################################
# Draw CVS series
#############################################################################################################

# st.write('### Les times series corrigées des variations saisonnières (CVS)')

# draw_clusters_cvs(train_periodique_q12)

#############################################################################################################
# Stationarization
#############################################################################################################

st.write('### Différenciation et stationnarisation')

clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3 = differeciate_cluster(train_periodique_q12)

#############################################################################################################
# ACF et PACF
#############################################################################################################

st.write('### ACF et PACF')
st.write("""Pour déterminer les ordres  p ,  q ,  P , et  Q  d'un modèle ARIMA saisonnier (SARIMA), les fonctions d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) sont des outils essentiels.""")

plot_acf_pacf(clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3)

st.write('Rechercher les paramètres optimaux pour un modèle SARIMAX en utilisant les graphiques ACF (Autocorrelation Function) et PACF (Partial Autocorrelation Function) peut être difficile et subjectif.')
st.write('Une approche alternative consiste à utiliser GridSearch pour automatiser et optimiser cette recherche.')

#############################################################################################################
# GridSearch pour SARIMAX
#############################################################################################################

st.write('### GridSearch pour SARIMAX')
st.write('Le GridSearch consiste à tester toutes les combinaisons possibles, d\'ordres (p, d, q) et de paramètres saisonniers (P, D, Q, s) pour trouver la combinaison qui minimise un critère d\'information tel que l\'AIC (Akaike Information Criterion).')  

if st.button('Lancer GridSearch'):
    grid_search_cluster_0()
    grid_search_cluster_1()
    grid_search_cluster_2()
    grid_search_cluster_3()


