import streamlit as st
import pandas as pd

from utils import load_df_sales_clean_ST, make_evolution_mensuelle_plots, kmean_par_coude, distrib_cluster
from utils import plot_explicatifs_par_cluster

#############################################################################################################
# Title
#############################################################################################################

st.title("Premières explorations des séries temporelles")
st.write("Prix de vente du m² sur l'ensemble de la France")

#############################################################################################################
# Load the dataset
#############################################################################################################

df_sales = load_df_sales_clean_ST()
st.dataframe(df_sales.sort_index().head())

#############################################################################################################
# Draw the distribution of the target variable
#############################################################################################################

# st.write("### Évolution mensuelle du prix moyen au m²")    
# st.image('./images/evo_mensuelle_prix_moyen_m2.png', caption='Évolution mensuelle du prix moyen au m²',width=800)

# Data preparation for the global plot
Train_pour_graph = pd.read_csv('./data/temp/Train_pour_graph.csv', sep=';', parse_dates=['date'], on_bad_lines='skip', low_memory=False)

# Data preparation for the department plot
Train_pour_graph_cp = pd.read_csv('./data/temp/Train_pour_graph_cp.csv', sep=';',  parse_dates=['date'], on_bad_lines='skip', low_memory=False)

st.write("## Évolution mensuelle du prix moyen au m²")

fig_mensuel = make_evolution_mensuelle_plots(Train_pour_graph, Train_pour_graph_cp)
st.plotly_chart(fig_mensuel)


st.write('### Tendances générales :  \n')
st.write('La plupart des départements montrent une stabilité relative dans les prix, avec des variations saisonnières mineures.')
st.write('Mais certains départements montrent des augmentations ou des diminutions plus prononcées et à certains moments')

#############################################################################################################
# Determmine the number of clusters by elbow method
#############################################################################################################

st.write("## Clustering ")
st.write('### Déterminer le nombre de clusters (méthode du coude)')

df_cluster_input = pd.read_csv('./data/temp/df_cluster_input.csv', sep=';')

number_ok_cluster = st.slider("Nombre de clusters", min_value=2, max_value=10, value=2)
kmean_by_elbow = kmean_par_coude(df_cluster_input, number_ok_cluster)

st.pyplot(kmean_by_elbow)

st.write('Le point où l\'ajout de nouveaux clusters ne réduit plus significativement l\'inertie est souvent considéré comme le nombre optimal de clusters.')

#############################################################################################################
# dsitribution of the indicators by cluster
#############################################################################################################

st.write("### Distribution des indicateurs par cluster (4) ")

distrib_cluster(df_cluster_input)

st.write('\n')

st.markdown("""
| Cluster | Couleur | Niveau de prix     | Volatilité       | Croissance (`tc_am_reg`) | Interprétation économique                                         |
|---------|---------|--------------------|------------------|---------------------------|-------------------------------------------------------------------|
| 0       | Bleu    | Moyen à élevé       | Faible           | Modérée                   | Centres urbains établis, zones résidentielles stables            |
| 1       | Orange  | Moyen à élevé       | Élevée           | Modérée à forte           | Banlieues, zones mixtes ou périurbaines en transformation         |
| 2       | Vert    | Faible              | Très faible      | Faible à négative         | Zones rurales, petites villes stagnantes ou décroissantes        |
| 3       | Rouge   | Très élevé          | Très élevée      | Modérée à forte           | Zones tendues : luxe, hypercentres, littoraux, secteurs spéculatifs |
""")


#############################################################################################################
# Map
#############################################################################################################
st.write('### Visualisation des clusters sur la carte')
st.image('./images/carte_clusters.png', width=600)

#############################################################################################################
# Explanations
#############################################################################################################
st.write('### Boxplots explicatifs par cluster')

geo_df = pd.read_csv('./data/temp/geo_df.csv', sep=';')

plot_explicatifs_par_cluster(geo_df)

st.markdown("""
Nous sommes bien sur les clusters suivants :

- **Cluster 3 (rouge) — zone de luxe / tendue**
  - Clairement séparé en haut à droite de presque tous les nuages de points.
  - Prix très élevés (mean, max, min), dispersion (std) forte.
  - TCAM (tc_am_reg) souvent positif.
  - Très cohérent avec des zones chères, touristiques ou spéculatives.

- **Cluster 0 (bleu) — ville dense, mature**
  - Zones à prix modérément élevés mais stables.
  - Prix moyens comparables au cluster orange, voire légèrement supérieurs.
  - Variance (écart-type) plus faible : le marché est plus homogène.
  - TCAM souvent modéré → zones matures et stabilisées, comme des centres-villes établis ou zones résidentielles stables.

- **Cluster 1 (orange) — zones périurbaines ou dynamiques secondaires**
  - Ville mixte, périurbaine ou segmentée.
  - Prix moyens similaires au cluster bleu.
  - Variance plus forte → marché plus dispersé, peut-être entre anciens et nouveaux quartiers, zones périurbaines en mutation.
  - TCAM modérément positif → marchés dynamiques ou en rattrapage, mais moins structurés que le bleu.

- **Cluster 2 (vert) — zones rurales / peu dynamiques**
  - Prix très bas et très homogènes.
  - Très peu de variance → marché stagnant.
  - TCAM parfois négatif → zones en décroissance ou stagnation.
""")




