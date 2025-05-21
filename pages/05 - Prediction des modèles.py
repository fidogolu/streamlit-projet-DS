import streamlit as st
import joblib

#############################################################################################################
# SARIMAX models summary
#############################################################################################################

cluster_0_model = joblib.load('./models/best_sarimax_cluster0.joblib')
st.write('### Résumé du modèle SARIMAX pour le cluster 0')
st.write(cluster_0_model.summary())

fig = cluster_0_model.plot_diagnostics(figsize=(15, 12))
st.pyplot(fig)

cluster_1_model = joblib.load('./models/best_sarimax_cluster1.joblib')
st.write('### Résumé du modèle SARIMAX pour le cluster 1')
st.write(cluster_1_model.summary())

fig = cluster_1_model.plot_diagnostics(figsize=(15, 12))
st.pyplot(fig)

cluster_2_model = joblib.load('./models/best_sarimax_cluster2.joblib')
st.write('### Résumé du modèle SARIMAX pour le cluster 2')
st.write(cluster_2_model.summary())

fig = cluster_2_model.plot_diagnostics(figsize=(15, 12))
st.pyplot(fig)

cluster_3_model = joblib.load('./models/best_sarimax_cluster3.joblib')
st.write('### Résumé du modèle SARIMAX pour le cluster 3')
st.write(cluster_3_model.summary())

fig = cluster_3_model.plot_diagnostics(figsize=(15, 12))
st.pyplot(fig)