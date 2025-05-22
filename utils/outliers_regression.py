import pandas as pd
import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


################################################################################
########################## FILTRAGE DES COLONNES NUMERIQUES ####################
################################################################################
def get_numeric_columns(data, group_col, excluded_cols=None):
    """
    Retourne les colonnes numériques en excluant une colonne de regroupement et d'autres colonnes spécifiées.
    """
    if excluded_cols is None:
        excluded_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']
    return [
        col for col in data.select_dtypes(include='number').columns
        if col != group_col and col not in excluded_cols
    ]

####################################################################################
########################## AFFICHAGE DES BOXPLOTS ##################################
####################################################################################

def plot_boxplots(data, numeric_cols, selected_cols=None):
    """
    Trace des boxplots pour les colonnes sélectionnées avec une couleur différente par variable.
    """
    if selected_cols is None:
        selected_cols = numeric_cols

    # Nombre de colonnes par ligne
    cols_per_row = 2

    # Calcul du nombre de lignes nécessaires
    num_cols = len(selected_cols)
    num_rows = math.ceil(num_cols / cols_per_row)

    # Création des sous-graphiques
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
    axes = axes.flatten()  # Aplatir pour un accès plus simple

    # Générer une couleur unique pour chaque variable
    colors = plt.cm.tab10.colors

    # Boucle pour tracer les boxplots
    for i, col in enumerate(selected_cols):
        color = colors[i % len(colors)]  # Réutiliser les couleurs si nécessaire
        data.boxplot(column=col, ax=axes[i], boxprops=dict(color=color), medianprops=dict(color=color))
        axes[i].set_title(f"Boxplot de la colonne '{col}'")

    # Supprimer les axes inutilisés si le nombre de colonnes est impair
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig

##################################################################################
########################## ANALYSE DES RÉSULTATS #################################
##################################################################################

def display_outlier_analysis():
    """
    Affiche une analyse des étapes de traitement des outliers.
    """
    st.markdown("""
    Les boxplots montrent des distributions étonnantes.
    Par ailleurs, certaines variables semblent montrer des problèmes d'unités d'échelle.
    Il s'agit en particulier des variables (non exhaustif):    
    - 'charges_copro'
    - 'loyer_m2_median_n6'
    - 'loyer_m2_median_n7'
    - 'taux_rendement_n6' 
    - 'taux_rendement_n7'
    - 'nb_log_n6'
    - 'nb_log_n7'

    Dans l'ordre, nous allons :
    - Éliminer les valeurs aberrantes via la détection d'anomalies logiques.
    - Éliminer les valeurs aberrantes via les anomalies visuelles (suite aux boxplots).
    - Séparer les données train et test pour éviter le data leakage.
    - Traiter les valeurs extrêmes en créant des fonctions de détection des "outliers" et d'imputation par médiane de code INSEE.
    """)

#################################################################################
########################## DÉTECTION DES ANOMALIES LOGIQUES ########################
#################################################################################


def detect_logical_anomalies(data):
    """
    Supprime les lignes contenant des anomalies logiques dans un DataFrame selon des règles prédéfinies.

    Args:
        data (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé des lignes contenant des anomalies logiques.
    """
    # Liste des règles d'anomalies logiques
    rules = [
        # Règle 1 : nb_toilettes > nb_pieces (pas logique dans un logement classique)
        (data['nb_toilettes'] > data['nb_pieces']),
        
        # Règle 2 : surface trop petite (< 10 m²) ou démesurée (> 1000 m²)
        (data['surface'] < 10) | (data['surface'] > 1000),
        
        # Règle 3 : nb_etages = 0 alors que etage > 0 (impossible sans étage)
        (data['nb_etages'] == 0) & (data['etage'] > 0),
        
        # Règle 4 : logement neuf mais année de construction ancienne (avant 2000)
        (data['logement_neuf'] == True) & (
            data['annee_construction'].isin([
                "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988", "1989-2000"
            ])
        ),
        
        # Règle 5 : prix_m2_vente très bas ou nul (hors outlier déjà traité)
        (data['prix_m2_vente'] < 100)
    ]

    # Combiner toutes les règles pour identifier les lignes à supprimer
    combined_rule = pd.concat(rules, axis=1).any(axis=1)

    # Supprimer les lignes contenant des anomalies logiques
    data = data[~combined_rule].copy()

    # Résumé : Nombre total de lignes supprimées
    nb_anomalies = combined_rule.sum()
    print(f"{nb_anomalies} lignes contenant des anomalies logiques ont été supprimées.")

    return data

####################################################################################
########################## DÉTECTION DES ANOMALIES DE SAISIE #######################
####################################################################################

def remove_improbable_values(data):
    """
    Supprime les lignes contenant des valeurs improbables selon des seuils définis
    et retourne un DataFrame nettoyé ainsi qu'un rapport des suppressions.

    Args:
        data (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
        pd.DataFrame: Un rapport des suppressions par colonne.
    """
    # Définir les colonnes suspectes et les seuils
    cols_suspectes = ['charges_copro', 'loyer_m2_median_n6', 'loyer_m2_median_n7', 
                      'taux_rendement_n6', 'taux_rendement_n7', 'nb_log_n6', 'nb_log_n7']
    seuils_max = {
        'charges_copro': 10000,
        'loyer_m2_median_n6': 100,
        'loyer_m2_median_n7': 100,
        'taux_rendement_n6': 50,
        'taux_rendement_n7': 50,
        'nb_log_n6': 1000,
        'nb_log_n7': 1000
    }
    seuils_min = {
        'charges_copro': 0,
        'loyer_m2_median_n6': 5,
        'loyer_m2_median_n7': 5,
        'taux_rendement_n6': 0,
        'taux_rendement_n7': 0,
        'nb_log_n6': 1,
        'nb_log_n7': 1
    }

    problemes = {}
    mask_valeurs_improbables = pd.Series(False, index=data.index)  # Initialiser le masque

    for col in cols_suspectes:
        # Détection des valeurs au-dessus du seuil maximum
        if col in seuils_max:
            mask_above = data[col] > seuils_max[col]
            mask_valeurs_improbables |= mask_above
            n_anormaux_max = mask_above.sum()
        else:
            n_anormaux_max = 0

        # Détection des valeurs en dessous du seuil minimum
        if col in seuils_min:
            mask_below = data[col] < seuils_min[col]
            mask_valeurs_improbables |= mask_below
            n_anormaux_min = mask_below.sum()
        else:
            n_anormaux_min = 0

        # Ajouter au rapport si des valeurs aberrantes sont détectées
        if n_anormaux_max > 0 or n_anormaux_min > 0:
            problemes[col] = {
                'nb_anormaux_max': n_anormaux_max,
                'max_valeur': data[col].max(),
                'nb_anormaux_min': n_anormaux_min,
                'min_valeur': data[col].min()
            }

    # Création d'un DataFrame pour le rapport
    df_problemes = pd.DataFrame.from_dict(problemes, orient='index')

    # Supprimer les lignes contenant des valeurs improbables
    data = data[~mask_valeurs_improbables].copy()

    # Nombre total de lignes supprimées
    nb_lignes_supprimees = mask_valeurs_improbables.sum()
    print(f"{nb_lignes_supprimees} lignes contenant des valeurs improbables ont été supprimées.")

    return data, df_problemes



###################################################################################
########################## SPLIT TRAIN/TEST ######################################
####################################################################################    

def split_train_test(data):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.

    Returns:
        tuple: A tuple containing the training set (train_data) and the testing set (test_data).
    """
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data
    



###################################################################################
########################## TRAITEMENT DES OUTLIERS ################################
###################################################################################

def process_outliers_sequence(
    train_data,
    test_data,
    group_col="INSEE_COM",
    lower_perc=0.01,
    upper_perc=0.99,
    outlier_tag=-999,
    target_col="prix_m2_vente"
):
    """
    Traite les outliers dans une séquence complète en prenant en entrée train_data et test_data.
    
    Parameters:
        train_data (pd.DataFrame): Données d'entraînement.
        test_data (pd.DataFrame): Données de test.
        group_col (str): Colonne utilisée pour regrouper les données (par défaut "INSEE_COM").
        lower_perc (float): Quantile inférieur pour détecter les outliers (par défaut 0.01).
        upper_perc (float): Quantile supérieur pour détecter les outliers (par défaut 0.99).
        outlier_tag (int): Valeur utilisée pour marquer les outliers (par défaut -999).
        target_col (str): Nom de la colonne cible à séparer (par défaut "TARGET_COL").
    
    Returns:
        tuple: X_train, y_train, X_test, y_test (features et cibles pour train et test).
    """

    # Sous-fonction : Calculer les bornes
    def calculate_bounds(data, numeric_cols, lower_perc, upper_perc):
        """
        Calcule les bornes inférieures et supérieures pour chaque colonne numérique.
        """
        return {
            col: (
                data[col].quantile(lower_perc),
                data[col].quantile(upper_perc)
            )
            for col in numeric_cols
        }

    # Sous-fonction : Calculer les médianes
    def compute_medians(data, bounds, group_col):
        """
        Calcule les médianes par groupe (group_col) et globales.
        """
        group_meds = {
            col: data.groupby(group_col)[col].median()
            for col in bounds
        }
        global_meds = data[list(bounds)].median()
        return group_meds, global_meds

    # Sous-fonction : Marquer les outliers
    def mark_outliers(data, bounds, outlier_tag):
        """
        Marque les valeurs en dehors des bornes comme outliers.
        """
        for col, (low, high) in bounds.items():
            mask = (data[col] < low) | (data[col] > high)
            data[f'{col}_outlier_flag'] = mask.astype(int)
            data.loc[mask, col] = outlier_tag
        return data

    # Sous-fonction : Nettoyer les outliers
    def clean_outliers(data, bounds, group_meds, global_meds, group_col, outlier_tag):
        """
        Remplace les valeurs marquées comme outliers par les médianes de groupe ou globales.
        """
        for col in bounds:
            mask = data[col] == outlier_tag
            data.loc[mask, col] = (
                data.loc[mask, group_col]
                    .map(group_meds[col])
                    .fillna(global_meds[col])
                    .astype(data[col].dtype)  # Conserve le type d'origine
            )
        return data

    # Étape 1 : Obtenir les colonnes numériques
    numeric_cols = [
        col for col in train_data.select_dtypes(include='number').columns
        if col != group_col and col != target_col
    ]

    # Étape 2 : Calculer les bornes à partir des données d'entraînement
    bounds = calculate_bounds(train_data, numeric_cols, lower_perc, upper_perc)

    # Étape 3 : Marquer les outliers dans train_data
    train_marked = mark_outliers(train_data.copy(), bounds, outlier_tag)

    # Étape 4 : Calculer les médianes de groupe et globales à partir de train_data
    group_medians, global_medians = compute_medians(train_marked, bounds, group_col)

    # Étape 5 : Nettoyer les outliers dans train_data
    train_clean = clean_outliers(train_marked.copy(), bounds, group_medians, global_medians, group_col, outlier_tag)

    # Étape 6 : Marquer les outliers dans test_data
    test_marked = mark_outliers(test_data.copy(), bounds, outlier_tag)

    # Étape 7 : Nettoyer les outliers dans test_data en utilisant les médianes de train_data
    test_clean = clean_outliers(test_marked.copy(), bounds, group_medians, global_medians, group_col, outlier_tag)

    # Étape 8 : Reconstitution des jeux X (features) et y (cible)
    X_train = train_clean.drop(columns=[target_col])
    y_train = train_clean[target_col]
    X_test = test_clean.drop(columns=[target_col])
    y_test = test_clean[target_col]

    # Retourner les jeux X et y
    return X_train, y_train, X_test, y_test



################################################################################
########################## ORCHESTRATION DES ETAPES ############################
################################################################################

def process_outliers(data, group_col="INSEE_COM", target_col="prix_m2_vente", lower_perc=0.01, upper_perc=0.99, outlier_tag=-999):
    """
    Orchestration des étapes de traitement des outliers.
    Ajoute des graphiques pour visualiser les résultats à chaque étape.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame brut.
        group_col (str): Colonne utilisée pour regrouper les données (par défaut "INSEE_COM").
        target_col (str): Colonne cible à séparer (par défaut "prix_m2_vente").
        lower_perc (float): Quantile inférieur pour détecter les outliers (par défaut 0.01).
        upper_perc (float): Quantile supérieur pour détecter les outliers (par défaut 0.99).
        outlier_tag (int): Valeur utilisée pour marquer les outliers (par défaut -999).
    
    Returns:
        dict: Un dictionnaire contenant les différentes étapes des données et les graphiques associés.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    results = {}

    # Étape 1 : Visualisation initiale des boxplots
    numeric_cols = get_numeric_columns(data, group_col)
    fig_initial_boxplots = plot_boxplots(data, numeric_cols)
    results["plot_initial_boxplots"] = fig_initial_boxplots

    # Étape 2 : Suppression des anomalies logiques
    data_cleaned_logical = detect_logical_anomalies(data)
    results["data_cleaned_logical"] = data_cleaned_logical
    results["plot_logical_anomalies"] = None  # Pas de graphique pour cette étape

    # Étape 3 : Suppression des valeurs improbables
    data_cleaned_improbable, report_improbable = remove_improbable_values(data_cleaned_logical)
    results["data_cleaned_improbable"] = data_cleaned_improbable
    results["report_improbable"] = report_improbable
    results["plot_improbable_values"] = None  # Pas de graphique pour cette étape

    # Étape 4 : Split train/test
    train_data, test_data = split_train_test(data_cleaned_improbable)
    results["train_data"] = train_data
    results["test_data"] = test_data

    # Étape 5 : Traitement des outliers
    X_train, y_train, X_test, y_test = process_outliers_sequence(
        train_data=train_data,
        test_data=test_data,
        group_col=group_col,
        lower_perc=lower_perc,
        upper_perc=upper_perc,
        outlier_tag=outlier_tag,
        target_col=target_col
    )
    results["X_train"] = X_train
    results["y_train"] = y_train
    results["X_test"] = X_test
    results["y_test"] = y_test

    # Étape 6 : Visualisation des boxplots pour X_train
    fig_boxplots, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=X_train[numeric_cols], ax=ax, palette="Set2")
    ax.set_title("Boxplots des colonnes numériques (X_train)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
    plt.tight_layout()
    results["plot_boxplots_X_train"] = fig_boxplots

    # Étape 7 : Visualisation de la distribution de y_train
    fig_distribution, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=y_train, bins=150, kde=True, ax=ax)
    ax.set_title('Distribution de la cible (y_train)')
    ax.set_xlim(0, 20000)
    ax.set_xticklabels(ax.get_xticks(), rotation=45, fontsize=8)
    plt.tight_layout()
    results["plot_distribution_y_train"] = fig_distribution

    return results