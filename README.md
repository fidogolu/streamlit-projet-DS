# Projet Streamlit - Analyse et Prétraitement des Données

Ce projet Streamlit vise à analyser, prétraiter, visualiser des données immobilières puis entrainer des modèles permettant de faire des prévisions sur le prix de vente au m2 en France, à partir de données agrégées des 5 dernieres années de ventes immobilières sur le territoire métropolitain.

Deux types de modélisations font suite aux analyses et prétraitements : 
 - une modélisation de régression
 - une modélisation via une série temporelle

Il est structuré pour séparer les différentes étapes du traitement des données, de leur chargement à leur visualisation, en passant par le nettoyage et la détection des anomalies.

---

## **Fonctionnalités principales**

### **1. Chargement des données**
- **Fonction : `try_read_csv`**
  - **Description** : Permet de lire un fichier CSV avec différents encodages pour garantir la compatibilité.
  - **Sujet** : Chargement des données.

---

### **2. Prétraitement des données**
- **Gestion des doublons**
  - **Description** : Suppression des doublons dans le dataset pour éviter les redondances.
  - **Sujet** : Prétraitement.

- **Gestion des valeurs manquantes**
  - **Fonction : `missing_data_percentage_sales`**
  - **Description** : Analyse et traitement des valeurs manquantes dans les colonnes.
  - **Sujet** : Prétraitement.

- **Suppression des colonnes avec trop de valeurs manquantes**
  - **Description** : Suppression des colonnes contenant plus de 75% de valeurs manquantes.
  - **Sujet** : Prétraitement.

- **Identification des variables catégoriques**
  - **Description** : Identification des colonnes avec moins de 10 modalités pour les traiter comme des variables catégoriques.
  - **Sujet** : Prétraitement.

- **Nettoyage des colonnes spécifiques**
  - **Fonctions : `clean_classe`, `clean_exposition`, etc.**
  - **Description** : Nettoyage et standardisation des colonnes spécifiques (ex. : classes énergétiques, exposition).
  - **Sujet** : Prétraitement.

- **Détection des anomalies logiques**
  - **Description** : Identification des incohérences logiques dans les données (ex. : valeurs impossibles ou contradictoires).
  - **Sujet** : Prétraitement.

- **Détection et traitement des outliers**
  - **Description** : Identification et remplacement des valeurs aberrantes dans les colonnes numériques.
  - **Sujet** : Prétraitement.

---

### **3. Visualisation des données**
- **Visualisation des valeurs manquantes**
  - **Description** : Création d'un graphique pour visualiser les colonnes avec des valeurs manquantes.
  - **Sujet** : Visualisation.

- **Visualisation des distributions**
  - **Description** : Création de graphiques pour visualiser les distributions des variables.
  - **Sujet** : Visualisation.

---

### **4. Sauvegarde des données**
- **Sauvegarde du dataset nettoyé**
  - **Description** : Exportation du dataset nettoyé au format CSV pour une utilisation ultérieure.
  - **Sujet** : Sauvegarde.

---

## **Structure du projet**

Le projet est organisé en plusieurs fichiers pour séparer les responsabilités et améliorer la lisibilité du code :
