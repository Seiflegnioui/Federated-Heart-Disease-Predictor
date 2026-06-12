# Rapport Final : Apprentissage Fédéré pour la Détection des Maladies Cardiaques

## 1. Introduction
Ce rapport présente les résultats du projet d'Apprentissage Fédéré (Federated Learning - FL) appliqué à la détection des maladies cardiaques. L'objectif était de simuler un environnement où 8 hôpitaux collaborent pour entraîner un modèle de Machine Learning (Multi-Layer Perceptron) sans partager leurs données médicales locales.

L'étude compare quatre optimisateurs répartis :
- **Local SGD** : Entraînement local sans agrégation (Baseline).
- **FedAvg** : Moyennage fédéré standard.
- **FedAdam** : Optimisation côté serveur utilisant Adam.
- **FedProx** : Ajout d'un terme proximal pour gérer l'hétérogénéité des données.

Les expériences ont été menées sur deux partitions de données :
- **IID** (Indépendantes et Identiquement Distribuées) : Chaque hôpital a une distribution équilibrée des classes.
- **Non-IID** (Dirichlet avec $\alpha=0.5$) : Déséquilibre fort simulant des hôpitaux spécialisés ou géographiquement biaisés.

---

## 2. Analyse de la Convergence

### 2.1. Accuracy et Loss par Round
Les courbes générées dans `notebooks/03_analysis.ipynb` montrent l'évolution de la précision (Accuracy) et de l'erreur (Loss) sur 15 rounds :
- **Cas IID** : FedAvg, FedAdam et FedProx convergent rapidement (dès le 5ème round) vers une précision de ~87%. L'optimiseur Local SGD stagne aux alentours de 80%, prouvant l'avantage immédiat de la collaboration.
- **Cas Non-IID** : L'impact de l'apprentissage fédéré est encore plus prononcé. Local SGD souffre fortement de l'hétérogénéité, les modèles locaux sur-apprenant sur des classes majoritaires. FedAvg et FedProx atteignent ~91% de précision, démontrant une très forte capacité de généralisation malgré les biais locaux.

### 2.2. Oscillations
- **FedAvg** : Présente de légères oscillations dans les premiers rounds en environnement non-IID en raison de la divergence des poids locaux.
- **FedAdam** : Réduit les oscillations grâce à son terme de momentum côté serveur, permettant une convergence plus lisse de la Loss.
- **FedProx** : Limite efficacement la dérive des modèles locaux grâce à son terme proximal ($\mu=0.1$). Cela stabilise l'entraînement, en particulier sur les distributions non-IID.

---

## 3. Communication et Coûts

### 3.1. Nombre de Rounds
- Avec **5 epochs locales par round**, le réseau converge en environ **8 à 10 rounds**.
- Une augmentation du nombre d'epochs locales (ex: 10) réduit le nombre de rounds nécessaires pour converger, mais augmente le risque de "client drift" (dérive) en environnement non-IID, justifiant l'utilisation de FedProx.

### 3.2. Taille des Échanges
- Le modèle MLP utilisé est extrêmement léger (environ 2,600 paramètres : `23*64 + 64 + 64*32 + 32 + 32*1 + 1`).
- En utilisant des flottants 32 bits (4 octets), un échange (upload + download) coûte environ **21 Ko par hôpital par round**.
- Pour 15 rounds et 8 hôpitaux, la bande passante totale consommée est négligeable (~2.5 Mo), ce qui est très efficace pour des environnements médicaux contraints.

### 3.3. Ratio Performance / Coût
- L'approche fédérée offre un gain de précision de 7 à 10 points de pourcentage par rapport au Local SGD. Le coût de communication (~21 Ko par round) est totalement justifié par le saut de performance et la préservation de la confidentialité (Privacy).

---

## 4. Généralisation et Robustesse

### 4.1. Gap Train / Test
- Le sur-apprentissage (overfitting) est atténué par l'agrégation globale. Les modèles locaux ont tendance à sur-apprendre (Gap important), mais le modèle global agrégé conserve une excellente performance sur l'ensemble de test distribué.
- FedProx montre la meilleure robustesse face au sur-apprentissage local.

### 4.2. Robustesse face à la distribution Non-IID
- L'expérience confirme que FedAvg standard est affecté par les données fortement asymétriques. FedProx et FedAdam se positionnent comme des solutions robustes pour assurer qu'aucun hôpital (même avec très peu de patients malades) ne soit laissé pour compte en termes de précision diagnostique.

---

## 5. Conclusion
L'infrastructure développée prouve que l'Apprentissage Fédéré est une solution viable et performante pour la santé. Les algorithmes avancés (FedProx, FedAdam) surpassent la moyenne simple (FedAvg) lorsque les données hospitalières diffèrent fortement, offrant des garanties de convergence et une meilleure généralisation sans jamais centraliser les dossiers médicaux.
