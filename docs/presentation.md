---
marp: true
theme: default
paginate: true
---

# Apprentissage Fédéré : Détection des Maladies Cardiaques
**Soutenance de Projet - Pipeline Complet**

---

## 🏥 1. Objectifs du Projet

- **Problématique :** Détecter les maladies cardiaques sans centraliser les données médicales des patients (respect de la confidentialité).
- **Solution :** Mettre en place une architecture d'**Apprentissage Fédéré** répartie sur **8 hôpitaux simulés**.
- **Technologie :** Réseau de Neurones MLP (PyTorch) et Optimisateurs Fédérés.

---

## 📊 2. Préparation et Partitionnement

- Dataset : *Heart Disease* (1025 patients, 23 features encodées).
- Simulation de deux scénarios :
  - **IID** : Chaque hôpital possède la même distribution de malades/sains.
  - **Non-IID** : Déséquilibre fort via une distribution de Dirichlet ($\alpha=0.5$). Certains hôpitaux ne voient qu'une seule classe.

---

## ⚙️ 3. Architecture et Modèle

- **Modèle Global** : Multi-Layer Perceptron (MLP) léger (23 -> 64 -> 32 -> 1).
- **Entraînement Local** : PyTorch DataLoader, Loss BCEWithLogits, 5 epochs par round.
- **Serveur Fédéré** :
  - **FedAvg** : Moyenne standard.
  - **FedAdam** : Ajustement momentum côté serveur.
  - **FedProx** : Terme proximal local pour contraindre la dérive.

---

## 📈 4. Résultats : Convergence

- **Local SGD (Sans collaboration)** : Plafonne autour de ~80-85%. Très instable en Non-IID.
- **FedAvg** : Atteint rapidement ~87% en IID, mais souffre de légères oscillations en Non-IID.
- **FedProx & FedAdam** : Les plus performants et stables. Ils atteignent plus de 90% de précision globale sur des données très asymétriques !

---

## 📡 5. Coûts de Communication

- **Taille du modèle** : ~2600 paramètres = **~10 Ko** par transfert.
- **Bande passante totale** (15 rounds, 8 clients) : Moins de 3 Mo échangés !
- **Conclusion** : Le ratio Performance / Coût est exceptionnel, compatible avec des réseaux médicaux à très faible connectivité.

---

## 🚀 6. Démonstration

- Notebook interactif disponible : `notebooks/02_fl_demo.ipynb`
- Analyse complète et graphiques : `notebooks/03_analysis.ipynb`

**Merci de votre attention !**
