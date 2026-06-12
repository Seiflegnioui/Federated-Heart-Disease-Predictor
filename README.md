# Federated Learning: Heart Disease Detection

Welcome to the **Federated Learning** simulation project repository for heart disease detection. This project demonstrates how multiple hospitals can collaborate to train a robust Machine Learning model without ever needing to share or centralize the private medical data of their patients.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Step 1: Data Preparation and Simulation (Person 1)](#step-1-data-preparation-and-simulation)
3. [Step 2: Model Development and Infrastructure (Person 2)](#step-2-model-development-and-infrastructure)
4. [Step 3: Optimizers Implementation (Person 3)](#step-3-optimizers-implementation)
5. [Step 4: Results Analysis (Person 4)](#step-4-results-analysis)
6. [How to Run the Project](#how-to-run-the-project)
7. [Conclusion and Results](#conclusion-and-results)

---

## 🎯 Project Overview
The goal is to train a binary classification model capable of predicting whether a patient has heart disease, based on 13 clinical attributes (age, cholesterol, blood pressure, etc.). To respect patient privacy, we simulated **8 hospitals**. The central server never receives any medical data, only the mathematical weights of the models.

---

## 🛠 Step 1: Data Preparation and Simulation
*Responsibility: Data Preparation & Hospital Simulation*

### 1. Data Cleaning
The `heart.csv` dataset contains 1025 samples. The data was processed in `src/data_preparation.py`:
- Checked for missing values.
- Applied "One-Hot" encoding for categorical variables.
- Applied standard normalization (Z-score) to continuous features to facilitate neural network convergence.
- Final result: A feature matrix `X` with 1025 rows and 23 columns.

### 2. Federated Partitioning (Hospital Simulation)
We virtually separated this data among 8 "clients" (hospitals) in `src/federated_partition.py` using two methods to study the system's robustness:
- **IID (Independent and Identically Distributed)**: The data is shuffled and distributed in a perfectly balanced manner. Each hospital sees as many healthy cases as sick cases.
- **Non-IID (Asymmetric Distribution)**: Using a Dirichlet distribution ($\alpha=0.5$), the data is unevenly distributed. Some hospitals might be "specialized" and treat almost exclusively sick or healthy patients.

---

## 🏗 Step 2: Model Development and Infrastructure
*Responsibility: FL Model and Architecture*

### 1. The Model (PyTorch)
In `src/model.py`, we developed an artificial neural network (Multi-Layer Perceptron) perfectly sized for tabular data. It contains 3 dense layers with ReLU activation functions and Dropout to prevent overfitting. The model is extremely lightweight (~2600 parameters).

### 2. The Federated Infrastructure
The standard operation relies on a client-server architecture:
- **Client (`src/fl_client.py`)**: Each hospital is modeled by a `LocalClient` class. When requested, the hospital downloads the current weights of the global model, trains the model on its own local data for a defined number of epochs (5 epochs), and sends back only the newly updated weights.
- **Server (`src/run_experiments.py`)**: The server script coordinates the "Rounds". At each Round, it asks the hospitals to train, collects the weights, and aggregates them according to a specific strategy to create an improved global model.

*(Technical Note: Due to compatibility issues in Python 3.13 environments where the `ray` library used by the `Flower` framework simulation is not supported, we implemented a custom federated loop that exactly mimics the behavior of a federated simulation).*

---

## ⚡ Step 3: Optimizers Implementation
*Responsibility: Federated Optimization Strategies*

To understand how to efficiently aggregate the knowledge from different hospitals, we implemented and compared 4 strategies in the simulation engine:

1. **Local SGD (Baseline)**: Each hospital trains alone in its corner. No models are shared. This serves to measure the added value of collaboration.
2. **FedAvg (Federated Averaging)**: The classic method. The server calculates a weighted average (based on the number of patients in the hospital) of the received weights.
3. **FedAdam**: The server acts not just as a simple average calculator, but as an optimizer (Adam) that uses the average of the weights as a "pseudo-gradient". This accelerates convergence and smooths out oscillations.
4. **FedProx**: Specifically designed for Non-IID environments. It adds a local "proximal term" forcing the hospital's model not to drift too far from the global model. This prevents a biased hospital from "breaking" the overall learning.

---

## 📊 Step 4: Results Analysis
*Responsibility: Convergence, Communication, and Documentation*

A script `create_nb.py` automatically generates a Jupyter Notebook (`notebooks/03_analysis.ipynb`) allowing a detailed study of the trained models' behavior.

### Key Project Results
Training was performed over **15 Rounds**, with **5 local epochs per hospital**.

1. **Collaboration Defeats Isolation**: 
   In all scenarios, federated learning (global accuracy of ~87% to 91%) vastly outperforms models trained in isolation (Local SGD stagnates around 80%).
2. **Handling Heterogeneity (Non-IID)**:
   When hospitals have highly imbalanced data, a locally trained model collapses (because it only sees a part of reality). Federated Learning manages to achieve **over 90% accuracy** even under these extreme conditions. **FedProx** proved to be particularly robust in stabilizing the "loss" in these scenarios.
3. **Communication Efficiency**:
   Since the model only weighs 10 KB, a round-trip network exchange costs less than 25 KB. The entire federated training for all 8 hospitals consumes only ~3 MB of total bandwidth, making the system deployable even on slow or unstable medical networks.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
The project requires a Python 3.x environment (ideally 3.9 to 3.13). Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Quick "Turnkey" Execution
A `run_pipeline.py` script was created to orchestrate the entire project automatically (data cleaning, partitioning, full federated training for all 4 algorithms, and final charts generation):
```bash
python run_pipeline.py
```
*The complete pipeline takes about one minute to run.*

### 3. Step-by-Step Manual Execution
If you wish to run the project granularly:
- **Preparation**: `python run_all.py`
- **Federated Training**: `python src/run_experiments.py --rounds 15 --epochs 5`
- **Generate Charts**: `python create_nb.py` then open `notebooks/03_analysis.ipynb` with Jupyter Notebook or VSCode.

### 4. Useful Resources
- A comprehensive report (`docs/final_report.md` and `.tex`) is available in the `docs/` folder.
- A minimalist demonstration to see the code in action can be found in `notebooks/02_fl_demo.ipynb`.
