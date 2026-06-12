import os
import subprocess
import sys
import time

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  ETAPE : {desc}")
    print(f"{'='*60}")
    
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"\n  [SUCCES] Termine en {time.time() - t0:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERREUR] La commande a echoue avec le code {e.returncode}")
        sys.exit(1)

def main():
    start_total = time.time()
    
    print("\n" + "#" * 60)
    print("#   Federated Heart Disease - Execution Complete")
    print("#" * 60)

    # 1. Pipeline de preparation des donnees (Personne 1)
    run_command(f"{sys.executable} run_all.py", "Preparation des donnees et creation des hopitaux")

    # 2. Entrainement Federe (Personnes 2 et 3)
    # On utilise 15 rounds et 5 epochs par defaut
    run_command(f"{sys.executable} src/run_experiments.py --rounds 15 --epochs 5", "Entrainement des modeles federes (FedAvg, FedAdam, FedProx, LocalSGD)")

    # 3. Generation du notebook d'analyse (Personne 4)
    run_command(f"{sys.executable} create_nb.py", "Creation du notebook d'analyse")
    
    # 4. Execution du notebook d'analyse pour generer les graphiques
    run_command(f"{sys.executable} -m jupyter nbconvert --to notebook --execute notebooks/03_analysis.ipynb --inplace", "Generation des graphiques finaux")

    elapsed = time.time() - start_total
    print("\n" + "#" * 60)
    print("#   PROJET TERMINE AVEC SUCCES!")
    print(f"#   Temps total : {elapsed:.2f}s")
    print("#" * 60)
    print("Vous pouvez maintenant ouvrir le rapport final dans 'docs/final_report.pdf' ou explorer 'notebooks/03_analysis.ipynb'.")

if __name__ == "__main__":
    main()
