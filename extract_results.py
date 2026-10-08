import os
import re

import pandas as pd

# Specifica qui il percorso della cartella principale che contiene tutte le tue sottocartelle (4x64x16, ecc.)
base_dir = r"/mnt/c/Users/Alessio/Desktop/saved_models/kernel-parameters-estimation-input-uy-output-alpha-c/"

data = []
baselines_estratti = set()  # Set per tenere traccia delle baseline già salvate

# Espressioni regolari per catturare i dati
regex_model = re.compile(
    r"Model \((.*?)\) - Loss:\s*([\d.]+),\s*FIT:\s*([\d.]+)%,\s*DF:\s*([\d.]+),\s*SURE:\s*([\d.]+)"
)

regex_baseline = re.compile(
    r"Baseline (.*?) - FIT:\s*([\d.]+)%,\s*DF:\s*([\d.]+),\s*SURE:\s*([\d.]+)"
)

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    # Verifica che sia una cartella e che contenga il file di log
    if os.path.isdir(folder_path):
        log_file = os.path.join(folder_path, "training_script.log")

        if os.path.exists(log_file):
            with open(log_file, encoding="utf-8") as f:
                lines = f.readlines()

                for line in lines:
                    # Cerca i dati dei Model
                    match_model = regex_model.search(line)
                    if match_model:
                        metodo = f"Model ({match_model.group(1).strip()})"
                        # loss = float(match_model.group(2)) # Decommenta se ti serve
                        fit = float(match_model.group(3))
                        df = float(match_model.group(4))
                        sure = float(match_model.group(5))

                        data.append(
                            {
                                "Modello": folder,
                                "Metodo": metodo,
                                "FIT (%)": fit,
                                "DF": df,
                                "SURE": sure,
                            }
                        )

                    # Cerca i dati delle Baseline
                    match_baseline = regex_baseline.search(line)
                    if match_baseline:
                        metodo = f"Baseline {match_baseline.group(1).strip().replace('-', '').strip()}"

                        # Controlla se questa baseline è GIA' stata estratta
                        if metodo not in baselines_estratti:
                            baselines_estratti.add(
                                metodo
                            )  # Aggiungila al set per ricordarla

                            fit = float(match_baseline.group(2))
                            df = float(match_baseline.group(3))
                            sure = float(match_baseline.group(4))

                            data.append(
                                {
                                    "Modello": "-",  # Usiamo "-" per far capire che è globale
                                    "Metodo": metodo,
                                    "FIT (%)": fit,
                                    "DF": df,
                                    "SURE": sure,
                                }
                            )

# Crea il DataFrame e salva in Excel
df = pd.DataFrame(data)

output_file = "training.xlsx"
df.to_excel(output_file, index=False)

print(f"Estrazione completata. File salvato in: {output_file}")
