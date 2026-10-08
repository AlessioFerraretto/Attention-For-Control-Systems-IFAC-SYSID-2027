import os
import re

import pandas as pd
from openpyxl import load_workbook

# Specifica qui il percorso della cartella principale
base_dir = r"/mnt/c/Users/Alessio/Desktop/saved_models/dynamic-impulse-response-prediction-alpha095/"

data = []
baselines_estratti = set()  # Set per tenere traccia delle baseline già salvate

# Espressioni regolari modificate (il [-+\d.]+ cattura anche i valori negativi come il LS)
regex_model = re.compile(
    r"Model \((.*?)\).*?Loss:\s*([\d.]+),\s*FIT:\s*([-+\d.]+)%,\s*DF:\s*([\d.]+),\s*SURE:\s*([\d.]+)"
)

regex_baseline = re.compile(
    r"Baseline (.*?)FIT:\s*([-+\d.]+)%,\s*DF:\s*([\d.]+),\s*SURE:\s*([\d.]+)"
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
                        # loss = float(match_model.group(2)) # Ora è nel gruppo 2
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
                            baselines_estratti.add(metodo)  # Aggiungila al set

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

# 1. Crea il DataFrame
df = pd.DataFrame(data)

# 2. Ordina per Modello
df = df.sort_values(by="Modello")

output_file = "training.xlsx"

# 3. Salva il file base senza formattazione
df.to_excel(output_file, index=False)

# 4. Apri con openpyxl per la formattazione (Filtri e Auto-resize)
wb = load_workbook(output_file)
ws = wb.active

# Aggiungi il filtro automatico su tutta l'area dei dati
ws.auto_filter.ref = ws.dimensions

# Fai il resize automatico delle colonne
for col in ws.columns:
    max_length = 0
    column_letter = col[0].column_letter  # Ottiene la lettera (A, B, C, ecc.)

    # Calcola la larghezza massima del testo in quella colonna
    for cell in col:
        try:
            if cell.value:
                # Usa la lunghezza della stringa come stima
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
        except:
            pass

    # Imposta la larghezza (aggiungendo un piccolo margine di 2 caratteri)
    adjusted_width = max_length + 2
    ws.column_dimensions[column_letter].width = adjusted_width

# Salva il file definitivo sovrascrivendolo
wb.save(output_file)

print(
    f"Estrazione, ordinamento e formattazione completati. File salvato in: {output_file}"
)
