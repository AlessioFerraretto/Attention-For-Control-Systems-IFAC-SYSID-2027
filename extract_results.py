import os
import re

import pandas as pd
from openpyxl import load_workbook

# Specifica qui il percorso della cartella principale
base_dir = r"/mnt/c/Users/Alessio/Desktop/saved_models/architettura mean-pooling/n400/"

data = []
baselines_estratti = set()  # Set per tenere traccia delle baseline già salvate

# Espressioni regolari aggiornate per i nuovi log a blocchi
regex_model_header = re.compile(r"Model \((.*?)\)")
regex_baseline_header = re.compile(r"Baseline\s+([A-Za-z0-9_-]+)")
# Cattura Mean, Median, Std (gestisce anche l'eventuale % del FIT)
regex_stats = re.compile(
    r"(FIT|DF|SURE)\s*\|\s*Mean:\s*([-+\d.]+)[%]?,\s*Median:\s*([-+\d.]+)[%]?,\s*Std:\s*([-+\d.]+)"
)

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    # Verifica che sia una cartella e che contenga il file di log
    if os.path.isdir(folder_path):
        log_file = os.path.join(folder_path, "training_script.log")

        if os.path.exists(log_file):
            with open(log_file, encoding="utf-8") as f:
                lines = f.readlines()

            current_block = (
                None  # Dizionario per memorizzare i dati temporanei del blocco corrente
            )

            for line in lines:
                # 1. Cerca l'inizio di un blocco Model
                match_model = regex_model_header.search(line)
                if match_model:
                    current_block = {
                        "Type": "Model",
                        "Modello": folder,
                        "Metodo": f"Model ({match_model.group(1).strip()})",
                    }
                    continue

                # 2. Cerca l'inizio di un blocco Baseline
                match_baseline = regex_baseline_header.search(line)
                if match_baseline:
                    current_block = {
                        "Type": "Baseline",
                        "Modello": "-",  # Usiamo "-" per far capire che è globale
                        "Metodo": f"Baseline {match_baseline.group(1).strip()}",
                    }
                    continue

                # 3. Se siamo dentro un blocco, cerchiamo le metriche FIT, DF, SURE
                if current_block:
                    match_stats = regex_stats.search(line)
                    if match_stats:
                        metric_name = match_stats.group(1).strip()  # FIT, DF, o SURE
                        mean_val = float(match_stats.group(2))
                        median_val = float(match_stats.group(3))
                        std_val = float(match_stats.group(4))

                        # Salviamo i valori estratti nel blocco temporaneo
                        current_block[f"{metric_name} Mean"] = mean_val
                        current_block[f"{metric_name} Median"] = median_val
                        current_block[f"{metric_name} Std"] = std_val

                        # Se abbiamo appena letto la riga SURE, il blocco è completo!
                        if metric_name == "SURE":
                            # Riformattiamo il dizionario per il DataFrame finale
                            row_data = {
                                "Modello": current_block["Modello"],
                                "Metodo": current_block["Metodo"],
                                "FIT Mean (%)": current_block.get("FIT Mean"),
                                "FIT Median (%)": current_block.get("FIT Median"),
                                "FIT Std": current_block.get("FIT Std"),
                                "DF Mean": current_block.get("DF Mean"),
                                "DF Median": current_block.get("DF Median"),
                                "DF Std": current_block.get("DF Std"),
                                "SURE Mean": current_block.get("SURE Mean"),
                                "SURE Median": current_block.get("SURE Median"),
                                "SURE Std": current_block.get("SURE Std"),
                            }

                            if current_block["Type"] == "Model":
                                data.append(row_data)
                            elif current_block["Type"] == "Baseline":
                                # Se è una baseline, la salviamo solo se non l'abbiamo già fatto
                                if current_block["Metodo"] not in baselines_estratti:
                                    baselines_estratti.add(current_block["Metodo"])
                                    data.append(row_data)

                            # Resettiamo il blocco per il prossimo ciclo
                            current_block = None

# 1. Crea il DataFrame
df = pd.DataFrame(data)

# 2. Ordina per Modello
df = df.sort_values(by=["Modello", "Metodo"])

output_file = os.path.join(base_dir, "training.xlsx")

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
