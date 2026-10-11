from pathlib import Path

import pandas as pd
import torch


def convert_pt_to_excel(input_path: str | Path, output_path: str | Path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Carica i dati (il dizionario {key: tensor} salvato dallo script originale)
    print(f"Caricamento dati da {input_path}...")
    data = torch.load(input_path, map_location="cpu")

    excel_data = {}

    # 2. Converte i tensori in array NumPy per l'inserimento in Pandas
    for key, tensor in data.items():
        arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor

        # Gestione delle dimensioni del tensore
        if arr.ndim == 1:
            excel_data[key] = arr
        elif arr.ndim == 2:
            for col_idx in range(arr.shape[1]):
                excel_data[f"{key}_{col_idx}"] = arr[:, col_idx]
        else:
            # Reshape da [N, D1, D2, ...] a [N, D1*D2*...]
            flattened_2d = arr.reshape(arr.shape[0], -1)
            for col_idx in range(flattened_2d.shape[1]):
                excel_data[f"{key}_{col_idx}"] = flattened_2d[:, col_idx]

    # 4. Creazione del DataFrame ed esportazione in Excel
    df = pd.DataFrame(excel_data)

    # Assicuriamoci che le colonne siano in ordine numerico (da 0 a 63)
    colonne_presenti = [col for col in df.columns]
    df = df[colonne_presenti]

    # Taglio dei dati se il parametro sample è definito
    print(f"Salvataggio in corso su {output_path}...")

    # Salvataggio nel file Excel senza la colonna degli indici numerici
    df.to_excel(output_path, index=False)

    print(f"Salvataggio completato con successo: {output_path}")
    print(f"Righe esportate: {len(df)} | Colonne esportate: {len(df.columns)}")


def salva_file_errore(output_path: Path, messaggio: str):
    """Crea un file Excel di fallback con il messaggio di errore."""
    df_errore = pd.DataFrame([[messaggio]], columns=["Stato"])
    df_errore.to_excel(output_path, index=False)
    print(f"Salvato file di errore in: {output_path}")


if __name__ == "__main__":
    # BASE_DIR = Path(
    #    r"/mnt/c/Users/Alessio/Desktop/saved_models/architettura mean-pooling/kernel-parameters-estimation-input-uy-output-alpha-c/"
    # )
    BASE_DIR = Path(
        "/mnt/c/Users/Alessio/Desktop/Workspace/Attention For Control Systems/outputs/kernel-parameters-estimation-input-uy-output-alpha-c/transformer/"
    )

    OUTPUT_DIR = Path(r"/mnt/c/Users/Alessio/Desktop/saved_models/output")

    # Assicura che la directory di output esista
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cartelle da cercare all'interno di ogni progetto
    folders_to_check = [
        "DATA",
        "4x64x16",
        "4x64x32",
        "4x128x16",
        "4x128x32",
        "8x64x16",
        "8x64x32",
        "8x128x16",
        "8x128x32",
    ]

    # Scorri le sottocartelle specificate per ogni progetto
    for folder in folders_to_check:
        input_file = BASE_DIR / folder / "test_impulse_responses.pt"
        output_file = OUTPUT_DIR / f"{folder}.xlsx"

        if not input_file.exists():
            print(f"Saltato: Il file {input_file} non è stato trovato.")
            salva_file_errore(output_file, "ERRORE (File non trovato)")
            continue

        try:
            # Tenta la conversione
            convert_pt_to_excel(input_file, output_file)
        except Exception as e:
            # Cattura qualsiasi errore
            print(f"Errore durante l'elaborazione di {input_file}.")
            print(f"Dettaglio: {e}")
            salva_file_errore(output_file, f"ERRORE: {str(e)}")
