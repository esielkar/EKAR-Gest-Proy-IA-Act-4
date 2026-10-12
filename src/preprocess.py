import pandas as pd
from datasets import Dataset

def load_and_prepare_data(csv_path: str = "data/raw_issues.csv") -> Dataset:
    """Carga el dataset de incidencias desde un archivo CSV y lo convierte en un Dataset de Hugging Face."""
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    print(f"Dataset cargado correctamente desde {csv_path}. Registros totales: {len(dataset)}")
    return dataset

if __name__ == "__main__":
    load_and_prepare_data()
