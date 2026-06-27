"""
Data Processing Script — MLOps Pipeline
Limpia datos crudos y guarda el resultado en data/processed/
"""

import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Eliminar duplicados
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Duplicados eliminados: {before - len(df)}")

    # 2. Normalizar nombres de columnas
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # 3. Eliminar outliers de sqft y price
    before = len(df)
    df = df[(df["sqft"] > 300) & (df["sqft"] < 10000)]
    df = df[df["price"] > 10000]
    log.info(f"Outliers eliminados: {before - len(df)}")

    # 4. Eliminar valores negativos en columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(df[num_cols] >= 0).all(axis=1)]

    # 5. Eliminar nulos
    before = len(df)
    df = df.dropna()
    log.info(f"Nulos eliminados: {before - len(df)}")

    log.info(f"Datos limpios: {df.shape[0]} filas")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    log.info(f"Datos guardados en: {path}")


def main():
    parser = argparse.ArgumentParser(description="Data Processing — MLOps")
    parser.add_argument("--input",  required=True, help="Ruta al CSV de entrada (raw)")
    parser.add_argument("--output", required=True, help="Ruta al CSV de salida (processed)")
    args = parser.parse_args()

    df = load_data(args.input)
    df = clean_data(df)
    save_data(df, args.output)


if __name__ == "__main__":
    main()
