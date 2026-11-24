"""Orquestador reproducible de extremo a extremo para Entregable 2."""

import argparse
import subprocess
import sys

from src.evaluate import main as evaluate_main
from src.plots import main as plots_main
from src.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo Entregable 2 (PJM CNN1D-LSTM)")
    parser.add_argument("--skip-train", action="store_true", help="Omitir entrenamiento y solo evaluar/graficar.")
    parser.add_argument("--skip-tests", action="store_true", help="Omitir pytest previo.")
    args = parser.parse_args()

    if not args.skip_tests:
        print("[PASO 1/4] Ejecutando suite de tests anti-data-leakage...")
        subprocess.run([sys.executable, "-m", "pytest", "-v"], check=True)

    if not args.skip_train:
        print("[PASO 2/4] Entrenando modelos PureLSTM y CNN1D-LSTM (multi-semilla)...")
        train_main()

    print("[PASO 3/4] Evaluando modelos y líneas base sobre Validación...")
    evaluate_main()

    print("[PASO 4/4] Generando figuras científicas...")
    plots_main()
    print("[OK] Flujo completo de Entregable 2 ejecutado sin errores.")


if __name__ == "__main__":
    main()
