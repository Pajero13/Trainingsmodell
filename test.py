import argparse
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from engine import lade_trainiertes_modell, lade_test_daten, speichere_testergebnisse


def parse_argumente():
    parser = argparse.ArgumentParser(description="Evaluiert ein trainiertes Modell auf dem Test-Split")
    parser.add_argument("--name", required=True, help="Name des zu ladenden Modells (ohne .pt), aus MODELLE_ORDNER")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argumente()
    checkpoint_pfad = os.path.join(MODELLE_ORDNER, f"{args.name}.pt")

    model = lade_trainiertes_modell(MODELL_NAME, checkpoint_pfad, NUM_CLASSES, DEVICE)
    test_ds, test_loader = lade_test_daten(model, DATA_DIR, BATCH_SIZE, NUM_WORKERS)

    alle_labels = []
    alle_vorhersagen = []

    with torch.no_grad():
        for bilder, labels in test_loader:
            bilder = bilder.to(DEVICE)
            with torch.autocast(device_type="cuda"):
                outputs = model(bilder)
            vorhersagen = outputs.argmax(1).cpu()
            alle_labels.extend(labels.tolist())
            alle_vorhersagen.extend(vorhersagen.tolist())

    klassen_namen = [k for k, v in sorted(test_ds.class_to_idx.items(), key=lambda x: x[1])]
    konfusion = confusion_matrix(alle_labels, alle_vorhersagen)
    bericht_dict = classification_report(alle_labels, alle_vorhersagen, target_names=klassen_namen, output_dict=True)

    print(f"Klassenzuordnung: {test_ds.class_to_idx}")
    print("\nKonfusionsmatrix:")
    print(konfusion)
    print("\nKlassifikationsbericht:")
    print(classification_report(alle_labels, alle_vorhersagen, target_names=klassen_namen))

    auswertung_pfad = speichere_testergebnisse(
        AUSWERTUNG_ORDNER, args.name, test_ds.class_to_idx, konfusion, bericht_dict
    )
    print(f"\nAuswertung gespeichert unter: {auswertung_pfad}")