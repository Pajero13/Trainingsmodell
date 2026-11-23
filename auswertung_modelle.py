import argparse
import json
import os
from datetime import datetime

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from engine import lade_trainiertes_modell, speichere_testergebnisse


def parse_argumente():
    parser = argparse.ArgumentParser(description="Evaluiert ein trainiertes Modell auf mehreren Test-Datensätzen")
    parser.add_argument("--name", required=True,
                        help="Name des zu ladenden Modells (ohne .pt), aus MODELLE_ORDNER")
    parser.add_argument("--test_root", required=True,
                        help="Ordner, dessen Unterordner je einen Test-Datensatz enthalten "
                             "(jeder Unterordner mit den Klassenordnern ai/ und real/)")
    parser.add_argument("--auswertung", default=None,
                        help="Name der Auswertungsdateien (ohne .json). "
                             "Ohne Angabe: <modellname>_<Zeitstempel>")
    parser.add_argument("--struktur", type=int, default=0, choices=[0, 1],
                        help="0 = eine JSON-Datei pro Test-Datensatz (Standard), "
                             "1 = alles in einer einzigen JSON-Datei")
    return parser.parse_args()


def erstelle_test_daten(modell, datensatz_pfad, batch_size, num_workers):
    """Gleiche Transformation wie lade_test_daten() in engine.py, aber für einen beliebigen Ordner."""
    data_config = timm.data.resolve_model_data_config(modell)
    mean, std = data_config["mean"], data_config["std"]

    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_ds = ImageFolder(datensatz_pfad, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_ds, test_loader


def speichere_gesamtauswertung(ordner, auswertung_name, modell_name, ergebnisse):
    """--struktur 1: alle Test-Datensätze in einer einzigen JSON-Datei speichern."""
    os.makedirs(ordner, exist_ok=True)

    inhalt = {
        "name": auswertung_name,
        "modell": modell_name,
        "zeitstempel": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datensaetze": ergebnisse,
    }

    pfad = os.path.join(ordner, f"{auswertung_name}.json")
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(inhalt, f, indent=2, ensure_ascii=False)
    return pfad


if __name__ == "__main__":
    args = parse_argumente()
    checkpoint_pfad = os.path.join(MODELLE_ORDNER, f"{args.name}.pt")

    if args.auswertung:
        auswertung_basis = args.auswertung
    else:
        zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
        auswertung_basis = f"{args.name}_{zeitstempel}"

    model = lade_trainiertes_modell(MODELL_NAME, checkpoint_pfad, NUM_CLASSES, DEVICE)
    model.eval()

    # Alle Unterordner des Test-Ordners durchgehen, jeder Unterordner = ein Datensatz (ai/, real/)
    ergebnisse = {}  # nur relevant für --struktur 1

    for datensatz_name in sorted(os.listdir(args.test_root)):
        datensatz_pfad = os.path.join(args.test_root, datensatz_name)
        if not os.path.isdir(datensatz_pfad):
            continue

        test_ds, test_loader = erstelle_test_daten(model, datensatz_pfad, BATCH_SIZE, NUM_WORKERS)

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
        bericht_dict = classification_report(alle_labels, alle_vorhersagen,
                                             target_names=klassen_namen, output_dict=True,
                                             zero_division=0)

        print(f"\n========== Datensatz: {datensatz_name} ==========")
        print(f"Klassenzuordnung: {test_ds.class_to_idx}")
        print("\nKonfusionsmatrix:")
        print(konfusion)
        print("\nKlassifikationsbericht:")
        print(classification_report(alle_labels, alle_vorhersagen,
                                    target_names=klassen_namen, zero_division=0))

        ergebnis = {
            "modell": args.name,
            "klassenzuordnung": test_ds.class_to_idx,
            "konfusionsmatrix": konfusion.tolist(),
            "klassifikationsbericht": bericht_dict,
        }

        if args.struktur == 1:
            ergebnisse[datensatz_name] = ergebnis
        else:
            auswertung_pfad = speichere_testergebnisse(
                AUSWERTUNG_ORDNER, f"{auswertung_basis}_{datensatz_name}", args.name,
                test_ds.class_to_idx, konfusion, bericht_dict
            )
            print(f"Auswertung gespeichert unter: {auswertung_pfad}")

    if args.struktur == 1:
        auswertung_pfad = speichere_gesamtauswertung(
            AUSWERTUNG_ORDNER, auswertung_basis, args.name, ergebnisse
        )
        print(f"\nGesamtauswertung ({len(ergebnisse)} Datensätze) gespeichert unter: {auswertung_pfad}")