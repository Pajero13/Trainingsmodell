import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from engine import lade_trainiertes_modell, lade_test_daten

def main():
    model = lade_trainiertes_modell(MODELL_NAME, CHECKPOINT_PFAD, NUM_CLASSES, DEVICE)
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

    print(f"Klassenzuordnung: {test_ds.class_to_idx}")
    print("\nKonfusionsmatrix:")
    print(confusion_matrix(alle_labels, alle_vorhersagen))
    print("\nKlassifikationsbericht:")
    print(classification_report(alle_labels, alle_vorhersagen, target_names=klassen_namen))

if __name__ == "__main__":
    main()