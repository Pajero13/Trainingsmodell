import argparse
import os
import timm
import torch.nn as nn
import torch

from config import *
from engine import lade_modell, lade_daten, erstelle_optimizer_und_scheduler, trainiere, speichere_modellinformationen


def parse_argumente():
    parser = argparse.ArgumentParser(description="Trainiert das KI-Bild-Erkennungsmodell")
    parser.add_argument("--name", required=True, help="Dateiname für das beste Modell dieses Trainingslaufs (ohne .pt)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_argumente()
    os.makedirs(MODELLE_ORDNER, exist_ok=True)
    checkpoint_pfad = os.path.join(MODELLE_ORDNER, f"{args.name}.pt")

    model = lade_modell(MODELL_NAME, PRETRAINED_PFAD, NUM_CLASSES, DEVICE)
    bildgroesse = timm.data.resolve_model_data_config(model)["input_size"][-2:]

    train_ds, val_ds, train_loader, val_loader = lade_daten(
        model, DATA_DIR, BATCH_SIZE, NUM_WORKERS, SUBSET_GROESSE, AUGMENTIERUNG
    )
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = erstelle_optimizer_und_scheduler(
        model, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, WARMUP_EPOCHS, len(train_loader)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(PRECISION == "amp"))

    verlauf, best_epoch, best_val_acc, gesamtdauer = trainiere(
        model, train_loader, val_loader, train_ds, val_ds,
        optimizer, criterion, scheduler, scaler,
        DEVICE, EPOCHS, checkpoint_pfad,
        GRADIENT_ACCUMULATION_STEPS, PRECISION, EARLY_STOPPING_PATIENCE
    )

    info_pfad = speichere_modellinformationen(
        MODELLINFO_ORDNER, args.name, MODELL_NAME, NUM_CLASSES, bildgroesse,
        BATCH_SIZE, EPOCHS, WARMUP_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS,
        len(train_ds), len(val_ds), best_epoch, best_val_acc, gesamtdauer,
        GRADIENT_ACCUMULATION_STEPS, EARLY_STOPPING_PATIENCE, PRECISION, AUGMENTIERUNG,
        verlauf
    )
    print(f"\nModellinformationen gespeichert unter: {info_pfad}")