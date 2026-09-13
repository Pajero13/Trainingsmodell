import math
import time
import random
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from safetensors.torch import load_file
from collections import Counter


def lade_modell(modell_name, pretrained_pfad, num_classes, device):
    model = timm.create_model(modell_name, pretrained=False, num_classes=num_classes)
    state_dict = load_file(pretrained_pfad)
    state_dict.pop('head.fc.weight', None)
    state_dict.pop('head.fc.bias', None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Fehlende Keys (sollte nur head.fc sein):", missing)
    print("Unerwartete Keys (sollte leer sein):", unexpected)
    return model.to(device)

def lade_daten(model, data_dir, batch_size, num_workers, subset_groesse=None):
    data_config = timm.data.resolve_model_data_config(model)
    mean, std = data_config["mean"], data_config["std"]

    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ImageFolder(f"{data_dir}\\train", transform=train_transform)
    val_ds = ImageFolder(f"{data_dir}\\val", transform=val_transform)
    print(train_ds.class_to_idx)

    if subset_groesse is not None:
        train_ds = Subset(train_ds, random.sample(range(len(train_ds)), min(subset_groesse, len(train_ds))))
        val_ds = Subset(val_ds, random.sample(range(len(val_ds)), min(subset_groesse, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_ds, val_ds, train_loader, val_loader


def lade_test_daten(model, data_dir, batch_size, num_workers):
    """Wird erst in der Testrunde (M3) aufgerufen, nicht während des Trainings."""
    data_config = timm.data.resolve_model_data_config(model)
    mean, std = data_config["mean"], data_config["std"]

    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_ds = ImageFolder(f"{data_dir}\\test", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_ds, test_loader

def erstelle_optimizer_und_scheduler(model, learning_rate, weight_decay, epochs, warmup_epochs, steps_pro_epoche):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = epochs * steps_pro_epoche
    warmup_steps = warmup_epochs * steps_pro_epoche

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, device, dataset_size):
    model.train()
    total_loss = 0.0
    for bilder, labels in loader:
        bilder, labels = bilder.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(bilder)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * bilder.size(0)
    return total_loss / dataset_size


def evaluate(model, loader, criterion, device, dataset_size):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for bilder, labels in loader:
            bilder, labels = bilder.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(bilder)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * bilder.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / dataset_size, correct / dataset_size


def formatiere_dauer(sekunden):
    minuten, sek = divmod(int(sekunden), 60)
    stunden, minuten = divmod(minuten, 60)
    if stunden > 0:
        return f"{stunden}h {minuten}min {sek}s"
    elif minuten > 0:
        return f"{minuten}min {sek}s"
    return f"{sek}s"


def trainiere(model, train_loader, val_loader, train_ds, val_ds, optimizer, criterion,
              scheduler, scaler, device, epochs, checkpoint_pfad,DEVICE):
    best_val_acc = 0.0
    best_epoch = None
    verlauf = []
    trainingsstart = time.time()
    for epoch in range(epochs):
        epochenstart = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, device, len(train_ds))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, len(val_ds))

        epochendauer = time.time() - epochenstart
        ist_bestes = val_acc > best_val_acc
        status = "[NEU] bestes Modell" if ist_bestes else "(keine Verbesserung)"

        print(f"Epoche {epoch+1:>3}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Dauer: {formatiere_dauer(epochendauer):>10} | {status}")

        verlauf.append({
            "epoche": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "dauer_sek": epochendauer,
            "bestes_modell": ist_bestes,
        })

        if ist_bestes:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_pfad)

    gesamtdauer = time.time() - trainingsstart

    print("\n" + "=" * 75)
    print("TRAININGS-ÜBERSICHT")
    print("=" * 75)
    print(f"Verwendetes Gerät: {DEVICE}")
    print(f"Modell liegt auf: {next(model.parameters()).device}")
    print(Counter(val_ds.targets))
    print(f"Gesamtdauer: {formatiere_dauer(gesamtdauer)} für {epochs} Epochen\n")
    print(f"{'Epoche':>7} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Dauer':>10} | Beste?")
    print("-" * 75)
    for eintrag in verlauf:
        markierung = "*" if eintrag["bestes_modell"] else ""
        print(f"{eintrag['epoche']:>7} | {eintrag['train_loss']:>10.4f} | {eintrag['val_loss']:>10.4f} | "
              f"{eintrag['val_acc']:>8.4f} | {formatiere_dauer(eintrag['dauer_sek']):>10} | {markierung}")
    print("-" * 75)
    print(f"\nBestes Modell: Epoche {best_epoch}, Val Acc: {best_val_acc:.4f}")
    print(f"Gespeichert unter: {checkpoint_pfad}")

    return verlauf, best_epoch, best_val_acc


def lade_trainiertes_modell(modell_name, checkpoint_pfad, num_classes, device):
    model = timm.create_model(modell_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_pfad, map_location=device))
    model = model.to(device)
    model.eval()
    return model