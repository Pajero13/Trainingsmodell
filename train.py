import torch.nn as nn
import torch

from config import *
from engine import lade_modell, lade_daten, erstelle_optimizer_und_scheduler, trainiere

model = lade_modell(MODELL_NAME, PRETRAINED_PFAD, NUM_CLASSES, DEVICE)

train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = lade_daten(
    model, DATA_DIR, BATCH_SIZE, NUM_WORKERS
)

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = erstelle_optimizer_und_scheduler(
    model, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, WARMUP_EPOCHS, len(train_loader)
)
scaler = torch.cuda.amp.GradScaler()

trainiere(
    model, train_loader, val_loader, train_ds, val_ds,
    optimizer, criterion, scheduler, scaler,
    DEVICE, EPOCHS, CHECKPOINT_PFAD
)