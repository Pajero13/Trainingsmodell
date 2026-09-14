import torch

# Pfade
DATA_DIR = r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Datensatz"
MODELLE_ORDNER = r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Modelle"
PRETRAINED_PFAD = r"C:\Users\Andrin\Documents\ConvNet\Modelle\model.safetensors"
MODELLINFO_ORDNER = r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Modellinformationen"

# Modell
MODELL_NAME = "convnext_tiny.in12k_ft_in1k_384"
NUM_CLASSES = 2

# Training
BATCH_SIZE = 32
EPOCHS = 2
WARMUP_EPOCHS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
SUBSET_GROESSE = None  # None = ganzer Datensatz, Zahl = nur so viele Bilder pro Split (für Testdurchlauf)
GRADIENT_ACCUMULATION_STEPS = 1      # 1 = keine Accumulation, effektive Batch-Size = BATCH_SIZE * dieser Wert
EARLY_STOPPING_PATIENCE = None       # None = deaktiviert, sonst Anzahl Epochen ohne Verbesserung bis Abbruch
PRECISION = "amp"                    # "amp" (Mixed Precision) oder "fp32" (volle Genauigkeit)
AUGMENTIERUNG = "horizontal_flip"    # "horizontal_flip" oder "keine"

# Gerät
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")