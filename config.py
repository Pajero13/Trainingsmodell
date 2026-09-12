import torch

# Pfade
DATA_DIR = r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Datensatz"
CHECKPOINT_PFAD = r"C:\Users\Andrin\Documents\ConvNet\Modelle\bestes_modell.pt"
PRETRAINED_PFAD = r"C:\Users\Andrin\Documents\ConvNet\Modelle\model.safetensors"

# Modell
MODELL_NAME = "convnext_tiny.in12k_ft_in1k_384"
NUM_CLASSES = 2

# Training
BATCH_SIZE = 16
EPOCHS = 1
WARMUP_EPOCHS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
SUBSET_GROESSE = 20  # None = ganzer Datensatz, Zahl = nur so viele Bilder pro Split (für Testdurchlauf)

# Gerät
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")