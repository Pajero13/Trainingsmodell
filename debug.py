import hashlib
from pathlib import Path
from config import *


import imagehash
from PIL import Image
from pathlib import Path

def sammle_phashes(ordner):
    hashes = {}
    for pfad in Path(ordner).rglob("*.png"):
        h = imagehash.phash(Image.open(pfad))
        hashes.setdefault(h, []).append(pfad)
    return hashes

train_p = sammle_phashes(f"{DATA_DIR}\\train")
val_p = sammle_phashes(f"{DATA_DIR}\\val")
print(f"Ähnliche Bilder (phash) zwischen train und val: {len(set(train_p) & set(val_p))}")