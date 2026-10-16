# konsolidieren.py — führt flux1, flux1_schnell, flux2_klein zu einem Datensatz zusammen
import shutil
from pathlib import Path

QUELLE = Path(r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Datensatz temp")
ZIEL = Path(r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Training\Datensatz")

QUELLEN = ["flux1", "flux1_schnell", "flux2_klein"]
SPLITS = ["train", "val", "test"]
KLASSEN = ["ai", "real"]
BILDER = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PREFIX = {"flux1": "f1_", "flux1_schnell": "f1s_", "flux2_klein": "f2_"}


def main():
    # 1) Sicherstellen, dass im Ziel keine DATEIEN liegen (leere Ordner sind okay)
    if any(p.is_file() for p in ZIEL.rglob("*")):
        raise SystemExit("Im Zielordner liegen bereits Dateien – bitte vorher entfernen, damit nichts überschrieben wird.")

    # 2) Zielstruktur anlegen
    for split in SPLITS:
        for klasse in KLASSEN:
            (ZIEL / split / klasse).mkdir(parents=True, exist_ok=True)

    statistik = {}
    for quelle in QUELLEN:
        n_quelle = 0
        for split in SPLITS:
            for klasse in KLASSEN:
                src = QUELLE / quelle / split / klasse
                dst = ZIEL / split / klasse
                if not src.is_dir():
                    print(f"⚠️  Ordner fehlt, wird übersprungen: {src}")
                    continue
                for datei in sorted(src.iterdir()):
                    if datei.suffix.lower() not in BILDER:
                        continue
                    n_quelle += 1
                    ziel_datei = dst / datei.name
                    if ziel_datei.exists():
                        ziel_datei = dst / f"{PREFIX[quelle]}{datei.name}"
                    shutil.copy2(datei, ziel_datei)
        statistik[quelle] = n_quelle
        print(f"✅ {quelle}: {n_quelle} Bilder kopiert")

    print("\n— Statistik —")
    for quelle, n in statistik.items():
        print(f"  {quelle}: {n}")
    print(f"  Gesamt: {sum(statistik.values())} Bilder kopiert")
    for split in SPLITS:
        for klasse in KLASSEN:
            n = len([p for p in (ZIEL / split / klasse).iterdir() if p.is_file()])
            print(f"  Ziel {split}/{klasse}: {n}")


if __name__ == "__main__":
    main()