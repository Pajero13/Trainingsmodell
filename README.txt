============================================================
Trainingsparameter (train.py)
============================================================

Ausführung:
    python train.py --name <modellname>

------------------------------------------------------------
KOMMANDOZEILEN-ARGUMENTE
------------------------------------------------------------
--name          (Pflicht) Name für das trainierte Modell und die
                zugehörige Konfigurations-/Ergebnisdatei, ohne
                Dateiendung. Wird verwendet als:
                  - Modelle/<name>.pt
                  - Modellinformationen/<name>.json
                Beispiel: python train.py --name flux1_epochs40_lr5e-5
