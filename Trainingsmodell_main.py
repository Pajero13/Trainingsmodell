import timm
from safetensors.torch import load_file

pfad_zu_datei = r"C:\Users\Andrin\Documents\ConvNet\Modelle\model.safetensors"

model = timm.create_model('convnext_tiny.in12k_ft_in1k_384', pretrained=False, num_classes=2)
state_dict = load_file(pfad_zu_datei)

# Head-Gewichte aus dem Checkpoint entfernen, da sie nicht zu num_classes=2 passen
state_dict.pop('head.fc.weight', None)
state_dict.pop('head.fc.bias', None)

model.load_state_dict(state_dict, strict=False)
print("Modell geladen (Backbone aus Checkpoint, Head neu initialisiert)")

#Debug
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Fehlende Keys (sollte nur head.fc sein):", missing)
print("Unerwartete Keys (sollte leer sein):", unexpected)