import timm
from safetensors.torch import load_file
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

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

DATA_DIR = r"C:\Users\Andrin\Documents\Maturaarbeit_W11WS16\parktisches_Experiment\Datensatz\coco_project\experiment_dataset"  # Ordner mit train/, val/, test/

data_config = timm.data.resolve_model_data_config(model)
mean, std = data_config["mean"], data_config["std"]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_ds = ImageFolder(f"{DATA_DIR}\\train", transform=train_transform)
val_ds = ImageFolder(f"{DATA_DIR}\\val", transform=val_transform)
test_ds = ImageFolder(f"{DATA_DIR}\\test", transform=val_transform)

print(train_ds.class_to_idx)  # zeigt dir, welches Label zu 'ai' und welches zu 'real' gehört

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

print(train_ds[0])