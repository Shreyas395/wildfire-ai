import torch
from wildfire_ai.improved_model import ImprovedWildfireCNN

model = ImprovedWildfireCNN(num_classes=2)

checkpoint = torch.load(
    'wildfire_model_epoch7_acc0.720_20251116_194209.pth',
    map_location=torch.device('cpu')  
)

model.load_state_dict(checkpoint['model_state_dict'])
print(model.state_dict())
model.eval()