import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

class FloodNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        for img_name in os.listdir(img_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                self.images.append(os.path.join(img_dir, img_name))
                mask_name = img_name.replace('.jpg', '.png').replace('.png', '.png')
                self.masks.append(os.path.join(mask_dir, mask_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=1):
        super(TransferLearningModel, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), 1, 1, 1)
        x = self.upsample(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Época {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            if i < 5:  # Guardar las primeras 5 predicciones
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(images[0].cpu().permute(1, 2, 0))
                plt.title('Imagen Original')
                
                plt.subplot(1, 3, 2)
                plt.imshow(masks[0].cpu().squeeze(), cmap='gray')
                plt.title('Máscara Real')
                
                plt.subplot(1, 3, 3)
                plt.imshow(outputs[0].cpu().squeeze(), cmap='gray')
                plt.title('Predicción')
                
                plt.savefig(f'prediction_{i}.png')
                plt.close()

def main():
    # Rutas de los directorios
    img_dir_train = r'D:\FloodNet-Supervised_v1.0\train\train-org-img'
    mask_dir_train = r'D:\ColorMasks-FloodNetv1.0\ColorMasks-TrainSet'
    img_dir_test = r'D:\FloodNet-Supervised_v1.0\test\test-org-img'
    mask_dir_test = r'D:\ColorMasks-FloodNetv1.0\ColorMasks-TestSet'

    # Configurar transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    train_dataset = FloodNetDataset(img_dir_train, mask_dir_train, transform=transform)
    test_dataset = FloodNetDataset(img_dir_test, mask_dir_test, transform=transform)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Inicializar modelo, criterio y optimizador
    model = TransferLearningModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenar el modelo
    train_model(model, train_loader, criterion, optimizer)
    
    # Guardar el modelo
    torch.save(model.state_dict(), 'flood_net_transfer_learning_model.pth')

    # Evaluar el modelo
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()