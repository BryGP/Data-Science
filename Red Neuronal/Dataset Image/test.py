import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            # Guardar algunas predicciones para visualización
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