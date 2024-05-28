import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from NILMDataset import CustomImageDataset
from NILMModel import NILMModel
import torch.nn.functional as F


def calculate_loss(pred_class_count, pred_time, true_class_count, true_time):
    class_count_loss = F.mse_loss(pred_class_count, true_class_count.float())
    time_loss = F.mse_loss(pred_time, true_time)
    return time_loss + class_count_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train NILM Model')
    parser.add_argument('--data_dir', type=str, default='/home/hernies/Documents/tfg/full_model/data/REFIT_GAF', help='Directory with training data')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='nilm_model.pth', help='Path to save the trained model')
    return parser.parse_args()

def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor()
    ])
    
    print("Loading dataset...")
    dataset = CustomImageDataset(data_dir, transform=transform)
    print("Dataset loaded")
    
    train_size = int(0.8 * (len(dataset)))
    val_size = int(len(dataset) - train_size)
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader

def define_model(device):
    model = NILMModel().to(device)  # Ensure the model is on the GPU
    return model

def train(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        i = 0
        for inputs, onehot_labels, house_labels in train_loader:
            inputs, onehot_labels, house_labels = inputs.to(device), onehot_labels.to(device), house_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = calculate_loss(outputs[0], outputs[1], onehot_labels, house_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            i += 1
            print(f'Batch {i}/{len(train_loader)}', end='\r')  # Carriage return to overwrite line
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, onehot_labels, house_labels in val_loader:
                inputs, onehot_labels, house_labels = inputs.to(device), onehot_labels.to(device), house_labels.to(device)
                outputs = model(inputs)
                loss = calculate_loss(outputs[0], outputs[1], onehot_labels, house_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')
    
    print('Training completed')

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and torch.backends.mps.is_available():  # For macOS with Metal Performance Shaders
        device = torch.device('mps')

    train_loader, val_loader = load_data(args.data_dir, args.batch_size)
    model = define_model(device)  # Ensure the model is on the GPU
    print(f"Beginning training on Device: {device}")
    train(model, train_loader, val_loader, args.epochs, args.learning_rate, device)
    save_model(model, args.save_path)

if __name__ == '__main__':
    main()
