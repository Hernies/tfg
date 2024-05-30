import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from NILMDataset import CustomImageDataset
from NILMModel import NILMModel
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser(description='Train NILM Model')
    parser.add_argument('--data_dir', type=str, default='/home/hernies/Documents/tfg/full_model/data/REFIT_GAF', help='Directory with training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=31, help='Number of epochs to train')
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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader

def define_model(device):
    model = NILMModel().to(device)  # Ensure the model is on the GPU
    return model


def calculate_loss(pred_class_count, pred_time, true_class_count, true_time, alpha=0.5):
    class_count_loss = F.binary_cross_entropy(pred_class_count, true_class_count.float())
    time_loss = F.mse_loss(pred_time, true_time)
    print(f"Class Count Loss: {alpha * class_count_loss}, Time Loss: {(1 - alpha) * time_loss}")
    return alpha * class_count_loss + (1 - alpha) * time_loss

def train(model, train_loader, val_loader, epochs, learning_rate, device, alpha=0.5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (image, time, classes) in enumerate(train_loader):
            image, time, classes = image.to(device), time.to(device), classes.to(device)
            
            optimizer.zero_grad()
            class_outputs, time_outputs = model(image)

            # Debugging statements
            print("Class Outputs:", class_outputs)
            print("True Class Labels:", classes)
            print("Time Outputs:", time_outputs)
            print("True Time Labels:", time)
            
            loss = calculate_loss(class_outputs, time_outputs, classes, time, alpha)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, time, classes in val_loader:
                image, time, classes = image.to(device), time.to(device), classes.to(device)
                class_outputs, time_outputs = model(image)
                loss = calculate_loss(class_outputs, time_outputs, time, classes, alpha)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print('Training completed')
    model.load_state_dict(torch.load('best_model.pth'))

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
