import argparse
import torch
import math
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from NILMDataset import CustomImageDataset
from NILMModel import NILMModel
from sklearn.metrics import f1_score, silhouette_score, precision_recall_fscore_support



def parse_args():
    parser = argparse.ArgumentParser(description='Train NILM Model')
    parser.add_argument('--data_dir', type=str, default='/home/hernies/Documents/tfg/full_model/data/REFIT_GAF', help='Directory with training data')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
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


def calculate_loss(pred_class_count, pred_time, true_class_count, true_time):
    #print all parameters
    # print(f'pred_class_count: {pred_class_count}')
    # print(f'true_class_count: {true_class_count}')
    # print(f'pred_time: {pred_time}')
    # print(f'true_time: {true_time}')
    class_count_loss = F.binary_cross_entropy(pred_class_count, true_class_count.float())
    time_loss = F.mse_loss(pred_time, true_time)
    alpha = 0.1  # Weight for class count loss
    total_loss = alpha * class_count_loss +  time_loss
    return total_loss

def calculate_accuracy(true_labels, pred_labels):
    true_labels_flat = true_labels.view(-1)
    pred_labels_flat = pred_labels.view(-1)
    TOs = (true_labels_flat == pred_labels_flat).sum().item()
    FOs = (true_labels_flat != pred_labels_flat).sum().item()
    accuracy = TOs / (TOs + FOs)
    return accuracy



def calculate_partial_f1(true_labels, pred_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels.cpu(), pred_labels.cpu(), average=None)
    return precision, recall, f1


def calculate_f1_score(true_labels, pred_labels):
    f1_scores = f1_score(true_labels, pred_labels, average=None)  # Get F1 score for each class
    return f1_scores

def calculate_weighted_f1_score(true_labels, pred_labels):
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    return f1_weighted

def calculate_silhouette_score(features, labels):
    silhouette_avg = silhouette_score(features, labels)
    return silhouette_avg

def train(model, train_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, time_labels, class_labels) in enumerate(train_loader):
            inputs, class_labels, time_labels = inputs.to(device), class_labels.to(device), time_labels.to(device)
            
            optimizer.zero_grad()
            class_outputs, time_outputs = model(inputs)
            
            loss = calculate_loss(class_outputs, time_outputs, class_labels, time_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f'Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}', end='\r')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    print('Training completed')
    torch.save(model.state_dict(), 'trained_model.pth')

def evaluate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, time_labels, class_labels in val_loader:
            inputs, class_labels, time_labels = inputs.to(device), class_labels.to(device), time_labels.to(device)
            class_outputs, time_outputs = model(inputs)
            
            loss = calculate_loss(class_outputs, time_outputs, class_labels, time_labels)
            val_loss += loss.item()
            
            # Compute partial metrics
            _, predicted = torch.max(class_outputs, 1)
            batch_correct, batch_total = calculate_accuracy(class_labels, predicted)
            correct += batch_correct
            total += batch_total
            
            precision, recall, f1 = calculate_partial_f1(class_labels, predicted)
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            num_batches += 1

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    avg_precision = precision_sum / num_batches
    avg_recall = recall_sum / num_batches
    avg_f1 = f1_sum / num_batches

    return avg_val_loss, accuracy, avg_precision, avg_recall, avg_f1

# Training and evaluation functions
def train_and_evaluate(model, train_loader, val_loader, epochs, learning_rate, device):
    train(model, train_loader, epochs, learning_rate, device)
    
    # Load the trained model
    model.load_state_dict(torch.load('trained_model.pth'))
    
    # Evaluate the model
    avg_val_loss, accuracy, avg_precision, avg_recall, avg_f1 = evaluate_model(model, val_loader, device)
    
    print(f'Final Validation Loss: {avg_val_loss:.4f}')
    print(f'Final Accuracy: {accuracy:.4f}')
    print(f'Final Precision: {avg_precision.mean():.4f}')
    print(f'Final Recall: {avg_recall.mean():.4f}')
    print(f'Final F1 Score: {avg_f1.mean():.4f}')

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
    train_and_evaluate(model, train_loader, val_loader, args.epochs, args.learning_rate, device)
    save_model(model, args.save_path)

if __name__ == '__main__':
    main()
