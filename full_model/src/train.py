import argparse
import torch
import math
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from NILMDataset import CustomImageDataset
from NILMModel import NILMModel
from CustomCrossEntropy import CustomCrossEntropyLoss
from sklearn.metrics import f1_score, precision_recall_fscore_support

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

def calculate_loss(model, pred_class_count, pred_time, true_class_count, true_time, training, alpha=1):
    # Compute the losses
    class_loss = CustomCrossEntropyLoss()(pred_class_count, true_class_count)
    time_loss = F.mse_loss(pred_time, true_time)

    if training:
        # Zero out any existing gradients
        model.zero_grad()

        # Calculate gradients for class_loss
        class_loss.backward(retain_graph=True)
        class_grads = [p.grad.clone().detach() for p in model.parameters() if p.grad is not None]
        class_grad_norm = torch.sqrt(sum([g.norm() ** 2 for g in class_grads]))

        # Zero gradients before calculating for the next loss
        model.zero_grad()

        # Calculate gradients for time_loss
        time_loss.backward(retain_graph=True)
        time_grads = [p.grad.clone().detach() for p in model.parameters() if p.grad is not None]
        time_grad_norm = torch.sqrt(sum([g.norm() ** 2 for g in time_grads]))

        avg_grad_norm = (class_grad_norm + time_grad_norm) / 2
        class_weight = (class_grad_norm / avg_grad_norm).pow(alpha)
        time_weight = (time_grad_norm / avg_grad_norm).pow(alpha)

        sum_weights = class_weight + time_weight
        class_weight /= sum_weights
        time_weight /= sum_weights

        loss = class_loss * class_weight + time_loss * time_weight
    else:
        loss = class_loss + time_loss

    return loss


def calculate_accuracy(true_labels, pred_labels):
    true_labels_flat = true_labels.view(-1)
    pred_labels_flat = pred_labels.view(-1)
    correct_predictions = (true_labels_flat == pred_labels_flat).sum().item()
    total_predictions = true_labels_flat.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_partial_f1(true_labels, pred_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels.cpu(), pred_labels.cpu(), average=None)
    return precision, recall, f1

def calculate_f1_score(true_labels, pred_labels):
    f1_scores = f1_score(true_labels.cpu(), pred_labels.cpu(), average=None)  # Get F1 score for each class
    return f1_scores

def calculate_weighted_f1_score(true_labels, pred_labels):
    f1_weighted = f1_score(true_labels.cpu(), pred_labels.cpu(), average='weighted')
    return f1_weighted

def train_and_evaluate(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, time_labels, class_labels) in enumerate(train_loader):
            inputs, class_labels, time_labels  = inputs.to(device), class_labels.to(device), time_labels.to(device)
            
            optimizer.zero_grad()
            class_outputs, time_outputs = model(inputs)
            
            loss = calculate_loss(model, class_outputs, time_outputs, class_labels, time_labels,True)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f'Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}', end='\r')

        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # Validation phase
        # Validation phase
                # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_f1_accumulated = 0.0
        val_weighted_f1_accumulated = 0.0
        val_time_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for i, (inputs, time_labels, class_labels) in enumerate(val_loader):
                inputs, class_labels, time_labels = inputs.to(device), class_labels.to(device), time_labels.to(device)
                class_outputs, time_outputs = model(inputs)
                
                val_loss = calculate_loss(model, class_outputs, time_outputs, class_labels, time_labels, training=False)
                val_running_loss += val_loss.item()
                
                # Convert class_outputs to predicted labels
                _, predicted = torch.max(class_outputs, 1)
                
                # Convert class_labels from one-hot encoding to class indices
                class_labels_indices = torch.argmax(class_labels, dim=1)

                # Ensure predicted and class_labels_indices are of the same shape
                val_correct += (predicted == class_labels_indices).sum().item()
                val_total += class_labels.size(0)
                
                # Calculate batch-wise F1 and accumulate
                batch_f1 = calculate_f1_score(class_labels_indices.cpu(), predicted.cpu())
                batch_weighted_f1 = calculate_weighted_f1_score(class_labels_indices.cpu(), predicted.cpu())
                
                val_f1_accumulated += batch_f1.mean()  # Average F1 score for this batch
                val_weighted_f1_accumulated += batch_weighted_f1
                batch_count += 1
                
                # Calculate time loss
                time_loss = F.mse_loss(time_outputs, time_labels)
                val_time_loss += time_loss.item()
                
            val_avg_loss = val_running_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            avg_f1 = val_f1_accumulated / batch_count
            avg_weighted_f1 = val_weighted_f1_accumulated / batch_count
            avg_time_loss = val_time_loss / len(val_loader)
            
            print(f'Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}, Avg F1: {avg_f1:.4f}, Avg Weighted F1: {avg_weighted_f1:.4f}, Avg Time Loss: {avg_time_loss:.4f}')


    print('Training completed')
    torch.save(model.state_dict(), 'trained_model.pth')

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and torch.backends.mps is available():  # For macOS with Metal Performance Shaders
        device = torch.device('mps')

    train_loader, val_loader = load_data(args.data_dir, args.batch_size)
    model = define_model(device)  # Ensure the model is on the GPU
    print(f"Beginning training on Device: {device}")
    train_and_evaluate(model, train_loader, val_loader, args.epochs, args.learning_rate, device)
    save_model(model, args.save_path)

if __name__ == '__main__':
    main()
