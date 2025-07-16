import torch.nn as nn
import torch
import json

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for image classification using PyTorch.
    """
    def __init__(self, config):
        """
        Initialize the CNN with convolutional and fully connected layers.

        Args:
            config (dict): Configuration dictionary with keys:
                - img_size (int): Input image size (assumes square).
                - dropout (float): Dropout rate after the first FC layer.
                - num_classes (int): Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        img_size = config["img_size"]
        dropout = config["dropout"]
        num_classes = config["num_classes"]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (Tensor): Input image batch.

        Returns:
            Tensor: Output class scores.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def fit(self, train_loader, train_dataset, optimizer, criterion, epochs, device, test_loader=None, test_dataset=None):
        """
        Train the CNN model on the given data. The test_dataset and test_loader are optional for calculating overall accuracy during training.

        Args:
            train_loader (DataLoader): DataLoader for training set.
            train_dataset (Dataset): Training dataset.
            optimizer (Optimizer): Optimizer
            criterion (Loss): Loss function.
            epochs (int): Number of training epochs.
            device (torch.device): CPU or GPU.
            test_loader (DataLoader, optional): Test DataLoader.
            test_dataset (Dataset, optional): Test dataset.

        Returns:
            dict: Training history with loss and accuracy per epoch.
        """
        metrics_history = {
            "loss": [],
            "accuracy": []
        }

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            accuracy = correct / len(train_dataset)
            avg_loss = running_loss / len(train_loader)

            metrics_history["loss"].append(avg_loss)
            metrics_history["accuracy"].append(accuracy)

            if test_loader is not None and test_dataset is not None:
                self.evaluate_accuracy(test_loader, test_dataset, device)


            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")

        return metrics_history
    
    def evaluate_accuracy(self, test_loader, test_dataset, device):
        """
        Evaluate model accuracy on the test set.

        Args:
            test_loader (DataLoader): Test data.
            test_dataset (Dataset): Full test dataset for size.
            device (torch.device): Device to run on.
        """
        self.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / len(test_dataset)
        print(f"Overall accuracy: {test_accuracy * 100:.2f}%")

    def save_model(self, metrics_history, path, config):
        """
        Save the model weights, config, and training metrics to file.

        Args:
            metrics_history (dict): Dictionary with training loss and accuracy.
            path (str): Path prefix to save `.pth` and `.json` files.
            config (dict): Model configuration.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config,
            'metrics': metrics_history
        }, path + ".pth")
        
        # Save metrics history  
        with open(path + ".json", "w") as f:
            json.dump(metrics_history, f)   

    def get_true_and_pred_labels(self, test_loader, device):
        """
        Compute true and predicted class labels on the test set.

        Args:
            test_loader (DataLoader): Test data.
            device (torch.device): Device to run on.

        Returns:
            Tuple[List[int], List[int]]: (true_labels, predicted_labels)
        """
        self.eval()
        Y_true_int = []
        Y_pred_int = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = self(images)
                preds = torch.argmax(outputs, dim=1)

                Y_true_int.extend(labels.cpu().numpy())
                Y_pred_int.extend(preds.cpu().numpy())

        return Y_true_int, Y_pred_int
    
def load_SimpleCNN_model(path, device):
    """
    Load a pretrained SimpleCNN model and its training metrics from file.

    Args:
        path (str): Path prefix to `.pth` and `.json` files.
        device (torch.device): Device to load the model onto.

    Returns:
        Tuple[SimpleCNN, dict]: Loaded model and its training metrics.
    """
    checkpoint = torch.load(path + ".pth", map_location=device, weights_only=True)
    model = SimpleCNN(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics_history = checkpoint['metrics']
    return model, metrics_history