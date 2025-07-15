import torch.nn as nn
import torch
import json

class SimpleCNN(nn.Module):
    def __init__(self, config):
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
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def fit(self, train_loader, train_dataset, optimizer, criterion, epochs, device, val_loader=None, val_dataset=None):
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

            self.evaluate_accuracy(val_loader, val_dataset, device)

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")

        return metrics_history
    
    def evaluate_accuracy(self, val_loader, val_dataset, device):
        self.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / len(val_dataset)
        print(f"🎯 Overall accuracy: {val_accuracy * 100:.2f}%")

    def save_model(self, metrics_history, path, config):
        # Save model
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config,
            'metrics': metrics_history
        }, path + ".pth")
        # Save metrics history  
        with open(path + ".json", "w") as f:
            json.dump(metrics_history, f)   

    def get_true_and_pred_labels(self, val_loader, device):
        self.eval()
        Y_true_int = []
        Y_pred_int = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = self(images)
                preds = torch.argmax(outputs, dim=1)

                Y_true_int.extend(labels.cpu().numpy())
                Y_pred_int.extend(preds.cpu().numpy())

        return Y_true_int, Y_pred_int
    
def load_SimpleCNN_model(path, device):
    checkpoint = torch.load(path + ".pth", map_location=device, weights_only=True)
    model = SimpleCNN(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics_history = checkpoint['metrics']
    return model, metrics_history