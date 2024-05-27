import timm
import torch
import torch.nn as nn

class NILMModel(nn.Module):
    def __init__(self):
        super(NILMModel, self).__init__()
        # Load the pretrained CSPResNeXt50 model without the final classification layer
        self.backbone = timm.create_model('cspresnext50', pretrained=True, num_classes=0)  # Set num_classes=0 to remove the default head
        num_features = self.backbone.num_features  # Get the number of features from the backbone
        
        # Define fully connected layers
        self.fc_class_count = nn.Linear(num_features, 23)  # For class count prediction
        self.fc_time = nn.Linear(num_features, 9)     # For time frame prediction (9 instances * 23 classes)
        
    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Predict class counts
        class_count_out = self.fc_class_count(features)
        
        # Predict time frames, reshaped to (9, 23)
        time_out = self.fc_time(features)
        
        return class_count_out, time_out

# Create an instance of the model
model = NILMModel()
