import timm
import torch.nn as nn

class NILMModel(nn.Module):
    def __init__(self):
        super(NILMModel, self).__init__()
        # Load the pretrained CSPResNeXt50 model without the final classification layer
        self.backbone = timm.create_model('cspresnext50', pretrained=True, num_classes=0)  # Set num_classes=0 to remove the default head
        num_features = self.backbone.num_features  # Get the number of features from the backbone
        # add custom head to the model to accept 10 256x256 rgb images as input
        # Define two separate fully connected layers
        self.fc_class = nn.Linear(num_features, 23)  # For class prediction
        self.fc_time = nn.Linear(num_features, 23)   # For time frame prediction

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Predict class and time frame
        class_out = self.fc_class(features)
        time_out = self.fc_time(features)
        
        return class_out, time_out

# Create an instance of the model
model = NILMModel()
