import timm
import torch
import torch.nn as nn

class NILMModel(nn.Module):
    def __init__(self, num_classes=23, num_time_outputs=23):
        super(NILMModel, self).__init__()
        self.backbone = timm.create_model('cspresnext50', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        
        self.fc_class_count = nn.Linear(num_features, num_classes)
        self.fc_time = nn.Linear(num_features, num_time_outputs)
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)  # Flatten the features
        # print(f"Flattened features shape: {features.shape}")  # Debugging line
        
        class_count_out = self.fc_class_count(features)
        class_count_out = torch.sigmoid(class_count_out)  # Apply sigmoid for multi-label classification
        
        time_out = self.fc_time(features)
        
        return class_count_out, time_out

# Create an instance of the model
model = NILMModel()
