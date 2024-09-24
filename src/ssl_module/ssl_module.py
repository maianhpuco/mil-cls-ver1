
import torch
import torch.nn as nn
import torch.nn.functional as F

# BaseModel Class
class SSLModel(nn.Module):
    def __init__(self):
        super(BaseSSLModel, self).__init__()

    @staticmethod
    def get_model(model_name, feature_dim, projection_dim, **kwargs):
        """
        Factory method to return the correct model class (SimCLR or MAE).
        """
        if model_name.lower() == "simclr":
            pass 
#            return SimCLR(feature_dim, projection_dim, **kwargs)
        elif model_name.lower() == "mae":
            pass  
#            return MAE(feature_dim, projection_dim, **kwargs)
        else:
            raise ValueError(f"Model {model_name} not recognized. Available models: 'simclr', 'mae'.")

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the compute_loss method.")
       return loss.mean()



