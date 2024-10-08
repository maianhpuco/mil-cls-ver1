import torch.nn as nn  # Import the neural network module
import torch.nn.functional as F  # Import functional API for activations like ReLU
import torchvision.models as models  # Import pre-defined models from torchvision
import torch
import numpy as np


# Define the ResNetSimCLR model class which inherits from nn.Module
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(
            ResNetSimCLR, self
        ).__init__()  # Call the parent class constructor
        # Define a dictionary of available ResNet models (ResNet18 and ResNet50) with instance normalization
        self.resnet_dict = {
            "resnet18": models.resnet18(
                pretrained=False, norm_layer=nn.InstanceNorm2d
            ),
            "resnet50": models.resnet50(
                pretrained=False, norm_layer=nn.InstanceNorm2d
            ),
        }

        # Get the chosen base model (ResNet18 or ResNet50) from the dictionary
        resnet = self._get_basemodel(base_model)
        num_ftrs = (
            resnet.fc.in_features
        )  # Get the number of features from the final fully connected layer of ResNet

        # Extract all layers except the last fully connected layer (fc) as the feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Define the projection head (MLP with two layers)
        self.l1 = nn.Linear(
            num_ftrs, num_ftrs
        )  # First linear layer (output size = input size)
        self.l2 = nn.Linear(
            num_ftrs, out_dim
        )  # Second linear layer (projects to output dimension)

    # Private function to get the ResNet model based on the provided name (either resnet18 or resnet50)
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[
                model_name
            ]  # Retrieve the model from the dictionary
            print(
                "Feature extractor:", model_name
            )  # Print the name of the chosen base model
            return model  # Return the model
        except:  # If the model name is invalid
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )

    # Forward pass through the model
    def forward(self, x):
        # Pass input through the feature extractor (ResNet without the fully connected layer)
        h = self.features(x)
        h = (
            h.squeeze()
        )  # Remove any singleton dimensions (e.g., [batch_size, num_features, 1, 1] -> [batch_size, num_features])

        # Pass through the projection head
        x = self.l1(h)  # First linear layer
        x = F.relu(x)  # Apply ReLU activation function
        x = self.l2(x)  # Second linear layer (output projection)

        return (
            h,
            x,
        )  # Return both the feature vector (h) and the projection vector (x)


# Define the NT-Xent (Normalized Temperature-scaled Cross Entropy Loss) class
class NTXentLoss(torch.nn.Module):
    # Initialization function
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size  # Set the batch size
        self.temperature = temperature  # Set the temperature parameter (scaling factor for logits)
        self.device = device  # Device to run computations on (GPU/CPU)
        self.softmax = torch.nn.Softmax(
            dim=-1
        )  # Softmax for turning logits into probabilities
        # Create a mask to exclude positive sample pairs from similarity matrix (self-contrast avoidance)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(
            torch.bool
        )
        # Select similarity function based on whether cosine similarity or dot product is used
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity
        )
        # Cross entropy loss function with sum reduction
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # Function to return the similarity function (cosine similarity or dot product)
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(
                dim=-1
            )  # Define cosine similarity function
            return self._cosine_simililarity  # Return cosine similarity
        else:
            return (
                self._dot_simililarity
            )  # Return dot product similarity if cosine is not used

    # Function to create a mask to exclude comparisons of a sample with itself and its augmentations
    def _get_correlated_mask(self):
        # Create an identity matrix to exclude self-similarity
        diag = np.eye(
            2 * self.batch_size
        )  # Identity matrix of size (2*batch_size, 2*batch_size)

        # Create a matrix with ones that shift diagonally by `batch_size`, marking positive pairs
        l1 = np.eye(
            (2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size
        )  # For left diagonal
        l2 = np.eye(
            (2 * self.batch_size), 2 * self.batch_size, k=self.batch_size
        )  # For right diagonal

        # Combine the identity matrix and the diagonal shift matrices
        mask = torch.from_numpy((diag + l1 + l2))

        # Convert to a mask where 1 indicates the samples to be excluded (self or positive pairs)
        mask = (1 - mask).type(torch.bool)

        return mask.to(
            self.device
        )  # Move mask to the same device as the model

    # Static method to compute dot product similarity between two tensors
    @staticmethod
    def _dot_simililarity(x, y):
        # Perform tensordot between x and y for similarity calculation
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N) - similarity matrix between all representations
        return v

    # Method to compute cosine similarity between two tensors
    def _cosine_simililarity(self, x, y):
        # Calculate cosine similarity between x and y
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    # Forward pass for the loss calculation
    def forward(self, zis, zjs):
        # Concatenate positive pairs (zis and zjs) vertically to create a single batch
        representations = torch.cat([zjs, zis], dim=0)

        # Compute the similarity matrix for all pairs
        similarity_matrix = self.similarity_function(
            representations, representations
        )

        # Extract the positive pairs from the diagonal (left-positive and right-positive)
        l_pos = torch.diag(
            similarity_matrix, self.batch_size
        )  # Positive pairs along the left diagonal
        r_pos = torch.diag(
            similarity_matrix, -self.batch_size
        )  # Positive pairs along the right diagonal
        # Concatenate the positive similarities into a single tensor
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # Extract the negative samples (excluding self and positive pairs) using the mask
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        # Concatenate the positive and negative similarities into logits
        logits = torch.cat((positives, negatives), dim=1)
        logits /= (
            self.temperature
        )  # Scale the logits by the temperature parameter

        # Create a tensor of labels (all zeros, because the positive pairs are the first column in logits)
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        # Compute the cross-entropy loss between the logits and the labels
        loss = self.criterion(logits, labels)

        # Return the normalized loss (average loss per sample)
        return loss / (2 * self.batch_size)
