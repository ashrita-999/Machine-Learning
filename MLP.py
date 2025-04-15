import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# =========================================================
# 1. Prepare Data
# =========================================================

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
tracker.final = True  # Ensure this is set before scoring


# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# =========================================================
# 2. Define the MLP Model
# =========================================================
class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: define at least two nn.Linear layers
        #       (e.g. input_dim->hidden_dim, then hidden_dim->output_dim)
        #       and an activation function (e.g. nn.ReLU()).
        """
        self.layer1 = ...
        self.activation = ...
        self.layer2 = ...
        """
        self.layer1 = nn.Linear(input_dim, hidden_dim) 
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        # TODO: apply layer1, then activation, then layer2
        """
        x = ...
        return x
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
        
        
        

# TODO: instantiate the model with correct sizes
# e.g. model = MyMLP(input_dim=num_features, hidden_dim=64, output_dim=2)
model = MyMLP(input_dim=num_features, hidden_dim=64, output_dim=2)

# =========================================================
# 3. Define Loss and Optimizer
# =========================================================
# TODO: pick a loss function (nn.CrossEntropyLoss for classification)
# and an optimizer (e.g., optim.Adam)
"""
criterion = ...
optimizer = ...
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================================================
# 4. Training Loop
# =========================================================
num_epochs = 10

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    
    # Track total epoch loss for reporting
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # TODO: Move data to device
    
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        

        # TODO: Zero out the optimizer gradients
        
        optimizer.zero_grad()
        

        # TODO: Forward pass
        
        outputs = model(X_batch)
        

        # TODO: Compute the loss
        
        loss = criterion(outputs, y_batch)
        

        # TODO: Backpropagation
        
        loss.backward()
        optimizer.step()
        

        # Accumulate the total_loss (factor in batch size)
        total_loss += loss.item() * X_batch.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    
    # ================== Validation ==================
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            

            val_outputs = model(X_val_batch)
            
            # For classification, get predicted class via argmax
            predicted_labels = torch.argmax(val_outputs, dim=1)
            
            val_preds.extend(predicted_labels.cpu().numpy())
            val_targets.extend(y_val_batch.cpu().numpy())

    # Compute accuracy
    val_acc = accuracy_score(val_targets, val_preds)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Accuracy: {val_acc:.4f}")

# =========================================================
# 5. Evaluation
# =========================================================
# If you want to evaluate on the entire val set again or do further analysis:
model.eval()
with torch.no_grad():   
    test_outputs = model(X_test_t.to(device))  # Shape: [N, output_dim]
    probs = torch.softmax(test_outputs, dim=1)  # Shape: [N, 2]
    final_preds = probs.cpu().numpy()          # ← Keep this 2D!
    #final_preds = final_preds[:21191]
tracker.final = True


model_performance = tracker.score_prediction(final_preds, model_name='MLP')


