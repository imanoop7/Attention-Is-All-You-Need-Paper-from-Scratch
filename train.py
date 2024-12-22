import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer import Transformer  # Import the Transformer model

class TransformerTrainer:
    """
    A trainer class for the Transformer model
    
    Args:
        model (nn.Module): The Transformer model
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on (CPU/GPU)
    """
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, dataloader):
        """
        Train the model for one epoch
        
        Args:
            dataloader (DataLoader): DataLoader containing training data
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            # Move data to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(src, tgt[:, :-1])  # Teacher forcing
            
            # Calculate loss
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on validation/test data
        
        Args:
            dataloader (DataLoader): DataLoader containing evaluation data
            
        Returns:
            float: Average loss for the evaluation
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in dataloader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

def train_transformer():
    # Hyperparameters
    EPOCHS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, criterion, optimizer
    vocab_size = 10000  # Example vocabulary size, replace with actual size
    pad_idx = 0  # Example padding index, replace with actual index
    model = Transformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create data loaders
    train_dataset = ...  # Replace with your training dataset
    val_dataset = ...  # Replace with your validation dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize trainer
    trainer = TransformerTrainer(model, criterion, optimizer, DEVICE)
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate on validation data
        val_loss = trainer.evaluate(val_loader)
        
        # Print training and validation loss
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
if __name__ == "__main__":
    train_transformer()