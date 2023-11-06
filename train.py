from model import GPT, GPTConfig

import torch
import torch.nn.functional as F



# Load training pairs
train_pairs = torch.load("train_pairs.pt")

# Create model

config = GPTConfig()
model = GPT(config)

# Train model

train_epochs = 10


optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

train_loss = []

print(len(train_pairs))

for epoch in range(train_epochs):
    
        running_loss = 0
    
        for train_pair in train_pairs:
            
            optim.zero_grad()
            x, y = train_pair
            _, loss = model(x.unsqueeze(0), targets=y.long())
            loss.backward()
            optim.step()
            running_loss += loss.item()
    
        epoch_loss = running_loss/len(train_pairs)
        train_loss.append(epoch_loss)
        print(f"{epoch=} : {epoch_loss:.3f}")
        

import matplotlib.pyplot as plt

plt.plot(train_loss)