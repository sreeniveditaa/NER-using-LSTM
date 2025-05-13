# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

### Problem Statement
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

### Dataset
Size: Contains thousands of words grouped into sentences with entity annotations.

Columns:
Sentence # – Sentence ID

Word – Individual word/token in the sentence

POS – Part-of-speech tag

Tag – Named entity tag (e.g., O, B-PER, I-LOC, etc.)


## DESIGN STEPS

### STEP 1
Import necessary libraries and set up the device (CPU or GPU).

### STEP 2
Load the NER dataset and fill missing values.

### STEP 3
Create word and tag dictionaries for encoding.

### STEP 4
Group words into sentences and encode them into numbers.

### STEP 5
Build a BiLSTM model for sequence tagging.

### STEP 6
Train the model using the training data.

### STEP 7
Evaluate the model performance on test data.



## PROGRAM
### Name:SREE NIVEDITAA SARAVANAN
### Register Number: 212223230213
```python
class BiLSTMTagger(nn.Module):
    def __init__(self,vocab_size,tagset_size,embedding_dim=50,hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,tagset_size)


    def forward(self, x):
        x=self.embedding(x)
        x=self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)
        
model = BiLSTMTagger(len(word2idx) +1,len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001) 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss=0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        train_losses.append(total_loss)
        model.eval()
        val_loss=0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
                val_loss+=loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1} : Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![{F66117C5-FDB1-492B-849D-04ED0E019D3F}](https://github.com/user-attachments/assets/6ae40dd8-8688-4e19-850c-f76e22b1e90b)


### Sample Text Prediction
![{2C8324FA-4A8B-49BF-8FC9-C6C5813EA3A0}](https://github.com/user-attachments/assets/a8e6f50f-296f-44fa-ba7d-ed2398a2a77b)


## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
