import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
        super(LlamaZeroShotClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # Zero-shot classification does not require updating llama parameters.
        for param in self.llama.parameters():
            param.requires_grad = False
        assert len(label_names) == self.num_labels
        self.tokenizer = tokenizer
        self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]

    def forward(self, input_ids):
        # Compute the completion probability of each label string
        logits, _ = self.llama(input_ids)
        log_probabilities = F.log_softmax(logits, dim=-1)
        label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
        for i, label_token_ids in enumerate(self.label_name_ids):
            total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
            label_probabilities[:, i] = total_log_prob[:, 0]
        return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
    def __init__(self, config):
        super(LlamaEmbeddingClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # If we use pretrain mode, we freeze Llama parameters.
        for param in self.llama.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

    def forward(self, input_ids):
        '''
        1) Find the hidden state after the final token of the input sequence.
        2) Apply dropout (self.dropout) to the hidden state at training time to mitigate overfitting.
        3) Pass this through the classifier head (self.classifier_head), which will return
           logits (unnormalized probabilities) over all classes.
        4) Take the log-softmax of the logits and return log-probabilities over all classes.
        '''
        # Obtain logits and hidden states from the Llama model
        _, hidden_states = self.llama(input_ids)

        # Extract the hidden state of the final token in the sequence
        final_hidden_state = hidden_states[:, -1, :]

        # Apply dropout to the final hidden state
        final_hidden_state = self.dropout(final_hidden_state)

        # Pass the hidden state through the classifier head to get logits
        logits = self.classifier_head(final_hidden_state)

        # Apply log-softmax to the logits to get log-probabilities
        log_probabilities = F.log_softmax(logits, dim=-1)

        return log_probabilities

class LlamaSentClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super(LlamaSentClassifier, self).__init__()
        self.llama = load_pretrained(config.pretrained_model_path)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(self.llama.config.dim, config.num_labels)

    def forward(self, input_ids, attention_mask=None):
        '''
        1) Encode the sentences using Llama2 to obtain the hidden representation from the final word of the sentence.
        2) Apply dropout to the pooled-output.
        3) Project the pooled-output using a linear layer to classify the sentence.
        4) Return the logits (unnormalized probabilities).
        '''
        # Obtain hidden states from Llama2
        logits, hidden_states = self.llama(input_ids)
        # Extract the hidden state of the final token
        pooled_output = hidden_states[:, -1, :]
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Pass through the classifier head
        logits = self.classifier_head(pooled_output)
        return logits

def fine_tune(config, train_dataloader, val_dataloader, num_epochs, learning_rate):
    '''
    Pipeline to fine-tune the Llama2 model on a downstream sentence classification task.
    1) Load the pretrained model.
    2) Initialize the LlamaSentClassifier.
    3) Define the optimizer and loss function.
    4) Train the model on the training dataset.
    5) Validate the model on the validation dataset.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaSentClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for batch in train_dataloader:
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {total_loss:.4f}, Accuracy: {total_correct / len(val_dataloader.dataset):.4f}")
