from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, 
                                                    output_hidden_states=True, 
                                                    max_position_embeddings = cfg.max_len)
        else:
            self.config = torch.load(config_path)

        self.model = AutoModel.from_pretrained(cfg.model, config=self.config) 

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropouts = [self.dropout1, self.dropout2, self.dropout3, self.dropout4, self.dropout5]

        self.hidden_lstm_size = cfg.max_len
        self.LSTM1 = nn.LSTM(input_size=self.config.hidden_size, 
                        hidden_size=self.hidden_lstm_size, 
                        num_layers=1, 
                        batch_first=True, 
                        dropout = 0.3,
                        bidirectional = True)
      

        self.ff_pooling = nn.Sequential(
            nn.Linear(int(self.hidden_lstm_size * 2), 768),
            nn.Tanh(),
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Linear(self.hidden_lstm_size * 2, 2)
        
    def squash(self, inputs):
        weights = self.ff_pooling(inputs)
        sentence_embedding = torch.sum(inputs * weights, dim=1)
        return sentence_embedding
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        self.h0 = torch.rand(2, last_hidden_states.size(0), self.hidden_lstm_size).to(device)
        self.c0 = torch.rand(2, last_hidden_states.size(0), self.hidden_lstm_size).to(device)
        sentence_embedding, _ = self.LSTM1(last_hidden_states, (self.h0, self.c0))
        sentence_embedding = self.squash(sentence_embedding)
        sentence_with_dropout = (sum([w(sentence_embedding) for w in self.dropouts]) / 5)
        linear_output = self.fc(sentence_with_dropout)
        return linear_output