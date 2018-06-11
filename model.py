import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super(DecoderRNN, self).__init__()
        super().__init__()

        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.hidden = (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))
    
    def forward(self, features, captions):
        # features : (batch_size, embed_size)
        print("batch_size: ", features.shape[0])
        print("embed_size: ", features.shape[1])
        batch_size = features.shape[0]
        
        embeded_captions = self.word_embeddings(captions[:, :-1])
        embeded_captions = torch.cat((features.unsqueeze(1), embeded_captions), 1)
        ltsm_out, hidden = self.lstm(embeded_captions)
        outputs = self.linear(ltsm_out)
  
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_ids = []
        #inputs = inputs.unsqueeze(1)
        for i in range(max_len):
            # (batch_size, 1, hidden_size)
            outputs, states = self.lstm(inputs, states)
            outputs = self.linear(outputs.squeeze(1))
            predicted = outputs.max(1)[1]
            sample_ids.append(predicted.item())
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        
        return sample_ids
        