import torch
import torch.nn as nn
import torchvision.models as models


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
        
        super().__init__()
        self.layer_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias = True)
        # This layer turns words received into a vector of any size as mentioned
        self.turn_to_embeddings = nn.Embedding(vocab_size, embed_size)
        # Saving the size of hidden in order to use in later function calls
        self.size_middle = hidden_size
        # This a fully connected layer which maps the dimension of the hidden state output to the number of words we want as output
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        collection_size = features.shape[0] 
        # This line defines a hidden state with all zeros as that is the case when we start training
        self.middle = (torch.zeros((1, collection_size, self.size_middle), device=self.cuda_device), torch.zeros((1, collection_size, self.size_middle), device=self.cuda_device)) 
        
        # We certainly would like to avoid predicting when <end> is passed as input to our RNN, hence we drop it
        collection_of_embedded_words = captions[:, :-1]
        # Just like in the concept of one hot encoding, here we make embedded vector for caption words
        collection_of_embedded_words = self.turn_to_embeddings(collection_of_embedded_words) 
        # Concatenating the obtained captions and features
        collection_of_float_vectors = torch.cat((features.unsqueeze(1), collection_of_embedded_words), dim=1) 
        
        answer_lstm, self.middle = self.layer_lstm(collection_of_float_vectors, self.middle) 

        answer = self.fc1(answer_lstm) 

        return answer

    def sample(self, inputs, states=None, max_len=20):
        
        answer_sample = []
        new_inputs = inputs
        middle = (torch.zeros((1, inputs.shape[0], self.size_middle), device=self.cuda_device), torch.zeros((1, inputs.shape[0], self.size_middle), device=self.cuda_device)) 
    
        while True:
            layer_answer, middle = self.layer_lstm(new_inputs, middle) 
            fc_output = self.fc1(layer_answer)  
            fc_output = fc_output.squeeze(1) 
            list_of_values_at_sampled_indices, list_of_sampled_indices = torch.max(fc_output, dim=1) 
            
            answer_sample.append(list_of_sampled_indices.cpu().numpy()[0].item()) 
            
            if (list_of_sampled_indices == 1):
                return answer_sample
            
            new_inputs = (self.turn_to_embeddings(list_of_sampled_indices)).unsqueeze(1)
            
        return answer_sample