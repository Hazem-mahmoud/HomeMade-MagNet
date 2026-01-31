
import torch
import torch.nn as nn

class SequenceToScalerNetwork(nn.Module):
    """
    Bristol's LSTM Seq2One Architecture.
    """
    def __init__(self,
                 input_dim=1, # Ignored, hardcoded to 3 (B, Freq, Temp)
                 hidden_dim=30,
                 output_dim=1, # Power Loss
                 num_layers=3):
        super(SequenceToScalerNetwork, self).__init__()

        self.hidden_size = hidden_dim
        
        # Bristol uses input_size=1 but constructs a tensor of size 3 (B, F, T) inside, NOT quite.
        # Bristol's code actually expected input of size 3 (B, Freq, Temp) from the start?
        # "inputs = torch.zeros(64, waveStep, 3)" -> Yes.
        # But their LSTM init says input_size=1? 
        # "self.lstm = nn.LSTM(input_size, ..." -> If they used input_size=1, they only fed B?
        # Let's check their forward: "out, _ = self.lstm(in_b)" -> in_b is x[:,:,0:1]. 
        # SO LSTM ONLY SEES B-Field!
        
        # LSTM layer (Processing B-Field Only)
        self.lstm = nn.LSTM(1, # Fixed to 1 for B-field
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # Fully connected layer
        # Input to FC is: LSTM_Out + Freq + Temp = hidden_dim + 2
        self.fc1 = nn.Linear(hidden_dim+2, 128)
        self.fc2 = nn.Linear(128, 196)
        self.fc3 = nn.Linear(196, 128)
        self.fc4 = nn.Linear(128, 96)
        self.fc5 = nn.Linear(96, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, output_dim)

        # Activation function
        self.elu = nn.ELU()


    def forward(self, b_seq, scalars):
        """
        Unified Interface Wrapper.
        
        Args:
            b_seq (Tensor): (Batch, Seq, 1)
            scalars (Tensor): (Batch, 3) -> Freq, Temp, Hdc
        """
        
        # Unpack scalars
        Freq = scalars[:, 0].unsqueeze(1) # (bs, 1)
        Temp = scalars[:, 1].unsqueeze(1) # (bs, 1)
        
        # Bristol's logic:
        # 1. Feed only B-field (b_seq) into LSTM
        # out: (Batch, Seq, Hidden)
        out, _ = self.lstm(b_seq)
        
        # 2. Take last output
        out = out[:, -1, :]  # (Batch, Hidden)
        
        # 3. Concatenate Freq and Temp to the features
        out = torch.cat([out, Freq, Temp], dim=1) # (Batch, Hidden + 2)

        # 4. Deep MLP
        out = self.fc1(out) 
        out = self.elu(self.fc2(out))
        out = self.elu(self.fc3(out))
        out = self.elu(self.fc4(out))
        out = self.elu(self.fc5(out))
        out = self.elu(self.fc6(out))
        out = self.fc7(out)
        out = self.fc8(out)

        return out
