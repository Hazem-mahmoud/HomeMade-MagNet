
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerNetwork(nn.Module):
    """
    Fuzhou's Transformer-based architecture.
    """
    def __init__(self, 
        B_in_channel=1024, # Default sequence length
        dim_hidden=24,  
        dim_proj_fusion=40,
        n_encoder_layers=1,
        n_heads=4,
        dropout_encoder=0.0,
        dropout_pos_enc=0.0,
        dim_feedforward_encoder=40,
        ): 
        super().__init__() 

        # Projection for B-field input: maps scalar input to hidden dimension
        self.proj_B = nn.Sequential(
            nn.Linear(1, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden))

        self.positional_encoding_layer = PositionalEncoding(d_model=dim_hidden, 
                                                            dropout=dropout_pos_enc, 
                                                            max_len=B_in_channel)
        
        # Transformer Encoder Layer definition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            ) 
        
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                             num_layers=n_encoder_layers, 
                                             norm=None)
        
        # Fusion Layer: Combines Transformer output (B-field features) with Temp and Freq
        self.proj_fusion = nn.Sequential(
            nn.Linear(dim_hidden+2, dim_proj_fusion),
            nn.Tanh(),
            nn.Linear(dim_proj_fusion, dim_proj_fusion),
            nn.Tanh(),
            nn.Linear(dim_proj_fusion, 1))
        
        # Final Regressor to predict Power Loss from fused features
        self.regressor = nn.Sequential(
            nn.Linear(B_in_channel, 1))

    def forward(self, b_seq, scalars):
        """
        Unified Interface Wrapper.
        
        Args:
            b_seq (Tensor): Input B-field curve, shape (batch_size, seq_len, 1).
            scalars (Tensor): Shape (batch_size, 3) -> Freq, Temp, Hdc.
        """
        # Unpack scalars
        # Dataset returns: Freq, Temp, Hdc
        Freq = scalars[:, 0].unsqueeze(1) # (bs, 1)
        Temp = scalars[:, 1].unsqueeze(1) # (bs, 1)
        
        B_curve = b_seq
        
        batch_size, len_seq, feat_dim = B_curve.shape
        
        # Fuzhou Logic
        B_curve = self.proj_B(B_curve) # (bs,1024,1)->(bs,1024,24)

        # Add Positional Encoding
        B_curve = self.positional_encoding_layer(B_curve)
        B_curve = self.encoder(B_curve) 

        # Repeat Temp and Freq to match sequence length for concatenation
        Temp_rep = Temp.unsqueeze(1).repeat(1, len_seq, 1) # (bs,1)->(bs,1024,1)
        Freq_rep = Freq.unsqueeze(1).repeat(1, len_seq, 1) # (bs,1)->(bs,1024,1)

        # Fuse B-field features with Temp and Freq
        feat = self.proj_fusion(torch.cat([B_curve, Temp_rep, Freq_rep], dim=2)) # (bs,1024,26)->(bs,1024,1)
        feat = feat.reshape(batch_size, -1) 
        P_pred = self.regressor(feat) # (bs,1)

        return P_pred
