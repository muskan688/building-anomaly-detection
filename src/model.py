import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder for univariate hourly windows.

    Encoder compresses the window into a fixed latent vector (final hidden state);
    the decoder reconstructs the window from that vector, repeated at every step.
    Anomaly score = reconstruction MSE.
    """

    def __init__(self, seq_len: int = 24, n_features: int = 1, hidden_dim: int = 64,
                 latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])  # (batch, latent_dim)
        decoder_input = self.from_latent(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)  # (batch, seq_len, n_features)
