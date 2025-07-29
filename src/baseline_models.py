import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """A simple LSTM-based classifier for anomaly detection."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # h0 and c0 are initialized to zero by default
        lstm_out, _ = self.lstm(x)
        # We only need the output of the last time step for classification
        last_hidden_state = lstm_out[:, -1, :]
        output = self.output_layer(last_hidden_state)
        return output


class LSTMAutoencoder(nn.Module):
    """An LSTM-based autoencoder for anomaly detection."""

    def __init__(self, input_dim, encoding_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.bottleneck = nn.Linear(hidden_dim, encoding_dim)

        self.decoder_expand = nn.Linear(encoding_dim, hidden_dim)

        # --- FIX: The decoder's input size should be the hidden_dim ---
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, num_layers, batch_first=True
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Encode
        _, (hidden, _) = self.encoder(x)
        encoded = self.bottleneck(hidden[-1])

        # Decode
        decoded_expanded = self.decoder_expand(encoded)
        # Repeat the context vector for each time step in the sequence
        decoder_input = decoded_expanded.unsqueeze(1).repeat(1, x.size(1), 1)

        reconstructed, _ = self.decoder(decoder_input)

        return reconstructed