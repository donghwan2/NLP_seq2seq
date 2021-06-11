import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):

    # 
    def __init__(
        self,
        input_size,    # 입력 크기 : 28
        hidden_size,   # 
        output_size,   # 출력 크기 : 28
        n_layers=4,    # layer 수 : gradient vanishing 때문에 4개 이하 추천
        dropout_p=.2,  # 
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        # LSTM 모델 구성
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )

        # FC layer(CNN과 동일)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)   -> h: time step,  w: time step 당 들어오는 입력의 크기

        z, _ = self.rnn(x)
        # |z| = (batch_size, h, hidden_size * 2)

        z = z[:, -1]   # 맨 마지막 time step에 대한 결과만 가져와라.
        # |z| = (batch_size, hidden_size * 2)
        
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
