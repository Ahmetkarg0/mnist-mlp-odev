import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 128), dropout=0.0, input_size=28 * 28, num_classes=10):
        super().__init__()
        layers = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Softmax yok: CrossEntropyLoss logits bekler
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP()
    print(model)
    print(f"\nEğitilebilir parametre sayısı: {count_parameters(model):,}")

    dummy = torch.randn(4, 1, 28, 28)
    logits = model(dummy)
    print(f"Giriş boyutu: {tuple(dummy.shape)} -> Çıkış boyutu: {tuple(logits.shape)}")
    print("Softmax sonrası ilk örneğin olasılıkları:", torch.softmax(logits[0], dim=0).detach().numpy().round(3))
