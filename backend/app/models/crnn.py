import timm
import torch.nn as nn
import torch

class CRNN(nn.Module):
    def __init__(
            self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True
        
        self.mapSeq = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Dropout(dropout))

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional = True,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size*2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        # Add input validation
        if x.dim() != 4:
            raise ValueError("Expected 4D input (batch, channels, height, width)")
            
        # Add input normalization
        x = x.float() / 255.0
        
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1) # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.gru(x)  # Fixed: Pass x to GRU
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # Fixed: Changed from self.permute

        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
        self.eval()



