import torch
from IBGNN_Models.models_common import IBGConv, MLP
from IBGNN_Models.models_mp import IBGNN


def build_model(args, device, num_features):
    model = IBGNN(IBGConv(num_features, args, num_classes=2),
                  MLP(128, 128, 1, torch.nn.ReLU, n_classes=2),
                  pooling='sum')
    return model
