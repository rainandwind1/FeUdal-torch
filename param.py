import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 3
MAX_EPISODES = 100000
MAX_STEPS = 200
