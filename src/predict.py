import torch
import numpy as np
from src.model import DecisionTransformer
from src.utils import load_normalization_stats

_model = None
_stats = None

def load_model():
    global _model, _stats
    if _model is None:
        state_dim = 18   # matches the trained model
        embed_dim = 256  # matches the trained model
        context_len = 10

        _model = DecisionTransformer(
            state_dim=state_dim,
            embed_dim=embed_dim,
            context_length=context_len,
            num_classes=3
        )
        _model.load_state_dict(torch.load("models/model_epoch_5.pt", map_location="cpu"))
        _model.eval()

        mean, std = load_normalization_stats("models/normalization_stats.npz")
        _stats = {"mean": mean, "std": std}

    return _model, _stats

def predict_action(date_str):
    model, stats = load_model()

    #Dummy input: replace with actual date-based state later
    dummy_state = np.random.randn(30, 18).astype(np.float32)  # (K, state_dim)
    dummy_rtg = np.full((30, 1), 0.0, dtype=np.float32)        # (K, 1)
    dummy_action = np.zeros((30,), dtype=int)                 # (K,)
    dummy_timestep = np.arange(30)                            # (K,)

    # Normalize states
    dummy_state = (dummy_state - stats["mean"]) / stats["std"]

    # Convert to torch tensors
    rtg = torch.tensor(dummy_rtg).unsqueeze(0)        # (1, K, 1)
    states = torch.tensor(dummy_state).unsqueeze(0)   # (1, K, state_dim)
    actions = torch.tensor(dummy_action).unsqueeze(0) # (1, K)
    timesteps = torch.tensor(dummy_timestep).unsqueeze(0)  # (1, K)

    with torch.no_grad():
        logits = model(rtg, states, actions, timesteps)  # (1, K, 3)
        pred_action = torch.argmax(logits[0, -1]).item()

    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return f"Predicted action for {date_str}: {label_map[pred_action]}"
