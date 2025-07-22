import torch
import torch.onnx
from dreamer_model import Actor


obs_dim = 3
latent_dim = 16
action_dim = 1
onnx_filename = "actor.onnx"


actor = Actor(latent_dim, action_dim)
actor.load_state_dict(torch.load("actor.pt", map_location="cpu"))
actor.eval()


dummy_input = torch.randn(1, latent_dim)


torch.onnx.export(
    actor,
    dummy_input,
    onnx_filename,
    input_names=["latent"],
    output_names=["action"],
    dynamic_axes={
        "latent": {0: "batch_size"},
        "action": {0: "batch_size"}
    },
    opset_version=11
)

print(f"✅ Exported actor to {onnx_filename}")
