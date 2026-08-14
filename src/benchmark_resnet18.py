import time
import torch
from torchvision import models


# --------------------------------------------------
# 1. LOAD MODEL
# --------------------------------------------------

model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    4
)

model.load_state_dict(
    torch.load(
        "models/mobilenetv2_leaf_best.pth",
        map_location="cpu"
    )
)

model.eval()


# --------------------------------------------------
# 2. DUMMY IMAGE
# --------------------------------------------------

dummy = torch.randn(
    1, 3, 224, 224
)


# --------------------------------------------------
# 3. WARM-UP
# --------------------------------------------------

with torch.no_grad():

    for _ in range(10):
        model(dummy)


# --------------------------------------------------
# 4. BENCHMARK
# --------------------------------------------------

with torch.no_grad():

    start = time.perf_counter()

    for _ in range(100):
        model(dummy)

    elapsed = (
        time.perf_counter() - start
    ) / 100


# --------------------------------------------------
# 5. RESULT
# --------------------------------------------------

print(
    f"CPU inference: "
    f"{elapsed * 1000:.2f} ms/image"
)