#!/usr/bin/env python

# In[ ]:


import sys
import warnings

sys.path.append("../../")
warnings.filterwarnings("ignore")


# In[ ]:


import json

import onnx
import torch

from finetune.segment.pskelp_model import PSKelpSegmentor

# ### Define paths and parameters

# In[ ]:


CHECKPOINT_PATH = (
    "../../checkpoints/model-valiant-jazz-61.ckpt"
)
CLAY_CHECKPOINT_PATH = "../../checkpoints/clay-v1.5.ckpt"
METADATA_PATH = "../../configs/metadata.yaml"
OUTPUT_PATH_ONNX = "../../checkpoints/model-valiant-jazz-61.onnx"

TILE_SIZE = 224
DEVICE = torch.device("cpu")

NUM_BANDS = 8
BATCH_SIZE = 32
NUM_WORKERS = 1
PLATFORM = "planetscope-sr"


# ### Model Loading

# In[ ]:


def get_model(chesapeake_checkpoint_path, clay_checkpoint_path, metadata_path):
    model = PSKelpSegmentor.load_from_checkpoint(
        checkpoint_path=chesapeake_checkpoint_path,
        metadata_path=metadata_path,
        ckpt_path=clay_checkpoint_path,
    )
    model.eval()
    return model


# In[ ]:


# Load model
model = get_model(CHECKPOINT_PATH, CLAY_CHECKPOINT_PATH, METADATA_PATH).to(DEVICE)


# # Export model to ONNX
#

# In[ ]:


# Define dynamic axes for input and output
dynamic_axes = {
    "input": {
        0: "batch_size",
        # 2: "tile_size",
        # 3: "tile_size",
    },  # Dynamic batch size, height and width
    "output": {
        0: "batch_size",
        # 2: "tile_size",
        # 3: "tile_size",
    },  # Dynamic batch size, height and width
}
input_names = ["input"]
output_names = ["output"]
waves = torch.tensor(
    [0.443, 0.490, 0.531, 0.565, 0.610, 0.665, 0.705, 0.865]
)  # Planet SR wavelengths
gsd = torch.tensor(5.0)  # Planet SR GSD


class DeepnessModel(torch.nn.Module):
    def __init__(self, model):
        super(DeepnessModel, self).__init__()
        self.model = model

    def forward(self, x):
        b = x.shape[0]

        datacube = {
            "pixels": x,
            "time": torch.zeros((b, 4)),  # Placeholder for time information
            "latlon": torch.zeros((b, 4)),  # Placeholder for latlon information
            "waves": waves,
            "gsd": gsd,
        }
        logits = self.model(datacube)
        probs = torch.sigmoid(logits)

        # Convert class 1 probabilities to 2 class probs shape: # [batch_size, 2, height, width]
        probs = torch.cat(
            [
                1 - probs,  # Background class
                probs,  # Kelp class
            ],
            dim=1,
        )

        return probs


model = DeepnessModel(model.model).to(DEVICE)

x = torch.rand(
    1,
    NUM_BANDS,
    TILE_SIZE,
    TILE_SIZE,
    device=DEVICE,
    requires_grad=False,
)

torch.onnx.export(
    model,  # Model to export
    x,  # Example input
    OUTPUT_PATH_ONNX,  # Output file path
    export_params=True,  # Store model weights in the model file
    opset_version=14,  # ONNX opset version
    do_constant_folding=True,  # Optimize constants
    input_names=input_names,  # Input tensor names
    output_names=output_names,  # Output tensor names
    dynamic_axes=dynamic_axes,  # Dynamic axes specification
    verbose=False,
)

onnx_model = onnx.load(OUTPUT_PATH_ONNX)

class_names = {
    0: "bg",
    1: "kelp",
}

m1 = onnx_model.metadata_props.add()
m1.key = "model_type"
m1.value = json.dumps("Segmentor")

m2 = onnx_model.metadata_props.add()
m2.key = "class_names"
m2.value = json.dumps(class_names)

m3 = onnx_model.metadata_props.add()
m3.key = "resolution"
m3.value = json.dumps(300)  # cm/px

m4 = onnx_model.metadata_props.add()
m4.key = "tiles_overlap"
m4.value = json.dumps(40)  # 40% overlap

m5 = onnx_model.metadata_props.add()
m5.key = "standardization_mean"
m5.value = json.dumps(
    [
        v / 255.0
        for v in [1720.0, 1715.0, 1913.0, 2088.0, 2274.0, 2290.0, 2613.0, 3970.0]
    ]
)

m6 = onnx_model.metadata_props.add()
m6.key = "standardization_std"
m6.value = json.dumps(
    [v / 255.0 for v in [747.0, 698.0, 739.0, 768.0, 849.0, 868.0, 849.0, 914.0]]
)

onnx.save(onnx_model, OUTPUT_PATH_ONNX)
onnx.checker.check_model(onnx_model)
