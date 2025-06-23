#!/usr/bin/env python

# In[ ]:


import sys
import warnings

sys.path.append("../../")
warnings.filterwarnings("ignore")


# In[ ]:


import torch

from finetune.segment.pskelp_model import PSKelpSegmentor

# ### Define paths and parameters

# In[ ]:


CHECKPOINT_PATH = (
    "../../checkpoints/segment/kelp-1class-segment_epoch-76_val-iou-0.7532.ckpt"
)
CLAY_CHECKPOINT_PATH = "../../checkpoints/clay-v1.5.ckpt"
METADATA_PATH = "../../configs/metadata.yaml"
OUTPUT_MODEL_PATH = (
    "../../triton_server/models/kelp_segmentation_ps8b_model/2/model.onnx"
)
OUTPUT_PREMODEL_PATH = (
    "../../triton_server/models/kelp_segmentation_ps8b_preprocessing/2/model.onnx"
)

TRAIN_CHIP_DIR = "../../data/cvpr/ny/train/chips/"
TRAIN_LABEL_DIR = "../../data/cvpr/ny/train/labels/"
VAL_CHIP_DIR = "../../data/cvpr/ny/val/chips/"
VAL_LABEL_DIR = "../../data/cvpr/ny/val/labels/"
TILE_SIZE = 224
DEVICE = torch.device("cpu")

NUM_BANDS = 8
BATCH_SIZE = 32
NUM_WORKERS = 1
PLATFORM = "planetscope-sr"


# ### Model Loading

# In[ ]:


def get_model(checkpoint_path, clay_checkpoint_path, metadata_path):
    model = PSKelpSegmentor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        ckpt_path=clay_checkpoint_path,
    )
    model.eval()
    return model


# # Export model to ONNX
#

# In[ ]:


# Export the preprocessing model to ONNX format
class PreprocessingModel(torch.nn.Module):
    def __init__(self, mean, std, max_pixel_value=255.0):
        super(PreprocessingModel, self).__init__()
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)
        self.max_pixel_value = max_pixel_value

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x = x / self.max_pixel_value

        # Preprocess the input tensor
        x = (x - self.mean) / self.std  # Standardize

        return x


model = PreprocessingModel(
    mean=torch.tensor([1720.0, 1715.0, 1913.0, 2088.0, 2274.0, 2290.0, 2613.0, 3970.0]),
    std=torch.tensor([747.0, 698.0, 739.0, 768.0, 849.0, 868.0, 849.0, 914.0]),
    max_pixel_value=1.0,
).to(DEVICE)

x = torch.randint(
    low=0,
    high=65535,
    size=(
        1,
        NUM_BANDS,
        TILE_SIZE,
        TILE_SIZE,
    ),
    device=DEVICE,
    requires_grad=False,
)

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

# Export the segmentation model to ONNX format
torch.onnx.export(
    model,  # Model to export
    x,  # Example input
    OUTPUT_PREMODEL_PATH,  # Output file path
    export_params=True,  # Store model weights in the model file
    opset_version=14,  # ONNX opset version
    do_constant_folding=True,  # Optimize constants
    input_names=input_names,  # Input tensor names
    output_names=output_names,  # Output tensor names
    dynamic_axes=dynamic_axes,  # Dynamic axes specification
    verbose=False,
)

# class_names = {
#     0: "water",
#     1: "kelp",
#     2: "land",
# }
# onnx_model = onnx.load(OUTPUT_PATH_ONNX)
#
# onnx.checker.check_model(onnx_model)


# In[ ]:


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


model = get_model(CHECKPOINT_PATH, CLAY_CHECKPOINT_PATH, METADATA_PATH).to(DEVICE)
model = DeepnessModel(model.model).to(DEVICE)

x = torch.rand(
    1,
    NUM_BANDS,
    TILE_SIZE,
    TILE_SIZE,
    device=DEVICE,
    requires_grad=False,
)

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

# Export the segmentation model to ONNX format
torch.onnx.export(
    model,  # Model to export
    x,  # Example input
    OUTPUT_MODEL_PATH,  # Output file path
    export_params=True,  # Store model weights in the model file
    opset_version=14,  # ONNX opset version
    do_constant_folding=True,  # Optimize constants
    input_names=input_names,  # Input tensor names
    output_names=output_names,  # Output tensor names
    dynamic_axes=dynamic_axes,  # Dynamic axes specification
    verbose=False,
)
