#!/usr/bin/env python


import logging
from pathlib import Path

import rasterio
from segmentation_processor import ProcessingConfig, SegmentationProcessor
from tqdm.auto import tqdm

ROOT = Path("/mnt/geospatial/Working/Planet_ML/PSScene")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Label interpretation
labels_s2i = {
    "background": 0,
    "kelp": 1,
    "land": 2,
    "nodata": 3,
    "noise": 4,
}
labels_i2s = dict((v, k) for k, v in labels_s2i.items())

# Create configuration
config = ProcessingConfig(
    tile_size=224,
    stride=56,  # 75% overlap
    batch_size=5,
    probability_threshold=0.5,
    median_blur_size=5,
    morphological_kernel_size=0,  # Disable morphological ops
    apply_morphological_ops=False,
)

# Create processor
triton_url = "10.8.1.32:8001"
triton_model = "kelp_segmentation_ps8b_ensemble"
processor = SegmentationProcessor(triton_url, triton_model, config)


images = (ROOT / "adbc3808-36d6-4e1b-a67a-097dfcc7d201").glob(
    "**/*_AnalyticMS_SR_8b_harmonized_clip.tif"
)


logging.info("Starting segmentation processing...")
logging.info(f"Configuration: {config}")

for p in tqdm(list(images), desc="Processing images", unit="image"):
    out_path = ROOT.parent / "PSScene_kelp_20250623" / p.relative_to(ROOT)
    out_path = out_path.with_stem(out_path.stem + "_kelp")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Process the raster
        logging.info(f"Input: {str(p)}")
        logging.info(f"Output: {str(out_path)}")

        processor.process_raster(
            input_path=str(p), output_path=str(out_path), num_classes=2
        )

        # Transfer nodata and noise information
        qa_raster = str(p).replace("_AnalyticMS_SR_8b_harmonized_clip.tif", "_udm2_clip.tif")
        with rasterio.open(qa_raster) as qa, rasterio.open(out_path) as seg:
            udm = qa.read(8)
            out = seg.read(1)
            out[udm == 1] = labels_s2i['nodata']
            out[udm > 1] = labels_s2i['noise']
            profile = seg.profile

        # Write back changes
        profile.update(nodata=3)
        with rasterio.open(out_path, "w", **profile) as seg:
            seg.write(out, 1)
            seg.descriptions = ["Class"]
            colormap = {
                0: (0, 0, 255, 255),
                1: (0,255,0,255),
                2: (0,128,128,255),
                3: (0,0,0,0),
                4: (255,128,128,255),
            }
            seg.write_colormap(1, colormap)
            seg.update_tags(1, STATISTICS_MINIMUM=1, STATISTICS_MAXIMUM=4)

    except Exception as e:
        logging.error(f"Error during processing: {e}")

logging.info("Segmentation processing completed successfully!")
