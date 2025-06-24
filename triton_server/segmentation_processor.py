"""
Simplified segmentation processor based on the QGIS Deepness plugin architecture.
Implements tiled processing with overlap, batching, and post-processing for high-quality results.
"""

from collections.abc import Generator
from dataclasses import dataclass

import cv2
import numpy as np
import rasterio
import tritonclient.grpc as grpcclient
from rasterio.windows import Window
from tqdm.auto import trange


@dataclass
class ProcessingConfig:
    """Configuration for segmentation processing"""

    tile_size: int = 224
    stride: int = 112  # overlap = (tile_size - stride) / 2
    batch_size: int = 4
    probability_threshold: float = 0.5
    median_blur_size: int = 5
    morphological_kernel_size: int = 9
    apply_morphological_ops: bool = False


@dataclass
class TileInfo:
    """Information about a single tile"""

    row_start: int
    col_start: int
    row_end: int
    col_end: int
    window: Window
    is_edge_tile: bool = False


class SegmentationProcessor:
    """
    Processes large raster images using tiled segmentation with overlap handling.
    Based on the QGIS Deepness plugin architecture.
    """

    def __init__(self, server_url: str, model_name: str, config: ProcessingConfig):
        self.client = grpcclient.InferenceServerClient(url=server_url)
        self.model_name = model_name
        self.config = config
        self.overlap = (config.tile_size - config.stride) // 2

    def process_raster(
        self, input_path: str, output_path: str, num_classes: int
    ) -> None:
        """
        Process a raster file with segmentation.

        Args:
            input_path: Path to input raster
            output_path: Path to output segmentation raster
            num_classes: Number of segmentation classes
        """
        with rasterio.open(input_path) as src:
            # Get raster properties
            height, width = src.height, src.width
            profile = src.profile.copy()

            # Update profile for output
            profile.update({"dtype": "uint8", "count": 1, "compress": "lzw"})

            # Calculate extended dimensions to accommodate full tiles
            extended_height, extended_width = self._calculate_extended_dimensions(
                height, width
            )

            # Create output array
            result = np.zeros((extended_height, extended_width), dtype=np.uint8)

            # Generate tiles and process in batches
            tiles = list(self._generate_tiles(extended_height, extended_width))

            for batch_start in trange(0, len(tiles), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(tiles))
                batch_tiles = tiles[batch_start:batch_end]

                # Read tile data
                tile_data = []
                for tile in batch_tiles:
                    tile_img = self._read_tile(src, tile, height, width)
                    tile_data.append(tile_img)

                # Stack into batch array [b, c, h, w]
                batch_array = np.stack(tile_data, axis=0)

                # Process batch through model
                batch_results = self._process_batch_triton(batch_array)

                # Apply post-processing to each result
                for i, (tile, tile_result) in enumerate(
                    zip(batch_tiles, batch_results)
                ):
                    processed_result = self._postprocess_tile_result(tile_result)
                    self._place_tile_result(result, tile, processed_result)

            # Apply final post-processing
            result = self._apply_final_postprocessing(result)

            # Crop back to original dimensions
            result = result[:height, :width]

            # Write output
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(result, 1)

    def _process_batch_triton(self, batch: np.ndarray) -> np.ndarray:
        """
        Triton server inference for batches.

        Args:
            batch: Input batch with shape [b, c, h, w]

        Returns:
            Segmentation results with shape [b, num_classes, h, w]
        """
        # 1. Prepare Triton input tensors
        input_tensors = [grpcclient.InferInput("IMAGE", batch.shape, "INT64")]
        input_tensors[0].set_data_from_numpy(batch.astype(np.int64))

        # 2. Send gRPC request to Triton server
        results = self.client.infer(model_name=self.model_name, inputs=input_tensors)

        # 3. Parse response tensors
        return results.as_numpy("SEGMENTATION").astype(float)

    def _calculate_extended_dimensions(
        self, height: int, width: int
    ) -> tuple[int, int]:
        """Calculate extended dimensions to fit complete tiles"""
        # Calculate number of tiles needed
        tiles_y = ((height - self.config.tile_size) // self.config.stride) + 1
        tiles_x = ((width - self.config.tile_size) // self.config.stride) + 1

        # Calculate extended dimensions
        extended_height = (tiles_y - 1) * self.config.stride + self.config.tile_size
        extended_width = (tiles_x - 1) * self.config.stride + self.config.tile_size

        return max(extended_height, height), max(extended_width, width)

    def _generate_tiles(self, height: int, width: int) -> Generator[TileInfo]:
        """Generate tile information for processing"""
        tiles_y = ((height - self.config.tile_size) // self.config.stride) + 1
        tiles_x = ((width - self.config.tile_size) // self.config.stride) + 1

        for y in range(tiles_y):
            for x in range(tiles_x):
                row_start = y * self.config.stride
                col_start = x * self.config.stride
                row_end = min(row_start + self.config.tile_size, height)
                col_end = min(col_start + self.config.tile_size, width)

                # Create rasterio window
                window = Window(
                    col_start, row_start, col_end - col_start, row_end - row_start
                )

                # Check if it's an edge tile
                is_edge = x == 0 or x == tiles_x - 1 or y == 0 or y == tiles_y - 1

                yield TileInfo(
                    row_start=row_start,
                    col_start=col_start,
                    row_end=row_end,
                    col_end=col_end,
                    window=window,
                    is_edge_tile=is_edge,
                )

    def _read_tile(
        self,
        src: rasterio.DatasetReader,
        tile: TileInfo,
        full_height: int,
        full_width: int,
    ) -> np.ndarray:
        """Read a single tile from the raster"""
        # Read the tile data
        window_height = tile.row_end - tile.row_start
        window_width = tile.col_end - tile.col_start

        # Handle edge cases where tile extends beyond image
        if tile.row_start >= full_height or tile.col_start >= full_width:
            # Return zeros for tiles completely outside image
            return np.zeros(
                (src.count, self.config.tile_size, self.config.tile_size),
                dtype=np.float32,
            )

        # Read actual data
        data = src.read(window=tile.window)

        # Pad if necessary to reach tile_size
        if (
            window_height < self.config.tile_size
            or window_width < self.config.tile_size
        ):
            padded_data = np.zeros(
                (src.count, self.config.tile_size, self.config.tile_size),
                dtype=data.dtype,
            )
            padded_data[:, :window_height, :window_width] = data
            data = padded_data

        return data

    def _postprocess_tile_result(self, tile_result: np.ndarray) -> np.ndarray:
        """Post-process individual tile segmentation result"""
        # Apply probability threshold
        tile_result[tile_result < self.config.probability_threshold] = 0.0

        # Convert to class indices (argmax + 1 to avoid 0 values)
        if tile_result.shape[0] == 1:
            # Binary segmentation
            processed = (tile_result[0] != 0).astype(np.uint8)
        else:
            # Multi-class segmentation
            processed = np.argmax(tile_result, axis=0).astype(np.uint8)

        return processed

    def _place_tile_result(
        self, result: np.ndarray, tile: TileInfo, tile_result: np.ndarray
    ) -> None:
        """Place tile result into the full result array using overlap strategy"""
        # Calculate the region to copy (excluding overlap except for edge tiles)
        copy_row_start = tile.row_start
        copy_col_start = tile.col_start
        copy_row_end = tile.row_end
        copy_col_end = tile.col_end

        # For non-edge tiles, exclude overlap regions
        if not tile.is_edge_tile:
            if tile.row_start > 0:  # Not top edge
                copy_row_start += self.overlap
            if tile.col_start > 0:  # Not left edge
                copy_col_start += self.overlap
            if tile.row_end < result.shape[0]:  # Not bottom edge
                copy_row_end -= self.overlap
            if tile.col_end < result.shape[1]:  # Not right edge
                copy_col_end -= self.overlap

        # Calculate corresponding region in tile result
        tile_row_start = copy_row_start - tile.row_start
        tile_col_start = copy_col_start - tile.col_start
        tile_row_end = tile_row_start + (copy_row_end - copy_row_start)
        tile_col_end = tile_col_start + (copy_col_end - copy_col_start)

        # Copy the data
        result[copy_row_start:copy_row_end, copy_col_start:copy_col_end] = tile_result[
            tile_row_start:tile_row_end, tile_col_start:tile_col_end
        ]

    def _apply_final_postprocessing(self, result: np.ndarray) -> np.ndarray:
        """Apply final post-processing filters"""
        # Apply median blur (key for quality improvement)
        if self.config.median_blur_size > 1:
            blur_size = self.config.median_blur_size
            if blur_size % 2 == 0:
                blur_size += 1  # Ensure odd size
            result = cv2.medianBlur(result, blur_size)

        # Apply morphological operations if enabled
        if (
            self.config.apply_morphological_ops
            and self.config.morphological_kernel_size > 0
        ):
            kernel_size = self.config.morphological_kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Opening (remove noise)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            # Closing (fill gaps)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        return result


def create_default_config() -> ProcessingConfig:
    """Create a default processing configuration"""
    return ProcessingConfig(
        tile_size=224,
        stride=224,  # 50% overlap
        batch_size=4,
        probability_threshold=0.5,
        median_blur_size=5,
        morphological_kernel_size=3,
        apply_morphological_ops=True,
    )


if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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

    # Define file paths
    input_raster = "/Users/taylor.denouden/Downloads/20210831_181119_77_2206_3B_AnalyticMS_SR_8b_harmonized_northernCalifornia.tif"
    output_raster = "/Users/taylor.denouden/Documents/PycharmProjects/clay/triton_server/20210831_181119_77_2206_3B_AnalyticMS_SR_8b_harmonized_northernCalifornia_kelp_new.tif"
    num_classes = 2  # Adjust based on your model

    try:
        # Process the raster
        logging.info("Starting segmentation processing...")
        logging.info(f"Input: {input_raster}")
        logging.info(f"Output: {output_raster}")
        logging.info(f"Configuration: {config}")

        processor.process_raster(
            input_path=input_raster, output_path=output_raster, num_classes=num_classes
        )

        logging.info("Segmentation processing completed successfully!")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise
