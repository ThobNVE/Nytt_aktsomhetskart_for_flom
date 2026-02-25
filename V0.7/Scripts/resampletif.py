import fiona
import rasterio as rio
from rasterio.enums import Resampling

def resample_tif_by_resolution(input_tif, output_tif, target_resolution, resampling_method=Resampling.average):
    """
    Resamples a TIFF file to a specified target resolution.

    Args:
        input_tif (str): Path to the input TIFF file.
        output_tif (str): Path to the output (resampled) TIFF file.
        target_resolution (float): The desired resolution in meters per pixel.
        resampling_method (rasterio.enums.Resampling): The resampling algorithm to use.
            Defaults to Resampling.average (good for downscaling).
    """
    with rio.open(input_tif) as dataset:
        original_resolution = dataset.res[0] # Assuming square pixels, get X resolution

        # Calculate the scale factor
        # If original_resolution is 1m and target_resolution is 5m, factor = 1/5 = 0.2
        scale_factor = original_resolution / target_resolution

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
            ),
            resampling=resampling_method
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        # Set up the metadata for the output file
        profile = dataset.profile
        profile.update({
            'height': data.shape[-2],
            'width': data.shape[-1],
            'transform': transform,
            'dtype': data.dtype, # Ensure the data type is correct
            'compress': 'LZW' # Optional: add compression
        })

        # Write the resampled data to a new TIFF file
        with rio.open(output_tif, 'w', **profile) as dst:
            dst.write(data)

    print(f"Resampled TIFF saved to: {output_tif} with target resolution {target_resolution}m")
