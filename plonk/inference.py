"""Reusable PLONK inference and geolocation evaluation utilities."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image
import torch

from plonk.pipe import PlonkPipeline


MODEL_PATHS = {
    "yfcc": "nicolas-dufour/PLONK_YFCC",
    "osv5m": "nicolas-dufour/PLONK_OSV_5M",
    "inaturalist": "nicolas-dufour/PLONK_iNaturalist",
}


def extract_gps_from_exif(image_path: Path | str) -> tuple[float, float]:
    """Extract latitude and longitude in degrees from image EXIF metadata."""
    with Image.open(image_path) as image:
        exif = image.getexif()
        if not exif:
            raise ValueError(f"No EXIF metadata found in {image_path}")
        try:
            gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError):
            gps_info = exif.get(34853)

    if not gps_info:
        raise ValueError(f"No GPS coordinates found in the EXIF metadata of {image_path}")

    def decimal_degrees(coordinates, reference) -> float:
        degrees, minutes, seconds = (float(value) for value in coordinates)
        result = degrees + minutes / 60.0 + seconds / 3600.0
        if isinstance(reference, bytes):
            reference = reference.decode("ascii")
        return -result if reference.upper() in {"S", "W"} else result

    try:
        return (
            decimal_degrees(gps_info[2], gps_info[1]),
            decimal_degrees(gps_info[4], gps_info[3]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Incomplete or invalid GPS EXIF metadata in {image_path}"
        ) from error


def haversine_errors(gps_coords, ground_truth: tuple[float, float]):
    """Return distances in kilometres between predictions and ground truth."""
    predictions = np.radians(np.asarray(gps_coords, dtype=float))
    truth = np.radians(np.asarray(ground_truth, dtype=float))
    latitude_delta = predictions[:, 0] - truth[0]
    longitude_delta = predictions[:, 1] - truth[1]
    a = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(predictions[:, 0])
        * np.cos(truth[0])
        * np.sin(longitude_delta / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


@lru_cache(maxsize=len(MODEL_PATHS))
def load_pipeline(model_type: str) -> PlonkPipeline:
    """Load and cache one pipeline per supported model type."""
    if model_type not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model {model_type!r}; choose one of: {', '.join(MODEL_PATHS)}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PlonkPipeline(MODEL_PATHS[model_type], device=device)


def print_result(result: dict) -> None:
    """Print an inference result in a human-readable format."""
    latitude, longitude = result["ground_truth"]
    print(f"Ground truth from EXIF: {latitude:.6f}, {longitude:.6f}")
    print("Predicted locations (latitude, longitude, error in km):")
    for (latitude, longitude), error_km in zip(
        result["predictions"], result["errors_km"]
    ):
        print(f"{latitude:.6f}, {longitude:.6f}, {error_km:.2f} km")
    print(f"Mean error: {result['mean_error_km']:.2f} km")


def inference(
    image_path: Path | str,
    model_type: str,
    guidance_scale: float = 0.0,
    num_samples: int = 1000,
    pipeline: PlonkPipeline | None = None,
    verbose: bool = False,
) -> dict:
    """Run PLONK and return predictions, EXIF truth and errors in kilometres."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model_type not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model {model_type!r}; choose one of: {', '.join(MODEL_PATHS)}"
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")

    ground_truth = extract_gps_from_exif(image_path)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    if pipeline is None:
        pipeline = load_pipeline(model_type)
    with torch.inference_mode():
        gps_coords = pipeline(
            image,
            batch_size=num_samples,
            cfg=guidance_scale,
        )

    errors_km = haversine_errors(gps_coords, ground_truth)
    result = {
        "predictions": gps_coords,
        "ground_truth": ground_truth,
        "errors_km": errors_km,
        "mean_error_km": float(errors_km.mean()),
    }
    if verbose:
        print_result(result)
    return result
