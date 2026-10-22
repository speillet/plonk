"""Command-line interface for running PLONK geolocation inference."""

import argparse
from pathlib import Path

from plonk.inference import MODEL_PATHS, inference


def positive_int(value: str) -> int:
    """Argparse type accepting strictly positive integers."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict possible geographic locations for an image with PLONK."
    )
    parser.add_argument(
        "--image-path",
        "--image_path",
        required=True,
        type=Path,
        help="path to the image to geolocate",
    )
    parser.add_argument(
        "--model-type",
        "--model_type",
        "--model",
        dest="model",
        required=True,
        choices=MODEL_PATHS,
        help="pre-trained PLONK model to use",
    )
    parser.add_argument(
        "--guidance-scale",
        "--guidance_scale",
        required=False,
        default=0,
        type=float,
        help="classifier-free guidance scale (optional)",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        required=False,
        default=1000,
        type=positive_int,
        help="number of predicted locations to generate (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the ground truth, predictions and errors",
    )
    return parser.parse_args()


def display_predictions(gps_coords, ground_truth=None) -> None:
    """Display predicted latitude/longitude pairs on a world map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Map display requires matplotlib and cartopy. "
            "Install them with: pip install matplotlib cartopy"
        ) from error

    _, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_global()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.scatter(
        gps_coords[:, 1],
        gps_coords[:, 0],
        color="red",
        marker="o",
        s=20,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        label="Predicted locations",
    )
    if ground_truth is not None:
        ax.scatter(
            ground_truth[1],
            ground_truth[0],
            color="lime",
            edgecolor="black",
            marker="*",
            s=180,
            transform=ccrs.PlateCarree(),
            label="Ground truth (EXIF)",
            zorder=5,
        )

    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax.set_title("PLONK predicted locations")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    try:
        result = inference(
            image_path=args.image_path,
            model_type=args.model,
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    display_predictions(result["predictions"], result["ground_truth"])


if __name__ == "__main__":
    main()


# Exemples d'utilisation :
#
# Modèle OSV5M (adapté aux images Street View) :
# python cli_pipeline.py --image-path "/home/SPeillet/projects/sevilla/data/first_tests/street_view/argentina_835-Calle-57-La-Plata_11-2024/3f48-40c8-a835-9b015c7df755.jpg" --model-type osv5m --guidance-scale 0 --num-samples 1000
#
# Même image avec le modèle YFCC :
# python cli_pipeline.py --image-path "/home/SPeillet/projects/sevilla/data/first_tests/street_view/argentina_835-Calle-57-La-Plata_11-2024/3f48-40c8-a835-9b015c7df755.jpg" --model-type yfcc --guidance-scale 0 --num-samples 1000
#
# Inférence plus rapide avec moins d'échantillons :
# python cli_pipeline.py --image-path "/home/SPeillet/projects/sevilla/data/first_tests/street_view/argentina_835-Calle-57-La-Plata_11-2024/3f48-40c8-a835-9b015c7df755.jpg" --model-type osv5m --guidance-scale 1.0 --num-samples 1000
