"""Small command-line client for testing the PLONK REST API."""

import argparse
import json
import mimetypes
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the PLONK inference API.")
    parser.add_argument("image", type=Path, help="path to the image to upload")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/inference",
        help="inference endpoint URL",
    )
    parser.add_argument(
        "--model-type",
        choices=("yfcc", "osv5m", "inaturalist"),
        default="osv5m",
    )
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be greater than zero")

    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    with args.image.open("rb") as image:
        response = requests.post(
            args.url,
            files={"image": (args.image.name, image, content_type)},
            data={
                "model_type": args.model_type,
                "guidance_scale": args.guidance_scale,
                "num_samples": args.num_samples,
            },
            timeout=None,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise SystemExit(
            f"API returned HTTP {response.status_code}: {response.text}"
        ) from error

    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
