"""REST API for PLONK image geolocation inference."""

from pathlib import Path
import tempfile
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from plonk.inference import MODEL_PATHS, inference


app = FastAPI(
    title="PLONK Geolocation API",
    description="Predict locations and compare them with the image's EXIF GPS position.",
    version="1.0.0",
)
_inference_lock = Lock()


class Coordinate(BaseModel):
    latitude: float
    longitude: float


class Prediction(Coordinate):
    error_km: float


class InferenceResponse(BaseModel):
    model_type: str
    ground_truth: Coordinate
    predictions: list[Prediction]
    mean_error_km: float


def run_inference(
    image_path: Path,
    model_type: str,
    guidance_scale: float,
    num_samples: int,
) -> dict:
    # Avoid simultaneous access to a cached model/GPU from several requests.
    with _inference_lock:
        return inference(
            image_path=image_path,
            model_type=model_type,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
            verbose=False,
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/inference", response_model=InferenceResponse)
async def predict(
    image: UploadFile = File(..., description="Image containing GPS EXIF metadata"),
    model_type: str = Form("osv5m"),
    guidance_scale: float = Form(0.0),
    num_samples: int = Form(1000, gt=0),
) -> InferenceResponse:
    if model_type not in MODEL_PATHS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model; choose one of: {', '.join(MODEL_PATHS)}",
        )

    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := await image.read(1024 * 1024):
                temporary_file.write(chunk)

        try:
            result = await run_in_threadpool(
                run_inference,
                temporary_path,
                model_type,
                guidance_scale,
                num_samples,
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await image.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    truth_latitude, truth_longitude = result["ground_truth"]
    predictions = [
        Prediction(
            latitude=float(latitude),
            longitude=float(longitude),
            error_km=float(error_km),
        )
        for (latitude, longitude), error_km in zip(
            result["predictions"], result["errors_km"]
        )
    ]
    return InferenceResponse(
        model_type=model_type,
        ground_truth=Coordinate(
            latitude=truth_latitude,
            longitude=truth_longitude,
        ),
        predictions=predictions,
        mean_error_km=result["mean_error_km"],
    )
