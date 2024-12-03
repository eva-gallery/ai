"""The BentoML inference service for the AI API.

The BentoML service for the inference API which handles the image and text embeddings,
captioning, AI generation detection, watermark detection, and watermark addition.
"""

from __future__ import annotations

from typing import Any, List, cast

import bentoml
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image as PILImageObject
from PIL.Image import Image as PILImage
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import AutoModelForImageClassification, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor, pipeline

from ai_api import settings
from ai_api.model.api.process import AddWatermarkRequest, AIGeneratedStatus, ListAddWatermarkRequest
from ai_api.model.api.protocols import InferenceServiceProto
from ai_api.model.api.embed import EmbedRequest
from ai_api.util.logger import get_logger


@bentoml.service(
    name="evagellery_ai_inference",
    **settings.bentoml.service.embedding.to_dict(),
)
class InferenceService(InferenceServiceProto):
    """Inference service for the AI API."""

    def __init__(self) -> None:
        """Initialize the inference service."""
        self.logger = get_logger()
        self.device: torch.device = torch.device(f"cuda:{getattr(bentoml.server_context, 'worker_index', 1) - 1}" if is_available() else "cpu")
        self.device_map_string = "balanced"  # can't use 'cuda:0', bug in safetensors
        self.model_img = SentenceTransformer(settings.model.embedding.image.name, device=str(self.device), cache_folder=settings.model.cache_dir)
        self.model_text = SentenceTransformer(settings.model.embedding.text.name, device=str(self.device), cache_folder=settings.model.cache_dir)

        # Initialize BLIP captioning model and processor
        self.caption_processor: BlipProcessor = cast(
            BlipProcessor,
            BlipProcessor.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir, device=self.device),
        )
        self.caption_model: BlipForConditionalGeneration = cast(
            BlipForConditionalGeneration,
            BlipForConditionalGeneration.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir),
        )

        self.model_ai_watermark = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            settings.model.watermark.diffusion_model,
            vae=AutoencoderKL.from_pretrained(settings.model.watermark.encoder_name, cache_dir=settings.model.cache_dir, device_map=self.device_map_string),
            device_map=self.device_map_string,
        )
        self.model_ai_watermark_decoder = AutoModelForImageClassification.from_pretrained(
            settings.model.watermark.decoder_name,
            device_map=self.device_map_string,
            cache_dir=settings.model.cache_dir,
        )
        self.model_ai_watermark_processor = cast(
            BlipImageProcessor,
            BlipImageProcessor.from_pretrained(
                settings.model.watermark.decoder_name,
                cache_dir=settings.model.cache_dir,
                device_map=self.device_map_string,
            ),
        )

        self.model_img.eval()
        self.model_text.eval()
        self.caption_model.eval()

        self.model_ai_detection = pipeline(
            "image-classification",
            settings.model.detection.name,
            device=self.device,
        )

    @bentoml.api(route="/readyz")
    async def readyz(self) -> dict[str, str]:
        """Check if the service is ready."""
        return {"status": "ready"}

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        results = [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]
        self.logger.debug("Text embedding results: {results}", results=results)
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def embed_image(self, images: List[PILImage]) -> list[list[float]]:
        """Embed a list of images."""
        results = [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]  # type: ignore[arg-type]
        self.logger.debug("Image embedding results: {results}", results=results)
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def generate_caption(self, images: List[PILImage]) -> list[str]:
        """Generate captions for a list of images."""
        inputs = self.caption_processor(images, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_length=settings.model.captioning.max_length)  # type: ignore[arg-type]
        captions = self.caption_processor.decode(out[0], skip_special_tokens=True)
        self.logger.debug("Caption results: {captions}", captions=captions)
        return captions

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def detect_ai_generation(self, images: List[PILImage]) -> list[AIGeneratedStatus]:
        """Detect if a list of images are AI-generated."""
        raw_results: list[list[dict[str, Any]]] = cast(list[list[dict[str, Any]]], self.model_ai_detection(images))
        fake_scores = [
            next(pred["score"] for pred in result if pred["label"] == "fake")
            for result in raw_results
        ]

        if not fake_scores:
            self.logger.error("No AI generation detection results due to some failure")
            return [AIGeneratedStatus.NOT_GENERATED] * len(images)

        results = [AIGeneratedStatus.GENERATED if result > settings.model.detection.threshold else AIGeneratedStatus.NOT_GENERATED for result in fake_scores]
        self.logger.debug("AI generation detection results: {results}", results=results)
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def check_watermark(self, images: List[PILImage]) -> list[tuple[bool, str | None]]:
        """Check if a list of images contain an AI watermark."""
        gan_mark = WatermarkDecoder("bits", 32)
        results = []
        for image in images:
            try:
                watermark = gan_mark.decode(np.asarray(image), settings.model.watermark.method)
                if watermark:
                    watermark_bytes = int(watermark, 2).to_bytes((len(watermark) + 7) // 8, byteorder="big")
                    try:
                        watermark_str = watermark_bytes.decode("utf-8").rstrip("\x00")
                        results.append((True, watermark_str))
                    except UnicodeDecodeError:
                        results.append((True, None))
                else:
                    results.append((False, None))
            except Exception as e:  # noqa: PERF203
                self.logger.exception("Error checking watermark: {e}", e=e)
                results.append((False, None))
        self.logger.debug("Watermark results: {results}", results=results)
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def check_ai_watermark(self, images: List[PILImage]) -> list[bool]:
        """Check if a list of images contain an AI watermark."""
        ai_watermark_t = self.model_ai_watermark_processor(images, return_tensors="pt").to(self.model_ai_watermark_decoder.device)
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[:, 0] < 0  # type: ignore[arg-type]
        pred_bool = ai_watermark_pred > (1 - settings.model.detection.threshold)
        self.logger.debug("AI watermark prediction: {pred_bool}", pred_bool=pred_bool)
        return pred_bool.tolist()

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms,
        input_spec=ListAddWatermarkRequest,
        output_spec=EmbedRequest,
    )
    async def add_watermark(self, request: List[AddWatermarkRequest]) -> list[PILImage]:
        """Add a watermark to a list of images."""
        wm = WatermarkEncoder()
        binaries = "".join(format(ord(c.watermark_text), "08b") for c in request)
        binaries = [binary[:32] if len(binary) > 32 else binary.ljust(32, "0") for binary in binaries]  # noqa: PLR2004

        results = []
        for image, binary in zip(request, binaries):
            wm.set_watermark("bits", binary)
            results.append(PILImageObject.fromarray(wm.encode(np.asarray(image.image), settings.model.watermark.method)))

        self.logger.debug("Watermark results: {results}", results=results)
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
        output_spec=EmbedRequest,
    )
    async def add_ai_watermark(self, images: List[PILImage]) -> list[PILImage]:
        """Add an AI watermark to a list of images."""
        results = list(self.model_ai_watermark(image=images)[0])  # type: ignore[arg-type]
        self.logger.debug("AI watermark results: {results}", results=results)
        return results
