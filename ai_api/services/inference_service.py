from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import bentoml
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image as PILImage
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import AutoModelForImageClassification, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor, pipeline

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.util.logger import get_logger


class AddWatermarkRequest(BaseModel):
    """Request model for adding a watermark to an image."""

    image: PILImage.Image
    watermark_text: str


class InferenceServiceProto(Protocol):
    """Protocol defining the interface for the inference service."""

    async def readyz(self) -> dict[str, str]: ...
    async def embed_text(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]: ...
    async def generate_caption(self, images: list[PILImage.Image]) -> list[str]: ...
    async def detect_ai_generation(self, images: list[PILImage.Image]) -> list[AIGeneratedStatus]: ...
    async def check_watermark(self, images: list[PILImage.Image]) -> list[tuple[bool, str | None]]: ...
    async def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]: ...
    async def add_watermark(self, request: list[AddWatermarkRequest]) -> list[PILImage.Image]: ...
    async def add_ai_watermark(self, images: list[PILImage.Image]) -> list[PILImage.Image]: ...


@bentoml.service(
    name="evagellery_ai_inference",
    **settings.bentoml.service.embedding.to_dict(),
)
class InferenceService(InferenceServiceProto):
    """Inference service for the AI API."""

    def __init__(self) -> None:
        self.logger = get_logger()
        self.device: torch.device = torch.device(f"cuda:{getattr(bentoml.server_context, 'worker_index', 1) - 1}" if is_available() else "cpu")
        self.device_map_string = "balanced"  # can't use 'cuda:0', bug in safetensors
        self.model_img = SentenceTransformer(settings.model.embedding.image.name, device=self.device, cache_folder=settings.model.cache_dir)
        self.model_text = SentenceTransformer(settings.model.embedding.text.name, device=self.device, cache_folder=settings.model.cache_dir)

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
        return {"status": "ready"}

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        results = [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]
        self.logger.debug(f"Text embedding results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]:
        results = [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]
        self.logger.debug(f"Image embedding results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def generate_caption(self, images: list[PILImage.Image]) -> list[str]:
        inputs = self.caption_processor(images, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_length=settings.model.captioning.max_length)
        captions = self.caption_processor.decode(out[0], skip_special_tokens=True)
        self.logger.debug(f"Caption results: {captions}")
        return captions

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def detect_ai_generation(self, images: list[PILImage.Image]) -> list[AIGeneratedStatus]:
        results = self.model_ai_detection(images)[0]
        results = [AIGeneratedStatus.GENERATED if result["score"] > settings.model.detection.threshold else AIGeneratedStatus.NOT_GENERATED for result in results]
        self.logger.debug(f"AI generation detection results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def check_watermark(self, images: list[PILImage.Image]) -> list[tuple[bool, str | None]]:
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
            except Exception:
                results.append((False, None))
        self.logger.debug(f"Watermark results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]:
        ai_watermark_t = self.model_ai_watermark_processor(images, return_tensors="pt").to(self.model_ai_watermark_decoder.device)
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[:, 0] < 0
        pred_bool = ai_watermark_pred > (1 - settings.model.detection.threshold)
        self.logger.debug(f"AI watermark prediction: {pred_bool}")
        return pred_bool.tolist()

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms,
    )
    async def add_watermark(self, request: list[AddWatermarkRequest]) -> list[PILImage.Image]:
        wm = WatermarkEncoder()
        binaries = "".join(format(ord(c.watermark_text), "08b") for c in request)
        binaries = [binary[:32] if len(binary) > 32 else binary.ljust(32, "0") for binary in binaries]

        results = []
        for image, binary in zip(request, binaries):
            wm.set_watermark("bits", binary)
            results.append(PILImage.fromarray(wm.encode(np.asarray(image.image), settings.model.watermark.method)))

        self.logger.debug(f"Watermark results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms,
    )
    async def add_ai_watermark(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        results = list(self.model_ai_watermark(image=images)[0])
        self.logger.debug(f"AI watermark results: {results}")
        return results
