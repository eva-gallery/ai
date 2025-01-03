"""The inference service for the AI API.

This module provides a service for handling various AI-related operations including:
- Image and text embedding generation
- Image captioning
- AI generation detection
- Watermark detection and addition
- Image similarity search
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import AutoModelForImageClassification, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor, pipeline

from ai_api import settings
from ai_api.model.api.process import AddWatermarkRequest, AIGeneratedStatus
from ai_api.model.api.protocols import InferenceServiceProto
from ai_api.util.logger import get_logger


class InferenceService(InferenceServiceProto):
    """Service for AI inference operations.

    This service handles various AI operations including embedding generation,
    image captioning, AI detection, and watermark operations. It uses multiple
    pre-trained models for different tasks.

    Attributes:
        logger: Logger instance for service logging.
        device: PyTorch device (CPU/GPU) for model execution.
        model_img: SentenceTransformer model for image embedding.
        model_text: SentenceTransformer model for text embedding.
        caption_processor: BLIP processor for image captioning.
        caption_model: BLIP model for image captioning.
        model_ai_watermark: StableDiffusion model for AI watermarking.
        model_ai_watermark_decoder: Model for detecting AI watermarks.
        model_ai_detection: Pipeline for detecting AI-generated images.

    """

    def __init__(self, worker_index: int = 0) -> None:
        """Initialize the inference service with required models and processors.

        :param worker_index: Index of the worker for GPU device selection.
        :type worker_index: int
        """
        self.logger = get_logger()
        self.device: torch.device = torch.device(f"cuda:{worker_index}" if is_available() else "cpu")
        self.device_map_string = "balanced"  # can't use 'cuda:0', bug in safetensors
        self.model_img = SentenceTransformer(settings.model.embedding.image.name, device=str(self.device), cache_folder=settings.model.cache_dir)
        self.model_text = SentenceTransformer(settings.model.embedding.text.name, device=str(self.device), cache_folder=settings.model.cache_dir)

        # Initialize BLIP captioning model and processor
        self.caption_processor: BlipProcessor = cast(
            BlipProcessor,
            BlipProcessor.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir, device=self.device),
        )
        self.caption_model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir)

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
        self.model_ai_watermark_processor = BlipImageProcessor.from_pretrained(
            settings.model.watermark.decoder_name,
            cache_dir=settings.model.cache_dir,
            device_map=self.device_map_string,
        )

        self.model_img.eval()
        self.model_text.eval()
        self.caption_model.eval()

        self.model_ai_detection = pipeline(
            "image-classification",
            settings.model.detection.name,
            device=self.device,
        )

    async def readyz(self) -> dict[str, str]:
        """Check if the service is ready to handle requests.

        :returns: Dictionary containing service status.
        :rtype: dict[str, str]
        """
        return {"status": "ready"}

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text inputs.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        results = [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]
        self.logger.debug("Text embedding results: {results}", results=results)
        return results

    async def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]:
        """Generate embeddings for a list of images.

        :param images: List of PIL images to embed.
        :type images: List[PILImage.Image]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        results = [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]  # type: ignore[arg-type]
        self.logger.debug("Image embedding results: {results}", results=results)
        return results

    async def generate_caption(self, images: list[PILImage.Image]) -> list[str]:
        """Generate descriptive captions for a list of images.

        :param images: List of PIL images to generate captions for.
        :type images: List[PILImage.Image]
        :returns: List of generated caption strings.
        :rtype: list[str]
        """
        inputs = self.caption_processor(images, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_length=settings.model.captioning.max_length)  # type: ignore[arg-type]
        captions = self.caption_processor.decode(out[0], skip_special_tokens=True)
        self.logger.debug("Caption results: {captions}", captions=captions)
        return captions

    async def detect_ai_generation(self, images: list[PILImage.Image]) -> list[AIGeneratedStatus]:
        """Detect whether images were generated by AI.

        :param images: List of PIL images to check.
        :type images: List[PILImage.Image]
        :returns: List of AI generation status enums.
        :rtype: list[AIGeneratedStatus]
        :raises: May raise exceptions from the underlying AI detection model.
        """
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

    async def check_watermark(self, images: list[PILImage.Image]) -> list[tuple[bool, str | None]]:
        """Check for the presence of watermarks in images.

        :param images: List of PIL images to check for watermarks.
        :type images: List[PILImage.Image]
        :returns: List of tuples containing (has_watermark, watermark_text).
        :rtype: list[tuple[bool, str | None]]
        :raises: May raise exceptions during watermark decoding.
        """
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

    async def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]:
        """Check for the presence of AI-specific watermarks in images.

        :param images: List of PIL images to check for AI watermarks.
        :type images: List[PILImage.Image]
        :returns: List of boolean values indicating AI watermark presence.
        :rtype: list[bool]
        """
        ai_watermark_t = self.model_ai_watermark_processor(images, return_tensors="pt").to(self.model_ai_watermark_decoder.device)
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[:, 0] < 0  # type: ignore[arg-type]
        pred_bool = ai_watermark_pred > (1 - settings.model.detection.threshold)
        self.logger.debug("AI watermark prediction: {pred_bool}", pred_bool=pred_bool)
        return pred_bool.tolist()

    async def add_watermark(self, request: list[AddWatermarkRequest]) -> list[PILImage.Image]:
        """Add watermarks to a list of images.

        :param request: List of watermark requests containing images and watermark text.
        :type request: List[AddWatermarkRequest]
        :returns: List of watermarked PIL images.
        :rtype: list[PILImage.Image]
        """
        wm = WatermarkEncoder()
        binaries = "".join(format(ord(c.watermark_text), "08b") for c in request)
        binaries = [binary[:32] if len(binary) > 32 else binary.ljust(32, "0") for binary in binaries]  # noqa: PLR2004

        results = []
        for image, binary in zip(request, binaries, strict=False):
            wm.set_watermark("bits", binary)
            results.append(PILImage.fromarray(wm.encode(np.asarray(image.image), settings.model.watermark.method)))

        self.logger.debug("Watermark results: {results}", results=results)
        return results

    async def add_ai_watermark(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        """Add AI-specific watermarks to a list of images.

        :param images: List of PIL images to add AI watermarks to.
        :type images: List[PILImage.Image]
        :returns: List of watermarked PIL images.
        :rtype: list[PILImage.Image]
        """
        results = list(self.model_ai_watermark(image=images)[0])  # type: ignore[arg-type]
        self.logger.debug("AI watermark results: {results}", results=results)
        return results
