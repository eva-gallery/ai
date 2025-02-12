"""The inference service for the AI API.

This module provides a service for handling various AI-related operations including:
- Image and text embedding generation
- Image captioning
- AI generation detection
- Image similarity search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import AutoModelForImageClassification, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor, pipeline

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.model.api.protocols import InferenceServiceProto
from ai_api.util.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


SHARPEN_KERNEL = np.array([[-0.5, -0.5, -0.5],
                      [-0.5,  5, -0.5],
                      [-0.5, -0.5, -0.5]], dtype=np.float32)


def interleave_images(
    original_image: NDArray[np.uint8],
    watermarked_image: NDArray[np.uint8],
    original_ratio: float = settings.model.watermark.interleave_ratio,
    gamma: float = settings.model.watermark.gamma,
) -> NDArray[np.uint8]:
    """Interleave two images with 50% contribution from each.

    :param image1: First input image.
    :param image2: Second input image.
    :raises ValueError: If images are not the same size or type.
    :returns: Blended image as a numpy array.
    """
    # Blend images with equal weights
    result = cv2.addWeighted(src1=original_image, alpha=original_ratio, src2=watermarked_image, beta=1-original_ratio, gamma=gamma)

    # Convert the image to float32 for more precise calculations
    result_float = result.astype(np.float32)

    # Create a mask for shadows (darker areas)
    shadow_mask = cv2.inRange(result, 0, 127)  # type: ignore[arg-type]
    shadow_adjustment = cv2.convertScaleAbs(result_float, alpha=0.8, beta=-20)  # Darken shadows

    # Create a mask for highlights (brighter areas)
    highlight_mask = cv2.inRange(result, 128, 255)  # type: ignore[arg-type]
    highlight_adjustment = cv2.convertScaleAbs(result_float, alpha=1.2, beta=20)  # Lighten highlights

    # Apply the shadow adjustment
    shadow_result = cv2.bitwise_and(shadow_adjustment, shadow_adjustment, mask=shadow_mask).astype(np.float32)

    # Apply the highlight adjustment
    highlight_result = cv2.bitwise_and(highlight_adjustment, highlight_adjustment, mask=highlight_mask).astype(np.float32)

    # Combine the original image with the adjusted shadows and highlights
    result_image = cv2.addWeighted(result_float, 1.0, shadow_result, 0.5, 0.0)
    result_image = cv2.addWeighted(result_image, 1.0, highlight_result, 0.5, 0.0)

    # Ensure the result is within valid range and convert back to uint8
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    # Apply gentle sharpening using a kernel filter
    result_image = cv2.filter2D(result_image, -1, SHARPEN_KERNEL)
    return np.clip(result_image, 0, 255).astype(np.uint8)


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
            BlipProcessor.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir, device=self.device, torch_dtype=torch.float16),
        )
        self.caption_model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir)

        self.model_ai_watermark: StableDiffusionXLImg2ImgPipeline = cast(StableDiffusionXLImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline.from_pretrained(
            settings.model.watermark.diffusion_model,
            vae=AutoencoderKL.from_pretrained(settings.model.watermark.encoder_name, cache_dir=settings.model.cache_dir, device_map=self.device_map_string),
            device_map=self.device_map_string,
            add_watermarker=True,
            torch_dtype=torch.float16,
        ))
        self.model_ai_watermark_decoder = AutoModelForImageClassification.from_pretrained(
            settings.model.watermark.decoder_name,
            device_map=self.device_map_string,
            cache_dir=settings.model.cache_dir,
            torch_dtype=torch.float16,
        )
        self.model_ai_watermark_processor: BlipImageProcessor = BlipImageProcessor.from_pretrained(
            settings.model.watermark.decoder_name,
            cache_dir=settings.model.cache_dir,
            device_map=self.device_map_string,
            torch_dtype=torch.float16,
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

    @torch.inference_mode()
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text inputs.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        return [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]

    @torch.inference_mode()
    async def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]:
        """Generate embeddings for a list of images.

        :param images: List of PIL images to embed.
        :type images: List[PILImage.Image]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        results = [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]  # type: ignore[arg-type]
        self.logger.debug("Image embedding results: {results}", results=str(results)[:100])
        return results

    @torch.inference_mode()
    async def generate_caption(self, images: list[PILImage.Image]) -> list[str]:
        """Generate descriptive captions for a list of images.

        :param images: List of PIL images to generate captions for.
        :type images: List[PILImage.Image]
        :returns: List of generated caption strings.
        :rtype: list[str]
        """
        self.logger.debug("Generating caption for {images}", images=images)
        inputs = self.caption_processor(images, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_length=settings.model.captioning.max_length)  # type: ignore[arg-type]
        captions = self.caption_processor.batch_decode(out, skip_special_tokens=True)
        self.logger.debug("Caption results: {captions}, type: {type}", captions=captions, type=type(captions))
        return captions

    @torch.inference_mode()
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

    async def _resize_images(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        """Resize images to target resolution while maintaining aspect ratio.

        Only downscales images that are larger than the target resolution.

        :param images: List of PIL images to resize.
        :type images: List[PILImage.Image]
        :returns: List of resized PIL images.
        :rtype: list[PILImage.Image]
        """
        resized_images = []
        target_size = settings.model.watermark.resolution_longest_size

        for img in images:
            # Convert PIL to cv2
            cv_img = np.array(img)
            height, width = cv_img.shape[:2]

            # Skip resize if image is already smaller than target
            if max(height, width) <= target_size:
                resized_images.append(img)
                continue

            # Calculate new dimensions maintaining aspect ratio
            scale = target_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize using cv2
            resized = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_images.append(PILImage.fromarray(resized))

        return resized_images

    async def _sort_by_ar_type(self, images: list[tuple[int, PILImage.Image]]) -> list[list[tuple[int, PILImage.Image]]]:
        """Sort images into bins of identical resolutions.

        :param images: List of tuples containing index and PIL images to sort.
        :type images: list[tuple[int, PILImage.Image]]
        :returns: List of lists of tuples sorted by identical resolution.
        :rtype: list[list[tuple[int, PILImage.Image]]]
        """
        # Dictionary to hold lists of images with identical resolutions
        resolution_bins: dict[tuple[int, int], list[tuple[int, PILImage.Image]]] = {}

        for index, img in images:
            resolution = img.size  # (width, height)

            if resolution not in resolution_bins:
                resolution_bins[resolution] = []

            resolution_bins[resolution].append((index, img))

        # Convert the dictionary values to a list of lists
        return list(resolution_bins.values())

    @torch.inference_mode()
    async def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]:
        """Check for the presence of AI-specific watermarks in images.

        :param images: List of PIL images to check for AI watermarks.
        :type images: List[PILImage.Image]
        :returns: List of boolean values indicating AI watermark presence.
        :rtype: list[bool]
        """
        # Sort images into resolution bins while keeping track of their original indices
        resized_images = await self._resize_images(images)
        resolution_bins = await self._sort_by_ar_type(list(enumerate(resized_images)))

        processed_indices: dict[int, bool] = {}

        for bin_image_grp in resolution_bins:
            indices, bin_images = zip(*bin_image_grp, strict=True)

            ai_watermark_t = self.model_ai_watermark_processor(bin_images, return_tensors="pt").to(self.model_ai_watermark_decoder.device, dtype=torch.float16)
            logits = self.model_ai_watermark_decoder(**ai_watermark_t).logits.detach().cpu().numpy()[:, 0]  # type: ignore[arg-type]
            is_watermarked = logits < settings.model.watermark.threshold

            # Store results with their indices
            processed_indices.update(dict(zip(indices, is_watermarked.tolist(), strict=True)))

        # Reconstruct the list in original order
        return [processed_indices[i] for i in range(len(images))]

    @torch.inference_mode()
    async def add_ai_watermark(self, images: list[PILImage.Image], prompts: list[str]) -> list[PILImage.Image]:
        """Add AI-specific watermarks to a list of images.

        :param images: List of PIL images to add AI watermarks to.
        :type images: List[PILImage.Image]
        :param prompts: List of prompts to guide the watermarking process.
        :type prompts: list[str]
        :returns: List of watermarked PIL images.
        :rtype: list[PILImage.Image]
        """
        self.logger.debug("Adding watermark to images with size: {sizes}", sizes=[img.size for img in images])

        # Create a list of tuples (index, image) to track original order
        indexed_images = list(enumerate(images))

        # Sort images into resolution bins while keeping track of their original indices
        full_size_resolution_bins = await self._sort_by_ar_type(indexed_images)
        resized_images = await self._resize_images(images)
        resolution_bins = await self._sort_by_ar_type(list(enumerate(resized_images)))

        processed_indices: dict[int, PILImage.Image] = {}

        for bin_image_grp, full_size_bin in zip(resolution_bins, full_size_resolution_bins, strict=True):
            indices, bin_images = zip(*bin_image_grp, strict=True)
            _, full_size_bin_images = zip(*full_size_bin, strict=True)

            pre_processed_images: dict[str, torch.Tensor] = self.model_ai_watermark_processor.preprocess(
                bin_images, do_resize=False, return_tensors="pt",
            ).to(self.model_ai_watermark_decoder.device)

            results = self.model_ai_watermark(
                image=pre_processed_images["pixel_values"],
                prompt=prompts,
                strength=settings.model.watermark.strength,
                num_inference_steps=settings.model.watermark.num_inference_steps,
                guidance_scale=settings.model.watermark.guidance_scale,
                original_size=bin_images[0].size,
                target_size=bin_images[0].size,
            ).images  # type: ignore[arg-type]

            # Resize images to original size and interleave
            resized_images = [img.resize(full_size_bin_images[i].size) for i, img in enumerate(results)]
            resized_images_numpy = [np.array(img) for img in resized_images]
            original_images_numpy = [np.array(img) for img in full_size_bin_images]
            interleaved_results = [interleave_images(original_images_numpy[i], resized_images_numpy[i]) for i in range(len(full_size_bin_images))]

            # Store processed images with their indices
            for idx, interleaved_img in zip(indices, interleaved_results, strict=True):
                processed_indices[idx] = PILImage.fromarray(interleaved_img)

        # Reconstruct the list in original order
        return [processed_indices[i] for i in range(len(images))]
