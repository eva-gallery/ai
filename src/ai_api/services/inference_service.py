from __future__ import annotations

from typing import Literal, cast

import bentoml
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import pipeline, AutoModelForImageClassification, BlipImageProcessor, BlipProcessor, BlipForConditionalGeneration
from imwatermark import WatermarkEncoder, WatermarkDecoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline

from PIL import Image as PILImage
import numpy as np

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.util.logger import get_logger


@bentoml.service(
    name="evagellery_ai_inference",
    resources=settings.bentoml.embedding.resources.to_dict(),
    traffic={"timeout": 300}
)
class InferenceService:
    
    def __init__(self) -> None:
        self.logger = get_logger()
        self.device: Literal['cuda'] | Literal['cpu'] = "cuda" if is_available() else "cpu"
        self.model_img = SentenceTransformer(settings.model.embedding.image.name, device=self.device, cache_folder=settings.model.cache_dir)
        self.model_text = SentenceTransformer(settings.model.embedding.text.name, device=self.device, cache_folder=settings.model.cache_dir)
        
        # Initialize BLIP captioning model and processor
        self.caption_processor: BlipProcessor = cast(
            BlipProcessor,
            BlipProcessor.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir, device_map="balanced")
        )
        self.caption_model: BlipForConditionalGeneration = cast(
            BlipForConditionalGeneration,
            BlipForConditionalGeneration.from_pretrained(settings.model.captioning.name, cache_dir=settings.model.cache_dir)
        )
        
        self.model_ai_watermark = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            settings.model.watermark.diffusion_model,
            vae=AutoencoderKL.from_pretrained(settings.model.watermark.encoder_name, cache_dir=settings.model.cache_dir),
            device_map="balanced",
        )
        self.model_ai_watermark_decoder = AutoModelForImageClassification.from_pretrained(
            settings.model.watermark.decoder_name,
            device_map="balanced",
            cache_dir=settings.model.cache_dir
        )
        self.model_ai_watermark_processor = cast(
            BlipImageProcessor,
            BlipImageProcessor.from_pretrained(
                settings.model.watermark.decoder_name,
                cache_dir=settings.model.cache_dir,
                device_map="balanced",
            )
        )
        
        self.model_img.eval()
        self.model_text.eval()
        self.caption_model.eval()
        
        self.model_ai_detection = pipeline(
            "image-classification",
            settings.model.detection.name,
            device_map="auto",
        )
        
    @bentoml.api(route="/readyz")
    async def readyz(self, ctx: bentoml.Context) -> dict[str, str]:
        ctx.response.status_code = 200
        return {"status": "ready"}

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        results = [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]
        self.logger.debug(f"Text embedding results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]:
        results = [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]  # type: ignore
        self.logger.debug(f"Image embedding results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def generate_caption(self, images: list[PILImage.Image]) -> list[str]:
        inputs = self.caption_processor(images, return_tensors="pt")
        out = self.caption_model.generate(**inputs, max_length=settings.model.captioning.max_length)  # type: ignore
        captions = self.caption_processor.decode(out[0], skip_special_tokens=True)
        self.logger.debug(f"Caption results: {captions}")
        return captions

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def detect_ai_generation(self, images: list[PILImage.Image]) -> list[AIGeneratedStatus]:
        results = self.model_ai_detection(images)[0]  # type: ignore
        results = [AIGeneratedStatus.GENERATED if result['score'] > settings.model.detection.threshold else AIGeneratedStatus.NOT_GENERATED for result in results]  # type: ignore
        self.logger.debug(f"AI generation detection results: {results}")
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def check_watermark(self, images: list[PILImage.Image]) -> list[tuple[bool, str | None]]:
        gan_mark = WatermarkDecoder('bits', 32)
        results = []
        for image in images:
            try:
                watermark = gan_mark.decode(np.asarray(image), settings.model.watermark.method)
                if watermark:
                    watermark_bytes = int(watermark, 2).to_bytes((len(watermark) + 7) // 8, byteorder='big')
                    try:
                        watermark_str = watermark_bytes.decode('utf-8').rstrip('\x00')
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
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms
    )
    async def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]:
        ai_watermark_t = self.model_ai_watermark_processor(images, return_tensors="pt").to(self.device)
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[:, 0] < 0  # type: ignore
        pred_bool = ai_watermark_pred > (1 - settings.model.detection.threshold)
        self.logger.debug(f"AI watermark prediction: {pred_bool}")
        return pred_bool.tolist()

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms
    )
    async def add_watermark(self, images: list[PILImage.Image], watermark_text: str) -> list[PILImage.Image]:
        wm = WatermarkEncoder()
        binary = ''.join(format(ord(c), '08b') for c in watermark_text)
        binary = binary[:32] if len(binary) > 32 else binary.ljust(32, '0')
        wm.set_watermark('bits', binary)
        results = [PILImage.fromarray(wm.encode(np.asarray(image), settings.model.watermark.method)) for image in images]
        self.logger.debug(f"Watermark results: {results}")
        return results
    
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms
    )
    async def add_ai_watermark(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        results = [result for result in self.model_ai_watermark(image=images)[0]]  # type: ignore
        self.logger.debug(f"AI watermark results: {results}")
        return results
