from __future__ import annotations

from typing import Literal, cast

import bentoml
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
from transformers import pipeline, AutoModelForImageClassification, BlipImageProcessor
from imwatermark import WatermarkEncoder, WatermarkDecoder
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline

from PIL import Image as PILImage
import numpy as np

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus


@bentoml.service(
    name="evagallery_ai_embedding",
    resources=settings.bentoml.embedding.resources.to_dict(),
    traffic={"timeout": 300}
)
class EmbeddingService:
    
    def __init__(self) -> None:
        self.device: Literal['cuda'] | Literal['cpu'] = "cuda" if is_available() else "cpu"
        self.model_img = SentenceTransformer(settings.model.embedding.image.name, device=self.device)
        self.model_text = SentenceTransformer(settings.model.embedding.text.name, device=self.device)
        self.model_caption = pipeline("image-to-text", model=settings.model.captioning.name, device=self.device)
        self.model_ai_watermark = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            settings.model.watermark.diffusion_model,
            vae=settings.model.watermark.encoder_name,
            device=self.device
        )
        self.model_ai_watermark_decoder = AutoModelForImageClassification.from_pretrained(
            settings.model.watermark.decoder_name,
            device=self.device
        )
        self.model_ai_watermark_processor = cast(
            BlipImageProcessor,
            BlipImageProcessor.from_pretrained(settings.model.watermark.decoder_name)
        )
        
        self.model_img.eval()
        self.model_text.eval()
        
        self.model_ai_detection = pipeline(
            "image-classification",
            settings.model.detection.name,
            device=self.device
        )
        
    @bentoml.api(route="/readyz")
    async def readyz(self, ctx: bentoml.Context) -> dict[str, str]:
        ctx.response.status_code = 200
        return {"status": "ready"}

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=100
    )
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [result.tolist() for result in self.model_text.encode(texts, normalize_embeddings=True)]

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def embed_image(self, images: list[PILImage.Image]) -> list[list[float]]:
        return [result.tolist() for result in self.model_img.encode(images, normalize_embeddings=True)]  # type: ignore

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def generate_caption(self, images: list[PILImage.Image]) -> list[str]:
        return [result[0]['generated_text'] for result in self.model_caption(images)]  # type: ignore

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def detect_ai_generation(self, images: list[PILImage.Image]) -> list[AIGeneratedStatus]:
        results = self.model_ai_detection(images)
        return [AIGeneratedStatus.GENERATED if result['score'] > settings.model.detection.threshold else AIGeneratedStatus.NOT_GENERATED for result in results]  # type: ignore

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def check_watermark(self, images: list[PILImage.Image]) -> list[str | None]:
        gan_mark = WatermarkDecoder('bytes', 256)
        return cast(list[str | None], [gan_mark.decode(np.asarray(image), settings.model.watermark.method) for image in images])

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def check_ai_watermark(self, images: list[PILImage.Image]) -> list[bool]:
        ai_watermark_t = self.model_ai_watermark_processor(images, return_tensors="pt")
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[0,0] < 0  # type: ignore
        return [pred > (1 - settings.model.watermark.threshold) for pred in ai_watermark_pred]

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def add_watermark(self, images: list[PILImage.Image], watermark_text: str) -> list[PILImage.Image]:
        wm = WatermarkEncoder()
        wm.set_watermark('bytes', watermark_text.encode('utf-8'))  # type: ignore
        return [PILImage.fromarray(wm.encode(np.asarray(image), settings.model.watermark.method)) for image in images]
    
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=10000
    )
    def add_ai_watermark(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        return [result for result in self.model_ai_watermark(image=images)[0]]  # type: ignore
