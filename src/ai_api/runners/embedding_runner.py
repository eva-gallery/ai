from __future__ import annotations

import hashlib
import io
from typing import Optional, Literal, cast

import bentoml
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.cuda import is_available
from transformers import pipeline, AutoModelForImageClassification, BlipImageProcessor
from imwatermark import WatermarkEncoder, WatermarkDecoder
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from PIL import Image as PILImage
import numpy as np

from ai_api import settings
from ai_api.database.postgres_client import AIOPostgres
from ai_api.model.api.status import ResponseType
from ai_api.orm.gallery_embedding import GalleryEmbedding
from ai_api.orm.image_data import ImageData
from ai_api.util.logger import get_logger
from ai_api.model.api.process import AIGeneratedStatus


class EmbeddingRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:
        self.device: Literal['cuda'] | Literal['cpu'] = "cuda" if is_available() else "cpu"
        self.model_img = SentenceTransformer(bentoml.transformers.import_model(name="image_embedding", model_name_or_path=settings.model.embedding.image.name).path, device=self.device)
        self.model_text = SentenceTransformer(bentoml.transformers.import_model(name="text_embedding", model_name_or_path=settings.model.embedding.text.name).path, device=self.device)
        self.model_caption = pipeline("image-to-text", model=bentoml.transformers.import_model(name="image_captioning", model_name_or_path=settings.model.captioning.name).path, device=self.device)
        self.model_ai_watermark = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", vae=bentoml.transformers.import_model(name="ai_watermark_encoder", model_name_or_path=settings.model.watermark.encoder_name).path, device=self.device)
        self.model_ai_watermark_decoder = AutoModelForImageClassification.from_pretrained(bentoml.transformers.import_model(name="ai_watermark_decoder", model_name_or_path=settings.model.watermark.decoder_name).path, device=self.device)
        self.model_ai_watermark_processor = cast(BlipImageProcessor, BlipImageProcessor.from_pretrained(bentoml.transformers.import_model(name="ai_watermark_decoder", model_name_or_path=settings.model.watermark.decoder_name).path))
        
        self.model_img.eval()
        self.model_text.eval()
        
        self.model_ai_detection = pipeline("image-classification", 
            model=bentoml.transformers.import_model(
                name="ai_detection", 
                model_name_or_path=settings.model.detection.name
            ).path, 
            device=self.device
        )
        
        self.logger = get_logger()
        self.pg = AIOPostgres(url=settings.postgres.url)

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def embed_text(self, texts: list[str]) -> list:
        return self.model_text.encode(texts).tolist()  # type: ignore

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def embed_image(self, images: list[PILImage.Image]) -> list:
        return self.model_img.encode(images, device=self.device, normalize_embeddings=True).tolist()  # type: ignore

    @bentoml.Runnable.method(batchable=False)
    def generate_caption(self, image: PILImage.Image) -> str:
        return self.model_caption(image)[0]['generated_text']  # type: ignore

    @bentoml.Runnable.method(batchable=False)
    def detect_ai_generation(self, image: PILImage.Image) -> tuple[AIGeneratedStatus, Optional[float]]:
        result = self.model_ai_detection(image)
        score: Tensor = result[0]['score']  # type: ignore
        
        if score > settings.model.detection.threshold:
            return AIGeneratedStatus.GENERATED, score.float().item()
        return AIGeneratedStatus.NOT_GENERATED, score.float().item()

    @bentoml.Runnable.method(batchable=False)
    def check_watermark(self, image: PILImage.Image) -> tuple[bool, Optional[str]]:
        gan_mark = WatermarkDecoder('bytes', 256)
        watermark: str = gan_mark.decode(np.asarray(image), settings.model.watermark.method)  # type: ignore
        return True if watermark else False, watermark

    @bentoml.Runnable.method(batchable=False)
    def check_ai_watermark(self, image: PILImage.Image) -> bool:
        ai_watermark_t = self.model_ai_watermark_processor(image, return_tensors="pt")
        ai_watermark_pred = self.model_ai_watermark_decoder(**ai_watermark_t).logits[0,0] < 0  # type: ignore
        return ai_watermark_pred > (1 - settings.model.watermark.threshold)

    @bentoml.Runnable.method(batchable=False)
    def add_watermark(self, image: PILImage.Image, watermark_text: str) -> PILImage.Image:
        wm = WatermarkEncoder()
        wm.set_watermark('bytes', watermark_text.encode('utf-8'))  # type: ignore
        return PILImage.fromarray(wm.encode(np.asarray(image), settings.model.watermark.method)) 
