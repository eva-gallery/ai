#!/usr/bin/env python

from huggingface_hub import snapshot_download

from ai_api import settings
from ai_api.util.logger import get_logger

logger = get_logger()

# Pre-download models by instantiating them once
logger.info("Pre-downloading models...")

# Image embedding model
snapshot_download(settings.model.embedding.image.name, cache_dir=settings.model.cache_dir)
# Text embedding model
snapshot_download(settings.model.embedding.text.name, cache_dir=settings.model.cache_dir)
# Image captioning model
snapshot_download(settings.model.captioning.name, cache_dir=settings.model.cache_dir)
# AI watermark models
snapshot_download(settings.model.watermark.diffusion_model, cache_dir=settings.model.cache_dir)
snapshot_download(settings.model.watermark.encoder_name, cache_dir=settings.model.cache_dir)
snapshot_download(settings.model.watermark.decoder_name, cache_dir=settings.model.cache_dir)
# AI detection model
snapshot_download(settings.model.detection.name, cache_dir=settings.model.cache_dir)

logger.info("Model pre-download complete")
