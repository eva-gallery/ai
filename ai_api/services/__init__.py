"""Module containing BentoML service implementations.

This module provides the core service implementations for the AI API, including:
- Inference service for AI operations (embedding, detection, watermarking)
- Request/response models for service operations
- Service configuration and initialization

The services are implemented using BentoML for production-ready ML serving.
"""

from ai_api.services.inference_service import InferenceService

__all__ = ["InferenceService"]
