"""Root module for the AI API service.

This module exports the main APIService class which serves as the entry point
for the AI API service. The service provides endpoints for image processing,
search operations, and AI-based analysis.

The service is implemented using BentoML and provides a RESTful API interface.
"""

from ai_api.main import APIService

__all__ = ["APIService"]
