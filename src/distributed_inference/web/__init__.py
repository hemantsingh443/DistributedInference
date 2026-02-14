"""Web UI and HTTP gateway for distributed inference."""

from .app import app, create_app

__all__ = ["app", "create_app"]
