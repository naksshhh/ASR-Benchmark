"""
Model Registry — maps model names to inference callables.

Design: each model is a callable that takes an audio path and returns
a transcript string. The registry handles lazy loading so models are
only loaded when first used.
"""

import time
from typing import Callable, Dict, Optional
from pathlib import Path

import yaml


ModelFn = Callable[[str], str]  # audio_path → transcript


class ModelRegistry:
    """
    Registry of ASR models with lazy loading.

    Usage:
        registry = ModelRegistry.from_config("config.yaml")
        for name, model_fn in registry.enabled_models():
            transcript = model_fn("audio.wav")
    """

    def __init__(self):
        self._configs: Dict[str, dict] = {}
        self._loaded: Dict[str, ModelFn] = {}

    @classmethod
    def from_config(cls, config_path: str) -> "ModelRegistry":
        """Load registry from config.yaml."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        registry = cls()
        for name, model_config in config.get("models", {}).items():
            registry.register_config(name, model_config)
        return registry

    def register_config(self, name: str, config: dict):
        """Register a model configuration."""
        self._configs[name] = config

    def register_model(self, name: str, model_fn: ModelFn):
        """Register a pre-loaded model function."""
        self._loaded[name] = model_fn

    def _load_model(self, name: str) -> ModelFn:
        """Lazy-load a model based on its backend config."""
        # Configure PyTorch CPU threads to avoid Xeon multi-socket synchronization overhead
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        config = self._configs[name]
        backend = config.get("backend", "whisper")

        if backend == "whisper":
            from .inference.whisper_local import create_whisper_model
            return create_whisper_model(
                model_id=config["model_id"],
                language=config.get("language", "hi"),
                task=config.get("task", "transcribe"),
            )
        elif backend == "nemo":
            from .inference.nemo_local import create_nemo_model
            return create_nemo_model(model_id=config["model_id"])
        elif backend == "huggingface":
            from .inference.huggingface_generic import create_hf_model
            return create_hf_model(model_id=config["model_id"])
        elif backend == "sherpa-onnx":
            from .inference.sherpa_onnx_local import create_sherpa_onnx_model
            return create_sherpa_onnx_model(model_id=config["model_id"])
        elif backend == "voxtral":
            from .inference.voxtral_local import create_voxtral_model
            return create_voxtral_model(model_id=config["model_id"])
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_model(self, name: str) -> ModelFn:
        """Get a model by name, loading it if necessary."""
        if name not in self._loaded:
            if name not in self._configs:
                raise KeyError(f"Model '{name}' not registered")
            print(f"[ModelRegistry] Loading {name}...")
            t0 = time.time()
            self._loaded[name] = self._load_model(name)
            print(f"[ModelRegistry] {name} loaded in {time.time() - t0:.1f}s")
        return self._loaded[name]

    def enabled_models(self):
        """Yield (name, model_fn) for all enabled models."""
        for name, config in self._configs.items():
            if config.get("enabled", False):
                yield name, self.get_model(name)

    def list_models(self) -> Dict[str, bool]:
        """List all registered models and their enabled status."""
        return {
            name: config.get("enabled", False)
            for name, config in self._configs.items()
        }
