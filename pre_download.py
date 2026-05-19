"""
Utility script to pre-download and cache all enabled models on a node with internet access (e.g. login node).
"""

import sys
from banking_asr_eval.models import ModelRegistry


def main():
    print("="*60)
    print("Pre-downloading all enabled models to cache...")
    print("="*60)

    try:
        registry = ModelRegistry.from_config("config.yaml")
    except Exception as e:
        print(f"Error loading registry: {e}")
        sys.exit(1)

    enabled_models = registry.list_models()
    print(f"Found models in config: {enabled_models}")

    # We iterate and get each enabled model, which triggers lazy loading and downloading
    for name, enabled in enabled_models.items():
        if not enabled:
            continue
        print(f"\n[Download] Loading and caching: {name}...")
        try:
            _ = registry.get_model(name)
            print(f"[Success] {name} is cached and ready.")
        except Exception as e:
            print(f"[Error] Failed to download {name}: {e}")

    print("\nAll enabled models have been cached successfully.")


if __name__ == "__main__":
    main()
