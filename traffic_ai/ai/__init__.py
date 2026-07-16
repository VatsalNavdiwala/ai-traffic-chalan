"""AI modules: detection, tracking, speed, signal, OCR, violations, pipeline."""

__all__ = ["TrafficPipeline"]


def __getattr__(name: str):
    if name == "TrafficPipeline":
        from traffic_ai.ai.pipeline import TrafficPipeline

        return TrafficPipeline
    raise AttributeError(name)
