from functools import lru_cache

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)


SERVICE_NAME = "coreassist-5g-rag"


@lru_cache(maxsize=1)
def configure_tracing() -> None:
    """
    Configure OpenTelemetry tracing for local development.

    Traces are currently exported to the console. An OTLP exporter can
    replace the console exporter later without changing the RAG spans.
    """
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )

    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    configure_tracing()
    return trace.get_tracer(SERVICE_NAME)