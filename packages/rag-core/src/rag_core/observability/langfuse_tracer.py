"""Fail-open Langfuse tracing adapter."""

import logging
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


logger = logging.getLogger(__name__)


class _NoopObservation:
    trace_id = None

    def update(self, **kwargs: Any) -> None:
        return None


class _SafeObservation:
    def __init__(self, observation: Any):
        self.observation = observation

    def update(self, **kwargs: Any) -> None:
        try:
            self.observation.update(**kwargs)
        except Exception:
            logger.debug("Could not update Langfuse observation", exc_info=True)

    @property
    def trace_id(self) -> Optional[str]:
        return getattr(self.observation, "trace_id", None)


class LangfuseTracer:
    """Create Langfuse observations only when explicitly enabled and configured."""

    def __init__(
        self,
        enabled: bool = False,
        capture_content: bool = False,
        client: Optional[Any] = None,
    ):
        self.capture_content = capture_content
        self.client = None

        if not enabled:
            return

        if client is not None:
            self.client = client
            return

        if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            logger.warning("Langfuse tracing enabled but credentials are missing; tracing disabled")
            return

        try:
            from langfuse import Langfuse

            base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
            self.client = Langfuse(**({"base_url": base_url} if base_url else {}))
        except Exception as exc:
            logger.warning(
                "Could not initialize Langfuse; tracing disabled (%s)",
                type(exc).__name__,
            )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        if self.client is None:
            yield _NoopObservation()
            return

        options: Dict[str, Any] = {"as_type": as_type, "name": name}
        if self.capture_content and input_data is not None:
            options["input"] = input_data
        if metadata:
            options["metadata"] = metadata
        if model:
            options["model"] = model
        if model_parameters:
            options["model_parameters"] = model_parameters

        try:
            manager = self.client.start_as_current_observation(**options)
            observation = manager.__enter__()
        except Exception as exc:
            logger.warning(
                "Could not start Langfuse observation '%s' (%s)",
                name,
                type(exc).__name__,
            )
            yield _NoopObservation()
            return

        try:
            yield _SafeObservation(observation)
        except BaseException as exc:
            try:
                observation.update(level="ERROR", status_message=type(exc).__name__)
            except Exception:
                logger.debug("Could not annotate failed Langfuse observation", exc_info=True)
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.debug("Could not close failed Langfuse observation", exc_info=True)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.debug("Could not close Langfuse observation", exc_info=True)

    def flush(self) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:
            logger.debug("Could not flush Langfuse events", exc_info=True)

    def score(self, trace_id: Optional[str], name: str, value: Any) -> None:
        if self.client is None or not trace_id or value is None:
            return
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                return
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=numeric_value,
                data_type="NUMERIC",
            )
        except Exception as exc:
            logger.warning(
                "Could not attach Langfuse score '%s' (%s)",
                name,
                type(exc).__name__,
            )