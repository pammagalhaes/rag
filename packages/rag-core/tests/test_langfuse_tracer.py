from contextlib import contextmanager

from rag_core.observability.langfuse_tracer import LangfuseTracer


class FakeObservation:
    trace_id = "trace-123"

    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeClient:
    def __init__(self):
        self.options = []
        self.observations = []
        self.flushed = False
        self.scores = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.options.append(kwargs)
        observation = FakeObservation()
        self.observations.append(observation)
        yield observation

    def flush(self):
        self.flushed = True

    def create_score(self, **kwargs):
        self.scores.append(kwargs)


def test_disabled_tracer_uses_noop_without_langfuse_client():
    tracer = LangfuseTracer(enabled=False)

    with tracer.observation("test", input_data="private question") as observation:
        observation.update(output="answer")

    assert tracer.enabled is False


def test_content_is_not_captured_by_default():
    client = FakeClient()
    tracer = LangfuseTracer(enabled=True, client=client)

    with tracer.observation(
        "test",
        input_data="private question",
        metadata={"question_id": "42"},
    ):
        pass

    assert "input" not in client.options[0]
    assert client.options[0]["metadata"] == {"question_id": "42"}


def test_content_can_be_enabled_explicitly_and_flushes():
    client = FakeClient()
    tracer = LangfuseTracer(enabled=True, capture_content=True, client=client)

    with tracer.observation("test", input_data="evaluation question"):
        pass
    tracer.flush()

    assert client.options[0]["input"] == "evaluation question"
    assert client.flushed is True


def test_numeric_scores_are_linked_to_trace_and_non_finite_values_are_skipped():
    client = FakeClient()
    tracer = LangfuseTracer(enabled=True, client=client)

    tracer.score("trace-123", "retrieval_precision_at_k", 0.8)
    tracer.score("trace-123", "context_precision", float("nan"))

    assert client.scores == [
        {
            "trace_id": "trace-123",
            "name": "retrieval_precision_at_k",
            "value": 0.8,
            "data_type": "NUMERIC",
        }
    ]


def test_langfuse_start_failure_falls_back_to_noop():
    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("export unavailable")

    tracer = LangfuseTracer(enabled=True, client=BrokenClient())
    with tracer.observation("test") as observation:
        observation.update(metadata={"still_running": True})

    assert tracer.enabled is True