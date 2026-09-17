import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Lightweight stubs so the evaluator module can be imported without the full runtime.
langchain_openai = types.ModuleType("langchain_openai")

class DummyChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

class DummyOpenAIEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

langchain_openai.ChatOpenAI = DummyChatOpenAI
langchain_openai.OpenAIEmbeddings = DummyOpenAIEmbeddings
sys.modules.setdefault("langchain_openai", langchain_openai)

ragas = types.ModuleType("ragas")
ragas.evaluate = lambda *args, **kwargs: None
sys.modules.setdefault("ragas", ragas)

metrics = types.ModuleType("ragas.metrics")
metrics.faithfulness = object()
metrics.answer_relevancy = object()
metrics.context_precision = object()
metrics.context_recall = object()
sys.modules.setdefault("ragas.metrics", metrics)

llms = types.ModuleType("ragas.llms")
class DummyWrapper:
    def __init__(self, *args, **kwargs):
        pass
llms.LangchainLLMWrapper = DummyWrapper
sys.modules.setdefault("ragas.llms", llms)

embeddings = types.ModuleType("ragas.embeddings")
embeddings.LangchainEmbeddingsWrapper = DummyWrapper
sys.modules.setdefault("ragas.embeddings", embeddings)

datasets = types.ModuleType("datasets")
class DummyDataset:
    @staticmethod
    def from_list(records):
        return records

datasets.Dataset = DummyDataset
sys.modules.setdefault("datasets", datasets)

from rag_core.evaluation.evaluator import RAGEvaluator
from rag_core.ingestion.loaders import load_pdf


def test_expected_targets_expand_multiple_pages_and_sources():
    evaluator = RAGEvaluator.__new__(RAGEvaluator)

    qa = {
        "expected_sources": ["Machine Learning Engineering.pdf", "Other.pdf"],
        "expected_pages": "7;8",
        "expected_chunk_ids": "chunk-1,chunk-2",
        "expected_slide": None,
    }

    targets = evaluator._expected_targets(qa)

    assert targets == [
        {"chunk_id": "chunk-1"},
        {"chunk_id": "chunk-2"},
        {"source": "Machine Learning Engineering.pdf", "page": 7},
        {"source": "Machine Learning Engineering.pdf", "page": 8},
        {"source": "Other.pdf", "page": 7},
        {"source": "Other.pdf", "page": 8},
    ]


def test_expected_targets_ignores_empty_and_nan_values():
    evaluator = RAGEvaluator.__new__(RAGEvaluator)

    qa = {
        "expected_sources": ["", None],
        "expected_pages": None,
        "expected_chunk_ids": None,
        "expected_slide": float("nan"),
    }

    assert evaluator._expected_targets(qa) == []


def test_load_pdf_indexes_pages_as_one_based_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%abc\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 18 Tf 72 72 Td (Hello world) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000119 00000 n \n0000000206 00000 n \n0000000305 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF")

    docs = load_pdf(str(pdf_path))

    assert len(docs) == 1
    assert docs[0].metadata["page"] == 1
