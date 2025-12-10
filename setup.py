from setuptools import setup, find_packages

setup(
    name="rag_agentic",
    version="0.1.0",
    description="RAG agentic demo - streamlit + fastapi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
