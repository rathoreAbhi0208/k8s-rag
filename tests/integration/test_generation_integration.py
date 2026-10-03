from src.generation.generator import Generator
from src.generation.ollama_client import OllamaClient
from src.generation.prompts import PromptBuilder


def test_real_generation():
    ollama_client = OllamaClient(
        model="llama3.2:3b",
    )

    prompt_builder = PromptBuilder()

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
    )

    results = [
        {
            "heading": "Control Plane Components",
            "content": (
                "The Kubernetes control plane manages the overall "
                "state of the cluster. Its components include "
                "kube-apiserver, etcd, kube-scheduler, "
                "kube-controller-manager, and optionally "
                "cloud-controller-manager."
            ),
        }
    ]

    query = "What are the components of the Kubernetes control plane?"

    answer = generator.generate(
        query=query,
        results=results,
    )

    assert answer
    assert isinstance(answer, str)

    print()
    print("=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)
    print(answer)
    print("=" * 80)