from src.ingestion.loaders.markdown_loader import MarkdownLoader


def test_load_markdown():
    loader = MarkdownLoader()

    file_path = (
        "knowledge-base/kubernetes/"
        "content/en/docs/concepts/overview/components.md"
    )

    document = loader.load(file_path)

    assert document["content"]
    assert document["metadata"]["filename"] == "components.md"
    assert document["metadata"]["file_type"] == ".md"