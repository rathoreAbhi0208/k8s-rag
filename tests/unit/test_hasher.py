from src.ingestion.hasher import generate_chunk_id


def test_chunk_id_is_deterministic():
    first = generate_chunk_id(
        "components.md",
        0,
    )

    second = generate_chunk_id(
        "components.md",
        0,
    )

    assert first == second


def test_different_chunk_indexes_produce_different_ids():
    first = generate_chunk_id(
        "components.md",
        0,
    )

    second = generate_chunk_id(
        "components.md",
        1,
    )

    assert first != second


def test_different_sources_produce_different_ids():
    first = generate_chunk_id(
        "components.md",
        0,
    )

    second = generate_chunk_id(
        "architecture.md",
        0,
    )

    assert first != second


def test_chunk_id_is_positive_int64():
    chunk_id = generate_chunk_id(
        "components.md",
        0,
    )

    assert isinstance(chunk_id, int)
    assert 0 <= chunk_id <= (2**63 - 1)


def test_same_input_always_produces_same_id():
    results = [
        generate_chunk_id(
            "components.md",
            5,
        )
        for _ in range(10)
    ]

    assert len(set(results)) == 1