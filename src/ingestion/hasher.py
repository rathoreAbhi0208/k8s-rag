import hashlib


def generate_chunk_id(
    source: str,
    chunk_index: int,
) -> int:
    """Generate a deterministic integer ID for a document chunk."""

    value = f"{source}:{chunk_index}"

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    # Convert the first 8 bytes into a positive integer.
    chunk_id = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    # Milvus INT64 supports values up to 2^63 - 1.
    return chunk_id & ((1 << 63) - 1)