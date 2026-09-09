from harness_ladder.retriever import Chunk, load_corpus, render_context, retrieve


def test_retrieve_ranks_matching_filename_and_terms():
    chunks = [
        Chunk("repo/config.toml", "port = 8088"),
        Chunk("repo/README.md", "worker retries failed jobs three times"),
    ]
    found = retrieve("What port is in config.toml?", chunks, top_k=1)
    assert [item.source for item in found] == ["repo/config.toml"]


def test_retrieve_is_bounded_and_deterministic():
    chunks = [Chunk("a", "alpha beta"), Chunk("b", "alpha beta"), Chunk("c", "alpha")]
    assert [item.source for item in retrieve("alpha beta", chunks, top_k=2)] == ["a", "b"]
    assert retrieve("unmatched", chunks) == []
    assert retrieve("alpha", chunks, top_k=0) == []


def test_corpus_loading_and_rendering():
    chunks = load_corpus("corpus/task_passages.md")
    assert len(chunks) == 5
    rendered = render_context(retrieve("cache DEFAULT_TTL", chunks, top_k=1))
    assert "DEFAULT_TTL = 900" in rendered
    assert "scratch work" in rendered
