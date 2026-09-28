"""Guards against train/serve skew: our hand-rendered prompt must equal the official chat template."""
from transformers import AutoTokenizer

from batlm import rag
from batlm.prompt import BASE_MODEL, chat, render


def test_render_matches_chat_template():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    hits = [c for _, c in rag.load().search("Who is the Joker?")]
    assert render("Who is the Joker?", hits) == tok.apply_chat_template(
        chat("Who is the Joker?", hits), tokenize=False, add_generation_prompt=True)


def test_bpe_roundtrip():
    from batlm.tokenizer import BPETokenizer
    tok = BPETokenizer.train("Batman and Robin fight the Joker in Gotham. " * 20, vocab_size=300)
    s = "Gotham's Dark Knight — 1939!"
    assert tok.decode(tok.encode(s)) == s
