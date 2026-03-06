from cocoa.vocabulary import Vocabulary


def test_training_mode_adds_new_words():
    vocab = Vocabulary(("UNK", "BOS", "EOS"))
    assert vocab("hello") == 3
    assert vocab("world") == 4
    assert vocab("hello") == 3  # same word, same token
    assert len(vocab) == 5


def test_frozen_mode_returns_unk():
    vocab = Vocabulary(("UNK", "BOS", "EOS"), is_training=False)
    assert vocab("never_seen") == 0  # UNK index
    assert len(vocab) == 3  # no growth


def test_frozen_mode_without_unk_returns_none():
    vocab = Vocabulary(("a", "b"), is_training=False)
    assert vocab("c") is None


def test_round_trip_yaml():
    vocab = Vocabulary(("UNK", "BOS", "EOS"))
    vocab("alpha")
    vocab("beta")
    restored = Vocabulary.from_yaml(vocab.to_yaml())
    assert len(restored) == len(vocab)
    assert restored("alpha") == vocab("alpha")
    assert restored("beta") == vocab("beta")


def test_contains():
    vocab = Vocabulary(("UNK", "BOS", "EOS"))
    vocab("hello")
    assert "hello" in vocab
    assert "missing" not in vocab


def test_reverse_lookup_consistent():
    vocab = Vocabulary(("UNK",))
    for word in ("cat", "dog", "bird"):
        idx = vocab(word)
        assert vocab.reverse[idx] == word


def test_getitem_same_as_call():
    vocab = Vocabulary(("UNK",))
    assert vocab["UNK"] == vocab("UNK") == 0
    assert vocab["new"] == vocab("new")
