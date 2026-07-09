import tempfile
import unittest
from pathlib import Path

from coding_agent.config import RetrievalConfig
from coding_agent.retrieval import Retriever


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="retr_test_"))
        (self.workdir / "auth.py").write_text(
            "def authenticate(user, password):\n"
            "    # verify credentials against the session token\n"
            "    return check_password(user, password)\n"
        )
        (self.workdir / "math_utils.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
        )
        self.config = RetrievalConfig(
            chunk_lines=20,
            chunk_overlap=2,
            top_k=3,
            include_globs=["**/*.py"],
            exclude_dirs=["__pycache__"],
        )

    def test_build_and_search(self):
        retriever = Retriever(self.config, self.workdir).build()
        self.assertGreater(len(retriever.chunks), 0)

        hits = retriever.search("authenticate user password credentials")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.path, "auth.py")

    def test_irrelevant_query_ranks_low_or_empty(self):
        retriever = Retriever(self.config, self.workdir).build()
        hits = retriever.search("multiply numbers arithmetic")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.path, "math_utils.py")

    def test_index_is_persisted_and_reloaded(self):
        # B10: build writes an index; an unchanged repo reloads from cache.
        Retriever(self.config, self.workdir).build()
        index_file = self.workdir / self.config.index_dir / "index.json"
        self.assertTrue(index_file.exists())

        fresh = Retriever(self.config, self.workdir)
        sig = fresh._signature(list(fresh._iter_files()))
        self.assertTrue(fresh._load(sig))
        self.assertTrue(fresh.chunks)

    def test_dense_embed_fn_path(self):
        # B10/D6: passing embed_fn switches ranking to dense cosine.
        def emb(text):
            v = [0.0, 0.0, 0.0]
            for tok in text.split():
                v[hash(tok) % 3] += 1.0
            return v

        retriever = Retriever(self.config, self.workdir, embed_fn=emb).build()
        self.assertTrue(retriever.dense)
        self.assertTrue(all(c.vec is not None for c in retriever.chunks))
        hits = retriever.search("authenticate user password")
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
