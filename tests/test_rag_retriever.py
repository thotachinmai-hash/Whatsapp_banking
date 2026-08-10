import unittest

from app.rag.retriever import search


class RagRetrieverTests(unittest.TestCase):
    def test_finds_home_loan_documents(self) -> None:
        results = search("what documents do I need for a home loan")
        self.assertTrue(results)
        self.assertIn("Home Loan", results[0]["title"])

    def test_finds_kyc_info(self) -> None:
        results = search("how does kyc work")
        self.assertTrue(results)
        self.assertTrue(any("KYC" in r["title"] for r in results))

    def test_unrelated_query_returns_nothing(self) -> None:
        # Structurally can't answer this — it isn't in the indexed corpus.
        results = search("who is the ceo of google")
        self.assertEqual(results, [])

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(search(""), [])


if __name__ == "__main__":
    unittest.main()
