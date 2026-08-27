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

    # ─── new content, added alongside the RAG content enhancement ──────

    def test_finds_loan_process_and_emi_info(self) -> None:
        results = search("how does the loan application process work")
        self.assertTrue(results)
        self.assertIn("Loan Application", results[0]["title"])

        emi_results = search("what is EMI")
        self.assertTrue(emi_results)
        self.assertIn("EMI", emi_results[0]["title"])

    def test_finds_transfer_limit_and_fee_info(self) -> None:
        results = search("is there a fee to transfer money")
        self.assertTrue(results)
        self.assertIn("Money Transfer", results[0]["title"])

    def test_finds_beneficiary_info(self) -> None:
        results = search("what is a beneficiary")
        self.assertTrue(results)
        self.assertIn("Beneficiaries", results[0]["title"])

    def test_finds_transaction_category_info(self) -> None:
        results = search("what categories can I filter my transactions by")
        self.assertTrue(results)
        self.assertIn("Transactions", results[0]["title"])

    def test_finds_security_info(self) -> None:
        results = search("will you ever ask me for my PIN or OTP")
        self.assertTrue(results)
        self.assertTrue(any("Security" in r["title"] or "OTP" in r["title"] for r in results))

    def test_finds_services_not_offered_info(self) -> None:
        results = search("can I withdraw cash from an ATM")
        self.assertTrue(results)
        self.assertIn("ATM", results[0]["title"])

    def test_finds_cheque_status_meaning(self) -> None:
        results = search("what does pending mean for my cheque")
        self.assertTrue(results)
        self.assertIn("Status", results[0]["title"])

    def test_finds_fixed_deposit_not_offered(self) -> None:
        # FD/DD are not offered by any tool/workflow in this app -- the
        # correct RAG answer is "not offered", never a fabricated "how FDs
        # work" explanation implying the service exists.
        results = search("can I open a fixed deposit")
        self.assertTrue(results)
        self.assertIn("Fixed Deposit", results[0]["title"])

    def test_finds_demand_draft_not_offered(self) -> None:
        results = search("how do I get a demand draft")
        self.assertTrue(results)
        self.assertIn("Demand Draft", results[0]["title"])

    def test_finds_account_closure_not_offered(self) -> None:
        results = search("can I close my account")
        self.assertTrue(results)
        self.assertTrue(any("Closing An Account" in r["title"] for r in results))

    def test_finds_transfer_mode_clarification(self) -> None:
        results = search("is this NEFT or RTGS")
        self.assertTrue(results)
        self.assertIn("NEFT", results[0]["title"])

    def test_finds_passbook_as_transaction_history(self) -> None:
        results = search("can I get my passbook")
        self.assertTrue(results)
        self.assertIn("Passbook", results[0]["title"])

    def test_finds_no_minimum_balance_or_interest_info(self) -> None:
        results = search("what is the interest rate on my savings account")
        self.assertTrue(results)
        self.assertIn("Interest On Savings", results[0]["title"])

    def test_finds_wrong_account_transfer_info(self) -> None:
        results = search("I sent money to the wrong account number by mistake")
        self.assertTrue(results)
        self.assertIn("Wrong Account", results[0]["title"])

    def test_no_dollar_figures_invented_for_loan_rates(self) -> None:
        # RAG must never carry loan rate/fee/limit numbers -- those come
        # from get_loan_product_info (DB), not indexed documents, so a
        # rate change never needs a documents update in two places.
        results = search("what is the interest rate on a personal loan", top_k=5)
        for r in results:
            self.assertNotIn("%", r["text"])


if __name__ == "__main__":
    unittest.main()
