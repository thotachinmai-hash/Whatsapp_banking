import unittest

from app.conversation.intent.text_clean import clean_noisy_text


class TextCleanTests(unittest.TestCase):
    def test_strips_trailing_laughter(self) -> None:
        self.assertEqual(
            clean_noisy_text("check my balance ha ha ha he he he"),
            "check my balance",
        )

    def test_strips_lol_and_filler(self) -> None:
        self.assertEqual(
            clean_noisy_text("umm what is the loan interest rate lol"),
            "what is the loan interest rate",
        )

    def test_collapses_repeated_punctuation(self) -> None:
        self.assertEqual(clean_noisy_text("cancel this!!!"), "cancel this!")

    def test_leaves_normal_text_untouched(self) -> None:
        self.assertEqual(clean_noisy_text("transfer 1500 to Riya"), "transfer 1500 to Riya")

    def test_does_not_empty_out_pure_filler_message(self) -> None:
        # If cleaning would remove every word, keep the original rather
        # than handing downstream rules an empty string to guess from.
        self.assertEqual(clean_noisy_text("ha ha ha"), "ha ha ha")

    def test_empty_input(self) -> None:
        self.assertEqual(clean_noisy_text(""), "")


if __name__ == "__main__":
    unittest.main()
