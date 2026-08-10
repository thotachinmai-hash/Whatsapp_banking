import unittest

from app.workflows.nlu import interpret_confirmation


class InterpretConfirmationTests(unittest.TestCase):
    def test_exact_yes_no(self) -> None:
        self.assertEqual(interpret_confirmation("yes"), "yes")
        self.assertEqual(interpret_confirmation("y"), "yes")
        self.assertEqual(interpret_confirmation("no"), "no")
        self.assertEqual(interpret_confirmation("n"), "no")

    def test_natural_affirm_phrasing(self) -> None:
        for text in ("yeah go ahead", "sure", "sounds good", "please confirm", "that's correct", "ok"):
            self.assertEqual(interpret_confirmation(text), "yes", text)

    def test_natural_deny_phrasing(self) -> None:
        for text in ("nah cancel it", "nope", "that's wrong, cancel this", "don't do it"):
            self.assertEqual(interpret_confirmation(text), "no", text)

    def test_no_problem_idiom_is_affirmative(self) -> None:
        self.assertEqual(interpret_confirmation("no problem, go ahead"), "yes")
        self.assertEqual(interpret_confirmation("no worries"), "yes")

    def test_ambiguous_input_returns_none_rather_than_guessing(self) -> None:
        self.assertIsNone(interpret_confirmation("maybe"))
        self.assertIsNone(interpret_confirmation("what"))
        self.assertIsNone(interpret_confirmation("yeah no dont"))
        self.assertIsNone(interpret_confirmation("123"))
        self.assertIsNone(interpret_confirmation(""))


if __name__ == "__main__":
    unittest.main()
