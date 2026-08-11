import unittest

from app.workflows.nlu import interpret_confirmation, interpret_menu_choice


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

    def test_llm_fallback_only_consulted_on_rule_miss(self) -> None:
        calls = []

        def fallback(text):
            calls.append(text)
            return "yes"

        # Rule already resolves "yes" — fallback must not be called.
        self.assertEqual(interpret_confirmation("yes", llm_fallback=fallback), "yes")
        self.assertEqual(calls, [])

        # Rule returns None ("maybe") — fallback is consulted.
        self.assertEqual(interpret_confirmation("maybe", llm_fallback=fallback), "yes")
        self.assertEqual(calls, ["maybe"])

    def test_llm_fallback_exception_is_swallowed(self) -> None:
        def fallback(text):
            raise RuntimeError("boom")

        self.assertIsNone(interpret_confirmation("maybe", llm_fallback=fallback))

    def test_llm_fallback_invalid_value_is_ignored(self) -> None:
        self.assertIsNone(interpret_confirmation("maybe", llm_fallback=lambda text: "banana"))


class InterpretMenuChoiceTests(unittest.TestCase):
    def test_exact_match_case_insensitive(self) -> None:
        self.assertEqual(interpret_menu_choice("2", ["1", "2", "3"]), "2")
        self.assertEqual(interpret_menu_choice("Priya", ["Priya", "Amit"]), "Priya")

    def test_no_match_without_fallback_returns_none(self) -> None:
        self.assertIsNone(interpret_menu_choice("the second one", ["1", "2", "3"]))

    def test_llm_fallback_resolves_free_text(self) -> None:
        result = interpret_menu_choice(
            "the second one", ["1", "2", "3"], llm_fallback=lambda text: "2"
        )
        self.assertEqual(result, "2")

    def test_llm_fallback_result_must_be_a_valid_option(self) -> None:
        result = interpret_menu_choice(
            "the second one", ["1", "2", "3"], llm_fallback=lambda text: "5"
        )
        self.assertIsNone(result)

    def test_empty_options_returns_none(self) -> None:
        self.assertIsNone(interpret_menu_choice("1", []))


if __name__ == "__main__":
    unittest.main()
