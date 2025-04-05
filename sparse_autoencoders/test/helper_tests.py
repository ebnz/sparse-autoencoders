import unittest

from sparse_autoencoders.utils.helper_functions import (calculate_correlation_from_kv_dict,
                                                        apply_dict_replacement,
                                                        remove_leading_asterisk)


class HelperTests(unittest.TestCase):
    """
    calculate_correlation_from_kv_dict
    """
    kv_dict_gt = {
        "a": 5,
        "b": 10,
        "c": 5
    }

    kv_dict_simulated = {
        "a": 5,
        "b": 8,
        "c": 5
    }

    kv_dict_simulated_missing_items = {
        "a": 5,
        "b": 8
    }

    def test_calculate_correlation_from_kv_dict_normal(self):
        correlation = calculate_correlation_from_kv_dict(self.kv_dict_gt, self.kv_dict_simulated)
        self.assertEqual(1.0, correlation)

    def test_calculate_correlation_from_kv_dict_missing_items(self):
        correlation = calculate_correlation_from_kv_dict(self.kv_dict_gt, self.kv_dict_simulated_missing_items)
        self.assertEqual(1.0, correlation)

    def test_calculate_correlation_from_kv_dict_no_gt(self):
        correlation = calculate_correlation_from_kv_dict(self.kv_dict_gt, {})
        self.assertEqual(0.0, correlation)

    def test_calculate_correlation_from_kv_dict_no_simulation(self):
        correlation = calculate_correlation_from_kv_dict({}, self.kv_dict_simulated)
        self.assertEqual(0.0, correlation)

    """
    apply_dict_replacement
    """
    repl_dict = {
        "_": "",
        "<": "",
        ">": "",
        "|": " "
    }

    def test_apply_dict_replacement_empty_str(self):
        self.assertEqual("", apply_dict_replacement("", self.repl_dict))

    def test_apply_dict_replacement_empty_dict(self):
        self.assertEqual("example_str<>|", apply_dict_replacement("example_str<>|", {}))

    def test_apply_dict_replacement_normal(self):
        self.assertEqual("examplestr ", apply_dict_replacement("example_str<>|", self.repl_dict))

    """
    remove_leading_asterisk
    """
    def test_remove_leading_asterisk_no_asterisk(self):
        self.assertEqual(
            "example_line***",
            remove_leading_asterisk("* example_line***")
        )

    def test_remove_leading_asterisk_normal(self):
        self.assertEqual(
            "example_line***",
            remove_leading_asterisk("example_line***")
        )


if __name__ == '__main__':
    unittest.main()
