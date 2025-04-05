import unittest

from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import TokenScoreRegexFilter, RegexException


class TokenScoreRegexFilterTest(unittest.TestCase):
    true_line = '* "Fix": 10'
    false_line = '- <Fix>: abc'
    filter = (TokenScoreRegexFilter('".+": [0-9]?[0-9]?', '".+":', '[0-9]+\Z')
              .set_token_cropping(1, 2))

    def test_matching(self):
        self.assertEqual(True, self.filter.match(self.true_line))
        self.assertEqual(False, self.filter.match(self.false_line))

    def test_get_token(self):
        self.assertEqual("Fix", self.filter.get_token(self.true_line))
        self.assertRaises(RegexException, lambda: self.filter.get_token(self.false_line))

    def test_get_score(self):
        self.assertEqual("10", self.filter.get_score(self.true_line))
        self.assertRaises(RegexException, lambda: self.filter.get_score(self.false_line))


if __name__ == '__main__':
    unittest.main()
