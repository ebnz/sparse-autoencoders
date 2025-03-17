import re

class TokenScoreRegexFilter:
    def __init__(self, search_regex, token_regex, score_regex):
        """
        Takes a line of LLM-Output and parses the Token and its simulated Activation
        :type search_regex: str
        :type token_regex: str
        :type score_regex: str
        :param search_regex: Regex-String, checking, whether the Filter applies to this line
        :param token_regex: Regex-String, searching for the Token in the line
        :param score_regex: Regex-String, searching for the Score in the line
        """
        self.search_regex = search_regex
        self.token_regex = token_regex
        self.score_regex = score_regex

        self.crop_token_start = 0
        self.crop_token_end = 0
        self.crop_score_start = 0
        self.crop_score_end = 0

    def set_token_cropping(self, crop_token_start, crop_token_end):
        """
        Sets up cropping of a found Token by the token_regex.
        :rtype: TokenScoreRegexFilter
        :type crop_token_start: int
        :type crop_token_end: int
        :param crop_token_start: Number of chars to crop from the beginning of a found Token by the token_regex
        :param crop_token_end: Number of chars to crop from the end of a found Token by the token_regex
        :return: self
        """
        self.crop_token_start = crop_token_start
        self.crop_token_end = crop_token_end

        return self

    def set_score_cropping(self, crop_score_start, crop_score_end):
        """
        Sets up cropping of a found Token by the token_regex.
        :rtype: TokenScoreRegexFilter
        :type crop_score_start: int
        :type crop_score_end: int
        :param crop_score_start: Number of chars to crop from the beginning of a found Score by the score_regex
        :param crop_score_end: Number of chars to crop from the beginning of a found Score by the score_regex
        :return: self
        """
        self.crop_score_start = crop_score_start
        self.crop_score_end = crop_score_end

        return self

    def match(self, line):
        """
        Matches a given line using this Filter
        :rtype: bool
        :type line: str
        :param line: Line to match to
        :return: Boolean, whether the Filter matches a line generally (search_regex)
        """
        match_object = re.search(self.search_regex, line)

        if match_object is not None:
            return True
        return False

    def get_token(self, line):
        """
        Returns the Token, found in the line.
        :rtype: str
        :type line: object
        :param line:
        :return:
        """
        match_object = re.search(self.token_regex, line)

        if match_object is None:
            raise RegexException(f"Token-Regex did not Match on <{line}>")

        token = match_object.group(0)

        if self.crop_token_end == 0:
            return token[self.crop_token_start::]
        return token[self.crop_token_start:(-1) * self.crop_token_end]

    def get_score(self, line):
        """
        Returns the Score, found in the line.
        :rtype: str
        :type line: str
        :param line: Line to search for the Score
        :return: Found Score
        """
        match_object = re.search(self.score_regex, line)

        if match_object is None:
            raise RegexException(f"Score-Regex did not Match on <{line}>")

        score = match_object.group(0)

        if self.crop_score_end == 0:
            return score[self.crop_score_start::]
        return score[self.crop_score_start:(-1) * self.crop_score_end]

    def bulk_get_tokens_scores(self, lines):
        """
        Parse multiple Lines, extract Tokens and Scores from them.
        :rtype: (list[str], list[int])
        :type lines: list[str]
        :param lines: Lines to parse
        :return: List of parsed Tokens and List of parsed Scores
        """
        tokens, scores = [], []
        for line in lines:
            if self.match(line):
                try:
                    token = self.get_token(line)
                    score = self.get_score(line)
                except RegexException:
                    continue
                tokens.append(token)
                scores.append(score)

        return tokens, scores


class TokenScoreRegexFilterAverage(TokenScoreRegexFilter):
    def __init__(self, search_regex, token_regex, score_regex):
        """
        TokenScoreRegexFilter that finds ranges of Scores (e.g. 'The Score may be in the range of 3-5').
        This Filter Averages these Scores.
        :type search_regex: str
        :type token_regex: str
        :type score_regex: str
        :param search_regex: Regex-String, checking, whether the Filter applies to this line
        :param token_regex: Regex-String, searching for the Token in the line
        :param score_regex: Regex-String, searching for the Score in the line
        """
        super().__init__(search_regex, token_regex, score_regex)

    def get_score(self, line):
        score_range = super().get_score(line)

        score_sum = 0

        for score in score_range.split("-"):
            try:
                score_val = int(score)
                score_sum += score_val
            except ValueError:
                print(f"WARN: <{score}> can't be casted to int. Value can't be casted")
                continue
            except TypeError:
                print(f"WARN: <{score}> can't be casted to int. Type can't be casted")
                continue

        return str(score_sum / len(score_range.split("-")))


class RegexException(Exception):
    def __init__(self, *args):
        """
        Regex could not be matched to a line.
        """
        super().__init__(*args)

