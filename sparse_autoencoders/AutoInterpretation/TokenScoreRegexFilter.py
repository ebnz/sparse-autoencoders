import re

class TokenScoreRegexFilter:
    def __init__(self, search_regex, token_regex, score_regex):
        self.search_regex = search_regex
        self.token_regex = token_regex
        self.score_regex = score_regex

        self.crop_token_start = 0
        self.crop_token_end = 0
        self.crop_score_start = 0
        self.crop_score_end = 0

    def set_token_cropping(self, crop_token_start, crop_token_end):
        self.crop_token_start = crop_token_start
        self.crop_token_end = crop_token_end

        return self

    def set_score_cropping(self, crop_score_start, crop_score_end):
        self.crop_score_start = crop_score_start
        self.crop_score_end = crop_score_end

        return self

    def match(self, line):
        match_object = re.search(self.search_regex, line)

        if match_object is not None:
            return True
        return False

    def get_token(self, line):
        match_object = re.search(self.token_regex, line)

        if match_object is None:
            print(self.search_regex)
            raise RegexException(f"Token-Regex did not Match on <{line}>")

        token = match_object.group(0)

        if self.crop_token_end == 0:
            return token[self.crop_token_start::]
        return token[self.crop_token_start:(-1) * self.crop_token_end]

    def get_score(self, line):
        match_object = re.search(self.score_regex, line)

        if match_object is None:
            print(self.search_regex)
            raise RegexException(f"Score-Regex did not Match on <{line}>")

        score = match_object.group(0)

        if self.crop_score_end == 0:
            return score[self.crop_score_start::]
        return score[self.crop_score_start:(-1) * self.crop_score_end]

    def bulk_get_tokens_scores(self, lines):
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

        return score_sum / len(score_range.split("-"))


class RegexException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

