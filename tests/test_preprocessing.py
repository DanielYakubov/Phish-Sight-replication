import unittest

import pandas as pd

from data.preprocessing import (drop_bad_pages, drop_non_unique_text,
                                drop_nulls_and_duplicates, get_top_level_domain)


class TestPreprocessing(unittest.TestCase):

    def test_drop_nulls_and_duplicates(self):
        toy_data = ['a', 'a', pd.NA]
        df = pd.DataFrame(data=toy_data, columns=['URL'])
        new_df = drop_nulls_and_duplicates(df)
        self.assertEqual(len(new_df), 1)
        self.assertNotIn(pd.NA, new_df)

    def test_drop_non_unique_text(self):
        toy_data = [
            'Shall I compare thee to a summers day?',
            'Shall I compare thee to a summers day?',
            'Thou art more lovely and more temperate'
        ]
        df = pd.DataFrame(data=toy_data, columns=['text'])
        new_df = drop_non_unique_text(df)
        self.assertEqual(len(new_df), 1)
        self.assertIn('Thou art more lovely and more temperate', new_df)

    def test_drop_bad_pages(self):
        toy_data = [
            'Uh-oh 404', 'Internal Server Error detected',
            'Your connection is not private', 'Error'
        ]
        df = pd.DataFrame(data=toy_data, columns=['text'])
        new_df = drop_bad_pages(df)
        self.assertEqual(len(new_df), 1)
        self.assertIn('Error', new_df)

    def test_one_hot_encode(self):
        pass  # TODO implement? Not sure if it makes sense.

    def test_get_tld(self):
        url = 'http://docs.python.org:80/3/library/urllib.parse.html?highlight=params#url-parsing'
        tld = get_top_level_domain(url)
        self.assertEqual(tld, "docs.python.org:80")

    def test_get_url_char_bigram_probs(self):
        pass  # TODO implement (more complicated test needed)

    def test_get_score(self):
        pass  # TODO implement (more complicated test needed)


if __name__ == '__main__':
    unittest.main()
