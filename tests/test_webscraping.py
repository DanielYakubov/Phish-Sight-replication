import unittest

from web_scraping import (get_brand_name, get_colors, get_text_from_image,
                          rgb_to_decimal)


class TestWebScraping(unittest.TestCase):

    def test_rgb_to_decimal(self):
        self.assertIsInstance(rgb_to_decimal(255, 255, 255), int)
        self.assertEqual(rgb_to_decimal(255, 255, 255), 16777215)
        self.assertEqual(rgb_to_decimal(132, 23, 90), 8656730)
        self.assertEqual(
            rgb_to_decimal(112, 12, 79),
            7343183)  # purple.png is described as this color (though it's not)

    def test_get_colors(self):
        colors = get_colors("test_files/purple.png", num_colors=1)
        self.assertEqual(colors[0], 7343283)

    def test_get_image_from_text(self):
        img_text = get_text_from_image('test_files/ocr_test.png')
        self.assertEqual(img_text, 'PASS')

    def test_get_brand_name(self):
        brand_names = ["Microsoft", "Apple", "Meta"]

        text1 = "Yesterday I was at the Apple store and thought, hmmm can Microsoft do this?"
        self.assertEqual(get_brand_name(text1, brand_names), "Microsoft")

        text2 = "Yesterday I was at the Apple store and thought, hmmm can Meta do this?"
        self.assertEqual(get_brand_name(text2, brand_names), "Apple")

        text3 = "I think it's time Meta released Facebook2"
        self.assertEqual(get_brand_name(text3, brand_names), "Meta")

        text4 = "Oh Captain! My Captain!"
        self.assertEqual(get_brand_name(text4, brand_names), "no_brand")


if __name__ == "__main__":
    unittest.main()
