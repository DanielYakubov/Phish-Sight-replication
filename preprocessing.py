import glob
import os
from typing import List

import colorgram
import pandas as pd
import pytesseract
from PIL import Image
from selenium import webdriver

BRAND_NAMES = [
    "Microsoft",
    "Netflix",
    "Linkedin",
    "Apple",
    "DocuSign",
    "Zoom",
    "Ring Central",
    "eFax",
    "Xerox",
    "AT&T",
    "Amazon",
    "CVS",
    "Sam’s Club",
    "Walmart",
    "Intuit",
    "American Express Inc",
    "Paypal",
    "Chase Bank",
]


def get_colors(img_path, num_colors=8):
    """
    Extract num_colors from an image file, returned as a list of RGB Dictionaries
    """
    colors = colorgram.extract(img_path, num_colors)
    # In the paper they only care about the dominant color feature vector, not all 8 colors
    dom_proportion_and_color = max(
        [(c.proportion, c.rgb) for c in colors], key=lambda c: c[0]
    )
    dom_color = dom_proportion_and_color[1]
    # each color is encoded in decimal form?
    red, green, blue = (
        float(dom_color.r),
        float(dom_color.g),
        float(dom_color.b),
    )
    return red, green, blue


def get_text_from_image(img_path):
    """
    Extract text from an image file, returned as an str
    """
    img = Image.open(img_path)
    return pytesseract.image_to_string(img).strip()


def get_brand_name(text: str, brand_names: List[str]) -> str:
    """
    Checks if a text contains brand names, if yes, it returns the brand name
    assumes only one brand name
    :param text: str from text read in using OCR on a webpage
    :return: str, the brand name that is present if there is one, else 'no_brand'
    """
    for name in brand_names:
        # very naive, no normalizing text or accounting for OCR errors
        if name in text:
            return name
    return 'no_brand'


if __name__ == "__main__":
    # https://stackoverflow.com/questions/40555930/selenium-chromedriver-executable-needs-to-be-in-path
    # you will also need chrome for this
    driver = webdriver.Chrome(
        executable_path="../chromedriver_mac_arm64/chromedriver"
    )  # driver initialization

    # going through all the data files
    for data_file in glob.glob("data/links/*"):
        URLs, all_texts, brand_names = [], [], []
        reds, greens, blues = [], [], []  # will be a list of lists
        out_file_prefix = data_file.removeprefix("data/links/").removesuffix(
            ".csv"
        )
        df = pd.read_csv(data_file)
        for i, row in df.iterrows():
            URL = f"https://{row[1].strip()}"  # row[0] is the index
            img = "tmp.png"
            driver.get(URL)

            # "algo 1"
            driver.get_screenshot_as_file(img)
            red, green, blue = get_colors(img)

            # "algo 2"
            text = get_text_from_image(img)
            brand_name = get_brand_name(text, BRAND_NAMES)

            # updating lists
            URLs.append(URL)
            reds.append(red)
            greens.append(green)
            blues.append(blue)
            all_texts.append(text)
            brand_names.append(brand_name)

            # deleting tmp file
            os.remove(img)

        # creating the labels
        labels = [out_file_prefix] * len(df)
        out_df = pd.DataFrame(
            columns=[
                "URL",
                "red",
                "green",
                "blue",
                "text",
                "brand_name",
                "label",
            ],
            data=zip(
                URLs, reds, greens, blues, all_texts, brand_names, labels
            ),
        )
        out_df.to_csv(
            f"data/scraped/{out_file_prefix}_scraped.csv", index=False
        )

    driver.quit()
