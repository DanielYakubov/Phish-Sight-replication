import os
import re
from typing import List

import colorgram
import pandas as pd
import pytesseract
from PIL import Image
from selenium import webdriver
from tqdm.auto import tqdm

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
    driver.set_page_load_timeout(10)

    # going through all the data files
    data_file = 'data/links/all_data_links.csv'
    URLs, all_texts, brand_names = [], [], []
    reds, greens, blues = [], [], []
    statuses = []
    df = pd.read_csv(data_file)
    progress_bar = tqdm(range(len(df)))
    for i, row in df.iterrows():
        if not re.match(r'https?://', row[0]):
            URL = f"https://{row[0].strip()}"
        else:
            URL = row[0]
        img = "tmp.png"
        try:
            driver.get(URL)
        except Exception as e:
            # Any exception, we just want to continue
            print(f"{URL} caused an exception {e}, skipping...")
            continue

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
        statuses.append(row[1])

        # deleting tmp file
        os.remove(img)

        # creating the df (doing this for each loop, we want to save csv even in case of early stop)
        out_df = pd.DataFrame(
            columns=[
                "URL",
                "red",
                "green",
                "blue",
                "text",
                "brand_name",
                "status",
            ],
            data=zip(
                URLs, reds, greens, blues, all_texts, brand_names, statuses
            ),
        )
        out_df.to_csv(
            f"data/scraped/all_data_scraped.csv", index=False
        )
        progress_bar.update(1)

    driver.quit()
