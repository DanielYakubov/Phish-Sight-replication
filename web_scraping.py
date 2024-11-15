import os
import re
from multiprocessing import Process
from typing import List

import colorgram
import pandas as pd
import pytesseract
from PIL import Image
from selenium import webdriver
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from wrapt_timeout_decorator import timeout

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


def rgb_to_decimal(r, g, b):
    """converting an rgb value to a base10 value"""
    return (r << 16) + (g << 8) + b

def get_colors(img_path, num_colors=8):
    """
    Extract num_colors from an image file, returned as a list of RGB Dictionaries
    """
    colors = colorgram.extract(img_path, num_colors)
    colors = [(rgb_to_decimal(c.rgb.r, c.rgb.g, c.rgb.b)) for c in colors]
    output_colors = [0] * 8
    for i, c in enumerate(colors):
        output_colors[i] = c

    return output_colors


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


@timeout(60, timeout_exception=TimeoutError)
def run_algos(driver, img):

    # selenium gets stuck, will need to force timeouts sometimes
    driver.get(URL)

    # "algo 1"
    driver.get_screenshot_as_file(img)
    colors = get_colors(img)

    # "algo 2"
    text = get_text_from_image(img)
    brand_name = get_brand_name(text, BRAND_NAMES)

    return colors, text, brand_name


if __name__ == "__main__":
    # https://stackoverflow.com/questions/40555930/selenium-chromedriver-executable-needs-to-be-in-path
    # you will also need chrome for this
    time_out_secs = 60
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(time_out_secs)
    driver.set_page_load_timeout(time_out_secs)

    # going through all the data files
    data_file = 'data/links/additional_phishing.csv'
    URLs, all_texts, brand_names = [], [], []
    color1s, color2s, color3s, color4s, color5s, color6s, color7s, color8s = [], [], [], [], [], [], [], []
    statuses = []
    df = pd.read_csv(data_file)
    progress_bar = tqdm(range(len(df)))
    for i, row in df.iterrows():
        try:
            if not re.match(r'https?://', row[0]):
                URL = f"https://{row[0].strip()}"
            else:
                URL = row[0]

            img = "tmp.png"
            colors, text, brand_name = run_algos(driver, img)

        except Exception as e:
            # Any exception, we just want to continue
            print(f"{URL} caused an exception {e}, skipping...")
            driver.close()
            driver.quit()

            driver = webdriver.Chrome(ChromeDriverManager().install())  # driver initialization
            driver.implicitly_wait(time_out_secs)
            driver.set_page_load_timeout(time_out_secs)
            progress_bar.update(1)
            continue

        # Not graceful, but explicit
        color1, color2, color3, color4, color5, color6, color7, color8 = colors

        # updating lists
        URLs.append(URL)
        color1s.append(color1)
        color2s.append(color2)
        color3s.append(color3)
        color4s.append(color4)
        color5s.append(color5)
        color6s.append(color6)
        color7s.append(color7)
        color8s.append(color8)
        all_texts.append(text)
        brand_names.append(brand_name)
        statuses.append(row[1])

        # deleting tmp file
        os.remove(img)

        # creating the df (doing this for each loop, we want to save csv even in case of early stop)
        out_df = pd.DataFrame(
            columns=[
                "URL",
                "color1",
                "color2",
                "color3",
                "color4",
                "color5",
                "color6",
                "color7",
                "color8",
                "text",
                "brand_name",
                "status",
            ],
            data=zip(
                URLs, color1s, color2s, color3s, color4s, color5s, color6s, color7s, color8s, all_texts, brand_names, statuses
            ),
        )
        out_df.to_csv(
            f"data/scraped/all_data_scraped2.csv", index=False
        )
        progress_bar.update(1)
    driver.quit()

