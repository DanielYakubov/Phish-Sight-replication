import glob
import os

import colorgram
import pandas as pd
import pytesseract
from PIL import Image
from selenium import webdriver


def get_colors(img_path, num_colors=8):
    """
    Extract num_colors from an image file, returned as a list of RGB Dictionaries
    """
    colors = colorgram.extract(img_path, num_colors)
    return [c.rgb for c in colors]


def get_text_from_image(img_path):
    """
    Extract text from an image file, returned as an str
    """
    img = Image.open(img_path)
    return pytesseract.image_to_string(img).strip()


if __name__ == "__main__":
    N = int(input('Number of rows to scrape (for testing ONLY):'))
    # https://stackoverflow.com/questions/40555930/selenium-chromedriver-executable-needs-to-be-in-path
    # you will also need chrome for this
    driver = webdriver.Chrome(
        executable_path="../chromedriver_mac_arm64/chromedriver"
    )  # driver initialization

    # going through all the data files
    for data_file in glob.glob("data/links/*"):
        all_texts = []
        all_colors = []  # will be a list of lists
        out_file_prefix = data_file.lstrip("data/links/").rstrip(".csv")
        df = pd.read_csv(data_file)[:N]
        for i, row in df.iterrows():
            URL = f"https://{row[1].strip()}"  # row[0] is the index
            img = "tmp.png"
            driver.get(URL)
            driver.get_screenshot_as_file(img)
            colors = get_colors(img)
            text = get_text_from_image(img)
            all_colors.append(colors)
            all_texts.append(text)
            # deleting tmp file
            os.remove(img)
            # driver.close()

        out_df = pd.DataFrame(
            columns=["colors", "raw_text"], data=zip(all_colors, all_texts)
        )
        out_df.to_csv(
            f"data/scraped/{out_file_prefix}_scraped.csv", index=False
        )

    driver.quit()
