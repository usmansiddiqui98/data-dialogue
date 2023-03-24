import os

from src.data.preprocess import Preprocessor

fname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews.csv"))


def test_clean():
    text = Preprocessor.clean_sentence(
        "This is a very healthy dog food. Good for their digestion. Also good for small puppies. "
        "My dog eats her required amount at every feeding."
    )
    assert text == "healthy dog food good digestion also good small puppy dog eats require amount every feeding"


def test_clean_csv():
    preprocessor = Preprocessor(fname)
    preprocessor.clean_csv()
    df = preprocessor.clean_df
    assert (
        df["cleaned_text"][1] == "pleased natural balance dog food dog issue dog food past someone recommend "
        "natural balance grain free since possible allergic grain since switch issue "
        "also helpful different kibble size size dog"
    )
