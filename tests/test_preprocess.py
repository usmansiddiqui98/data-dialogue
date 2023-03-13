import pandas as pd

import src.data.preprocess as preprocess


def test_clean():
    text = preprocess.clean(
        "This is a very healthy dog food. Good for their digestion. Also good for small puppies. "
        "My dog eats her required amount at every feeding."
    )
    assert text == "healthy dog food good digestion also good small puppy dog eats require amount every feeding"


def test_clean_df():
    df = pd.read_csv("../data/raw/reviews.csv").head(10)
    df = preprocess.clean_df(df)
    assert (
        df["Cleaned Text"][5] == "like people mention coffee great taste try different instant coffee one one "
        "good one another one favorites want try though organic pure instant coffee"
    )


def test_clean_csv():
    df = preprocess.clean_csv("../data/raw/reviews.csv")
    assert (
        df["Cleaned Text"][1] == "pleased natural balance dog food dog issue dog food past someone recommend "
        "natural balance grain free since possible allergic grain since switch issue "
        "also helpful different kibble size size dog"
    )
