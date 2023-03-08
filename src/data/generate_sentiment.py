import random

import pandas as pd

# Define review categories and adjectives
categories = ["food", "service", "ambiance", "price"]
adjectives = ["amazing", "terrible", "mediocre", "expensive", "cheap", "friendly", "rude"]


# Define function to generate random reviews
def generate_review():
    category = random.choice(categories)
    rating = random.randint(1, 5)
    adjective1, adjective2 = random.sample(adjectives, k=2)
    review = (
        f"I went to the {category} and it was {adjective1}. The {adjective2} staff made the experience {rating} stars!"
    )
    return review


# Define main function
def main():
    # Generate test data
    data = {"reviews": [generate_review() for _ in range(50)]}
    df = pd.DataFrame(data)

    # Save test data as CSV
    df.to_csv("bert_sentiment.csv", index=False)


# Call main function
if __name__ == "__main__":
    main()
