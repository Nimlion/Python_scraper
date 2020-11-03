import pandas as pd

from controllers.database_controller import DBController
from data.created_reviews import CreatedReviews
from data_analyser import DataAnalyser
from data_cleaner import DataCleaner
from scrapers.agoda_scraper import AgodaScraper
from scrapers.booking_scraper import BookingScraper

if __name__ == '__main__':
    # Read the given hotel reviews
    csv_df = pd.read_csv('data/hotel_Reviews.csv')

    # Retrieve my own written reviews DataFrame
    created_df = CreatedReviews.created_df

    # Scrape the booking.com reviews and retrieve them
    booking_df = BookingScraper.scrape_booking()

    # Scrape the booking.com reviews and retrieve them
    agoda_df = AgodaScraper.scrape_agoda()

    # Combining all DataFrames
    completed_df = pd.concat([csv_df, created_df, booking_df, agoda_df], ignore_index=True)

    # completed_df.isnull().sum()

    # Drop all rows with nan values
    completed_df = completed_df.dropna()

    # Initiate a new DataCleaner Object
    cleaner = DataCleaner()

    # Clean the positive and negative reviews
    cleaner.string_cleaner(completed_df, 'Positive_Review')
    cleaner.string_cleaner(completed_df, 'Negative_Review')

    # Create an DB controller
    DBCon = DBController()

    # Send the dataframe to the database
    DBCon.df_to_sql(completed_df)

    # Query the database using a stored procedure
    db_reviews = DBCon.call_procedure(10000, 0)
    db_reviews.columns = ['hotel_name', 'average_score', 'review_score', 'positive', 'negative']

    # Making all positive and negative reviews lowercase
    db_reviews['positive'] = db_reviews['positive'].str.lower()
    db_reviews['negative'] = db_reviews['negative'].str.lower()

    # Group the response on hotel
    grouped = db_reviews.groupby('hotel_name').mean()

    # Initiate a new DataAnalyser object
    analyser = DataAnalyser()

    # Detect some languages
    analyser.detect_langs(db_reviews[:500])

    # Split the reviews into positive and negative reviews
    split_reviews = analyser.extract_reviews(db_reviews)
    split_reviews.columns = ['hotel', 'review', 'rating']

    # Generate a WordCloud for positive and negative reviews
    analyser.word_cloud(split_reviews.loc[split_reviews['rating'] == 1], "Positive")
    analyser.word_cloud(split_reviews.loc[split_reviews['rating'] == 0], "Negative")

    # Stem the split reviews
    split_reviews = cleaner.stemmed(split_reviews)

    # Predict the LR, NB and RF score
    analyser.lr_score(split_reviews)
    analyser.nb_score(split_reviews)
    analyser.rf_score(split_reviews)

    # Live testing
    analyser.test_sentence("A nightmare, disaster hotel")
