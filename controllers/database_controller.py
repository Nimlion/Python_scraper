from sqlalchemy import create_engine
import pandas as pd


class DBController:
    def __init__(self):
        self.engine = create_engine('mysql://user:username@localhost/database')

    # create a table in the local database under the big_data using MySQL
    def df_to_sql(self, df):
        df.to_sql(name='assignment_reviews', con=self.engine, if_exists='replace', index=False)

    # Call a stored procedure to retrieve reviews
    def call_procedure(self, limit, score):
        connection = self.engine.raw_connection()
        cursor = connection.cursor()
        cursor.callproc("review_model", [limit, score])
        results = pd.DataFrame(list(cursor.fetchall()))
        cursor.close()
        connection.commit()
        return results
