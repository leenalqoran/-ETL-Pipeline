import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def load_data_to_postgresql(csv_file_path, db_url, table_name):
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Connect to PostgreSQL using SQLAlchemy (to make things easier)
    engine = create_engine(db_url)
    
    # Load DataFrame to PostgreSQL
    df.to_sql(table_name, engine, index=False, if_exists='replace')  # Use 'replace' or 'append'
    print(f"Data from {csv_file_path} loaded into {table_name} successfully.")

def extract_data():
    csv_file_path = 'tfidf_features.csv'  # Path to your CSV file
    db_url = 'postgresql://postgres:Leen#2001@localhost:5432/Arabic-dataset'  # PostgreSQL connection URL
    table_name = 'Arabic articles'  # Table in PostgreSQL
    
    load_data_to_postgresql(csv_file_path, db_url, table_name)

if __name__ == "__main__":
    extract_data()
    
