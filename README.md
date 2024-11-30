# Arabic Text Preprocessing and Analysis Pipeline

## Overview
This project focuses on building a complete pipeline for web scraping, preprocessing, and analyzing Arabic text data. It includes three main stages:
1. **Extract:** Web scraping Arabic articles and storing them in a structured format.
2. **Transform:** Preprocessing the text data, including cleaning and feature extraction using TF-IDF.
3. **Load:** Storing the processed features into a PostgreSQL database.

## Features
- Scrape Arabic articles from specified URLs.
- Perform extensive preprocessing on Arabic text:
  - Remove stop words, punctuation, tashkeel, URLs, and numbers.
  - Handle HTML symbols, duplicate articles, and multiple whitespaces.
  - Extract Arabic-only sentences.
- Extract meaningful features using TF-IDF.
- Load preprocessed data into a PostgreSQL database.

---

## Files in the Repository
1. **`extract.py`:**  
   - Scrapes Arabic articles from a given URL.
   - Saves URLs and articles in a CSV file.

2. **`transform.py`:**  
   - Reads the scraped articles.
   - Preprocesses and cleans the text using custom functions for Arabic language handling.
   - Applies TF-IDF vectorization and saves the features in a CSV file.

3. **`load.py`:**  
   - Loads the processed features into a PostgreSQL database.

---

## Installation and Setup
### Prerequisites
- Python 3.8+
- PostgreSQL
- Libraries: `requests`, `BeautifulSoup`, `pandas`, `scikit-learn`, `sqlalchemy`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Arabic-Text-Pipeline.git
   cd Arabic-Text-Pipeline
2. Install the Required Python Packages
     ```bash
     pip install -r requirements.txt
3. Configure PostgreSQL
a. Create a PostgreSQL Database
Create a new database in PostgreSQL for storing the processed articles.
b. Update the Connection String
Edit the load.py file to match your PostgreSQL database credentials:
     ```bash
     db_url = 'postgresql://<username>:<password>@localhost:5432/<database>'
Replace:

<username>: Your PostgreSQL username.
<password>: Your PostgreSQL password.
<database>: The name of the database you created.

### Usage
# 1. Extract Data
Run the extract.py script to scrape and save Arabic articles to a CSV file:
   ```bash
python extract.py
# 2. Transform Data
Run the transform.py script to preprocess the scraped articles and generate TF-IDF features:
   ```bash
python transform.py
# 3. Load Data
Run the load.py script to load the TF-IDF features into PostgreSQL:
   ```bash
python load.py
### File Descriptions
extract.py: Contains the web scraping logic to extract Arabic articles and save them in Arabic_dataset.csv.
transform.py: Preprocesses the scraped data and performs TF-IDF vectorization, saving the results in tfidf_features.csv.
load.py: Loads the processed data from tfidf_features.csv into a PostgreSQL database.
### Data Pipeline Overview
Extraction: Web scraping using BeautifulSoup to fetch article titles and content.
Transformation: Preprocessing text to clean and prepare data for analysis, followed by feature extraction using TfidfVectorizer.
Loading: Storing the processed data in PostgreSQL for downstream tasks.
     
     

