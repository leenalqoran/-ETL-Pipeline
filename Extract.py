import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

# Function to find URLs on a given webpage
def find_urls(url):
    try:
        response = requests.get(url)
        response.encoding = 'utf-8-sig'  # Set the encoding for Arabic content
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        for a_tag in soup.find_all('a'):
            href = a_tag.get('href')
            if href:
                urls.append(href)
        return urls
    except:
        return []

# Specify the website URL
website_url = 'https://mawdoo3.com/%D8%AA%D8%B5%D9%86%D9%8A%D9%81:%D8%B1%D9%8A%D8%A7%D8%B6%D8%A7%D8%AA_%D9%85%D9%86%D9%88%D8%B9%D8%A9'  # Replace with the desired website URL

# Find URLs on the website
urls = find_urls(website_url)

# Specify the file path
file_path = "mawdoo3.csv"

# Open the file in write mode and create a CSV writer
with open('mawdoo3.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows([[f"https://mawdoo3.com{url}"] for url in urls])

print(f"URLs written to {file_path} successfully.")


# Read the list of URLs from the CSV file
url_file = 'mawdoo3.csv'

# Create a CSV file to store the scraped data
output_file = 'Arabic_dataset.csv'

# Define the header for the CSV file
header = ['Title', 'Article']

# Function to scrape data from a given URL
def scrape_data(url):
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'  # Set the encoding for Arabic content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the title
        title = soup.title.text.strip()
        # Extract the article content
        article = ''
        article_tags = soup.find_all("p")
 # Update this based on the HTML structure of the articles
        for tag in article_tags:
            article += tag.text.strip() + ' '
        return title, article
    except:
        return None, None

# Open the CSV file to write the scraped data
with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Read the URLs from the CSV file and scrape data for each URL
    with open(url_file, 'r', encoding='utf-8-sig') as url_csv:
        reader = csv.reader(url_csv)
        for row in reader:
            url = row[0]
            # Scrape data from the URL
            title, article = scrape_data(url)
            if title and article:
                # Write the scraped data to the CSV file
                writer.writerow([title, article])
                print(f"Scraped data for {url} successfully.")
            else:
                print(f"Failed to scrape data for {url}.")

print("Web scraping and data saving completed.")
