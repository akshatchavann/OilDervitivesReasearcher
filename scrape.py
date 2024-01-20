import requests
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_and_save_text(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed

    # Parse the content of the response with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the webpage title (if available)
    title = soup.title.string if soup.title else 'No Title'

    # Extract all text from <p> tags
    paragraphs = soup.find_all('p')
    text = '\n'.join([para.get_text() for para in paragraphs])

    # Combine the title, URL, timestamp, and extracted text
    full_text = f"Title: {title}Scraped at: {datetime.now()}\n{text}"

    # Save the extracted text to a .txt file
    with open(filename, 'w') as file:
        file.write(full_text)

    return filename

# Example usage
url = 'https://www.mosaicml.com/blog/mpt-7b'
filename = 'output.txt'
path_to_file = scrape_and_save_text(url, filename)
