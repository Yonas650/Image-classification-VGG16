import os
import requests
from bs4 import BeautifulSoup

# Replace this URL with the actual URL of the website containing pants images
base_url = 'https://www.amazon.ae/s?k=pants+for+men&i=fashion&page=21&crid=15ANUJ29R4MP5&qid=1682966766&sprefix=pant%2Cfashion%2C429&ref=sr_pg_21'

# Define the number of pages you want to scrape (set to 20)
num_pages = 25

# Create a directory to store downloaded images
os.makedirs('pants_images', exist_ok=True)

# Function to download images from a single page
def download_images(url, image_count):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for img in soup.find_all('img'):
        img_url = img['src']
        
        if not img_url.startswith('http'):
            img_url = f"{base_url}/{img_url}"
        
        img_data = requests.get(img_url).content
        with open(f'pants_images/pants_{image_count}.jpg', 'wb') as f:
            f.write(img_data)
        
        image_count += 1

    return image_count

# Loop through the pages and download images
image_count = 1
for page_num in range(21, num_pages + 1):
    # Modify this line to match the URL structure of the website's paginated pages
    page_url = f"{base_url}?page={page_num}"
    
    print(f"Scraping images from page {page_num}")
    image_count = download_images(page_url, image_count)

print("Image scraping completed.")
