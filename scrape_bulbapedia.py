from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib.parse import urljoin

import bs4
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

index_url = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
base_url = "https://bulbapedia.bulbagarden.net"

# Set up a HTTP request session with retries.

session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.1)))

# Parse the index to get links to all the pokemon pages.
print("Looking up links to all the pokemon pages")
index_soup = bs4.BeautifulSoup(session.get(index_url).text)
name_to_link = {}
tables_soup = index_soup.find_all("table", attrs={"class": "roundy"})
for table_soup in tables_soup:
    table_rows = table_soup.find_all("tr")[1:]
    page_links = [row.find("a") for row in table_rows]
    name_to_link.update({page_link.attrs["title"]: page_link.attrs["href"] for page_link in page_links})
print(f"Found {len(name_to_link)} pokemon")

# Scrape description text for each of the pokemon
def get_pokemon_description(link):
    full_link = urljoin(base_url, link)
    pokemon_soup = bs4.BeautifulSoup(session.get(full_link).text)
    full_description = "".join([x.text for x in pokemon_soup.find_all("p")]).strip()
    
    # Drop the first two lines, which are name and evolution details, not description.
    full_description = full_description.split("\n", maxsplit=2)[-1]

    # Take only up to the first big break.
    initial_description = full_description.split("\n\n", maxsplit=1)[0]
    
    return initial_description

names, links = zip(*name_to_link.items())
with ThreadPool(10) as pool:
    res_iter = pool.imap(get_pokemon_description, links)
    res_iter_with_pbar = tqdm(res_iter, total=len(links), desc="Scaping descriptions", unit="pokemon")
    name_to_description = dict(zip(names, res_iter_with_pbar))
df = pd.Series(name_to_description).to_frame().reset_index()
df.columns = ["name", "description"]
print(f"Here's an excerpt of the data:\n{df.head()}")
out_path = Path(__file__).parent / "all_the_pokemon.csv"
df.to_csv(out_path)
print(f"Saved {out_path}")