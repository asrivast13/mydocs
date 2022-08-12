import re
import requests
from bs4 import BeautifulSoup

base_url = 'https://media.talkbank.org/ca/CallHome'
lang     = 'eng'

vgm_url = base_url + "/" + lang
html_text = requests.get(vgm_url).text
soup = BeautifulSoup(html_text, 'html.parser')

for link in soup.find_all('a'):
    fname = link.get('href')
    if re.match('^\d+\.mp3$', fname):
        audio = requests.get(vgm_url+"/"+fname)
        with open(fname, "wb") as f:
            f.write(audio.content)
        print(fname)

