import requests #for making HTTP requests in Python
from bs4 import BeautifulSoup # pulling data from HTML or XML files

query = "deep neural network"
r = requests.get('https://www.google.com/search?q={}'.format(query))
soup = BeautifulSoup(r.text, "html.parser")

links = []
for item in soup.find_all('a'):
    links.append(item.get('href'))

final = []
for item in links:
    str = item[0:7]
    if str=="/url?q=":
        final.append(item)

webpage1 = requests.get('https://www.google.com/' + final[0])
wptext = BeautifulSoup(webpage1.text, "html.parser")
allp = wptext.find_all('p')
text = ""
for item in allp:
    text = text + item.get_text()
print(text)
