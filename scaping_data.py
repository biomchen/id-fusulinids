from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv

name_url = 'https://www.eecs.mit.edu/people/faculty-advisors'
data_url = 'https://arxiv.org'

html =  urlopen(name_url)
soup =  BeautifulSoup(html)

# scrape the faculty names of the EECS department
name_all =  []
tags =  soup('br')

for n in range(len(tags)):
    name_temp = tags[n].previous_sibling
    name_all.append(name_temp)

# open csv for saving the scraping data
with open('abstract_data.csv', mode='w') as csvFile:
    features = ['name', 'id', 'url', 'abstract']
    writer = csv.DictWriter(csvFile, fieldnames = features)
    writer.writeheader()

    # clean up the faculty names that scaped from MIT website
    for name in name_all:
        name_split =  name.split(' ')
        if len(name_split)!= 2:
            continue
        faculty_name = name_split[1] + '_' + name_split[0]
        
        # scraping the abstract contnet from abstact links
        search_page  = urlopen(data_url + "/find/all/1/au:+" + faculty_name + "/0/1/0/all/0/1?per_page=50")
        soup = BeautifulSoup(search_page)
        
        for title in soup(search_page, title = 'Abstract'):
            id =  title.contents[0]
            abstractUrl = data_url + title['href']
            abstractPage = urlopen(abstractUrl)
            abstract = BeautifulSoup(abstractPage)
            abstract_content = abstract.find('blockquote', {'class': 'abstract'}).text.strip()

            writer.writerow({'name': name, 'id': id, 'url': abstractUrl, 'abstract':abstract_content})

print('Finished :')
