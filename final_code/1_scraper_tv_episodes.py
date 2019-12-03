import json, requests, time
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool

#This is a scraper to create a csv file of all television show transcripts on 
#springfieldspringfield.co.uk


#creating list of links to all of the pages containing episodes
#the url is the same for all pages just the number changes
page_base = "https://www.springfieldspringfield.co.uk/tv_show_episode_scripts.php"
base = "https://www.springfieldspringfield.co.uk"

page_list = []

for i in range(1, 282):
	url = page_base + "?page=" +str(i)
	page_list.append(url)

#looping through each page and creating a list of urls for every tv show
show_list = []
for page in page_list:
    res = requests.get(page)
    soup = BeautifulSoup(res.content, "lxml")
    show_page = soup.html
    shows = show_page.find_all("a",  {"class":"script-list-item"})
    for show in shows:
    	show_list.append(base + str(show.attrs["href"]))

#looping though each tv show in the list of show urls and creating a list of every episode of each show
episode_list = []
for show in show_list:
	res = requests.get(show)
	soup = BeautifulSoup(res.content, "lxml")
	episode_page = soup.html
	episodes = episode_page.find_all("a", {"class": "season-episode-title"})
	for episode in episodes:
		episode_list.append(base + "/" + str(episode.attrs["href"]))


#creating a dictionary containing the episode transcript, show name and episode name
#some episodes have no data because they have yet to be transcribed and will be filled in
#with NaNs
def episode_text(episode):
	script_dictionary = {}
	res = requests.get(episode)
	soup = BeautifulSoup(res.content, "lxml")
	script_page = soup.html
	try:
		script_dictionary["text"] = script_page.find("div", {"class": "scrolling-script-container"}).text.strip()
	except:
		script_dictionary["text"] = np.nan
	try:
		script_dictionary["episode_name"] = script_page.find("h3").text.strip()
	except:
		script_dictionary["episode_name"] = np.nan
	try:
		script_dictionary["show_name"] = script_page.find("h1").text.strip()
	except:
		script_dictionary["show_name"] = np.nan
	return script_dictionary


#function to run the episode scrape in parallel
def multithread_parallel(episode, threads = 3):
	pool = Pool(threads)
	results = pool.map(episode_text, episode)
	pool.close()
	pool.join()
	return results


#create a final list from the output of the threading function
output_episodes = multithread_parallel(episode_list)
final_list = []
for episode in output_episodes:
	final_list.append(episode)

#save as dataframs and output to csv
df = pd.DataFrame(final_list)

df.to_csv("./data/springfield_scrape.csv", index=False)













