import json, requests, time, csv
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

#attempting to add multithreading to make scrape go faster
#idea came from here: 
#https://stackoverflow.com/questions/25373167/multithreading-in-python-beautifulsoup-scraping-doesnt-speed-up-at-all
#turning the scrape loops into a function that will then feed into a Pool class that will combine them into one df
#i think?


def scrape_function():
    base_url = "https://www.springfieldspringfield.co.uk"
    res = requests.get(base_url)
    page_show_list = []

    if res.status_code == 200:
        start = time.time()
        #there are 281 pages total
        base = "https://www.springfieldspringfield.co.uk/tv_show_episode_scripts.php"
        for i in range(100,111):
            
            url = base + "?page=" +str(i)
            res = requests.get(url)
            soup = BeautifulSoup(res.content, "lxml")
            episode_page = soup.html
            episodes = episode_page.find_all("a",  {"class":"script-list-item"}) 
            time.sleep(5)
            #print(episodes)
            for episode in episodes:
                episode_url = base_url +str(episode.attrs["href"])
                #print(episode_url)
                res_episode = requests.get(episode_url)
                soup_episode = BeautifulSoup(res_episode.content, "lxml")
                show_page = soup_episode.html
                shows = show_page.find_all("a", {"class": "season-episode-title"})
                time.sleep(5)

                for show in shows:
                    scripts = {}
                    script_url = base_url+"/" + str(show.attrs["href"])
                    #print(script_url)
                    res_show = requests.get(script_url)
                    soup_show = BeautifulSoup(res_show.content, "lxml")
                    script_page = soup_show.html

                    scripts["text"] = script_page.find("div", {"class": "scrolling-script-container"}).text.strip()
                    scripts["episode_name"] = script_page.find("h3").text.strip()
                    scripts["show_name"] = script_page.find("h1").text.strip()
                    #print(scripts)
                    page_show_list.append(scripts)
                    time.sleep(5)
    return page_show_list

if __name__ == "main":

    csv_name = "springfield-scrape-pooled.csv"
    pool = Pool(cpu_count() * 2)
    with open(csv_name, "rb") as f:
        results = pool.map(scrape_function, f)
    with open("springfield-scrape-pooled-test.csv", "ab") as f:
        write_file = csv.writer(f)
        for result in results:
            write_file.writerow(result)

    #script_df = pd.DataFrame(page_show_list)
    #script_df.to_csv("./data/springfield/springfield-scrape-page-" + str(i) + ".csv", index=False)
    #elapsed = round((time.time() - start) / 60, 2)
    #print(f"Page {i} is done and it took {elapsed} minutes")


