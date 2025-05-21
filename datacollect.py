import os
import random
import signal
import subprocess
import time
from collections import Counter
from time import sleep
import netifaces
import pyshark
from envs.python36.Lib.symbol import pass_stmt
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium import webdriver
from selenium.webdriver.common.by import By
def readDomains(path="./data/top-1000000-domains"):
    with open(path,encoding="utf8") as f:
        return [item.strip() for item in f.readlines()]

def openBrowser(num,domainnames,domain_table):
    proxy_ip = "127.0.0.1"
    proxy_port = "7897"
    proxy = Proxy()
    proxy.proxy_type = ProxyType.MANUAL
    proxy.http_proxy = f"{proxy_ip}:{proxy_port}"
    proxy.ssl_proxy = f"{proxy_ip}:{proxy_port}"
    options = webdriver.ChromeOptions()
    options.add_argument('--incognito')  
    options.add_argument('--proxy-server=http://{}:{}'.format(proxy_ip, proxy_port))
    options.add_argument("--disable-blink-features=AutomationControlled")  
    options.add_argument('--headless')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)
    interface = "Adapter for loopback traffic capture"
    tab_index=[str(domain_table[item]) for item in domainnames]
    tab_index="-".join(tab_index)
    tab_index=f"[{num}]"+tab_index
    output_file = f"./res/{tab_index}.pcap"
    capture_filter = "tcp port 7897"
    try:
        capture_process = start_capture(interface, output_file, capture_filter)
        for d_i,domain in enumerate(domainnames):
            driver.get("https://"+domain)
            first_window = driver.current_window_handle
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])
            random_delay = random.uniform(0, 2)
            sleep(random_delay)
        sleep(5)
        driver.quit()
    except Exception as e:
        print(e)
    finally:
        stop_capture(capture_process)

def start_capture(interface, output_file, capture_filter=None):
    command = [
        r"C:\Program Files\Wireshark\tshark",
        "-i", interface,  
        "-w", output_file,  
    ]
    if capture_filter:
        command.extend(["-f", capture_filter])  

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def stop_capture(process):
    process.terminate()
    process.wait()  

def collectData():
    domains = readDomains()
    domains_ids = dict()
    for domain in domains:
        if domain not in domains_ids:
            domains_ids[domain] = len(domains_ids)

    for i in range(len(domains)):
        labelnum=random.randint(2, 10) 
        number_range = list(range(0, i)) 
        if labelnum >i-1:
            index_select = [j for j in range(i)]
        else:
            index_select = random.sample(number_range, labelnum)
            index_select.append(i)
            index_select.sort()
        num = i
        openBrowser(num,[domains[k] for k in index_select], domains_ids)

def collectEachData():
    domains = readDomains()
    domains_ids = dict()
    for domain in domains:
        if domain not in domains_ids:
            domains_ids[domain] = len(domains_ids)

    for i in range(len(domains)):
        index_select=[i]
        num = i
        openBrowser(num, [domains[k] for k in index_select], domains_ids)

if __name__ == '__main__':

    # 1、mult
    # collectData()
    # 2、single
    collectEachData()






