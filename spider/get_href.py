# download vsqx files and their ranks from vsqx.top
# get herfs from the vsqx.top
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import sys
import argparse
sys.path.append('.')
import get_file

args = argparse.ArgumentParser()
args.add_argument('--account', type=str, required=True)
args.add_argument('--password', type=str, required=True)
args.add_argument('--page', type=int, default=1)
args = args.parse_args()
ACCOUNT = args.account
PASSWORD = args.password
PAGE = args.page

options = webdriver.ChromeOptions()
prefs = {'profile.default_content_settings.popups':0,'download.default_directory':'.'}
options.add_experimental_option('prefs',prefs)

driver = webdriver.Chrome()
# login
driver.get("https://www.vsqx.top/login")
driver.find_element(By.CLASS_NAME, value='el-dialog__close').click()
time.sleep(0.5)
username, password = driver.find_elements(By.CLASS_NAME, value='el-input__inner')
username.send_keys(ACCOUNT)
password.send_keys(PASSWORD)
driver.find_element(By.CLASS_NAME, value='login_button').click()
time.sleep(1)

driver.get("https://www.vsqx.top/project?language=1&singer=1&level=3&synthesizer=1")
time.sleep(0.5)

# classes = ['el-button', 'el-button--primary', 'el-button--small']
# buttons = driver.find_elements(By.CLASS_NAME, value='el-button')
# buttons = [button for button in buttons if all([c in button.get_attribute('class') for c in classes])]
# time.sleep(0.5)
# buttons[0].click()

time.sleep(0.5)
check = driver.find_elements(By.CLASS_NAME, value="el-checkbox__input")[1]
check.click()
time.sleep(0.5)
pageButton = driver.find_elements(By.CLASS_NAME, value='number')[PAGE-1]
pageButton.click()
time.sleep(0.5)
links = driver.find_elements(By.CLASS_NAME, value="vsqx-name")

for index, link in enumerate(links, 0):
    links[index] = link.get_property("href")
for link in links:
    get_file.GetFile(driver, link)