import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


def GetFile(driver, url):
    try:
        driver.get(url)
        # time.sleep(0.5)
        # driver.find_element(By.CLASS_NAME, value='el-dialog__close').click()
        time.sleep(0.5)
        driver.find_element(By.XPATH, "//span[text()='下载 Vocaloid 工程']").click()
        time.sleep(0.5)
        driver.find_element(By.CLASS_NAME, 'download_start').click()
        time.sleep(0.5)
        driver.find_element(By.CLASS_NAME, 'el-checkbox__inner').click()
        time.sleep(3)
        driver.find_element(By.XPATH, "//span[text()=' 下载 ']").click()
        time.sleep(1)
        driver.get('https://www.vsqx.top/project')
    except:
        print("Error")

if __name__ == "__main__":
    GetFile(webdriver.Chrome(), 'https://www.vsqx.top/project/vn513')