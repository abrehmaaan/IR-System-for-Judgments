import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Set the URL of the webpage
url = "https://mis.ihc.gov.pk/frmSrchOrdr"

# Set the search keyword and other parameters
keyword = "tax"

# Initialize a Chrome driver and navigate to the webpage
driver = webdriver.Chrome()
driver.get(url)

# Find the search box and button by their IDs
search_box = driver.find_element("id","txtKyWrd")
search_button = driver.find_element("id","btnSearch")

# Set the value of the search box to the keyword and click the search button
search_box.send_keys(keyword)
search_button.click()

# Wait for at least one minute for the search results to load
WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CLASS_NAME, "info-box")))
# time.sleep(50)
# Find all divs with the class "info-box"
info_boxes = driver.find_elements(By.CLASS_NAME, "info-box")

# Iterate over the info boxes
for info_box in info_boxes:
    try:
        # Find the i tag with class "GrdB" and click it
        i_tag = info_box.find_element(By.CLASS_NAME,"GrdB")
        i_tag.click()
        pdf_url = ""
        # Get the URL of the PDF file from the new page
        WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
        driver.switch_to.window(driver.window_handles[-1])
        pdf_link = driver.find_element(By.CSS_SELECTOR,"#mTeam > div:nth-child(1) > div:nth-child(4) > a")
        if pdf_link:
            pdf_url = pdf_link.get_attribute("href")
        # Download the PDF file to a folder
        if pdf_url:
            response = requests.get(pdf_url)
            with open("./tax" + pdf_url.split("/")[-1], "wb") as f:
                f.write(response.content)
        # Close the new tab and switch back to the main window
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
    except Exception as ex:
        # Close the new tab and switch back to the main window
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
