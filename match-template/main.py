import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np



THRESHOLD = 0.8

# Opening JSON file
jEFTItems = open('../eft-items.json');

eftItems = json.load(jEFTItems)['data']['items']

# Load grid image for each item
for item in eftItems:
    itemID = item['id']
    gridImageName = item['gridImageLink'].split('/')[-1]
    image = cv.imread(f'../image-downloader/train/{itemID}/{gridImageName}')
    item['grid-image'] = image
    item['dimensions'] = image.shape[::-1]

print('Loaded grid images.')

screenshot_name= '2024-05-09[09-06]_205.2, 9.2, -98.7_-0.1, 0.5, 0.0, -0.9_11.79 (0).png'

input_path = f'./screenshots/{screenshot_name}'
output_path = f'./output/{screenshot_name}'

screenshot = cv.imread(input_path)

def match_image(item, reference_image, image):
    matchResult = cv.matchTemplate(reference_image, image, cv.TM_CCOEFF_NORMED)
    
    loc = np.where(matchResult >= THRESHOLD)

    found = (len(loc[0]) > 0)

    return [item, found, loc]

def startMatching():
    print('Starting matching...')

    matchResults = []

    with ThreadPoolExecutor(20) as executor:
        matchImageResult = {executor.submit(match_image, item, screenshot, item['grid-image']): item for item in eftItems}

        for executorResult in as_completed(matchImageResult):
            result = executorResult.result()
            
            item = result[0]
            found = result[1]
            loc = result[2]

            if found:
                matchResults.append([item, loc])

    # for item in eftItems:
    #     matchResult = match_image(screenshot, item['grid-image'])
    #     matchResults.append([item, matchResult])

    print('Finished matching.')

    return matchResults

def main():
    matchResults = startMatching()

    for matchResult in matchResults:
        item = matchResult[0]
        loc = matchResult[1]
        
        channels, w, h = item['dimensions']

        for pt in zip(*loc[::-1]):
            cv.putText(screenshot, str(item['avg24hPrice']), (pt[0] + w / 2, pt[1] + h / 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv.rectangle(screenshot, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            break

        # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchTemplateResult)

        # #top_left = min_loc
        # top_left = max_loc

        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # cv.rectangle(screenshot, top_left, bottom_right, 255, 2)

    cv.imwrite(output_path, screenshot)

main()