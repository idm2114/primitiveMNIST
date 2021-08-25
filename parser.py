import numpy as np 
import matplotlib.pyplot as plt
import cv2

class parser():
    def __init__(self):
        pass

    def parse(self, path, save_dir):
        img = cv2.imread(path)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (thresh, bwimg) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(bwimg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # find contours and bounding boxes
        bboxes = []
        bboxes_img = img.copy()
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        contours = contours[0]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            bboxes.append((x,y,w,h))
    
        # sorting bboxes
        bboxes = sorted(bboxes, key=lambda x: x[0])
        num = 0
        for count in range(len(bboxes)):
            x,y,w,h = bboxes[count]
            if (w * h) < 20: 
                continue
            number = img[y:y+h,x:x+w]
            gray = cv2.cvtColor(number,cv2.COLOR_BGR2GRAY)
            (thresh, bwimg) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            ret, thresh = cv2.threshold(bwimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow("a", thresh)
            cv2.waitKey(0)
            cv2.imwrite(f'{save_dir}/digit{num}.png', thresh)
            num += 1

        cv2.imwrite("numwithboxes.png", bboxes_img)

if __name__=='__main__':
    p = parser() 
    p.parse("phonenumber.png", "test_images")
