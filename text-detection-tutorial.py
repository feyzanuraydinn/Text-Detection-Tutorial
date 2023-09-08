import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = 'C:/Users/Aydin/VSCodeProjects/internshipProjects/text-detection-tutorial/data/image_1.png'

img = cv2.imread(image_path)

reader = easyocr.Reader(['en'], gpu=False)

text_ = reader.readtext(img)

threshold = 0.25

for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        top_right = tuple(map(int, top_right))
        bottom_right = tuple(map(int, bottom_right))
        bottom_left = tuple(map(int, bottom_left))

        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
        cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
