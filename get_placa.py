import cv2
import numpy as np


# ============================================================================

def clean_image(img):
    
    resized_img = cv2.resize(img
        , None
        , fx=8.0
        , fy=8.0
        , interpolation=cv2.INTER_CUBIC)

    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)
    cv2.imwrite('licence_plate_large.png', resized_img)
    

    #Original Threshold
    #mask = cv2.adaptiveThreshold(resized_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #cv2.imwrite('licence_plate_mask.png', mask) 

    
    ret, mask = cv2.threshold(resized_img,100,255,cv2.THRESH_BINARY)
    cv2.imwrite('licence_plate_mask.png', mask) 

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('licence_plate_mask1.png', mask) 

    return mask

# ============================================================================

def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        if (area > 5000) and (area < 10000):
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)
    
    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])
    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))

    return clean, characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)

    return output_img

# ============================================================================    

img = cv2.imread("pruebas_placas/pr0.png", 0)
#img = cv2.imread("first_placa.png", 0)
r = 60.0 / img.shape[1]
dim = (60, int(img.shape[0] * r))
rez = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('resized.png', rez)

rez = clean_image(rez)
clean_img, chars = extract_characters(rez)

output_img = highlight_characters(clean_img, chars)
cv2.imwrite('licence_plate_out.png', output_img)

# Load from num
samples_num = np.loadtxt('char_samples_num.data',np.float32)
responses_num = np.loadtxt('char_responses_num.data',np.float32)
responses_num = responses_num.reshape((responses_num.size,1))

# Load from let 
samples_let = np.loadtxt('char_samples_let.data',np.float32)
responses_let = np.loadtxt('char_responses_let.data',np.float32)
responses_let = responses_let.reshape((responses_let.size,1))

# Both models
model_num = cv2.ml.KNearest_create()
model_let = cv2.ml.KNearest_create()
model_num.train(samples_num, cv2.ml.ROW_SAMPLE, responses_num)
model_let.train(samples_let, cv2.ml.ROW_SAMPLE, responses_let)

plate_chars = ""

if (len(chars) != 6): print 'Warning, bad number of chars detected: ', len(chars)
plate_lets = chars[0:3]
plate_nums = chars[3:6]

for bbox, char_img in plate_lets:
    small_img = cv2.resize(char_img,(10,10))
    small_img = small_img.reshape((1,100))
    small_img = np.float32(small_img)
    retval, result, neigh_resp, dists = model_let.findNearest(small_img, k = 5)
    predict = model_let.predict(small_img)
    plate_chars += str(chr((result[0][0])))

for bbox, char_img in plate_nums:
    small_img = cv2.resize(char_img,(10,10))
    small_img = small_img.reshape((1,100))
    small_img = np.float32(small_img)
    retval, result, neigh_resp, dists = model_num.findNearest(small_img, k = 5)
    predict = model_num.predict(small_img)
    plate_chars += str(chr((result[0][0])))

print("Licence plate: %s" % plate_chars)