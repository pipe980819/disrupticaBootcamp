#######
# https://stackoverflow.com/questions/36407608/recognize-the-characters-of-license-plate

import cv2
import numpy as np
import glob

NUMBERS = [chr(ord('0') + i) for i in range(10)] 
LETTER = [chr(ord('A') + i) for i in range(26)]

# ============================================================================

def load_char_num():
    characters = {}
    for char in NUMBERS:
        l_char = char.lower()
        f_list = glob.glob("charsImgs/" + l_char+  "*")
        cv_list = []
        for f in f_list:
            cv_list.append(cv2.imread(f, 0))
        #f = glob.glob("charsImgs/" + l_char+  "*.png")[0]
        #print("Getting " + str(char) + ' ' + str(len(f_list)))
        #char_img = cv2.imread(f, 0)
        characters[char] = cv_list
    return characters


def load_char_let():
    characters = {}
    for char in LETTER:
        l_char = char.lower()
        f_list = glob.glob("charsImgs/" + l_char+  "*")
        cv_list = []
        for f in f_list:
            cv_list.append(cv2.imread(f, 0))
        characters[char] = cv_list
    return characters
# ============================================================================
# Numeros
characters = load_char_num()
samples =  np.empty((0,100))
for char in NUMBERS:
    char_img_list = characters[char]
    for char_img in char_img_list:
        small_char = cv2.resize(char_img,(10,10))
        sample = small_char.reshape((1,100))
        samples = np.append(samples,sample,0)

responses = np.array([], np.float32)
for c in NUMBERS:
    lenChar = len(characters[c])
    for i in range(lenChar):
        responses = np.insert(responses, len(responses), ord(c))
    #responses = p.array([ord(c) for c in CHARS],np.float32)
#responses = np.array([ord(c) for c in CHARS],np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('char_samples_num.data',samples)
np.savetxt('char_responses_num.data',responses)

# ============================================================================
# Letras
characters = load_char_let()
samples =  np.empty((0,100))
for char in LETTER:
    char_img_list = characters[char]
    for char_img in char_img_list:
        small_char = cv2.resize(char_img,(10,10))
        sample = small_char.reshape((1,100))
        samples = np.append(samples,sample,0)

responses = np.array([], np.float32)
for c in LETTER:
    lenChar = len(characters[c])
    for i in range(lenChar):
        responses = np.insert(responses, len(responses), ord(c))

responses = responses.reshape((responses.size,1))
np.savetxt('char_samples_let.data',samples)
np.savetxt('char_responses_let.data',responses)