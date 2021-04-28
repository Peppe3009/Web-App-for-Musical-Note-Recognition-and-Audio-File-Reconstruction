import cv2
import re
from pydub import AudioSegment
from PIL import Image, ImageFilter
import numpy as np
from os import listdir
from os.path import isfile, join

path_sound_sample = "audio_sample/"
voc_file = "vocabulary_semantic.txt"

counter = 0

# Read the dictionary
dict_file = open(voc_file, 'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
cont = 0
for word in dict_list:
    int2word[cont] = word
    cont += 1
dict_file.close()


def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]
    strs = [[] for i in range(dense_shape[0])]
    string = []
    ptr = 0
    b = 0
    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]
        string.append(values[ptr])
        ptr = ptr + 1
    strs[b] = string
    return strs


def normalize(image):
    return (255. - image) / 255.


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])

    sample_img = cv2.resize(image, (width, height))
    return sample_img


def recreate_sound(bpm, array_of_notes):
    print(array_of_notes)
    time_quarter = (60.0 / bpm) + 0.2
    time_half = time_quarter * 2
    time_eight = time_quarter / 2
    time_sixteenth = time_eight / 4
    time_whole = time_quarter * 4

    prev_sound = AudioSegment.from_wav(path_sound_sample + "rest.wav")[0:0]
    curr_sound = AudioSegment.from_wav(path_sound_sample + "rest.wav")[0:0]

    for note in array_of_notes:
        splitted = re.split('-|_', note)
        if splitted[0] == "note" or splitted[0] == "rest":
            if splitted[0] == 'note':
                curr_sound = AudioSegment.from_wav(path_sound_sample + splitted[1] + ".wav")
                if splitted[2] == 'whole':
                    curr_sound = curr_sound[0:time_whole * 1000]
                elif splitted[2] == 'half':
                    curr_sound = curr_sound[0:time_half * 1000]
                elif splitted[2] == 'half.':
                    curr_sound = curr_sound[0:(time_half + time_half / 2) * 1000]
                elif splitted[2] == 'quarter':
                    curr_sound = curr_sound[0:time_quarter * 1000]
                elif splitted[2] == 'quarter.':
                    curr_sound = curr_sound[0:(time_quarter + time_quarter / 2) * 1000]
                elif splitted[2] == "eight":
                    curr_sound = curr_sound[0:time_eight * 1500]
                elif splitted[2] == "eight.":
                    curr_sound = curr_sound[0:(time_eight + time_eight / 2) * 1500]
                elif splitted[2] == "sixteenth":
                    curr_sound = curr_sound[0:time_sixteenth * 1500]
                else:
                    curr_sound = curr_sound[0:(time_sixteenth + time_sixteenth / 2) * 1500]
            elif splitted[0] == 'rest':
                curr_sound = AudioSegment.from_wav(path_sound_sample + splitted[0] + ".wav")
                if splitted[1] == 'half':
                    curr_sound = curr_sound[0:time_half * 1000]
                elif splitted[1] == 'quarter':
                    curr_sound = curr_sound[0:time_quarter * 1000]
                elif splitted[1] == "eight":
                    curr_sound = curr_sound[0:time_eight * 1500]
                else:
                    curr_sound = curr_sound[0:time_sixteenth * 1500]
            prev_sound = prev_sound + curr_sound
    return prev_sound


def pre_processing(file, height):
    image = Image.open(file).convert('L')
    image = image.filter(ImageFilter.SHARPEN)
    image = np.array(image)
    image = resize(image, height)
    image = normalize(image)

    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
    return image


def from_prediction_to_note(str_predictions):
    array_of_notes = []

    for w in str_predictions[0]:
        figure = int2word[w]
        array_of_notes.append(figure)

    return array_of_notes


def update_counter():
    global counter
    counter = counter + 1
    return counter

def read_directory(except_track):
    mypath = 'static'
    print(except_track)
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] =='.wav' and f != except_track]
