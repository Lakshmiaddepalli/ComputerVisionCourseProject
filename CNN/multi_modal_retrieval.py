from __future__ import division

import sys,os
import random
import json
import numpy as np
from model import TextTopicNetCNN
import torch.backends.cudnn as cudnn
import torch.nn
import torch.optim
import torch.utils.data
from termcolor import colored
from PIL import Image
import gensim
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import scipy.stats as sp

import torch
import torchvision.transforms as transforms, utils
from torch.autograd import Variable

from TextTopicNet.experiments.preprocess_text import preprocess_imageclef

n_topics = 40
layer = 'prob'
type_data_list = ['test']

# Function to compute average precision for text retrieval given image as input
def get_AP_img2txt(sorted_scores, given_image, top_k):
        consider_top = sorted_scores[:top_k]
        top_text_classes = [GT_txt2img[i[0]][1] for i in consider_top]
        class_of_image = GT_img2txt[given_image][1]
        T = top_text_classes.count(class_of_image)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_text_classes[i] != class_of_image:
                        pass
                else:
                        p_r = top_text_classes[:i+1].count(class_of_image)
                        sum_term = sum_term + float(p_r/len(top_text_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)

# Function to compute average precision for image retrieval given text as input
def get_AP_txt2img(sorted_scores, given_text, top_k):
        consider_top = sorted_scores[:top_k]
        top_image_classes = [GT_img2txt[i[0]][1] for i in consider_top]
        class_of_text = GT_txt2img[given_text][1]
        T = top_image_classes.count(class_of_text)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_image_classes[i] != class_of_text:
                        pass
                else:
                        p_r = top_image_classes[:i+1].count(class_of_text)
                        sum_term = sum_term + float(p_r/len(top_image_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)


if len(sys.argv) < 2:
        print('Please enter the type of query. Eg txt2img, img2txt')
        quit()

query_types=sys.argv[1:]

### Start : Generating image representations of wikipedia dataset for performing multi modal retrieval

layer = 'prob' # Note : Since image and text has to be in same space for retieval. CNN layer has to be 'prob'
# Model parameters
n_topics = 40  # Number of topics for the corresponding LDA model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_filename = './BEST_checkpoint_ssl.pth.tar'
checkpoint = checkpoint_filename if os.path.exists(checkpoint_filename) else None # path to model checkpoint, None if none

# Initialize caffe model instnce with given weights and model prototxt
checkpoint = torch.load(checkpoint, map_location='cpu')
start_epoch = checkpoint['epoch'] + 1
epochs_since_improvement = checkpoint['epochs_since_improvement']
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()
optimizer = checkpoint['optimizer']

IMG_SIZE = 256
MODEL_INPUT_SIZE = 227
MEAN = np.array([104.00698793, 116.66876762, 122.67891434])

text_dir_wd = 'TextTopicNet/data/Wikipedia/texts_wd/' # Path to wikipedia dataset text files
img_root = 'TextTopicNet/data/Wikipedia/images_wd_256/' # Path to wikipedia dataset image files
image_categories = ['art', 'geography', 'literature', 'music', 'sport', 'biology', 'history', 'media', 'royalty', 'warfare'] # List of document (image-text) categories in wikipedia dataset

with torch.no_grad():
    # Generate representation for each image in wikipedia dataset
    for type_data in type_data_list:
        # Specify path to wikipedia dataset image folder and output folder to store features
        out_root = './generated_data/multi_modal_retrieval/image/' + str(layer) + '/' + str(n_topics) + '/' + str(type_data) + '/'
        if not os.path.exists(out_root):    
            os.makedirs(out_root)
            
        im_txt_pair_wd = open('TextTopicNet/data/Wikipedia/'+str(type_data)+'set_txt_img_cat.list', 'r').readlines() # Image-text pairs
        img_files = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd] # List of image files in wikipedia dataset
        for sample in img_files:
                im_filename = img_root+sample
                print(colored('Generating image representation for : ' + im_filename, 'green'))
                im = Image.open(im_filename).convert('RGB')     
                transform = transforms.Compose([
                                                    transforms.Resize(IMG_SIZE),
                                                    transforms.CenterCrop(227),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=MEAN, std=[1,1,1]),
                                                ])

                t_img = Variable(transform(im))
                t_img = t_img.unsqueeze_(0)
                t_img = t_img.to(device=device)
                output = model.forward(t_img)
                # output_prob = net.blobs[layer].data[0] # the output feature vector for the first image in the batch            
                np.save(out_root+sample, output)
        print('Finished generating representation for wikipedia dataset images')
        ### End : Generating image representations of wikipedia dataset for performing multi modal retrieval

        ### Start : Generating text representation of wikipedia dataset for performing multi modal retrieval

        choose_set_list = type_data_list

        # IMPORTANT ! Specify minimum probability for LDA
        min_prob_LDA = None

        # load id <-> term dictionary
        dictionary = gensim.corpora.Dictionary.load('TextTopicNet/LDA/dictionary.dict')

        # load LDA model
        ldamodel = gensim.models.ldamulticore.LdaMulticore.load('TextTopicNet/LDA/ldamodel'+str(n_topics)+'.lda', mmap='r')

        for choose_set in choose_set_list:
                # Read image-text document pair ids
                im_txt_pair_wd = open('TextTopicNet/data/Wikipedia/'+str(choose_set)+'set_txt_img_cat.list', 'r').readlines()
                text_files_wd = [text_dir_wd + i.split('\t')[0] + '.xml' for i in im_txt_pair_wd]
                output_path_root = './generated_data/multi_modal_retrieval/text/'
                if not os.path.exists(output_path_root):
                        os.makedirs(output_path_root)
                output_file_path = 'wd_txt_' + str(n_topics) + '_' + str(type_data) + '.json'
                output_path = output_path_root + output_file_path

                # transform ALL documents into LDA space
                TARGET_LABELS = {}
                
                for i in text_files_wd:
                        print(colored('Generating text representation for document number : ' + str(len(TARGET_LABELS.keys())), 'green'))
                        raw = open(i,'r').read()
                        process = preprocess_imageclef(raw)
                        if process[1] != '':
                                tokens = process[0]
                                bow_vector = dictionary.doc2bow(tokens)
                                lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=None)
                                #lda_vector = ldamodel[bow_vector]
                                lda_vector = sorted(lda_vector,key=lambda x:x[1],reverse=True)
                                topic_prob = {}
                                for instance in lda_vector:
                                        topic_prob[instance[0]] = instance[1]
                                labels = []
                                for topic_num in range(0,n_topics):
                                        if topic_num in topic_prob.keys():
                                                labels.append(topic_prob[topic_num])
                                        else:
                                                labels.append(0)
                                list_name = list_name = i.split('/')
                                TARGET_LABELS[list_name[len(list_name) -1 ].split('.xml')[0]] = labels
                
                # Save thi as json.
                json.dump(TARGET_LABELS, open(output_path,'w'))


        # ### End : Generating text representation of wikipedia dataset for performing multi modal retrieval

### Start : Perform multi-modal retrieval on wikipedia dataset.

for type_data in type_data_list:
    # Wikipedia data paths
    im_txt_pair_wd = open('TextTopicNet/data/Wikipedia/'+str(type_data)+'set_txt_img_cat.list', 'r').readlines()
    image_files_wd = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd]

    # Read the required Grount Truth for the task.
    GT_img2txt = {} # While retrieving text, you need image as key.
    GT_txt2img = {} # While retrieving image, you need text as key.
    for i in im_txt_pair_wd:
            GT_img2txt[i.split('\t')[1]] = (i.split('\t')[0], i.split('\t')[2]) # (Corresponding text, class)
            GT_txt2img[i.split('\t')[0]] = (i.split('\t')[1], i.split('\t')[2]) # (Corresponding image, class)

    # Load image representation
    image_rep = './generated_data/multi_modal_retrieval/image/' + str(layer) + '/' + str(n_topics) + '/' + str(type_data) + '/'

    # Load text representation
    data_text = json.load(open('./generated_data/multi_modal_retrieval/text/wd_txt_' + str(n_topics) + '_' + str(type_data) + '.json','r'))

    image_ttp = {}
    for i in GT_img2txt.keys():
            sample = i
            value = np.load(image_rep + i + '.jpg.npy')
            image_ttp[sample] = value

    # Convert text_rep to numpy format
    text_ttp = {}
    for i in data_text.keys():
            text_ttp[i] = np.asarray(data_text[i], dtype='f')
    # IMPORTANT NOTE : always sort the images and text in lexi order !!
    # If Query type is input=image, output=text
    mAP = 0
    if 'img2txt' in query_types:
            counter = 0
            order_of_images = sorted(image_ttp.keys())
            order_of_texts = sorted(text_ttp.keys())
            for given_image in order_of_images:
                    print(colored('Performing retrieval for image number : ' + str(counter), 'green'))
                    score_texts = []
                    image_reps = image_ttp[given_image][0]
                    for given_text in order_of_texts:
                            text_reps = text_ttp[given_text]
                            given_score = sp.entropy(text_reps, image_reps)
                            score_texts.append((given_text, given_score))
                    sorted_scores = sorted(score_texts, key=lambda x:x[1],reverse=False)
                    mAP = mAP + get_AP_img2txt(sorted_scores, given_image, top_k=10)
                    counter += 1

            print(colored('MAP img2txt : ' + str(float(mAP/len(image_ttp.keys()))), 'red'))

    mAP = 0
    if 'txt2img' in query_types:
            counter = 0
            order_of_images = sorted(image_ttp.keys())
            order_of_texts = sorted(text_ttp.keys())
            for given_text in order_of_texts:
                    print(colored('Performing retrieval for document number : ' + str(counter), 'green'))
                    score_images = []
                    text_reps = text_ttp[given_text]
                    for given_image in order_of_images:
                            image_reps = image_ttp[given_image][0]
                            given_score = sp.entropy(text_reps, image_reps)
                            score_images.append((given_image, given_score))
                    sorted_scores = sorted(score_images, key=lambda x:x[1],reverse=False)
                    mAP = mAP + get_AP_txt2img(sorted_scores, given_text, top_k=10)
                    counter += 1
            print(colored('MAP txt2img : ' + str(float(mAP/len(text_ttp.keys()))), 'red'))

### End : Perform multi modal retrieval on wikipedia dataset
