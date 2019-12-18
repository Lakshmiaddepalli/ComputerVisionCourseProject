# import json

# with open('./training_labels40.json') as json_file:
#     # parse json_file:
#     data = json.load(json_file)

#     for path in data:
#         labels = data[path]
#         for label_probability in labels:
#             print(type(label_probability))
#         break

from datasets import ImageCLEFWikipediaDataset

dataset_train = ImageCLEFWikipediaDataset('train')
dataset_test = ImageCLEFWikipediaDataset('test')

print(dataset_train.range())
print(dataset_test.range())
