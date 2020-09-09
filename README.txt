DATA SET: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

data should be stored as follows:

chest_xray
│   └───train
│   |    └───0
|   |    |   normal1.jpeg
|   |    |   normal2.jpeg
|   |    |   ...
|   |    └───1
|   |    |   pneum1.jpeg
|   |    |   pneum2.jpeg
|   |    |   ...
│   └───val
│   |    └───0
|   |    |   normal1.jpeg
|   |    |   normal2.jpeg
|   |    |   ...
|   |    └───1
|   |    |   pneum1.jpeg
|   |    |   pneum2.jpeg
|   |    |   ...

0 is for normal lungs
1 is for lungs with pneumonia

Name of actual image files does not matter but folder names must be as above.
Train is used as training data
Val can be used to test model or have it categorise an image.