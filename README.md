# Phenumonia Detector
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Training

`train.py` yields the output below:
```pwsh
Moved 158 images to data\test\NORMAL
Moved 427 images to data\test\PNEUMONIA
epoch     train_loss  valid_loss  accuracy  time    
0         0.494486    0.407573    0.880444  08:28
epoch     train_loss  valid_loss  accuracy  time    
0         0.237182    0.165131    0.955594  09:52
```

## Test predictions

Test on two random samples from `test/NORMAL` and two random samples from `test/PNEUMONIA`.
Both samples were **excluded** from the training set.

```pwsh
Image: test/NORMAL\NORMAL2-IM-1406-0001.jpeg
Prediction: NORMAL
Probabilities: tensor([0.7957, 0.2043])

Image: test/NORMAL\NORMAL2-IM-1326-0001.jpeg
Prediction: NORMAL
Probabilities: tensor([0.8174, 0.1826])

Image: test/PNEUMONIA\person136_bacteria_654.jpeg
Prediction: PNEUMONIA
Probabilities: tensor([0.0047, 0.9953])

Image: test/PNEUMONIA\person1910_bacteria_4814.jpeg
Prediction: PNEUMONIA
Probabilities: tensor([0.0144, 0.9856])
```