
# Project 2 - Covolutional Neural Network for MNIST classification
- The goal of this homework is for you to become familiar with a deep learning framework.
- You will be asked to design and implement a CNN for MNIST classification.

##Tuning
-python main.py --tune

## Training
- python main.py --train --activation "func_name/s-['relu','sigmoid','tanh']"

## Testing
- python main.py --test --activation "func_name/s-['relu','sigmoid','tanh']"

##Example
-To run final model- python main.py --train

###To train/test other models 
-python main.py --train --activation "sigmoid"
-python main.py --train --activation "relu" "sigmoid"
-python main.py --train --test --activation "relu" "sigmoid" "tanh"
-python main.py --test --activation "relu" "sigmoid"