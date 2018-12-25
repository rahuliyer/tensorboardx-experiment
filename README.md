# tensorboardx-experiment

A simple script to explore the abilities of TensorboardX. The script creates a simple convolutional neural net and trains it sending events to be logged to TensorboardX.

## Usage ##
Set up the required environment 
```
conda env create -f environment.yml 
```

Activate the environment
```
conda activate tensorboardx-experiment
```

Start Tensorboard
```
tensorboard --logdir=runs/
```

Run the script
```
python tensorboard_demo.py 
```

## Screenshots ##
Visualizing training and validation loss over number of epochs
![Training and validation loss](images/loss.png)

Logging average loss per epoch and accuracy
![Average Loss in an epoch](images/loss_text.png)
![Accuracy](images/accuracy.png)

Visualizing incorrect predictions
![Incorrect predictions](images/incorrect_predictions.png)
