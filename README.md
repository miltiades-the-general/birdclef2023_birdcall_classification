# BirdClef 2023 Kaggle Machine Learning Competition
*source: https://www.kaggle.com/miltiadesgeneral/tensorflow-melspectro-efficientnetb0/edit*

### Competition Overview from Kaggle
____
"For this competition, you'll use your machine-learning skills to identify Eastern African bird species by sound. Specifically, you'll develop computational solutions to process continuous audio data and recognize the species by their calls. The best entries will be able to train reliable classifiers with limited training data. If successful, you'll help advance ongoing efforts to protect avian biodiversity in Africa, including those led by the Kenyan conservation organization NATURAL STATE."

The goal is to use the audio recordings of 264 classes of birds to train a model to make predictions on a continuous test feed of an unspecified number of bird classes within the dataset. 

### Guidelines
____

"The evaluation metric for this contest is padded cmAP, a derivative of the macro-averaged average precision score as implemented by scikit-learn. In order to support accepting predictions for species with zero true positive labels and to reduce the impact of species with very few positive labels, prior to scoring we pad each submission and the solution with five rows of true positives. This means that even a baseline submission will get a relatively strong score."

cmAP - cumulative mean average precision. Calculates the precision and recall for each category. Includes instances with 0 class predictions rather than ignoring them. This means the model will have to apply a classification even if it does not produce a prediction, or it predicts null.

*Submission Format:*
"For each row_id, you should predict the probability that a given bird species was present. There is one column per bird species so you will need to provide 264 predictions per row. Each row covers a five second window of audio."

This means *OneHotEncoding* the categories that belong to each unique bird species which is defined in the "primary_label" column. The model will apply its categorical prediction to each 5 second audio interval from the test data. Each row will have n=NUM_CLASSES one-hot-encoded prediction labels.

### Current Methodology
____

One of the main challenges that I have discovered in this competition is that the large number of classes (264) makes it fairly slow to train a neural network. If the classes are split into smaller chunks in makes it much quicker and more accurate to train. One of the problems embedded is the class imbalances with some categories of birds having very few data entries and some having many. Without balancing the network is prone to overfitting.

#### *Class Distribution Without Balancing*
![Classes Unbalanced](/unbalanced_classes.png)


#### *Class Distribution With Balancing*
![Classes Balanced](/class_balances.png)

*Dataset Preparation* 
Data is prepared and preprocessed using Tensorflow.
The data is presented in the training dataframe as an audio file path which must be opened, loaded and preprocessed. Tensorflow has built in functions to read and decode the file into a numpy array. Once the audio path is loaded in full, it is resized to a standard time of 5 seconds, and then resampled to the desired sample rate (32,000kHz). The clip is then time shifted a random amount to add variety. If the shift extends the clip beyond the 5 second threshold the audio is relooped to the beginning of the sample.

In order for the neural network to train on the data properly the data must be presented as an image-like tensor. A standard technique to accomplish this is by converting the audio array into a Mel Frequency Cepstral Coefficient (MFCC) spectrogram. To summarize, an MFCC spectrogram is an image representation of the frequencies present in an audio sample along the y-axis and time on the x-axis. It creates a picture that a neural network can learn from to classify categories. 

To read more on MFCCs and other audio deep learning:
https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc

Once the preprocessing steps are mapped onto the audio, the data is trained using a tensorflow EfficientNetB0 model using the one-hot-encoded class labels.

EfficientNetB0 Model:
https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
