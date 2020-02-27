# Identify The Sentiments - Practice Problem 

This problem is a practise hackathon problem taken from Analytics Vidhya. More details [here](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/).

The task here is to identify if the provided tweets have a negative sentiment towards such companies or products.

**Evaluation Metric**: The metric used for evaluating the performance of classification model would be weighted F1-Score.

**Data**

- **train.csv** - For training the models, we provide a labelled dataset of 7920 tweets. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet.

- **test.csv** - The test data file contains only tweet ids and the tweet text with each tweet in a new line.

- **sample_submission.csv** - The exact format for a valid submission

Most profane and vulgar terms in the tweets have been replaced with “$&@*#”. However, please note that the dataset still might contain text that may be considered profane, vulgar, or offensive.

# Solution Approach

We will be using **fastai** in order to build a sentiment analysis model in order to predict the sentiment score of tweets.

**fastai** is a libray in python built on top of PyTorch for performing Deep Learning tasks developed by Rachel Thomas, Jeremy Howard and others which is easy to use, intuitive to understand and also gives decent results which are comparable with state-of-the-art models.

# Setting up fastai

You can refer [this website](https://course.fast.ai/start_colab.html) to set fastai for any environment you may be working on. It's better to refer these over other personal tutorials and stuff because it is directly from the developers and they update it periodically with new releases and stuff. 

The left margin contains options to configure fastai for various instances. Check that out depending on which system you're working on.

# Data Preparation

In any deep learning application, building the model is relatively a trivial job compared to the task of preprocessing it and bringing it into a form where you can feed it to the network/model.

fastai provides a lot of different options to facilitate easy loading of data. 

Our data here is given in a `csv` file. We can load it into a pandas dataframe with an *appropriate encoding scheme* and then build our dataset based on these dataframes. I would highly recommend to follow this notebook along with [this tutorial](https://www.youtube.com/watch?v=qqt3aMPB81c) as everything would make sense pretty easily after you've watched this video.

# Language Model

A Language Mode simply stated is a network which learns to predict the next word in a sentence based on current word and this iteratively continues forever until you manually put a stopping condition.

**Why would a language model help here?** 

The words in the review, they are meant to convey the emotion of a person based on which the label is assigned. A sequence of words is the input in this case and the sentiment underlying that sequence is the output. In order to build a network that learns to determine the sentiment, we will have to build a model that can handle a sequence input and return only one output. 

RNN *i.e. Recurrent Neural Networks* like LSTM networks or GRU Networks are capable of handling such data. But in order to build these, it's essential to express words as vectors as the network doesn't understand anything except numbers.

>In the process of building a language model, we learn the best way to numerically represent words as one-dimensional tensors which encapsulate contextual worldly knowledge which subsequently helps in classification.

Well, it's always better to start with a good set of weights than arbitrarily anyhow. So, fastai provides a language model which is trained on the entire Wikipedia Corpus and one start with the weights/coefficients of this model and then let the language model learn on the user-specified corpus. This can fine-tune the word-vector weights for a more targeted application.


For building a loader to fit a language model learner, fastai provides a class called `TextLMDataBunch` which creates a databunch object for systematically loading data into model for learning. There are several methods which could be used to initialize this object
- *from_csv*: If your data is in a csv format, you can use this method.

- *from_df*: If you've got a pandas dataframe in which you have loaded data and you want to build your dataset using this dataframe, this is the option you select.

- *from folder*: If your data is neatly organized in train, test folders and within them there are sub-folders for every class, then this method is well-suited to load your data.


Once you've loaded the data, you need to build a language model learner object to which you'll have to pass the databunch you just defined above and an architecture. This architecture could be the WikiText103 model learner which you can specify saying `AWD_LSTM` while passing the arch parameter of the language_model_learner.

Subsequently you can find the learning rate and train the model first with the end layers unfreezed and subsequently with all the layers unfreezed and you should see the accuracy metric rise. 

**Remember that even an accuracy of 30% or more is great for this model because this isn't our generic classification model; It's a language model. What this means is that given a sentence and if asked to predict the next word then the model is gonna predict a sensible output once every three times which is commendable.**

# Text Classification Model

Now that you've created a learner model which has learned quite a bit in the world of tweets, you are ready to build a classifier.


To provide data to the classifier model for training, we can use the `TextClasDataBunch` class and use the `from_df` method to load the dataframes that we built earlier as train, test and validation dataloaders.

**Note: It's important here that you also define the vocabulary based on the language model learner's vocabulary. It must be the same!**


Learning process is very much similar to the language model learner. A slight difference here is that we first unfreeze only the last layer, train for a few epochs, then unfreeze last two layers and train for several epochs and eventually unfreeze all the layers and train for several epochs. This approach, Jeremy says works best in his empirical experience. So let's stick with it.

The `get_preds` method helps obtain a prediction for every sequence that it's passed. it returns a probability distribution which in this case is simply two values i.e. $P(Sentiment=-|Tweet) \  \& \ P(Sentiment = +|Tweet)$. We have to select an appropriate threshold to classify the sentiment positive or negative and take a decision.

Once we obtain the predictions on the validation dataset, we can plot an ROC Curve and find out the best threshold for the problem by determining the point where the $f_1$ score is the best out of all the thresholds. The ROC Curve looks pretty good with an AUC of around 0.95 

![](roc_auc_curve.PNG)

The $f_1$ gives weightage to both Recall and Precision and hence is a decent metric to look for when looking for a good threshold value.

![](threshold.PNG)

The best value of $f_1$ score i.e. 0.833 is obtained when the threshold is set to around 0.429 for validation data. Utilize this and use the same for the test data as well.

*Besides, the hackathon also grades based on the weighted $f_1$ score, so that's that.*

# Result

After several hits and tries, I obtained a best $f_1$ score of 0.8949 for this problem which got me into the 84th percentile (Rank 101) as on 27th Feb 2020. It's a pretty decent score for very little hyperparameter tuning and such small code too.

![](leaderboard.PNG)

*fastai* is quite good and without much external preprocessing, I could comfortably manage to get a decent enough $f_1$ score. It's easy to use, convenient and also takes less time i.e. it's quite fast.
