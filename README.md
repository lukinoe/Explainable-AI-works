
# Explainable AI Assignment 2 - Model Explanations
In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

## General Information Submission due on 02.12.2021, 23:59

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** CodeBusters

## Intermediate Presentation on the 09.12.2021 (optional)
The presentations will be held during the Q&A sessions on that day.
For more details, please refer to the assignment description.

## Final Submission due on 15.12.2021, 23:59
The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2021-<GROUP_NAME>.git
cd xai_model_explanation_2021-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_model_explanation
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.


## Report

### Model & Data

**Model:** DistilBERT (https://huggingface.co/docs/transformers/model_doc/distilbert)</br>
**Data:**  IMDB-Dataset (https://huggingface.co/datasets/imdb)</br>

### Explainability Approaches
Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

#### Approach 1 : Showing how the attention changes when the model is trained

* Breifly summarize the approach. 

For our first approach, we use the library BertViz (https://github.com/jessevig/bertviz) to visualize the attention of the model for each layer and head. 
We use the model view of the library BertViz, which shows you can see the attention for every head for every layer at once. The visualization is interactive and you can see by clicking or hovering in the visualization to which words the attention flows.

Visualizing the attention of the DistilBERT trained on Imdb looks like the following:

<img src="img/model_trained_positive.png" alt="Attention visualized from a trained model on a positive text" title="Attention from trained model on a positive text" />

With that we can learn about the model and how it works on attention level, but we wanted to apply this visualization to explain what the model has learned. So, we took the attention from a pretrained and untrained model on Imdb and visualize the difference between both attention matrices. Then we only see the attention connections, which got stronger by training the model on Imdb.

With that the following got visualized:

<img src="img/model_diff_positive.png" alt="Attention difference betwen trained and untrained model visualized on a positive text" title="Attention difference between trained and untrained model on a positive text" />

* Categorize this explainability approach according to the criteria by Hohman et al.

**Why?**      Interpretability & Explainability</br>
**What?**     Learned Model Parameters | Computational Graph & Network Architecture</br>
**When?**     After Training</br>
**Who?**      Model Developers & Builders</br>
**How?**      Algorithms for Attribution & Feature Visualization</br>
**Where?**    Interpretability of NLP Tasks</br>

* Interpret the results here. How does it help to explain your model?

If we interact with the visualization, we can see what attention values got bigger, the following is an example:

<img src="img/model_diff_positive_with_example.png" alt="Attention difference betwen trained and untrained model visualized on a positive text" title="Attention difference between trained and untrained model on a positive text" />

We can see that in this example there is more attention for the words "highly" and "recommend" in layer 4 for head 5. This is something we would expect for a trained model on imdb.

The following is one example on a negative sentence:

<img src="img/model_diff_negative_with_example.png" alt="Attention difference betwen trained and untrained model visualized on a negative text" title="Attention difference between trained and untrained model on a negative text" />

Here we can see that the attention got stronger for words like "not", "really" and "good", which is also something we would expect.

This can help explain how the model is working and what it has learned to classify the sentences correctly in positive and negative. This could be helpful for more complex language classification tasks to see where the model puts its attention too to get to that classification. This helps us to understand the model and how it works and what it has learned better.

***

#### Approach 2 : Visualize the embeddings

* Breifly summarize the approach. 

In the second approach we created prediction of the IMDB dataset to calculate the shap values which give explaination how much a word contributed to the sentiment of a review.
Then we downproject the embedding of the words and color them according to their shap value.

* Categorize this explainability approach according to the criteria by Hohman et al.

**Why?**    Interpretability & Explainability </br>
**What?**   Aggregated Information </br>
**When?**   After Training </br>
**Who?**    Model Users </br>
**How?**    Dimensionality Reduction & Scatter Plots </br>
**Where?**  NLP Sentiment Research </br>

* Interpret the results here. How does it help to explain your model?

The color of the words indicate wheter a word contributed more to positive or negative reviews. 
So it can be interpreted that certain words are more often used in a positive or negative way.
In the figure blue words represent positive sentiment, whereas red colored words represent negative words.
Black colored words are neutral.
The shading indicates how strong the sentiment of a word is. 

<img src="img/word_cloud_shap.PNG" alt="Word cloud colored by their shap values (Blue = positive | Red = Negative | Black = Neutral)" title="Word cloud colored by their shap values (Blue = positive | Red = Negative | Black = Neutral)" />

***

#### Approach 3 : Sentiment Analysis for every layer and epoch

* Breifly summarize the approach. 

We trained a pre-trained DistilBERT sentiment model for 5 epochs. After each epoch we generated embeddings for 200 samples and saved the hidden states of each layer for each sample in a numpy array of shape (epoch, sample, layer, hidden_dim). 
For each of the embeddings of the 5 epochs and the 6 layers we performed downprojection via PCA und visualized them in scatter plots, where the colors represent the labels, positive (1) and negative(0).



* Categorize this explainability approach according to the criteria by Hohman et al.

**Why?** Interpretability & Explainability </br>
**What?** Aggregated Information </br>
**When?** During/After Training </br>
**Who?** Model Users </br>
**How?** Dimensionality Reduction & Scatter Plots </br>
**Where?** NLP Sentiment Research </br>

* Interpret the results here. How does it help to explain your model?

Here the downprojected embeddings are visualized. The rows of the visualization represent the epochs, the columns represent the layers. 
The plots illustrate well how the model learns sentiment analysis on the training samples by adapting the embeddings layer by layer, epoch by epoch.

<img src="img/embeddings_layerwise.png" alt="Downprojected Embeddings visualized per epoch and layer" title="Downprojected Embeddings visualized per epoch and layer" />

***

#### Approach 4 : Sentiment Analysis with SHAP

* Breifly summarize the approach. 

Based on the work of Lundberg & Lee (https://proceedings.neurips.cc//paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf), the idea is to use the Shapely value that originally comes from game theory to determine the marginal contributions of input factors to the result. In our specific use case, this means that each DistilBERT token gets a certain Shapely value assigned, based on it’s impact on the prediction (positive or negative impact on sentiment). For this the SHAP library is used.

* Categorize this explainability approach according to the criteria by Hohman et al.

**Why?** Increases interpretability & explainability of the model/the input-output mapping, as well as for improving or comparing models </br>
**What?** Learned model parameters, aggregated information </br>
**When?** After training </br>
**Who?** Model Developers & Builders | Model Users </br>
**How?** Various plots like barcharts, but also interactive input/output mapping for feature importance </br>
**Where?** NLP, Image classification/object detection, Tabular data </br>

* Interpret the results here. How does it help to explain your model?

The results show which DistilBERT tokens have overall the biggest impact on positive or negative sentiment. This leads to a better understanding of how DistilBERT determines the overall sentiment of a sequence, focusing on Shapely-based feature importance. Not only can the importance be analyzed on a global level (e.g. obtaining the on average most influential tokens overall), but also on a sentence level. This means that we can determine which features are in general important for DistilBERT to decide if a sentiment is positive or negative, but also look at a specific sequence and get explanations for the related prediction.

The following two bar charts visualize the DistilBERT tokens that have the most impact on positive & on negative sentiment on a global level, sorted by their Shapely value. Interestingly, we can not only observe some general vocabulary that indicates strong sentiment ('dump' or 'benefits'), but also movie-related vocabulary ('loops', 'precursor').

![grafik](https://user-images.githubusercontent.com/69237838/145988221-804e04c0-a6f0-459c-9176-0353a96f9a52.png)

![grafik](https://user-images.githubusercontent.com/69237838/145988250-511e013d-13f2-4a1e-a5f5-1d68f0ba0a91.png)

The visualization below is an interactive tool provided by the SHAP library. It allows a text-level analysis of the DistilBERT token's impact on sentiment, based on it's Shapely value. In our sample text sequence we can see that tokens like 'good' or 'fantastic' are obviously strongly positively contributing to the sentiment of the text. What we should also note is that the somewhat negative sentiment in the 1st part of the 3rd sentence is not really captured. This could be an interesting observation for developers in order for them to understand better how the model works. 
Furthermore, we can see that the phrase got an overall Shapely value of ~1.54. This means that the general sentiment of the phrase is positive since the positive tokens overweigh the negative tokens (considering Shapely value).

<img width="1181" alt="Bildschirmfoto 2021-12-14 um 12 18 10" src="https://user-images.githubusercontent.com/69237838/145988608-5b8609d8-f0a8-4cb0-840c-d6e9d77c1318.png">

***


### Summary of Approaches
Write a brief summary reflecting on all approaches.

Our goal was to get both a general understanding about the model, as well as task-specific insights which are in this case about Sentiment Analysis. Approaches 1 & 2 (Attention changes & Embedding visualisation) aim to target the former while approaches 3 & 4 (Sentiment per layer/epoch & SHAP) explain the latter. 

Furthermore, we believe that the mix of methods we used present a neat picture of Sentiment Analysis using DistilBERT. Not only do we cover details about the training process with approaches 1 & 3 (Attention changes & Sentiment per layer/epoch), we also show how the obtained results from DistilBERT can be used to explain feature importance and clusters in the embedding space with our approaches 2 & 4 (Embedding visualisation & SHAP). 
