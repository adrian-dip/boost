# Boost

Boost is a model that predicts the engagement (shares, likes, likelihood of going viral) that text and images will generate. Boost achieves 87-92% prediction accuracy on social media and news, and 71%–88% accuracy on internet comments, depending on the topic of discussion.

This repo includes the model and dataset for text and a JS endpoint to serve it. 

This is a partial release of the Boost models, app and dataset. Due to their size, they have to be downloaded from here:  . 

## Serving the model

To serve the model (backbone: `deberta-xsmall`) using this code, you need to upload it to a Flask server that supports Pytorch. The endpoint is a Chrome extension that detects typing and automatically sends the content a few seconds after you stop. Inference is fast enough on a single CPU core, which can process a couple of queries per second. 

## Full model

If you want to use the full model, which processes text + images, you need to train it yourself using the scripts on the directory `full_model/` using images you scrape. You can find scraping repos on my profile. The model on the link is only the text model and is less accurate (-5% to -10%). 

## Architecture

The text model is a text transformer + LSTM + feed-forward pooling + linear layer. The full model has the following architecture:

![Diagram](diagram1.png)
