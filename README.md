# Boost

Boost is a model that predicts the potential engagement (shares, likes, likelihood of going viral) that text and images will generate independent of the serving algorithm or initial exposure. This repo includes the model and dataset for text and a website-agnostic JS endpoint to serve it. 

Boost achieves 87% prediction accuracy for social media and structured content like news. It achieves 71%-88% accuracy on anonymous internet comments, with great variation depending on the topic of discussion. For details, see the error analysis and methodology sections below.

This is a partial release of the text Boost models, app and dataset. We're waiting to find out if we're allowed to release the images or the model trained on them. Due to their size, they have to be downloaded from here:  . The code on this repo can be used to replicate the results on the text data.

## Serving the model
