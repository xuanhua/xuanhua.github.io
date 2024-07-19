---
layout: post
title:  "Deep Neural Networks for YouTube Recommendations"
date:   2024-07-16 10:29:33 +0000
categories: jekyll update
---

The whole system is comprised of two neural networks, one for candidate generation and another for ranking.


## Candidate Generation

This problem is modeled as a classification problem, the target is to predict next video in a million-sized vedio pool based on user's past behavior.

$$
P(w_t = i | U, C) = \frac{e^{v_i u}}{\sum_{j \in V} e^{v_j u}}
$$

Where: $w_t$ is the vedio at time $t$; And $V$ is the video corpus;$U$ is the set of users; And $C$ is the context. $u \in \mathbb{R}^N$ respresents a high-dimensional user embedding 

For the data used for training, a completed video is a positive example in the concext $C$ (we suppose that the previous watched, clicked vedios or other exposed videos are attributed as context) 

**Data sampling**: 
* Candidate sampling and correct this sampling via importance weighting.

**Model architecture**
![](/assets/img/2024-07-19-youtube_recall_network.png)

### Why use neural network as a genralization of matrix factorization?

The answer is with neural networks, arbitrary continuous and categorical features can be easily added into the model:
* Search history
  * Tokens from both Unigram, bigram
* User demographic information
  * Age, gender, location, and this is useful for new users.
* Watch history
* Simple binary and continuous features such as user's gender, logged-in state and age are input directly into network as real value normalized to [0,1]
* "Example age" Feature, for video the lastest upload one is usually preferred by users. Without it, the model would usually recommend older videos which may not be what user wants. Here is a comparison with and without this feature for online watching:

![alt text](/assets/img/2024-07-16-youtube_recall_age_feature.png)


## Ranking model

In ranking model, relationship between the user and video are more concerned. More features describing the user and the video relationship are added. The ranking model use the similar architecture with the recall model.
* Watch history
* last search query
* User's previous interaction with the item and other similar items
  * When was the user watch a video on this channel
* Propagate information from candidate generation into ranking in the form of feature. Which sources nominated this video candidate

**Model Architecture**
![alt text](/assets/img/2024-07-19-youtube_ranking_network.png)

**Positive and Negative examples**

In ranking model, it use the [weighted logistic regression](https://stats.stackexchange.com/questions/442796/what-does-weighted-logistic-regression-mean) for positive and negative examples:  
* Positive examples are annotated with the amount of time $t_{p_i}$ the user spent watching the video.
* Negative examples are annotated with unit watch time $t_{n_i}=1$.
After applying the scoring function: $f(t) = e^t$, we could get:
$$
\begin{align*}
s_{p_i} &= f(t_{p_i}) = e^{t_{p_i}} \\
s_{n_i} &= f(t_{n_i}) = e^{t_{n_i}} = e \\
\end{align*}
$$

After the training, the odds learned by the logistic regression are: $\frac{\sum T_i}{N-k}$, where $N$ is the number of training examples, $k$ is the number of positive impressions. Where $P$ is the click probability, and it is small; $T_i$ is the watch time of the $i$ th impression. And we should have:

$$
\frac{\sum T_i}{N -k} \approx E[T](1 + P) \approx E[T]
$$


## Reference

* [Notes for youtube DNN](https://github.com/luweiagi/machine-learning-notes/blob/master/docs/recommender-systems/industry-application/youtube/youtube-dnn/Deep-Neural-Networks-for-YouTube-Recommendations.md)
* [One implementation of youtube DNN](https://github.com/hyez/Deep-Youtube-Recommendations) 