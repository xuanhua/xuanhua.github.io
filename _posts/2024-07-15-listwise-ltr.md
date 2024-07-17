---
layout: post
title:  "Learning to rank for listwise based method"
date:   2024-07-15 10:29:33 +0000
categories: jekyll update
---

## Basic concepts in machine learning

First let's review three mathimatical concepts:
* Entropy
* Relative entropy
* Cross-entropy

Entropy is a measure of uncertainty of the system. For example, now we are training a model to do sentiment classfication. And we have 4 examples with lables either as 'pos' or 'neg' (to represent 'Postive' or 'Negative'): $\{ \text{pos},\text{neg},\text{pos},\text{neg}\}$. 

To measure the uncertainty of picking 'pos' or 'neg' examples, we can use entropy:

$$ H(X) = - \sum_{i} P_i log_2 (P_i) = -0.5 \times log_2 (0.5) - 0.5 \times log_2(0.5) = 1 $$

And note that $i \in [0,1]$ because we have two different classes 'pos' and 'neg'. So, the entropy for this case is $1$. Or say If we want to encode these labels, we need as least 1 bit.

Now, lets look at relative entropy or Jensen–Shannon divergence. Relative entropy is a measure of how one probability distribution diverges from a second, expected probability distribution.

Still from a machine learning perspective, we have our sentiment classifcation dataset with 4 examples, with labels  $\{ \text{pos},\text{neg},\text{pos},\text{neg}\}$, and the label distribution is $P$ ($P(pos)=\frac{1}{2},\ P(neg) = \frac{1}{2}$); And we are gonna to train a model for prediction. the model's prediction will follow a distribution $Q$；

we hope our model $Q_\theta$'s prediction will be as acurate as possible. But how can we achieve that? The answer is we define the difference measurement between distribution of model $Q_\theta$'s prediction and the ground truth.

W can use the KL divergence between these two distributions:

$$
\begin{align*}
D_{KL}(P || Q)  &= \sum_i P(i) log\left(\frac{P(i)}{Q(i)}\right) \\
                &= \sum_i \left[ P(i) log\left( P(i) \right) - P(i) log\left( Q(i) \right) \right] \\
                &= \sum_i \left[ P(i) log\left( P(i)  \right) \right ] - \sum_i  \left[ P(i) log\left( Q(i)   \right)  \right] \\
\end{align*}
$$

Here the index $i$ refers to the classes of labels. After summing over all possible classes, we get the KL divergence between $P$ and $Q$.

Our goal is to minimize this divergence. We could just use $D_{KL}$ as the loss function or say object function: 

$$
\begin{align*}
\underset{\theta}{argmin} \ D_{KL}(P || Q_\theta)  &= \underset{\theta}{argmin} \sum_i P(i) log\left(\frac{P(i)}{Q_\theta(i)}\right) \\
&= \underset{\theta}{argmin} \left( \sum_i P(i) log\left(P(i)\right) - \sum_i P(i) log\left(Q_\theta(i)\right) \right) \\
&= \underset{\theta}{argmin}  \left( - \sum_i P(i) log\left(Q_\theta(i)\right)  \right) \\
&= \underset{\theta}{argmin}\: H(P, Q_\theta)  \\
\end{align*}
$$

And here $H(P, Q_\theta)$ is the cross entropy between $P$ and $Q_\theta$. The index $i$ refers to each example of the training data.


## Learning to rank

Usually there are three types of learning to rank methods:
* Pointwise learning to rank (PLR): The model(we just named it as $f_\theta$) learns to predict a relevance score for each query-document pair.

$$
\underset{\theta}{argmin} \sum_i L(y_i, f(\mathbf{x}_i; \theta))
$$

  And $L$ is the loss function (e.g., squared error for regression).

* Pairwise learning to rank (PLR): The model learns to order the documents in pairs, rather than predicting scores for individual documents. Suppose that we have a query $q$ two documents $d_1$ and $d_2$, the relevance score for $q$ and $d_1$,$d_2$ are $s_1$,$s_{2}$ respectively. And in ground truth, $s_1 > s_2 $, then we say that $d_1$ is ranked higher than $d_2$ for query $q$. The model $f_\theta$ we want to learn should give following predicttions:

$$
f_\theta(q, d_1, d_2) = \begin{cases} 1 & \text{if } s_1 > s_2 \\ 
0 & \text{if  } s_1 < s_2 
\end{cases}
$$

* Listwise learning to rank: The model use the whole list of documents as input, rather than single or pair of documents.

## Listwise learning to rank

In this section, we will first give a formal definition of listwise based learning to rank. 

First lets give some definitions about for following symbols

* **Query**: $Q = \{  q^{(1)}, q^{(2)}, ... q^{(m)}\}$ is a set of queries, and we use superscripts to denote different queries.
* **Documents**: Each query $q^{(i)}$ is associated with a list of documents $D^{(i)} = \{ d_1^{(i)}, d_2^{(i)}, ..., d_{n^{(i)}}^{(i)}\}$.
* **Relevant Score** Each list of documents $d^{(i)}$ is associated with a relevance score $y^{(i)} = \left( y_1^{(i)}, y_2^{(i)}..., y_{n^{(i)}}^{(i)}\right)$, which indicates how relevant the document is to query
* **Feature Vector of score list** A feature vector $x_j^{(i)} = \Psi(q^{(i)}, d_j^{(i)})$ is created from each document query pair $(q^{(i)}, d_j^{(i)})$. And the feature of a single list is: $x^{(i)} = \left( x_1^{(i)}, x_2^{(i)}, ... x_{n^{(i)}}^{(i)}\right)$.
* **True scores for list**: And the corresponding list of scores $y^{(i)} = \left( y_1^{(i)}, y_2^{(i)}, ..., y_{n^{(i)}}^{(i)}\right)$.
* **Training set**: The whole traing set could be denoted as $\mathcal{\tau} = \{ x^{(i)}, y^{(i)} \}_1^m $ .
* **Ranking function** A ranking function $f$, for each feature vector $x_j^{(i)}$, outputs a score $f(x_j^{(i)})$. For list of feature vectors $x^{(i)}$, we get a list of scores $z^{(i)} = \left(  f(x_1^{(i)}), f(x_2^{(i)}),... f(x_{n^{i}}^{(i)}) \right) $.
* **Loss function**: $\sum_1^m  L(y^{(i)}, z^{(i)})$

### Permutation probability

Suppose that we have a list of intergers $\{ 1,2...n \}$, we denote the permutation as $\pi = \langle \pi(1), \pi(2), ... \pi(n) \rangle$. And here $\pi(j)$ denotes the object at position $j$ in the permutation. So mathematically, there will be totally $ n! $ possible permutations.

Each permutation $\pi$ happens with some probability $P(\pi)$, and $\sum_{\pi \in \Omega } P(\pi) = 1$.And here $\Omega$ is the set of all possible permutations, $\text{size}(\Omega) = n!$. 

But here we want to create a probability distribution of "the score of each pumertation".(or say **each permutation's score will take a probability**) Can we find a such a distribution ? The answer is yes. We can define a score distribution for each permutation as follows:

$$
P_s(\pi) = \prod_{j=1}^{n} \frac{ \phi(s_{\pi(j)}) }{ \sum_{k=j}^{n} \phi(s_{\pi(k)})}
$$

Here is an example, suppose we have three objects $\{1,2,3\}$ having scores $s = (s_1,s_2,s_3)$. The probability of $\pi = \langle 1,2,3 \rangle$ and $\pi' = \langle 3,2,1 \rangle$ are calculated as follows:

$$
P_s(\pi) = \frac{\phi(s_1)}{\phi(s_1)+\phi(s_2)+\phi(s_3)} \cdot  \frac{\phi(s_2)}{\phi(s_2)+\phi(s_3)} \cdot   \frac{\phi(s_3)}{\phi(s_3)} = \frac{1}{6}\times \frac{2}{5} \times 1 = \frac{1}{15}
$$

And

$$
P_s(\pi') =  \frac{\phi(s_3)}{\phi(s_1)+\phi(s_2)+\phi(s_3)}  \cdot   \frac{\phi(s_2)}{\phi(s_2)+\phi(s_1)}  \cdot    \frac{\phi(s_1)}{\phi(s_1)} = \frac{3}{6}\times \frac{2}{5} \times 1  = \frac{1}{5}.
$$

To prove that this $P_s(\pi)$ is a real probability measure, we need to ensure that the sum of all possible permutations is 1. The provement is not that straightforward, but for some special case, we can show it is a probability measure. Suppose that $\phi(s_{\pi(i)}) \equiv 1$ for all $i \in \{1,2,...n!\}$ then we will have:

$$
\sum_{\pi \in \Omega} P_s(\pi) = \frac{1}{n!} + \frac{1}{n!}+...+\frac{1}{n!} = \frac{|\Omega|}{n!} = 1
$$

This probability distribution is mathematically beautifully, and it is easy to show that for the n objects, if $s_1 > s_2 > ... > s_n$, then $P_s(\langle 1,2,...n \rangle)$ is the highest permutation probability and $P_s(\langle n, n-1, ...,1 \rangle)$ is the lowest permutation probability.

If we consider to use current distribution of $P_s(\pi)$ to do learning (and the object function is to do distribution comparison, like KL divergence image below). If we have ground truth label list is $\{ 3,2,1 \}$，and we have a prediction score list is $\{ 0.8, 1.1,0.1 \}$，and if we want to compare these two from the perpective of $P_s(\pi)$ distribution, we have to compute all $P_s(\pi_t) $ with $t \in \{1,2,...3!\}$, that is totally $3!=6$ computations, which is not feasible for large n.

![](https://dibyaghosh.com/blog/assets/posts/kldivergence/forwardkl.png)

So what is the feasible alternatives probability distribution?

The answer is: **Top One Probability**, it respresent the probability of its being ranked on the top given the scores of all objects. Here is the definition:

$$
P_s(j) = \sum_{ \pi(1) = j, \pi \in \Omega} P_s(\pi)
$$

Where $P_s(\pi)$ is permutation probability of $\pi$ given s.

And from the original paper, we know that:

$$
P_s(j) = \frac{\phi(s_j)}{\sum_{k=1}^n \phi(s_k)}
$$
The proof is very easy if you check the appendix A of this blog

So we can use Top One Probability to do the learning from permutation distribution, which is feasible for large n. As we disscussed before, we can use cross entropy as metric. The listwise loss function becomes:

$$
L(y^{(i)}, z^{(i)}) = - \sum_{j=1}^n y_j^{(i)} log(P_{z^{(i)}}(j))
$$
And $y^{(i)}$ is the ground truth permutation, and $z^{(i)}$ is the predicted distribution by the model. 

### Learning method

Suppose we use a neural network model $\omega$ as $f_\omega$, Given a feature vector $x_j^{(i)}$, $f_\omega(x_j^{(i)})$ assign a score to it:

Given a query $q^{(i)}$, we can use the model to get a score list:
$$
z^{(i)}(f_\omega) = \left( f_{\omega}(x_1^{(i)}),  f_{\omega}(x_2^{(i)}), ... f_{\omega}(x_{n^{(i)}}^{(i)})   \right)
$$

Then the top one probability $P_{z^{(i)}}$ is:



**References**

* [What is the difference between Cross-entropy and KL divergence?](https://stats.stackexchange.com/questions/357963/what-is-the-difference-between-cross-entropy-and-kl-divergence)
* [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)

**Appendix**: 

**A: Prove of $\sum_{\pi \in \Omega } P_s(\pi) = 1$** 

$Proof$ by mathematical induction, if there is only one document $d_1$ with score $s_1$ in the list. This equation holds true because there are no other permutations to sum up to make it equal to 1. 

If we add another document $d_2$ with score $s_2$ to the list,  then we have a score list $\{ s_1, s_2\}$ given $s_1 > s_2$. And we can have two permutations: 
$$
\begin{align*}
\pi_1 &= [s_1, s_2] \\
\pi_2 &= [s_2, s_1]
\end{align*}
$$

So here $\Omega = \{ \pi_1, \pi_2 \} $

Thus we can write:

$$
\begin{align*}
\sum_{\pi \in \Omega} P_s(\pi) &= P_s(\pi_1) + P_s(\pi_2) \\
&= \frac{s_1}{s_1 + s_2} \cdot \frac{s_2}{s_2} + \frac{s_2}{s_1 + s_2} \cdot \frac{s_1}{s_1} \qquad \#\text{by definition of } P_s(\pi) \\
&= \frac{s_1}{s_1 + s_2} \sum_{\pi_k \in \Omega_k } P_s(\pi_k)  + \frac{s_2}{s_1 + s_2}  \sum_{\pi_l \in \Omega_l } P_s(\pi_l) \qquad \#\text{by definition of } \sum_{\pi  \in \Omega} P_s(\pi), \text{we create two permutation sets } \Omega_k, \Omega_l, \text{that both has size} 1   \\
&= \frac{s_1}{s_1 + s_2}  + \frac{s_2}{s_1 + s_2}  = 1  \\
\end{align*}
$$

Suppose that we have prove the case of documents list $D_N$ with score list $S_N = \{ s_1, s_2,...s_N \} $ with length $N$, so that we have $\sum_{\pi \in \Omega, n=N } P_s(\pi) = 1 $. Now suppose we add a new document to this set, we can have:

$$
\begin{align*}

\sum_{\pi \in \Omega', n = N+1} P_s(\pi) &=  \frac{s_1}{s_1 + ... s_{N+1}} \sum_{ \pi_1 \in \Omega_1, n=N} P_s(\pi_1)  + ... + \frac{s_{N+1}}{s_1 + ... s_{N+1}}   \sum_{\pi_{N+1}  \in  \Omega_{N+1}, n=N} P_s(\pi_{N+1}) \qquad (1)

\end{align*}
$$

As all these $\sum_{\pi_k \in \Omega_k, n=N} P_s(\pi_k)$ are equal to $1$ by assumption. So we can simplify equation (1):

$$
\begin{align*}
\sum_{\pi \in \Omega', n = N+1} P_s(\pi) &=  1 \\
\end{align*}
$$

If you are still not clear about above proof, This is an example for the case with 3 documents, with scores for each document being $S =\{ s_1, s_2, s_3 \}$ and with constraints: $s_1 > s_2 > s_3$:

$$
\begin{align*}

\sum_{\pi \in \Omega, n = 3} P_s(\pi)&= \left( \sum_{\pi \in \{ [s_1,s_2,s_3],[s_1,s_3,s_2] \}} P_s(\pi) \right) + \left( \sum_{\pi  \in \{ [s_2,s_1,s_3],[s_2,s_3,s_1]\} } P_s(\pi) \right) + \left(  \sum_{\pi  \in \{ [s_3,s_1,s_2],[s_3,s_2,s_1] \} } P_s(\pi) \right)\\

 &=  \frac{s_1}{s_1 + s_2 + s_3}\left( \frac{s_2}{s_2+s_3} \cdot \frac{s_3}{s_3}  + \frac{s_3}{s_3+s_2} \cdot \frac{s_2}{s_2}  \right)   +  \frac{s_2}{s_1 + s_2 + s_3}\left( \frac{s_1}{s_1+s_3} \cdot \frac{s_3}{s_3}  + \frac{s_3}{s_1+s_3} \cdot \frac{s_1}{s_1}  \right)    +\frac{s_3}{s_1 + s_2 + s_3}\left( \frac{s_1}{s_1+s_2}  \cdot \frac{s_2}{s_2}  + \frac{s_2}{s_2+s_1} \frac{s_1}{s_1}  \right)\\

&= \frac{s_1}{s_1 + s_2 + s_3} \sum_{s=\{s_2,s_3\},\pi \in \Omega(s)} P_s(\pi) + \frac{s_2}{s_1 + s_2 + s3} \sum_{s=\{s_1,s_3\},\pi \in \Omega(s)} P_s(\pi) + \frac{s_3}{s_1 + s_2 + s_3} \sum_{s=\{s_1,s_2\},\pi \in \Omega(s)} P_s(\pi)\\

&= \frac{s_1}{s_1 + s_2 + s_3} \cdot 1 + \frac{s_2}{s_1 + s_2 + s_3} \cdot 1 + \frac{s_3}{s_1 + s_2 + s_3} \cdot 1\\

&= 1

\end{align*}
$$




