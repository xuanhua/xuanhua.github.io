---
layout: post
title:  "Simple analysis for birthday paradox"
date:   2024-07-31 10:29:33 +0000
categories: jekyll update
---

In the field of probability, the Birthday Paradox is a problem that illustrates the surprising results when dealing with randomness. The paradox states that in a set of randomly chosen 23 people, there's a 50% chance that at least two will have their birthdays on the same day of the year.

I saw the birthday paradox on a post from Scientific American, but the analysis was not that streightforward and I thought it would be interesting to analyze it myself. So here's my simple analysis:

If we set the probability that there does not exist any two people in the group sharing the same birthday as $P_{\text{opposite}}$, then the probability that at least two people share the same birthday is $1 - P_{\text{opposite}}$.

So let's calculate $P_{\text{opposite}}$ first. We need to find out the numerator and denominator for this probability, intuitively it should be like this:

$$
\begin{align*}
P_{\text{opposite}} &= \frac{ \#\text{Birthdays assignment that no pair shares same birthday} }{ \#\text{Total Birthday assignments}}  \\
&\overset{def}{=} \frac{N_{\text{no share}}}{N_{\text{total}}}
\end{align*}
$$

The $N_{\text{no share}}$ as numerator equal to the total number of ways to distribute 23 people across 365 slots (days in a year). That equals to $365 \times 364 \times ... (365-22)$ . Because each person can be put into one of 365 positions, and this slot could not be shared with others.

For $N_{\text{total}}$, There are 365 days in a year, so there are 365 possible birthdates for each person. Therefore, we have $365^{23}$ possible combinations. 

So finally we have:


$$
\begin{align*}
P &= 1 - \frac{N_{\text{no share}}}{N_{\text{total}}} \\
&=1 - \frac{\prod_{k=0}^{22}(365-k)}{365^{23}}   \\
&=1 -  \frac{365 \times 364 \times ... \times (365 - 22)}{365^{23}} \\
&=0.5072972343239854 \qquad \# \text{Calculated by below python code}
\end{align*}
$$

The result looks like impossible, and it is really anti-intuitive. But it is correct mathematically. This result means that the probability of at least one pair having same birthday in a group of 23 people is about 50%. It's not as small as we might expect.

And you can also use Python to calculate it. Here is the code snippet for that:

```python
# Verification of birthday paradox

n_all = pow(365, 23)
print("All possible combinations: ", n_all)
n_negs = 1
for idx, n_choices in enumerate(range(365, 365-23, -1)):
  n_negs *= n_choices
print("All possible combinations that no pair with the same birthday ", n_negs)

prob = 1 - (n_negs / n_all)
print("\nProbability: ", prob)
```

