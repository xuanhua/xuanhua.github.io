---
layout: post
title:  "How does Taylor series derived"
date:   2024-08-19 10:29:33 +0000
categories: jekyll update
---

Taylor series is a method of finding an approximation for a function. It's based on the idea that any function can be approximated by a polynomial, and this polynomial will have better accuracy as we increase the number of terms in our sum.

Taylor series of different functions take different forms and it's hard to remember all of them. 

So is there a general method for deriving Taylor series? Yes, there is. Here's how it works: (for simplity, we use Maclaurin series here. Maclaurin series is a special case of Taylor series where $n=0$ in Taylor series formula.)

Suppose we have a function $f(x)$ that is infinitely differentiable at $0$; From the definition of calculation. we have:

$$
f(x) - f(0) = \int_{0}^{x} f'(t) dt \qquad (1)
$$

Move $f(0)$ to the right side, we get:

$$
\begin{align*}
f(x) &= f(0) + \int_{0}^{x} f'(t) dt  \\
& \overset{t \mapsto x-t }{=} f(0) + \int_{x-0}^{x-x} f'(x - t ) d(x-t)  \\
&= f(0) + \int_{x}^0 f'(x-t)(-1)dt \\
&= f(0) + \int_{0}^{x} f'(x - t ) dt   \\
\end{align*} \qquad (2)
$$ 

Where $t \mapsto x - t$ is a variable replacement.

Before we apply integrating by parts on $(2)$, we review how it works.

Given two functions $u(x)$ and $v(x)$, derivatives of their products is:

$$
(u(x)v(x))' = u(x)'v(x) + u'(x)v(x)  \qquad (3)
$$

For more compactly, we can write:

$$
(uv)' = u'v + uv'   \qquad  (4)
$$

Applying integration by $x$ on both sides, we get:

$$
\begin{align*}
(uv) &= \int u'v dx + \int uv' dx \\
&= \int vdu + \int udv
\end{align*} \qquad (5)
$$

So we can have formula of integration by parts by re-aranging $(5)$ and use $u(x),v(x)$ to replace $u,v$ in necessary part:

$$
\int u(x) dv = v(x)u(x) - \int v(x)du   \qquad  (6)
$$

Now we turn back to formula $(2)$, applying integration by parts on $\int_0^x f'(x - t ) dt$, and let $u(t)=f'(x-t)$ and $v(t)=t$ so that we have:

$$
\begin{align*}
\int_0^{x} f'(x - t ) dt &= \int_0^x u(t) dv \\
&= u(t)v(t)\bigg\rvert_{t=0}^{x} - \int_0^{x} v(t)du   \\
&= t\cdot f'(x-t) \bigg\rvert_{t=0}^{x} - \int_0^{x} v(t)d(f'(x-t))   \\
&= t\cdot f'(x-t) \bigg\rvert_{t=0}^{x} - \int_0^{x} v(t)f''(x-t)d(x-t)   \\
&=t\cdot f'(x-t) \bigg\rvert_{t=0}^{x} - \int_0^{x} t \cdot f''(x-t)(-1)dt    \\
&=x\cdot f'(0) + \int_0^x t \cdot f''(x-t)dt
\end{align*} \qquad (7)
$$

Bring equation $(7)$ back to equation $(2)$, we have:

$$
f(x) = f(0) +  f'(0)\cdot x + \int_0^{x} t  \cdot f''(x-t)dt   \qquad (8)
$$

Again we can apply integration by parts on $\int_0^{x} t   \cdot f''(x-t)dt$, let $u(t) = f''(x-t)$ and $v(t) = \frac{1}{2}t^2$, so that we have:

$$
\begin{align*}

\int_0^{x} t \cdot f''(x-t)dt &= u(t)v(t)\bigg\rvert_{t=0}^{x} -  \int_0^{x} v(t)du \\
&= f''(x-t)\cdot \frac{1}{2}x^2 \bigg |_{t=0}^x - \int_0^x \frac{1}{2}t^2 d(f''(x-t)) \\
&= \frac{1}{2}x^2 \cdot f''(0) + \int_0^x \frac{1}{2}t^2 \cdot f'''(x-t)dt

\end{align*} \qquad (9)
$$

Bring equation $(9)$ back to equation $(8)$, we have:

$$
\begin{align*}
f(x) &= f(0) + f'(0)\cdot x  + \frac{1}{2}x^2\cdot f''(0) +  \int_0^x  \frac{1}{2}t^2  \cdot f'''(x-t)dt \\
&= f(0) + f'(0)\cdot x + \frac{1}{2}x^2 \cdot f''(0) + ... + \frac{1}{n!}x^n\cdot f^n(0) + \int_0^x \frac{1}{n!}t^n \cdot f^{n+1}(x-t)dt \\
\end{align*}   \qquad (10)
$$







**Reference**

* [How are the Taylor Series derived?](https://math.stackexchange.com/questions/706282/how-are-the-taylor-series-derived)
* [Integration by parts](https://en.wikipedia.org/wiki/Integration_by_parts)
