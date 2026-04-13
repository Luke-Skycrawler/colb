#### Edge-Edge Cross Product
Mollifier form: $$p = ||e_0 \times e_1||^2$$
Let $a = e_0, b = e_1 - \alpha e_, \alpha = \frac{e_1 \cdot e_0}{||e_0 || ^ 2}$, we have $a\cdot b = 0$.
$$
\begin{aligned}
&\frac{\partial p} {\partial a} \\
= & -2[b][b]a \\
= & -2(bb^T  -I) a
\end{aligned}
$$
This comes from the vector triple product 
$$
a \times b \times c = a\cdot c \, b - a\cdot b\,c
$$
$$
\begin{aligned}
\frac{\partial p} {\partial b} &=  -2(aa^T  -I) b \\
\frac{\partial^2 p} {\partial a^2} &=  -2(bb^T  -I) \\
\frac{\partial^2 p} {\partial b^2} &=  -2(aa^T  -I) \\
\frac{\partial^2 p} {\partial a \partial b} &=  -2(ba^T - 2ab^T + a\cdot b I_3) \\
\end{aligned}
$$
Let's ignore the common -2 coefficient. Write out the determinant of $J = |H - \lambda I|$ as 
$$
\begin{vmatrix}
b^2 - \lambda & & & & 2 ab & \\
& -\lambda & & -ab & & \\
& & b^2 - \lambda & & & \\
& -ab & & -\lambda & & \\
2ab & & & & a^2 - \lambda & \\
& & & & & a^2 - \lambda
\end{vmatrix}
$$
(shortened $||a||, ||b||$ as $a, b$)
$$
\begin {aligned}
J &= (a^2 - \lambda)(b^2 - \lambda) \begin {vmatrix} 
b^2  - \lambda & & & 2ab\\
& -\lambda & -ab & \\
& -ab & -\lambda & \\
2ab & & & a^2 - \lambda
\end{vmatrix} \\
&= (a^2 - \lambda)(b^2 - \lambda)(\lambda ^ 2 - a^2b^2) \begin{vmatrix} 
b^2 - \lambda & 2ab  \\
2ab & a^2 - \lambda
\end{vmatrix} \\
&=(\lambda - a ^ 2)(\lambda - b^2)(\lambda - ab) (\lambda + ab)(\lambda - \lambda_4) (\lambda - \lambda_5)  \\
\text{where}\, &\lambda_{4,5} = \frac{(a^2 + b^2) \pm \sqrt{(a^2 + b^2)^2 + 12 a^2 b^2}}{2} 
\end {aligned} 
$$
where the second line expands the determinant from row 2. The full eigen system is
$$
H_p = \begin{pmatrix}q_0 & \dots &q_5\end{pmatrix}diag(\lambda_0 \dots \lambda_5)\begin{pmatrix}q_0 & \dots &q_5\end{pmatrix}^{-1}
$$

$$
\begin {aligned}
&\lambda_0 = b^2, && q_0^T = (n^T, 0) \\
&\lambda_1 = a^2, && q_1^T = (0, n^T) \\
&\lambda_2 = ab, && q_2^T = (-\hat{b}^T, \hat{a}^T)\\
&\lambda_3 = -ab, && q_3^T = (\hat{b}^T, \hat{a}^T) \\
&q_{4, 5} = (2|b|a^T, (\lambda_{4, 5} - b^2)\hat{b}^T) \\
\end {aligned}
$$
where $n$ is the unit normal of $span(a, b)$. 

Now we compute $\frac{\partial ^ 2 p}{\partial x^2}$. Due to the orthogonality of the frame $c = (a, b)$, we apply the same trick in [1] by decomposing $(\frac{\partial c}{\partial x})$ into $(\frac{\partial c}{\partial x})_{s}$ and $(\frac{\partial c}{\partial x})_{\Delta}$. 

We have 
$$
(\frac{\partial p}{\partial c})^T (\frac{\partial c}{\partial x})_{\Delta} = 0
$$
i.e. $(\frac{\partial c}{\partial x})_{\Delta}$ lies in the nullspace of $(\frac{\partial p}{\partial c})$.

By numerical experiment we have 
$$
\begin{aligned}
\frac{\partial ^ 2 p}{\partial x^2} &= (\frac{\partial c}{\partial x})_s^T(\frac{\partial ^ 2 p}{\partial c^2})(\frac{\partial c}{\partial x})_s\\
\frac{\partial p}{\partial x} &= (\frac{\partial c}{\partial x})_{s} (\frac{\partial p}{\partial c}) \\
(\frac{\partial c}{\partial x})_{s} &= \begin{pmatrix}1  & 0 \\
-\alpha &1\end{pmatrix} \otimes I_3
\end{aligned}
$$
#### Point-Edge
Let $c = (e_0, e_{1\perp}, e_{2\perp})$, the unsigned distance $u = ||e_{2\perp} ||$. Let $x = (p, edge_0, edge_1)$. $e_2 = x_0 - x_1, e_0 = x_2 - x_1$. $e_1$ is selected randomly, we can assign it as $e_0 \times e_2$ for convenience.

$$
(\frac{\partial c}{\partial x})_{s} = \begin{pmatrix}0 &-1 &1 \\
0& 0 & 0 \\
1 &\alpha - 1 & -\alpha
\end{pmatrix} \otimes I_3
$$
$$
(\frac{\partial c}{\partial x})_{\Delta} = \begin{pmatrix}0_9 \\
0_9 \\
e_0 \otimes \frac{\partial\alpha}{\partial x}^T
\end{pmatrix} \otimes I_3
$$

$$
\frac{\partial ^ 2 u}{\partial x^2} = (\frac{\partial c}{\partial x})_s^T(\frac{\partial ^ 2 u}{\partial c^2})(\frac{\partial c}{\partial x})_s - (\frac{\partial c}{\partial x})_{\Delta}^T(\frac{\partial ^ 2 u}{\partial c^2})(\frac{\partial c}{\partial x})_{\Delta}
$$

#### Reference
[1]: Alvin Shi and Thedore Kim. A Unified Analysis of Penalty-Based Collision Energies. 