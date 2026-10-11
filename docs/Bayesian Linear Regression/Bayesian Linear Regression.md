# Bayesian Linear Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 440
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Christopher M. Bishop
Date: January 1, 2006
url: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf

> **Hồi quy tuyến tính Bayes** trả lời một câu mà hồi quy tuyến tính thường không trả lời
> được: *"đường nào cũng hợp lý, và ta chắc tới đâu?"* Đầu ra không phải một bộ hệ số mà là
> một **phân phối trên hệ số**, và từ đó một **phân phối trên dự báo** — mỗi điểm dự báo đi
> kèm một độ rộng riêng của nó.
>
> Phần thứ hai, và là phần ít ai nhắc: cùng bộ máy đó cho một cách **chọn mức chính quy hoá
> bằng chính dữ liệu huấn luyện**, không cần tập giữ lại và không cần kiểm định chéo. Bài này
> đo cả hai phần.

---

## 0. Trước khi đọc

### 0.1 Bài này nhận gì, trả ra gì

| | Hồi quy tuyến tính thường (bình phương tối thiểu) | Hồi quy tuyến tính Bayes |
|---|---|---|
| **Vào** | `X` shape `(N, D)`, `t` shape `(N,)` | thêm **hai số**: `alpha` (độ chính xác tiên nghiệm) và `beta` (độ chính xác nhiễu) |
| **Ra** | `w` shape `(M,)` — **một** bộ hệ số | `m_N` shape `(M,)` **và** `S_N` shape `(M, M)` — trung bình và hiệp phương sai của **một phân phối** trên hệ số |
| **Dự báo tại `x*`** | một số | một số **và** một độ lệch chuẩn `sigma(x*)`, đổi theo `x*` |
| **Chọn siêu tham số** | kiểm định chéo trên tập giữ lại | tối đa hoá **evidence** trên chính tập huấn luyện |

Ở đây $M$ là **số hàm cơ sở**, kể cả hàm hằng $\phi_0(x)=1$. Với dữ liệu một chiều và
chín hàm cơ sở Gauss thì $M = 10$; `X` shape `(N, 1)` nhưng ma trận thiết kế
$\mathbf\Phi$ shape `(N, 10)`.

Một ví dụ chạy được, không cần thư viện nào ngoài `numpy`:

```python
# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

# ba điểm, mô hình y = w0 + w1*x
X = np.array([0.0, 1.0, 2.0])
t = np.array([1.0, 2.0, 2.0])
Phi = np.c_[np.ones(len(X)), X]          # (3, 2) — cột đầu là hàm cơ sở hằng
alpha, beta = 1.0, 2.0                   # tiên nghiệm N(0, 1/alpha) · nhiễu N(0, 1/beta)

SN_inv = alpha * np.eye(2) + beta * Phi.T @ Phi        # (3.54)
SN = np.linalg.inv(SN_inv)
mN = beta * SN @ Phi.T @ t                             # (3.53)

x_moi = np.array([3.0])
Phi_moi = np.c_[np.ones(1), x_moi]
tb = Phi_moi @ mN                                       # (3.58) trung bình dự báo
var = 1.0 / beta + np.einsum("ij,jk,ik->i", Phi_moi, SN, Phi_moi)   # (3.59)

print("m_N      =", np.round(mN, 6))          # [0.926829 0.585366]
print("S_N      =", np.round(SN, 6).tolist())
print("dự báo tại x = 3: %.6f ± %.6f" % (tb[0], var[0] ** 0.5))
```

Đầu ra: `m_N = [0.926829 0.585366]`, và dự báo tại $x=3$ là
$2{,}682927 \pm 1{,}194500$. Mục 0.4 tính lại đúng ba con số đó **bằng phân số**, không
dùng số thực dấu phẩy động, nên chúng kiểm được bằng mắt.

**Điều duy nhất phải nhớ về hình dạng.** Đầu ra không phải một điểm trong không gian
$M$ chiều mà là một **hình ellipsoid** trong không gian đó, và ellipsoid ấy co lại khi
thêm dữ liệu. Mọi thứ còn lại của bài là hệ quả của câu này.

### 0.2 Bảng ký hiệu — và sáu chỗ trùng tên

Chương 3 của Bishop dùng khoảng hai mươi ký hiệu, và phần chặn người mới nhiều nhất
không phải công thức mà là **không đọc nổi chúng thành tiếng**.

| Ký hiệu | Đọc là gì | Nghĩa trong bài này | Chiều |
|---|---|---|---|
| $t$ | "tê" | giá trị mục tiêu của **một** quan sát | vô hướng |
| $\mathsf{t}$ | "tê in thẳng" | vectơ gom mọi $t_n$ của tập huấn luyện | $N$ |
| $\mathbf w$ | "vê kép đậm" | vectơ tham số — thứ cần suy ra | $M$ |
| $\phi_j(x)$ | "phi chỉ số j của x" | hàm cơ sở thứ $j$, một hàm số | vô hướng |
| $\boldsymbol\phi(x)$ | "phi đậm của x" | gom mọi $\phi_j(x)$ cho **một** điểm $x$ | $M$ |
| $\boldsymbol\varphi_j$ | "phi cong chỉ số j" | cột thứ $j$ của $\mathbf\Phi$ — hàm cơ sở $j$ đo tại **mọi** điểm | $N$ |
| $\mathbf\Phi$ | "phi hoa đậm" | ma trận thiết kế, $\Phi_{nj}=\phi_j(x_n)$ | $N \times M$ |
| $\alpha$ | "alpha" | **độ chính xác tiên nghiệm** — $\alpha$ lớn = tin rằng $\mathbf w$ nhỏ | vô hướng |
| $\beta$ | "beta" | **độ chính xác nhiễu** = $1/\sigma^2_{\text{nhiễu}}$ | vô hướng |
| $\mathbf m_N$ | "em en" | trung bình hậu nghiệm sau $N$ điểm — cũng là điểm MAP | $M$ |
| $\mathbf S_N$ | "ét en" | hiệp phương sai hậu nghiệm sau $N$ điểm | $M \times M$ |
| $\mathbf A$ | "a" | ma trận Hesse của hàm lỗi, $\mathbf A = \mathbf S_N^{-1}$ | $M \times M$ |
| $\sigma_N^2(x)$ | "sigma bình phương en của x" | phương sai **dự báo** tại $x$ | vô hướng |
| $\lambda$ | "lambda" | hệ số chính quy hoá $= \alpha/\beta$ | vô hướng |
| $\lambda_i$ | "lambda chỉ số i" | **trị riêng** thứ $i$ của $\beta\mathbf\Phi^{\!\top}\mathbf\Phi$ | vô hướng |
| $\gamma$ | "gamma" | **số tham số hiệu dụng**, $0 \le \gamma \le M$ | vô hướng |
| $k(x,x')$ | "ka của x, x phẩy" | nhân tương đương — trọng số mà $t$ tại $x'$ nhận khi dự báo tại $x$ | vô hướng |
| $E_D(\mathbf w)$ | "e đê" | nửa tổng bình phương phần dư | vô hướng |
| $E_W(\mathbf w)$ | "e vê kép" | $\tfrac12\mathbf w^{\!\top}\mathbf w$ | vô hướng |
| $\mathcal D$ | "đê viết hoa kiểu thư pháp" | tập dữ liệu, xem như **một** đối tượng | — |
| $\mathcal M_i$ | "em chỉ số i" | mô hình thứ $i$ trong một họ mô hình | — |
| $\mathcal N(x \mid \mu, \Sigma)$ | "en của x, cho biết mu và Sigma" | phân phối Gauss của $x$ với trung bình $\mu$, hiệp phương sai $\Sigma$ | — |
| $\mid$ | "cho biết" / "biết rằng" | vạch điều kiện: $p(a\mid b)$ = xác suất của $a$ **khi đã biết** $b$ | — |
| $\propto$ | "tỉ lệ với" | bằng nhau **sai khác một hằng số** không phụ thuộc biến đang xét | — |
| $\sim$ | "tuân theo phân phối" | $x \sim \mathcal N(0,1)$: $x$ được rút từ phân phối đó | — |
| $\boldsymbol\psi(x)$ | "psi của x" | ánh xạ đặc trưng ở (3.65), $k(x,z)=\boldsymbol\psi(x)^{\!\top}\boldsymbol\psi(z)$ | $M$ |
| $\Delta w_{\text{posterior}}$ | "delta vê kép hậu nghiệm" | bề rộng của hậu nghiệm theo một hướng | vô hướng |

**Bốn chỗ trùng tên nằm NGAY TRONG chương 3**, và cả bốn đều gây lỗi thật khi viết mã:

| Chữ | Nghĩa ở chỗ này | Nghĩa ở chỗ kia | Cách nhau |
|---|---|---|---|
| $\lambda$ | (3.27) tr.144: **hệ số chính quy hoá** | (3.87) tr.168: **trị riêng** của $\beta\mathbf\Phi^{\!\top}\mathbf\Phi$ | 24 trang |
| $\sigma$ | (3.6) tr.139: **hàm sigmoid logistic** $\sigma(a)$ | (3.59) tr.156: $\sigma_N^2(x)$ là **phương sai dự báo**; (3.96) tr.171: $\sigma_{\text{ML}}^2$ là phương sai ước lượng | 17 trang |
| $M$ | (3.86) tr.167: **số chiều của $\mathbf w$** | trục ngang Hình 3.14 tr.168: **bậc của đa thức**, tức $M_{\text{chiều}} = M_{\text{bậc}}+1$ | 1 trang |
| $\phi$ | $\phi_j(x)$ là một **hàm**; $\boldsymbol\phi(x_n)$ là một **hàng** của $\mathbf\Phi$ | $\boldsymbol\varphi_j$ là một **cột** của $\mathbf\Phi$, dài $N$ | cùng trang 143 |

Dòng $M$ là dòng tốn nhất: mục 2.24 của bài này phải dựng lại Hình 3.14, và nếu đọc
$M$ của trục ngang thành số chiều của $\mathbf w$ thì mọi con số lệch đi một tham số.
Dòng $\phi$ thì chính Bishop tự cảnh báo ở tr.143 — *"$\boldsymbol\varphi_j$ corresponds
to the jth column of $\mathbf\Phi$, whereas $\boldsymbol\phi(x_n)$ corresponds to the nth
row"* — và đó là một câu phải đọc hai lần.

**Hai chỗ trùng tên với các bài KHÁC của database**, nguy hiểm hơn vì nó làm người đọc
chép mã sai mà không báo lỗi:

| Chữ | Ở bài này | Ở bài khác |
|---|---|---|
| $\beta$ | **độ chính xác nhiễu**, một số dương lớn (ở đây 11,1) | [Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) · [Lasso Regression](<../Lasso Regression/Lasso Regression.md>) · [Quantile Regression](<../Quantile Regression/Quantile Regression.md>): $\beta$ là **vectơ hệ số**, tức thứ bài này gọi là $\mathbf w$ |
| $\alpha$ | **độ chính xác tiên nghiệm** | Ridge/Lasso của `scikit-learn`: `alpha` là **mức phạt**; và trong `BayesianRidge` thì `alpha_` lại là $\beta$ của bài này — xem mục 3 |

Chỗ `alpha_` của `scikit-learn` là chỗ đảo nghĩa hoàn toàn: `alpha_` của `BayesianRidge`
**chính là $\beta$** của Bishop (độ chính xác nhiễu), còn `lambda_` mới là $\alpha$. Đọc
nhầm hai thuộc tính đó cho một mô hình hoàn toàn khác, và không có gì báo lỗi. Mục 3 đo
phần chênh.

### 0.3 Mười hai từ mới, định nghĩa tại chỗ

| Từ | Nghĩa đủ dùng ngay trong bài này |
|---|---|
| **tiên nghiệm** (prior) | phân phối ta đặt lên $\mathbf w$ **trước khi** xem dữ liệu. Ở đây nó nói "hệ số nên nhỏ", mạnh hay yếu tuỳ $\alpha$. |
| **hậu nghiệm** (posterior) | phân phối của $\mathbf w$ **sau khi** đã xem dữ liệu. Nó là tiên nghiệm nhân hàm hợp lẽ rồi chuẩn hoá lại. |
| **hàm hợp lẽ** (likelihood) | $p(\mathsf t \mid \mathbf w)$ — xác suất thấy đúng bộ dữ liệu này **nếu** tham số đúng là $\mathbf w$. Đọc theo chiều $\mathbf w$, không theo chiều $\mathsf t$. |
| **liên hợp** (conjugate) | tiên nghiệm mà khi nhân với hàm hợp lẽ thì hậu nghiệm **cùng họ phân phối**. Gauss liên hợp với Gauss ở mô hình này, nên hậu nghiệm có công thức đóng. |
| **độ chính xác** (precision) | nghịch đảo phương sai. $\beta = 1/\sigma^2$. Dùng nó thay phương sai vì mọi công thức của bài gọn hơn hẳn. |
| **MAP** | *maximum a posteriori* — điểm cao nhất của hậu nghiệm. Với hậu nghiệm Gauss thì MAP **trùng** trung bình. |
| **phân phối dự báo** (predictive distribution) | phân phối của $t$ tại một $x$ mới, đã **tích ra** hết $\mathbf w$: nó gộp cả nhiễu lẫn độ bất định về tham số. |
| **evidence** / **marginal likelihood** | $p(\mathsf t\mid\alpha,\beta)$ — xác suất của dữ liệu khi $\mathbf w$ đã bị tích ra. Đây là đại lượng dùng để **chọn** $\alpha$, $\beta$ và chọn mô hình. |
| **Bayes factor** | tỉ số hai evidence của hai mô hình. Lớn hơn 1 nghĩa là dữ liệu nghiêng về mô hình thứ nhất. |
| **nhân tương đương** (equivalent kernel) | trọng số $k(x,x_n)$ mà giá trị $t_n$ nhận được khi ta dự báo tại $x$. Dự báo Bayes là một **tổ hợp tuyến tính của các $t_n$**, và $k$ là bộ trọng số đó. |
| **số tham số hiệu dụng** $\gamma$ | không phải $M$, mà là "bao nhiêu tham số thật sự được dữ liệu quyết định". Nằm trong $[0, M]$. |
| **khoảng tin cậy Bayes** (credible interval) | khoảng chứa $x\,\%$ xác suất **hậu nghiệm**. Khác *confidence interval* của thống kê tần suất: ở đây câu "95 % xác suất giá trị nằm trong khoảng này" là một câu **hợp lệ**; ở kia thì không. |

### 0.4 Ví dụ chạy tay — ba điểm, hai tham số, tính bằng phân số

Bishop không cho một ví dụ số nhỏ nào chạy tay được. Mục này dựng một ví dụ như thế, và
mọi con số dưới đây do `tools/v_bayes.py` mục 5 in ra bằng **số hữu tỉ chính xác**
(`fractions.Fraction`), không phải số thực — nên đọc bằng mắt là kiểm được.

Dữ liệu: ba điểm $(0,1)$, $(1,2)$, $(2,2)$. Mô hình $y = w_0 + w_1 x$. Chọn $\alpha=1$,
$\beta=2$.

**Bước 1 — ma trận thiết kế.** $\mathbf\Phi$ có cột đầu toàn số 1 (hàm cơ sở hằng
$\phi_0=1$) và cột thứ hai là $x$:

$$\mathbf\Phi = \begin{pmatrix}1 & 0\\ 1 & 1\\ 1 & 2\end{pmatrix},\qquad
\mathbf\Phi^{\!\top}\mathbf\Phi = \begin{pmatrix}3 & 3\\ 3 & 5\end{pmatrix},\qquad
\mathbf\Phi^{\!\top}\mathsf t = \begin{pmatrix}5\\ 6\end{pmatrix}$$

**Bước 2 — nghịch đảo hiệp phương sai hậu nghiệm**, theo (3.54):

$$\mathbf S_N^{-1} = \alpha\mathbf I + \beta\mathbf\Phi^{\!\top}\mathbf\Phi
= \begin{pmatrix}1&0\\0&1\end{pmatrix} + 2\begin{pmatrix}3&3\\3&5\end{pmatrix}
= \begin{pmatrix}7&6\\6&11\end{pmatrix}$$

Phép cộng ấy là toàn bộ nội dung của bước này: **độ chính xác của tiên nghiệm cộng độ chính
xác mà dữ liệu mang vào**. Ma trận đơn vị là phần niềm tin ban đầu, ma trận $2\,
\mathbf\Phi^{\!\top}\mathbf\Phi$ là phần ba điểm dữ liệu đóng góp, và chúng cộng thẳng được
vì thông tin cộng được — xem mục 2.3.

Định thức bằng $7\cdot11-6\cdot6 = 41$, nên

$$\mathbf S_N = \frac{1}{41}\begin{pmatrix}11&-6\\-6&7\end{pmatrix}$$

Đây là **hình dạng của đám mây niềm tin** về $\mathbf w$: hai số trên đường chéo là độ rộng
theo từng hệ số, và số âm $-6/41$ ngoài đường chéo nói rằng hai hệ số **tương quan âm** — nếu
đoán hệ số chặn cao hơn thì phải đoán độ dốc thấp hơn để đường vẫn đi qua đám dữ liệu.

**Bước 3 — trung bình hậu nghiệm**, theo (3.53):

$$\mathbf m_N = \beta\mathbf S_N\mathbf\Phi^{\!\top}\mathsf t
= \frac{2}{41}\begin{pmatrix}11&-6\\-6&7\end{pmatrix}\begin{pmatrix}5\\6\end{pmatrix}
= \frac{1}{41}\begin{pmatrix}38\\24\end{pmatrix}
= \begin{pmatrix}0{,}926829\\0{,}585366\end{pmatrix}$$

**Bước 4 — đối chiếu với ridge.** Giải bài toán bình phương tối thiểu có phạt
$\lambda = \alpha/\beta = 1/2$ trên cùng dữ liệu cho **đúng** $\left[\tfrac{38}{41},
\tfrac{24}{41}\right]$ — bằng nhau theo **phân số**, không phải xấp xỉ. Đó là nội dung
(3.55), và mục 2.5 nói vì sao.

Không có tiên nghiệm thì nghiệm là $\mathbf w_{\text{ML}} = \left[\tfrac76, \tfrac12\right]
= [1{,}166667;\,0{,}500000]$ — cả hai hệ số đều **lớn hơn** bản Bayes. Đó là phép co
(*shrinkage*) mà tiên nghiệm gây ra, nhìn thấy bằng mắt.

**Bước 5 — dự báo tại $x^\ast = 3$**, theo (3.58) và (3.59):

$$\text{trung bình} = \mathbf m_N^{\!\top}\boldsymbol\phi(3) = \frac{38}{41}+3\cdot\frac{24}{41}
= \frac{110}{41} = 2{,}682927$$

$$\sigma_N^2(3) = \underbrace{\frac1\beta}_{1/2}
+ \underbrace{\boldsymbol\phi(3)^{\!\top}\mathbf S_N\boldsymbol\phi(3)}_{38/41}
= \frac{117}{82} = 1{,}426829$$

Hai số hạng đó là **hai nguồn khác nhau** của độ bất định, và ở đây tỉ lệ của chúng là
**35,04 %** cho nhiễu và **64,96 %** cho độ bất định tham số. Độ lệch chuẩn dự báo bằng
$1{,}194500$, nên khoảng 95 % là $2{,}682927 \pm 2{,}341219$.

**Chỗ đáng nhớ nhất của ví dụ này**, và là chỗ hồi quy thường không có: cùng phép tính
tại $x^\ast=1$ — một điểm nằm **giữa** dữ liệu — cho $\boldsymbol\phi^{\!\top}\mathbf
S_N\boldsymbol\phi = \tfrac{6}{41} = 0{,}146341$, tức **nhỏ hơn 6,3333 lần** so với ở
$x^\ast=3$. Cùng một mô hình, cùng một tập dữ liệu, mà độ bất định **đổi theo chỗ ta
hỏi**. Bình phương tối thiểu trả về một con số duy nhất và không có chỗ nào để nói điều
đó.

### 0.5 Nguồn của bài, và ba hạn chế phải nói trước

**Nguồn ruột là chương 3 của Bishop 2006**, không phải một bài báo. Hồi quy tuyến tính
Bayes không có một bài báo gốc đơn lẻ: phần hậu nghiệm liên hợp là kết quả cổ điển của
thống kê Bayes, phần evidence approximation thì Bishop tự dẫn về Gull (1989) và
MacKay (1992a). Chương 3 là chỗ **mọi mảnh được viết ra đầy đủ và nhất quán một hệ ký
hiệu**, nên nó là nguồn của bài này, và bài đi theo đúng thứ tự của nó.

Đếm bằng `v_bayes.py` mục 2, trên chính lớp text của PDF:

| Đo | Con số |
|---|---|
| số trang của chương 3 | **41** (sách tr.137–177) |
| Figure 3.x | **17** |
| **Table 3.x** | **0** |
| công thức đánh số ở thân chương | **99** — từ (3.1) tới (3.99) |
| công thức đánh số ở phần bài tập | **20** — từ (3.100) tới (3.119) |
| trong đó thuộc mục 3.3–3.5 của sách, tức phần ruột của bài này | **52** — từ (3.48) tới (3.99) |
| mục đánh số | 3.1 · 3.1.1 · 3.1.2 · 3.1.3 · 3.1.4 · 3.1.5 · 3.2 · 3.3 · 3.3.1 · 3.3.2 · 3.3.3 · 3.4 · 3.5 · 3.5.1 · 3.5.2 · 3.5.3 · 3.6 |

**Dòng `Table 3.x = 0` quyết định hình dạng của cả bài.** Chương 3 không có một bảng số
nào, nên **không con số nào của nguồn có thể trích bằng cách đọc một ô bảng**. Mọi con
số của Bishop nằm trong hình. Vì thế phần lớn công việc kiểm chứng của bài này là **đọc
lớp vectơ của PDF** rồi dựng lại — cùng kỹ thuật đã dùng ở bài
[Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) và
[Lasso Regression](<../Lasso Regression/Lasso Regression.md>), nhưng ở đây nó không phải một lựa chọn mà là
đường duy nhất.

Ba hạn chế, ghi ra ngay để không bị đọc thành lời hứa:

1. **Hình 3.7 và Hình 3.10 là ẢNH RASTER.** `v_bayes.py` mục 3 đếm được: trang sách 155
   (Hình 3.7) có **2 đối tượng ảnh và 0 đường vectơ**, còn trang 157 (Hình 3.8) có
   **56 đường và 1 243 đoạn**. Hai hình raster đó không đọc được toạ độ; bài chỉ dựng
   lại được **thiết lập** của chúng bằng mô phỏng, không đối chiếu được từng giá trị.
2. **Hình 3.16 KHÔNG CÓ trục dọc.** Lớp text của trang 172 chỉ chứa nhãn `ln α` và
   `−5 0 5`; không một con số nào trên trục đứng của cả hai ô. Với hình đó chỉ đọc được
   **hoành độ** — ba hoành độ, xem mục 2.30.
3. **Bốn tham số của mô hình vẽ hình thì bài báo không khai**: tâm $\mu_j$ và bề rộng
   $s$ của các hàm cơ sở Gauss, giá trị $\alpha$ của Hình 3.8, và giá trị $\beta$ của
   Hình 3.14. Bài này **dò ra** chúng từ chính các đường được vẽ, và ghi rõ chỗ nào là
   dò ra, chỗ nào là con số bài báo tự in.

**Thời gian chạy của `v_bayes.py` không được ghi vào bài.** Nó là một con số đồng hồ,
chỉ tái lập trong đúng một lần chạy; ghi nó ra là tạo một phép kiểm báo xanh giả.
Script chạy dưới nửa phút trên máy dựng bài, và mọi con số khác của bài **không** phụ
thuộc đồng hồ.

---

## 1. Trước Bayes người ta làm sao, và tắc ở đâu

Cách cũ là **bình phương tối thiểu**, có hoặc không có phạt. Nó cho một bộ hệ số, và mọi
thứ khác phải xử lý bên ngoài: mức phạt chọn bằng kiểm định chéo, độ bất định thì không có.
Bishop nêu đúng ba chỗ tắc, và nêu bằng lời:

![Mở đầu mục 3.3 của sách của Bishop 2006](<paper_bishop2006_mo_dau_3_3.png>)

**Hình 1 — mở đầu mục 3.3 của sách, nơi bài báo nêu bế tắc của cực đại hoá hợp lẽ.** Nguồn:
Bishop 2006, trang 152, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"This leaves the issue of deciding the appropriate model complexity for the particular
> problem, which cannot be decided simply by maximizing the likelihood function, because
> this always leads to excessively complex models and over-fitting."*

> *"Independent hold-out data can be used to determine model complexity, as discussed in
> Section 1.3, but this can be both computationally expensive and wasteful of valuable
> data."*

**Dịch:** *"Còn lại vấn đề quyết định độ phức tạp mô hình phù hợp cho từng bài toán cụ
thể, và vấn đề đó không giải được bằng cách chỉ cực đại hoá hàm hợp lẽ, vì làm thế thì
luôn dẫn tới mô hình quá phức tạp và quá khớp."* · *"Dữ liệu giữ lại độc lập có thể dùng
để xác định độ phức tạp mô hình, như đã bàn ở mục 1.3, nhưng cách đó vừa tốn kém về tính
toán vừa lãng phí dữ liệu quý."*

**Cách đọc khung:** ba tính từ trong đoạn này — *excessively complex*, *computationally
expensive*, *wasteful* — là ba lời hứa sẽ được đo ở ba tiểu mục dưới đây, trên **chính
mười điểm dữ liệu của Bishop**.

### 1.1 "Excessively complex" — đo bằng số

Mười điểm dữ liệu của mục 1.1 sách Bishop không được in ở đâu trong sách. Bài
[Polynomial Regression](<../Polynomial Regression/Polynomial Regression.md>) đã dựng lại chúng từ cột $M=9$ của
Bảng 1.1 tr.8 — cột đó là đa thức **đi qua đúng** mười điểm nên hệ số của nó mã hoá chính
dữ liệu. Bài này dùng lại kết quả đó:

$$x_n = 0,\ \tfrac19,\ \tfrac29,\ \ldots,\ 1 \qquad (\text{tức } \texttt{linspace(0,1,10)})$$

$$\mathsf t = (+0{,}3500,\ +0{,}8315,\ +1{,}0081,\ +0{,}9723,\ +0{,}1338,\ +0{,}1673,\ -0{,}8481,\ -0{,}4458,\ -0{,}5642,\ +0{,}2600)$$

Khớp một đa thức bậc 9 vào mười điểm đó **không** có chính quy hoá:

| Đại lượng | Không tiên nghiệm (bình phương tối thiểu) | $\alpha$, $\beta$ lấy từ evidence |
|---|---:|---:|
| RMSE trên tập huấn luyện | $9{,}192\times10^{-11}$ | — |
| RMSE trên 2 000 điểm kiểm tra | **1,291709** | **0,501726** |
| tỉ số RMSE kiểm tra / RMSE huấn luyện | $1{,}405\times10^{10}$ | — |
| $\lvert w \rvert$ lớn nhất | $1{,}062\times10^{6}$ | **0,9596** |
| số tham số hiệu dụng $\gamma$ trên $M = 10$ | 10 (mọi tham số tự do) | **2,9511** |

Hai dòng phải đọc cùng nhau. Dòng đầu là định nghĩa của quá khớp: sai số huấn luyện gần
như **bằng không**, sai số kiểm tra lớn gấp mười tỉ lần. Dòng $\lvert w\rvert$ thì tự
kiểm: con số $1{,}062\times10^{6}$ dựng lại được **đúng** bằng hệ số lớn nhất mà chính
Bảng 1.1 tr.8 của Bishop in ra, $-1\,061\,800{,}52$.

Cùng bậc 9, cùng dữ liệu, nhưng lấy $\alpha$ và $\beta$ từ evidence (mục 2.26) thì RMSE
kiểm tra **giảm 2,575 lần** và hệ số lớn nhất tụt từ hơn một triệu xuống **dưới 1**. Số
tham số hiệu dụng khi đó là **2,9511** chứ không phải 10 — mô hình có mười tham số nhưng
dữ liệu chỉ quyết định được gần ba.

### 1.2 "Computationally expensive" — đếm số lần giải hệ

Phép tính đắt nhất của cả hai đường là **giải một hệ $M\times M$**. Đếm nó:

| Cách chọn mức chính quy hoá | Số lần giải hệ $M\times M$ |
|---|---:|
| kiểm định chéo 5 mảnh × lưới 20 giá trị $\lambda$ | 100 |
| kiểm định chéo 5 mảnh × lưới 100 giá trị $\lambda$ | 500 |
| kiểm định chéo 10 mảnh × lưới 100 giá trị $\lambda$ | **1 000** |
| kiểm định chéo 10 mảnh × lưới 1 000 giá trị $\lambda$ | 10 000 |
| **evidence, công thức (3.92) + (3.95)** | **34** (17 vòng lặp × 2 lần giải) |

Rẻ hơn **29,4 lần** so với dòng in đậm ở giữa. Và chỗ đáng kể hơn con số: đường evidence
**không cần một lưới nào** — nó không phải tìm kiếm mà là một phép lặp điểm bất động,
nên không phải chọn trước biên của lưới.

### 1.3 "Wasteful of valuable data" — đo độ ổn định

Với $N = 10$ thì kiểm định chéo 5 mảnh huấn luyện mỗi lần trên **8 điểm**. Chạy 60 cách
chia mảnh khác nhau, mỗi lần chọn $\lambda$ trên lưới 120 giá trị:

| Đo | Kiểm định chéo 5 mảnh | Evidence |
|---|---|---|
| $\lambda$ chọn được | từ **0,331901** tới **1,1231** | **0,565684** |
| trải rộng | **3,4 lần** | không trải — một giá trị |
| số giá trị phân biệt được chọn | **8** | 1 |
| phụ thuộc cách chia mảnh | có | **không** |
| RMSE trên 2 000 điểm kiểm tra | 0,398823 (dùng trung vị) | **0,393342** |

Câu đúng ở đây **không** phải "evidence tốt hơn kiểm định chéo": chênh lệch RMSE chỉ
**1,39 %**, quá nhỏ để kết luận từ một bộ dữ liệu. Câu đúng là **evidence trả lời tất
định, kiểm định chéo thì không** — cùng dữ liệu, chỉ đổi cách chia mảnh, là $\lambda$ đổi
3,4 lần. Với một tập 10 điểm thì đó là một lượng nhiễu người dùng không kiểm soát được.

### 1.4 Thứ bình phương tối thiểu KHÔNG trả về được

Lấy chín hàm cơ sở Gauss trên mười điểm ấy, $\alpha$ và $\beta$ từ evidence:

| $x^\ast$ | Bayes | Bình phương tối thiểu |
|---|---|---|
| 0,05 | $+0{,}5561 \pm 0{,}3545$ (95 %: $\pm0{,}6948$) | $+0{,}4480$, không có thanh sai số |
| 0,50 | $+0{,}1221 \pm 0{,}3496$ (95 %: $\pm0{,}6853$) | $+0{,}2219$, không có thanh sai số |
| 0,95 | $-0{,}0961 \pm 0{,}3545$ (95 %: $\pm0{,}6948$) | $-0{,}1438$, không có thanh sai số |
| **1,30** (ngoài dữ liệu) | $+0{,}1439 \pm 0{,}3724$ | $+1{,}1454$, không có thanh sai số |

Dòng cuối là dòng dạy được nhiều nhất, và nó **không** dạy điều người ta chờ đợi. Giá trị
thật ở đó là $\sin(2\pi\cdot 1{,}3) = +0{,}951057$ — **dương**. Bình phương tối thiểu trả
về $+1{,}145412$, sai **0,194356**; bản Bayes trả về $+0{,}143942$, sai **0,807115**. Tức
là ở điểm này **bản Bayes TỆ HƠN, sai số lớn hơn 4,1528 lần**: phép co về 0 kéo dự báo
xuống đúng lúc sự thật đang ở xa 0.

Đó là kết quả thật và nó được giữ nguyên. Nhưng nó **không** bác được điều mục này nói,
vì điều mục này nói là chuyện khác: bình phương tối thiểu trả về $+1{,}145412$ **và không
có chỗ nào trong đầu ra của nó để nói rằng con số đó đáng tin hay không**. Bản Bayes thì
có, và điều đáng học nằm ở chính con số đó:

> khoảng 95 % của Bayes tại $x = 1{,}3$ là $[-0{,}586046;\ +0{,}873930]$, và giá trị thật
> $+0{,}951057$ nằm **NGOÀI** khoảng đó — vượt cận trên **0,077127** đơn vị.

Nghĩa là mô hình **tự tin quá mức** đúng ở chỗ nó nên khiêm tốn nhất. $\sigma$ tại
$x = 1{,}3$ chỉ lớn hơn $\sigma$ ở giữa dữ liệu **1,0652 lần**, trong khi dự báo đã sai
bốn lần nhiều hơn. Mục 2.11 đo đúng cơ chế gây ra chuyện đó, và mục 6 xếp nó là chỗ hỏng
nguy hiểm thứ hai của phương pháp. Đây là một giới hạn thật, không phải một chỗ để nói
nhẹ đi.

---

## 2. Phần ruột — đi theo đúng chương 3 của Bishop

Từ đây bài đi theo thứ tự của nguồn: mục 3.3 của sách (hậu nghiệm, dự báo, nhân tương
đương), mục 3.4 của sách (so sánh mô hình), mục 3.5 của sách (xấp xỉ evidence), và
mục 3.6 của sách (giới hạn). **Cả 52 công thức
đánh số từ (3.48) tới (3.99) đều có mặt**, và mỗi công thức có đủ ba phần mà chuẩn viết
đòi: công thức kèm số hiệu · mọi ký hiệu giải nghĩa kèm chiều · và ý nghĩa trừu tượng.

Ký hiệu của mục 2 là ký hiệu **của Bishop**, không phải của database: $\mathbf w$ chứ
không phải $\beta$; $\alpha$ là độ chính xác tiên nghiệm chứ không phải mức phạt; $M$ là
số hàm cơ sở. Bảng ở mục 0.2 là bảng quy đổi.

### 2.1 Nền cần nhắc lại: mô hình hàm cơ sở tuyến tính

Bốn công thức của mục 3.1 của sách phải có mặt vì mọi thứ sau đó dùng chúng. Chúng không thuộc
phần ruột của bài này — đó là phần ruột của bài
[Linear Regression](<../Linear Regression/Linear Regression.md>) — nhưng Luật 1 đòi nhắc lại đủ dùng ngay tại
đây.

$$y(x,\mathbf w) = \sum_{j=0}^{M-1} w_j\phi_j(x) = \mathbf w^{\!\top}\boldsymbol\phi(x) \tag{3.3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $w_j$ | tham số thứ $j$ | vô hướng |
| $\phi_j(x)$ | hàm cơ sở thứ $j$, **cố định trước khi xem dữ liệu** | vô hướng |
| $\phi_0(x)$ | quy ước $\equiv 1$, để $w_0$ đóng vai hệ số chặn | vô hướng |
| $M$ | tổng số tham số, kể cả $w_0$ | — |

**Ý nghĩa trừu tượng.** Mô hình là tuyến tính **theo tham số**, không nhất thiết tuyến
tính theo $x$: $\phi_j$ có thể là $x^j$, có thể là hình chuông Gauss, có thể là sigmoid.
Chính vì tuyến tính theo $\mathbf w$ mà cả chương này có công thức đóng — và chính vì
$\phi_j$ phải **chọn trước** mà mục 3.6 của sách phải nêu giới hạn.

$$\phi_j(x) = \exp\!\left(-\frac{(x-\mu_j)^2}{2s^2}\right) \tag{3.4}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mu_j$ | tâm của hàm cơ sở thứ $j$ trong không gian đầu vào | vô hướng |
| $s$ | bề rộng không gian, dùng chung cho mọi $j$ | vô hướng |

**Ý nghĩa trừu tượng.** Đây là hàm cơ sở **định xứ**: nó chỉ khác 0 đáng kể trong một
vùng quanh $\mu_j$. Bishop nói rõ nó *"không cần có nghĩa xác suất"*, và hệ số chuẩn hoá
không quan trọng vì $\phi_j$ sẽ được nhân với một $w_j$ học được. Tính định xứ đó là lý
do mục 2.11 tồn tại: nó làm mô hình **tự tin quá mức** khi ngoại suy.

![Hình 3.1 của Bishop 2006](<paper_bishop2006_hinh_3_1.png>)

**Hình 2 — Hình 3.1 nguyên bản: ba họ hàm cơ sở.** Nguồn: Bishop 2006, trang 140,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Examples of basis functions, showing polynomials on the left, Gaussians of the form (3.4) in
> the centre, and sigmoidal of the form (3.5) on the right."*

**Dịch:** *"Ví dụ về các hàm cơ sở: đa thức ở bên trái, hàm Gauss dạng (3.4) ở giữa, và hàm
sigmoid dạng (3.5) ở bên phải."*

**Cách đọc hình, và vì sao hình này thuộc phần ruột của bài.** Ba ô, mỗi ô vẽ vài hàm cơ sở
trên $x \in (-1,1)$. Ô giữa là hình chuông Gauss — **định xứ**, mỗi hàm chỉ khác 0 quanh $\mu_j$
của nó. Ô trái là $x^j$ và ô phải là sigmoid — cả hai **không** định xứ, và vươn ra khắp trục.

Hình này nằm ở mục 3.1 của sách, tức ngoài phần ruột của bài. Nó phải có mặt vì **hai chú thích của phần
ruột trỏ thẳng vào nó**: Hình 3.10 nói *"the Gaussian basis functions in Figure 3.1"* và Hình
3.11 nói *"the polynomial basis functions and ... the sigmoidal basis functions shown in Figure
3.1"*. Không có hình này thì hai hình đó không đọc được.

**Điều hình KHÔNG nói ra, và nó gây hệ quả về sau.** Chú thích **không** cho một con số nào:
không $\mu_j$, không $s$, không bậc đa thức, không số hàm cơ sở. Đó là chỗ mục 2.13 phải dừng
lại: nhân tương đương của Hình 3.11 chỉ kiểm được **hình dạng**, không kiểm được biên độ.

$$p(t \mid x, \mathbf w, \beta) = \mathcal N\!\left(t \mid y(x,\mathbf w), \beta^{-1}\right) \tag{3.8}$$

$$p(\mathsf t \mid \mathbf X, \mathbf w, \beta) = \prod_{n=1}^{N}\mathcal N\!\left(t_n \mid \mathbf w^{\!\top}\boldsymbol\phi(x_n), \beta^{-1}\right) \tag{3.10}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\beta^{-1}$ | phương sai nhiễu — **giả thiết như nhau ở mọi $x$** | vô hướng |
| $N$ | số quan sát | — |

**Ý nghĩa trừu tượng.** (3.10) là hàm hợp lẽ, và điều quyết định cho cả chương là: đọc
theo chiều $\mathbf w$ thì nó là **luỹ thừa của một hàm bậc hai của $\mathbf w$**. Chính
tính chất đó — và chỉ nó — làm tiên nghiệm Gauss thành liên hợp, tức cho hậu nghiệm có
công thức đóng. Đổi (3.8) sang phân phối khác là mất toàn bộ chương 3.

$$E_D(\mathbf w) = \tfrac12\sum_{n=1}^{N}\left\{t_n - \mathbf w^{\!\top}\boldsymbol\phi(x_n)\right\}^2 \tag{3.12}$$

$$E_W(\mathbf w) = \tfrac12\mathbf w^{\!\top}\mathbf w \tag{3.25}$$

$$\tfrac12\sum_{n=1}^{N}\left\{t_n-\mathbf w^{\!\top}\boldsymbol\phi(x_n)\right\}^2 + \frac{\lambda}{2}\mathbf w^{\!\top}\mathbf w \tag{3.27}$$

**Ý nghĩa trừu tượng.** (3.27) là hàm lỗi của **ridge regression**. Nó xuất hiện ở đây
không phải như một mẹo mà như một hệ quả: mục 2.5 chứng minh rằng cực đại hậu nghiệm của
mô hình Bayes **chính là** cực tiểu của (3.27) với $\lambda = \alpha/\beta$. Chữ $\lambda$
trong công thức này là **mức phạt**; chữ $\lambda_i$ ở (3.87) là **trị riêng** — hai thứ
khác nhau, xem bảng trùng tên ở mục 0.2.

### 2.2 (3.48) — tiên nghiệm liên hợp là một Gauss

![Công thức 3.48 của Bishop 2006](<paper_bishop2006_ct_3_48.png>)

**Hình 3 — mục 3.3.1 của sách và công thức (3.48).** Nguồn: Bishop 2006, trang 152,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"First note that the likelihood function p(t|w) defined by (3.10) is the exponential of
> a quadratic function of w. The corresponding conjugate prior is therefore given by a
> Gaussian distribution of the form"*

**Dịch:** *"Trước tiên hãy để ý rằng hàm hợp lẽ p(t|w) định nghĩa ở (3.10) là hàm mũ của
một hàm bậc hai của w. Tiên nghiệm liên hợp tương ứng vì thế là một phân phối Gauss có
dạng"*

$$p(\mathbf w) = \mathcal N(\mathbf w \mid \mathbf m_0, \mathbf S_0) \tag{3.48}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf m_0$ | trung bình tiên nghiệm — ta tin $\mathbf w$ ở quanh đâu trước khi xem dữ liệu | $M$ |
| $\mathbf S_0$ | hiệp phương sai tiên nghiệm — ta tin điều đó **chắc tới đâu** | $M\times M$ |

**Ý nghĩa trừu tượng.** Câu quan trọng không phải "tiên nghiệm là Gauss" mà là **lý do**
nó phải là Gauss: liên hợp không phải một lựa chọn thẩm mỹ mà là điều kiện để hậu nghiệm
**cùng họ** với tiên nghiệm, và do đó để cả chương có công thức đóng thay vì phải lấy mẫu.
Chỉ số 0 ở $\mathbf m_0$, $\mathbf S_0$ đọc là "sau **không** quan sát" — nó ăn khớp với
$\mathbf m_N$, $\mathbf S_N$ ở (3.49), và chính cách đánh chỉ số đó là chỗ (3.49) trở
thành một phép **cập nhật** thay vì một công thức mới.

### 2.3 (3.49)–(3.51) — hậu nghiệm, viết thẳng ra không cần tính tích phân

![Công thức 3.49 đến 3.51 của Bishop 2006](<paper_bishop2006_ct_3_49_3_51.png>)

**Hình 4 — hậu nghiệm (3.49)–(3.51).** Nguồn: Bishop 2006, trang 153,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Due to the choice of a conjugate Gaussian prior distribution, the posterior will also
> be Gaussian. We can evaluate this distribution by the usual procedure of completing the
> square in the exponential, and then finding the normalization coefficient using the
> standard result for a normalized Gaussian."*

**Dịch:** *"Vì đã chọn tiên nghiệm Gauss liên hợp, hậu nghiệm cũng sẽ là Gauss. Ta tính
được phân phối đó bằng thủ tục thường dùng là gom thành bình phương đúng ở số mũ, rồi tìm
hệ số chuẩn hoá bằng kết quả chuẩn cho một Gauss đã chuẩn hoá."*

$$p(\mathbf w \mid \mathsf t) = \mathcal N(\mathbf w \mid \mathbf m_N, \mathbf S_N) \tag{3.49}$$

$$\mathbf m_N = \mathbf S_N\!\left(\mathbf S_0^{-1}\mathbf m_0 + \beta\mathbf\Phi^{\!\top}\mathsf t\right) \tag{3.50}$$

$$\mathbf S_N^{-1} = \mathbf S_0^{-1} + \beta\mathbf\Phi^{\!\top}\mathbf\Phi \tag{3.51}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf m_N$ | trung bình hậu nghiệm sau $N$ quan sát | $M$ |
| $\mathbf S_N$ | hiệp phương sai hậu nghiệm sau $N$ quan sát | $M\times M$ |
| $\mathbf\Phi$ | ma trận thiết kế, $\Phi_{nj} = \phi_j(x_n)$ | $N\times M$ |
| $\beta\mathbf\Phi^{\!\top}\mathbf\Phi$ | lượng **thông tin** dữ liệu mang vào | $M\times M$ |

**Ý nghĩa trừu tượng.** (3.51) là câu đáng nhớ nhất của cả mục 3.3 của sách, và nó nói về phép
**cộng**: nghịch đảo hiệp phương sai — tức *độ chính xác* — của hậu nghiệm bằng độ chính
xác của tiên nghiệm **cộng** độ chính xác mà dữ liệu mang vào. Thông tin cộng được; độ
rộng thì không. Đó là lý do mọi phép cập nhật Bayes của mô hình này viết được thành một
phép cộng ma trận, và là lý do thứ tự nạp dữ liệu không quan trọng (mục 2.6).

(3.50) thì nói trung bình hậu nghiệm là một **bình quân có trọng số** giữa niềm tin cũ
$\mathbf m_0$ và bằng chứng mới $\mathbf\Phi^{\!\top}\mathsf t$, trọng số đúng bằng độ
chính xác của mỗi bên. Khi $\mathbf S_0^{-1}\to 0$ — tiên nghiệm vô cùng rộng, tức không
biết gì — số hạng đầu biến mất và (3.50) tụt về nghiệm bình phương tối thiểu.

### 2.4 Ba nhận xét ngay sau đó, và cả ba đều là cơ chế

![Ba nhận xét sau công thức 3.51](<paper_bishop2006_nhan_xet_map.png>)

**Hình 5 — ba nhận xét về hậu nghiệm.** Nguồn: Bishop 2006, trang 153,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Note that because the posterior distribution is Gaussian, its mode coincides with its
> mean. Thus the maximum posterior weight vector is simply given by wMAP = mN."*

> *"Furthermore, if data points arrive sequentially, then the posterior distribution at any
> stage acts as the prior distribution for the subsequent data point, such that the new
> posterior distribution is again given by (3.49)."*

**Dịch:** *"Để ý rằng vì hậu nghiệm là Gauss, đỉnh của nó trùng với trung bình. Vậy vectơ
trọng số cực đại hậu nghiệm đơn giản là $\mathbf w_{\text{MAP}} = \mathbf m_N$."* ·
*"Hơn nữa, nếu các điểm dữ liệu đến lần lượt thì hậu nghiệm ở bất kỳ giai đoạn nào cũng
đóng vai tiên nghiệm cho điểm dữ liệu tiếp theo, sao cho hậu nghiệm mới lại được cho bởi
(3.49)."*

**Cách đọc khung — ba nhận xét, ba hệ quả khác nhau:**

1. **MAP trùng trung bình.** Đúng *vì* hậu nghiệm là Gauss, và **chỉ** vì thế. Mục 2.7
   cho thấy khi đổi tiên nghiệm sang dạng (3.56) với $q \ne 2$ thì hai thứ tách ra ngay.
2. **$\alpha \to 0$ cho nghiệm hợp lẽ cực đại.** Tiên nghiệm rộng vô hạn nghĩa là không
   ràng buộc gì, nên (3.50) trở về (3.15).
3. **Cập nhật tuần tự bằng cập nhật một lượt.** Đây là một đẳng thức, không phải một
   xấp xỉ, và nó kiểm được. `v_bayes.py` mục 7 nạp 20 điểm theo hai cách — một lượt, và
   từng điểm một, mỗi lần lấy hậu nghiệm cũ làm tiên nghiệm mới:

| Cách nạp | $\mathbf m_N$ |
|---|---|
| một lượt cả 20 điểm | $[-0{,}2848494816;\ +0{,}4371278617]$ |
| tuần tự 20 bước | $[-0{,}2848494816;\ +0{,}4371278617]$ |

Lệch lớn nhất của $\mathbf m_N$ là $5{,}551\times10^{-17}$ và của $\mathbf S_N$ là
$5{,}421\times10^{-20}$ — tức bằng nhau tới mức làm tròn của `float64`. Hệ quả thực hành:
mô hình này học **trực tuyến** được mà không mất gì, và thứ tự dữ liệu không ảnh hưởng.

### 2.5 (3.52)–(3.55) — tiên nghiệm đẳng hướng, và MAP CHÍNH LÀ ridge

![Công thức 3.52 đến 3.55 của Bishop 2006](<paper_bishop2006_ct_3_52_3_55.png>)

**Hình 6 — tiên nghiệm đẳng hướng (3.52) và cầu nối sang ridge (3.55).** Nguồn:
Bishop 2006, trang 153, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Maximization of this posterior distribution with respect to w is therefore equivalent
> to the minimization of the sum-of-squares error function with the addition of a quadratic
> regularization term, corresponding to (3.27) with"*

**Dịch:** *"Cực đại hoá hậu nghiệm này theo w vì thế tương đương với cực tiểu hoá hàm lỗi
tổng bình phương có thêm một số hạng chính quy hoá bậc hai, ứng với (3.27) với"*
$\lambda = \alpha/\beta$.

$$p(\mathbf w \mid \alpha) = \mathcal N(\mathbf w \mid \mathbf 0, \alpha^{-1}\mathbf I) \tag{3.52}$$

$$\mathbf m_N = \beta\mathbf S_N\mathbf\Phi^{\!\top}\mathsf t \tag{3.53}$$

$$\mathbf S_N^{-1} = \alpha\mathbf I + \beta\mathbf\Phi^{\!\top}\mathbf\Phi \tag{3.54}$$

$$\ln p(\mathbf w \mid \mathsf t) = -\frac{\beta}{2}\sum_{n=1}^{N}\left\{t_n - \mathbf w^{\!\top}\boldsymbol\phi(x_n)\right\}^2 - \frac{\alpha}{2}\mathbf w^{\!\top}\mathbf w + \text{const} \tag{3.55}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\alpha^{-1}\mathbf I$ | hiệp phương sai tiên nghiệm **đẳng hướng**: mọi hướng của $\mathbf w$ bị ràng buộc như nhau | $M\times M$ |
| $\text{const}$ | phần không phụ thuộc $\mathbf w$ — hằng chuẩn hoá | vô hướng |

**Ý nghĩa trừu tượng.** (3.55) là chỗ hai thế giới gặp nhau, và nó đáng đọc theo chiều
ngược lại của cách người ta thường dạy. Thông thường ridge được giới thiệu như một *mẹo*:
thêm $\lambda\|\mathbf w\|^2$ để tránh quá khớp. (3.55) nói ngược: phạt bậc hai **không
phải mẹo**, nó là logarit của một tiên nghiệm Gauss, và hệ số phạt không phải một con số
tự do mà là **tỉ số** $\alpha/\beta$ — tỉ số giữa "ta tin hệ số nhỏ tới đâu" và "nhiễu ít
tới đâu". Từ đó có hai điều mà cách nhìn "mẹo" không cho:

* $\lambda$ **có đơn vị và có nghĩa**: nó là tỉ số hai độ chính xác, nên có thể ước lượng
  từ dữ liệu (mục 2.26) thay vì phải quét lưới.
* MAP chỉ là **một điểm** của hậu nghiệm. Ridge dừng ở đó; Bayes còn lại cả $\mathbf S_N$,
  và đó là chỗ (3.59) lấy độ bất định ra.

**Đo lại cho chắc.** Mục 0.4 đã kiểm bằng phân số đúng: trên ba điểm với $\alpha=1$,
$\beta=2$ thì $\mathbf m_N = \left[\tfrac{38}{41},\tfrac{24}{41}\right]$ và nghiệm ridge
với $\lambda=\tfrac12$ cho **cùng hai phân số ấy**. Không phải "gần bằng" — bằng.

Một chỗ dễ bỏ qua và nó quan trọng: (3.52) đặt tiên nghiệm lên **mọi** thành phần của
$\mathbf w$, **kể cả $w_0$**. Trong khi đó `Ridge` và `BayesianRidge` của `scikit-learn`
mặc định **không** phạt hệ số chặn. Mục 3 đo phần chênh đó, và mục 2.11 cho thấy chính
$w_0$ bị phạt là thứ quyết định cận dưới của phương sai dự báo khi ngoại suy.

### 2.6 Hình 3.7 — sáu ô, và thiết lập duy nhất bài báo khai ĐẦY ĐỦ

![Thiết lập của Hình 3.7](<paper_bishop2006_thiet_lap_hinh_3_7.png>)

**Hình 7 — đoạn khai thiết lập của Hình 3.7.** Nguồn: Bishop 2006, trang 154,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"We generate synthetic data from the function f(x, a) = a0 + a1x with parameter values
> a0 = −0.3 and a1 = 0.5 by first choosing values of xn from the uniform distribution
> U(x|−1, 1), then evaluating f(xn, a), and finally adding Gaussian noise with standard
> deviation of 0.2 to obtain the target values tn."*

> *"We assume here that the noise variance is known and hence we set the precision parameter
> to its true value"*

**Dịch:** *"Ta sinh dữ liệu tổng hợp từ hàm f(x, a) = a₀ + a₁x với giá trị tham số
a₀ = −0,3 và a₁ = 0,5, bằng cách trước tiên chọn các giá trị xₙ từ phân phối đều
U(x|−1, 1), rồi tính f(xₙ, a), và cuối cùng cộng nhiễu Gauss có độ lệch chuẩn 0,2 để được
giá trị mục tiêu tₙ."* · *"Ở đây ta giả thiết phương sai nhiễu là đã biết, và do đó đặt
tham số độ chính xác bằng giá trị thật của nó"* — $\beta = (1/0{,}2)^2 = 25$, và $\alpha$
được cố định ở $2{,}0$.

**Đây là hình DUY NHẤT của chương mà bài báo khai đủ mọi tham số**: $a_0$, $a_1$, phân
phối của $x$, độ lệch chuẩn nhiễu, $\beta$, $\alpha$. Phép kiểm rẻ nhất là chính số học
mà câu trên nói: $(1/0{,}2)^2 = 25{,}000000$ — khớp con số bài báo in.

![Hình 3.7 của Bishop 2006](<paper_bishop2006_hinh_3_7.png>)

**Hình 8 — Hình 3.7 nguyên bản: học Bayes tuần tự cho mô hình $y = w_0 + w_1x$.** Nguồn:
Bishop 2006, trang 155, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Illustration of sequential Bayesian learning for a simple linear model of the form"*
> $y(x, \mathbf w) = w_0 + w_1x$*"A detailed description of this figure is given in the
> text."*

**Dịch:** *"Minh hoạ việc học Bayes tuần tự cho một mô hình tuyến tính đơn giản dạng
$y(x,\mathbf w)=w_0+w_1x$. Phần mô tả chi tiết hình này nằm trong thân bài."*

**Nhận xét của chính tác giả về hình này**, nguyên văn từ thân bài tr.154–156:

> *"The first row of this figure corresponds to the situation before any data points are
> observed and shows a plot of the prior distribution in w space together with six samples
> of the function"*

> *"Note that the likelihood function provides a soft constraint that the line must pass
> close to the data point, where close is determined by the noise precision"*

> *"Note that this is exactly the same posterior distribution as would be obtained by
> combining the original prior with the likelihood function for the two data points."*

> *"In the limit of an infinite number of data points, the posterior distribution would
> become a delta function centred on the true parameter values, shown by the white cross."*

**Dịch bốn câu trên:** *"Hàng đầu của hình ứng với tình huống trước khi quan sát điểm dữ
liệu nào, và vẽ tiên nghiệm trong không gian w cùng sáu mẫu của hàm"* · *"Để ý rằng hàm
hợp lẽ cho một ràng buộc MỀM rằng đường thẳng phải đi gần điểm dữ liệu, gần tới mức nào
thì do độ chính xác nhiễu quyết định"* · *"Để ý rằng đây đúng là cùng một hậu nghiệm mà ta
sẽ có nếu kết hợp tiên nghiệm ban đầu với hàm hợp lẽ của hai điểm dữ liệu."* · *"Trong
giới hạn vô số điểm dữ liệu, hậu nghiệm sẽ thành một hàm delta đặt tại giá trị tham số
thật, chỗ dấu chữ thập trắng."*

**Cách đọc hình.** Ba cột, bốn hàng. Cột trái là **hàm hợp lẽ của riêng điểm mới nhất**
trong không gian $(w_0, w_1)$ — nó là một dải, không phải một đỉnh, vì một điểm dữ liệu
không đủ xác định hai hệ số. Cột giữa là **hậu nghiệm** sau khi nhân dải đó vào phân phối
của hàng trên. Cột phải là sáu đường $y(x,\mathbf w)$ với $\mathbf w$ rút từ hậu nghiệm
đó. Bốn hàng ứng với $N = 0, 1, 2, 20$.

**Điều hình KHÔNG nói ra.** Ba chuyện:

* Hàng thứ ba đã khá hẹp và Bishop giải thích bằng câu *"two points are sufficient to
  define a line"*. Nhưng hậu nghiệm ở đó **không** là một điểm — nó vẫn là một ellipse,
  vì nhiễu làm hai điểm không xác định đường một cách chắc chắn.
* Dấu chữ thập trắng ở $(-0{,}3;\,0{,}5)$ là **tham số thật**, không phải nghiệm. Hậu
  nghiệm không có lý do gì phải đi qua nó ở $N$ hữu hạn.
* Hình vẽ **độ rộng** co lại, nhưng không nói gì về **sai số của trung bình**, và hai
  thứ đó không cùng đơn điệu.

Chỗ cuối là chỗ dựng lại được và cần được nói rõ. Chạy đúng thiết lập của Bishop
($\alpha = 2$, $\beta = 25$, $x_n \sim U(-1,1)$, nhiễu sd 0,2):

| $N$ | $\mathbf m_N$ | đường chéo $\mathbf S_N$ | $\lvert\mathbf m_N - \mathbf a\rvert$ |
|---:|---|---|---:|
| 0 | $[+0{,}000000;\ +0{,}000000]$ | $[0{,}500000;\ 0{,}500000]$ | 0,500000 |
| 1 | $[-0{,}163411;\ -0{,}044762]$ | $[0{,}067112;\ 0{,}467519]$ | **0,544762** |
| 2 | $[-0{,}207446;\ +0{,}076022]$ | $[0{,}020149;\ 0{,}114187]$ | 0,423978 |
| 20 | $[-0{,}284849;\ +0{,}437128]$ | $[0{,}001995;\ 0{,}004792]$ | 0,062872 |

Cột **đường chéo $\mathbf S_N$ giảm đơn điệu** — cả hai thành phần, ở cả ba bước. Cột
**sai số thì KHÔNG**: nó đi $0{,}500 \to 0{,}545 \to 0{,}424 \to 0{,}063$, tức **tăng** ở
bước đầu. Lý do hiển nhiên khi nghĩ kỹ: một điểm dữ liệu kéo $\mathbf m_N$ về phía nó, và
phía đó không nhất thiết là phía của $\mathbf a$. Câu đúng là *"hậu nghiệm CO LẠI đơn
điệu"*, không phải *"hậu nghiệm tiến về giá trị thật một cách đơn điệu"*.

Câu *"trong giới hạn vô số điểm thì hậu nghiệm thành hàm delta"* thì đo được, và tốc độ
đo được luôn:

| $N$ | 20 | 100 | 500 | 2 500 |
|---|---:|---:|---:|---:|
| $\lvert\mathbf m_N - \mathbf a\rvert$ trung bình trên 60 hạt giống | 0,070195 | 0,030264 | 0,012816 | 0,006959 |
| tỉ số so với cột trước | — | 0,4311 | 0,4235 | 0,5430 |

Mỗi bước tăng $N$ gấp 5 lần, nên nếu sai số giảm như $1/\sqrt N$ thì tỉ số phải là
$5^{-1/2} = 0{,}4472$. Hai tỉ số đầu khớp; tỉ số thứ ba là **0,5430**, tức cao hơn. Ba
điểm đo với 60 hạt giống mỗi điểm **không đủ** để kết luận về số mũ; con số đọc được là
*"sai số giảm, cỡ $1/\sqrt N$"*, không phải *"đúng $1/\sqrt N$"*. Cùng thiết lập với 200
hạt giống ở $N=20$ cho trung bình 0,072374 và khoảng $[0{,}000914;\ 0{,}256945]$ — độ
trải rộng gấp **280,9728 lần** giữa hai đầu, nên một lần chạy đơn lẻ không nói được gì.

### 2.7 (3.56) — tiên nghiệm tổng quát, và chỗ MAP tách khỏi trung bình

![Công thức 3.56 của Bishop 2006](<paper_bishop2006_ct_3_56.png>)

**Hình 9 — tiên nghiệm tổng quát (3.56).** Nguồn: Bishop 2006, trang 156,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"in which q = 2 corresponds to the Gaussian distribution, and only in this case is the
> prior conjugate to the likelihood function (3.10). Finding the maximum of the posterior
> distribution over w corresponds to minimization of the regularized error function"*

**Dịch:** *"trong đó q = 2 ứng với phân phối Gauss, và chỉ trong trường hợp đó tiên nghiệm
mới liên hợp với hàm hợp lẽ (3.10). Tìm cực đại của hậu nghiệm theo w ứng với cực tiểu hoá
hàm lỗi có chính quy hoá"* (3.29).

$$p(\mathbf w\mid\alpha) = \left[\frac{q}{2}\left(\frac{\alpha}{2}\right)^{1/q}\frac{1}{\Gamma(1/q)}\right]^{M}\exp\!\left(-\frac{\alpha}{2}\sum_{j=1}^{M}\lvert w_j\rvert^{q}\right) \tag{3.56}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $q$ | bậc của chuẩn dùng trong tiên nghiệm; $q=2$ Gauss, $q=1$ Laplace | vô hướng |
| $\Gamma(\cdot)$ | hàm gamma, xuất hiện ở hằng chuẩn hoá | — |

**Ý nghĩa trừu tượng.** (3.56) là chỗ cả họ phương pháp chính quy hoá hiện ra thành **một
họ tiên nghiệm**: ridge là $q=2$, [lasso](<../Lasso Regression/Lasso Regression.md>) là $q=1$. Nhưng câu quan
trọng nhất của mục này là câu **phủ định** mà Bishop viết ngay sau đó:

> *"In the case of the Gaussian prior, the mode of the posterior distribution was equal to
> the mean, although this will no longer hold if q"* $\ne 2$.

**Dịch:** *"Trong trường hợp tiên nghiệm Gauss, đỉnh của hậu nghiệm bằng trung bình, dù
điều đó sẽ không còn đúng nếu q khác 2."*

Đó là một câu cần đọc rất kỹ, vì nó nói rằng **"chính quy hoá là một tiên nghiệm" chỉ đúng
ở mức MAP, không đúng ở mức trung bình hậu nghiệm**. Với $q=1$, nghiệm lasso là đỉnh của
hậu nghiệm Laplace và nó **thưa**; nhưng trung bình của cùng hậu nghiệm ấy thì **không
thưa** — xác suất để một hệ số bằng đúng 0 trong một phân phối liên tục là 0. Vậy câu
"lasso = tiên nghiệm Laplace" đúng về hàm mục tiêu và sai về đầu ra Bayes. Bài
[Lasso Regression](<../Lasso Regression/Lasso Regression.md>) nói phía còn lại của cùng chuyện này.

Và hệ quả kỹ thuật: $q\ne2$ thì mất liên hợp, mất công thức đóng, mất luôn cả chương 3.
Toàn bộ phần còn lại của chương giả thiết $q=2$.

### 2.8 (3.57)–(3.59) — phân phối dự báo, và hai nguồn của độ bất định

![Công thức 3.57 đến 3.59 của Bishop 2006](<paper_bishop2006_ct_3_57_3_59.png>)

**Hình 10 — mục 3.3.2 của sách và phân phối dự báo (3.57)–(3.59).** Nguồn: Bishop 2006, trang 156,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"In practice, we are not usually interested in the value of w itself but rather in making
> predictions of t for new values of x."*

> *"The first term in (3.59) represents the noise on the data whereas the second term
> reflects the uncertainty associated with the parameters w. Because the noise process and
> the distribution of w are independent Gaussians, their variances are additive."*

**Dịch:** *"Trên thực tế ta thường không quan tâm tới bản thân giá trị của w mà quan tâm
tới việc dự báo t cho những giá trị x mới."* · *"Số hạng thứ nhất trong (3.59) biểu diễn
nhiễu trên dữ liệu, còn số hạng thứ hai phản ánh độ bất định gắn với các tham số w. Vì quá
trình nhiễu và phân phối của w là hai Gauss độc lập, phương sai của chúng cộng được."*

$$p(t \mid \mathsf t, \alpha, \beta) = \int p(t\mid\mathbf w,\beta)\,p(\mathbf w\mid\mathsf t,\alpha,\beta)\,\mathrm d\mathbf w \tag{3.57}$$

$$p(t \mid x, \mathsf t, \alpha, \beta) = \mathcal N\!\left(t \mid \mathbf m_N^{\!\top}\boldsymbol\phi(x),\ \sigma_N^2(x)\right) \tag{3.58}$$

$$\sigma_N^2(x) = \frac{1}{\beta} + \boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x) \tag{3.59}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\int \cdots \mathrm d\mathbf w$ | tích ra toàn bộ $\mathbf w$ — không chọn một $\mathbf w$ nào cả | — |
| $1/\beta$ | phần nhiễu, **không phụ thuộc $x$** | vô hướng |
| $\boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x)$ | phần bất định tham số, **phụ thuộc $x$** | vô hướng |

**Ý nghĩa trừu tượng.** (3.57) là chỗ phân biệt Bayes với MAP: MAP chọn một $\mathbf w$
rồi dự báo; (3.57) **không chọn gì** mà lấy bình quân dự báo trên mọi $\mathbf w$, có
trọng số là xác suất hậu nghiệm. Kết quả là một phân phối rộng hơn — và rộng hơn đúng bằng
một lượng có nghĩa.

(3.59) là công thức trung tâm của cả bài viết này, và nó nên đọc như một phép **tách**:

* Số hạng $1/\beta$ là thứ **không bao giờ giảm được** bằng cách thêm dữ liệu. Đó là nhiễu
  của chính thế giới. Với $\beta = 11{,}1$ thì nó bằng $0{,}090090$, tức $\sigma = 0{,}300150$.
* Số hạng $\boldsymbol\phi^{\!\top}\mathbf S_N\boldsymbol\phi$ là thứ **giảm được**: nó là
  hệ quả của việc ta chưa biết $\mathbf w$, và nó co lại khi $\mathbf S_N$ co lại. Nó cũng
  là **thứ duy nhất phụ thuộc vào chỗ ta hỏi**: dạng toàn phương đó nhỏ ở nơi có nhiều dữ
  liệu và lớn ở nơi không có.

Phân biệt hai số hạng ấy là điều dùng được ngay: nếu $\sigma_N^2$ lớn vì số hạng thứ hai,
**thu thêm dữ liệu sẽ giúp**; nếu lớn vì số hạng thứ nhất thì không, phải đổi mô hình hoặc
đổi phép đo.

Bishop khẳng định thêm một tính chất, dẫn về Qazaz và cộng sự 1997:

> *"Note that, as additional data points are observed, the posterior distribution becomes
> narrower. As a consequence it can be shown (Qazaz et al., 1997) that"*
> $\sigma_{N+1}^2(x) \le \sigma_N^2(x)$.

**Dịch:** *"Để ý rằng khi quan sát thêm điểm dữ liệu, hậu nghiệm hẹp lại. Hệ quả là có thể
chứng minh được rằng"* $\sigma_{N+1}^2(x) \le \sigma_N^2(x)$.

`v_bayes.py` mục 10 kiểm điều đó: quét 3 bề rộng hàm cơ sở × 3 giá trị $\alpha$ × 24 bước
$N$ × 71 hoành độ, **kể cả các hoành độ nằm ngoài $[0,1]$** mà Bishop không thử —
**15 336 phép so, 0 chỗ vi phạm**. Khẳng định đúng, và đúng cả ở vùng ngoại suy.

### 2.9 Hình 3.8 — bài báo không in dữ liệu, nhưng LỚP VECTƠ thì có

![Hình 3.8 của Bishop 2006](<paper_bishop2006_hinh_3_8.png>)

**Hình 11 — Hình 3.8 nguyên bản: phân phối dự báo ở $N = 1, 2, 4, 25$.** Nguồn:
Bishop 2006, trang 157, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Examples of the predictive distribution (3.58) for a model consisting of 9 Gaussian
> basis functions of the form (3.4) using the synthetic sinusoidal data set of Section 1.1.
> See the text for a detailed discussion."*

**Dịch:** *"Ví dụ về phân phối dự báo (3.58) cho một mô hình gồm 9 hàm cơ sở Gauss dạng
(3.4), dùng tập dữ liệu hình sin tổng hợp của mục 1.1. Xem thân bài để có phần bàn luận
chi tiết."*

![Bình luận của Bishop về Hình 3.8](<paper_bishop2006_binh_hinh_3_8.png>)

**Hình 12 — đoạn thân bài Bishop dùng để bình luận Hình 3.8.** Nguồn: Bishop 2006,
trang 157, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nhận xét của chính tác giả, nguyên văn:**

> *"For each plot, the red curve shows the mean of the corresponding Gaussian predictive
> distribution, and the red shaded region spans one standard deviation either side of the
> mean."*

> *"Note that the predictive uncertainty depends on x and is smallest in the neighbourhood
> of the data points. Also note that the level of uncertainty decreases as more data points
> are observed."*

**Dịch:** *"Ở mỗi ô, đường đỏ là trung bình của phân phối dự báo Gauss tương ứng, và vùng
tô đỏ trải một độ lệch chuẩn về mỗi phía của trung bình."* · *"Để ý rằng độ bất định dự báo
phụ thuộc x và nhỏ nhất ở vùng quanh các điểm dữ liệu. Cũng để ý rằng mức độ bất định giảm
khi quan sát thêm điểm dữ liệu."*

**Cách đọc hình.** Bốn ô, $N = 1, 2, 4, 25$. Đường xanh lá là $\sin(2\pi x)$ — hàm thật.
Vòng tròn xanh là dữ liệu. Đường đỏ là trung bình dự báo $\mathbf m_N^{\!\top}
\boldsymbol\phi(x)$. Dải hồng là $\pm$ một $\sigma_N(x)$, tức là **cả hai** số hạng của
(3.59) gộp lại.

**Điều hình KHÔNG nói ra — và điều đọc được từ lớp vectơ.** Trang này là hình **vectơ**:
`v_bayes.py` mục 3 đếm được **56 đường và 1 243 đoạn** trên trang 157. Từ đó đọc được thứ
bài báo không in ở đâu.

**Một — 25 điểm dữ liệu.** Mỗi vòng tròn của MATLAB là một khối đúng 13 đoạn đường; ô
$N=25$ có **325 đoạn**, tức đúng 25 vòng. Quy đổi toạ độ bằng các vạch chia của chính hình
cho ra:

| $k$ | 0 | 5 | 10 | 15 | 20 | 24 |
|---|---|---|---|---|---|---|
| $x_k$ | 0,0000 | 0,2083 | 0,4167 | 0,6250 | 0,8333 | 1,0000 |
| $t_k$ | $+0{,}3501$ | $+1{,}4768$ | $+0{,}0669$ | $-0{,}8165$ | $-0{,}8804$ | $-0{,}0171$ |

Hoành độ đọc được lệch khỏi `linspace(0,1,25)` nhiều nhất **0,000348**. Con số đó không
chỉ là một phép làm sạch — nó **xác nhận cả phép quy đổi trục**: nếu quy đổi lệch thì các
$x$ đọc ra sẽ không rơi thành một lưới đều, và chúng rơi đúng lưới đều.

Phép kiểm chéo thứ hai, độc lập: đường **xanh lá** của cùng ô đọc lại và so với
$\sin(2\pi x)$ cho lệch lớn nhất **0,003277** trên 74 điểm. Đó là **sai số của khâu đọc
vectơ**, và mọi con số dựng lại về sau phải so với nó, không so với 0.

Nhiễu của 25 điểm đó: trung bình $+0{,}027686$, độ lệch chuẩn **0,277763**, giá trị tuyệt
đối lớn nhất 0,540655. Mục 1.1 của sách dùng nhiễu độ lệch chuẩn 0,3; tr.171 của cùng
sách ghi *"we have set β to its true value of 11.1"*, và $1/0{,}3^2 = 11{,}111111$. Ba con
số đó khớp nhau.

**Hai — bốn ô là TẬP CON LỒNG NHAU.** Bài báo chỉ nói *"Data sets of size N = 1, N = 2,
N = 4, and N = 25"*, không nói gì về quan hệ giữa chúng. Đối chiếu toạ độ từng vòng tròn:

| Ô | Dùng những điểm nào (chỉ số trong tập 25) |
|---|---|
| $N=1$ | $\{10\}$ |
| $N=2$ | $\{10,\ 14\}$ |
| $N=4$ | $\{0,\ 10,\ 14,\ 24\}$ |
| $N=25$ | tất cả |

Ba tập đầu **lồng trong nhau** và lồng trong tập 25 điểm. Điều đó làm bốn ô đọc được như
**một** quá trình học tuần tự chứ không phải bốn thí nghiệm rời, và nó khớp với thông điệp
của Hình 3.7. Thứ tự chọn thì lạ — điểm giữa trước, rồi hai đầu — nên đây không phải "lấy
$N$ điểm đầu tiên".

**Ba — dò ba tham số bài báo không khai.** Chú thích nói "9 hàm cơ sở Gauss", nhưng không
nói tâm $\mu_j$, bề rộng $s$, hay $\alpha$. Dải hồng cho một phép đo trực tiếp cho
$\sigma_N(x)$: đa giác tô là **một nhánh khép kín**, nên một đường dọc $x = c$ cắt nó đúng
hai lần, và hiệu hai giao điểm chia đôi chính là $\sigma_N(c)$.

| Đo từ dải hồng của ô $N=25$ | Giá trị |
|---|---|
| $\sigma$ nhỏ nhất | **0,335545** tại $x = 0{,}220$ |
| $\sigma$ lớn nhất | **0,357665** tại $x = 0{,}980$ |
| $1/\sqrt\beta$ với $\beta = 11{,}1$ | 0,300150 |
| tâm dải so với **đường đỏ** (hai đối tượng PDF khác nhau) | lệch lớn nhất **0,002249** |

Dòng cuối là một phép kiểm chéo nữa: tâm của dải và đường trung bình là hai đối tượng vẽ
riêng trong PDF, và chúng trùng nhau tới 0,002249.

Lấy $\mu_j = $ `linspace(0,1,9)` và $\beta = 11{,}1$, rồi dò $(s, \alpha)$ sao cho vừa
khớp đường đỏ vừa khớp $\sigma(x)$ của dải:

| Kết quả dò | Giá trị |
|---|---|
| bề rộng $s$ | **0,123000** |
| $\alpha$ | **0,513333** |
| lệch lớn nhất của trung bình dự báo | **0,003296** |
| lệch lớn nhất của $\sigma(x)$ | **0,001580** |
| sai số của khâu đọc vectơ (mốc so sánh) | 0,003277 |

Lệch của trung bình **ngang đúng** sai số đọc (tỉ số 1,01) và lệch của $\sigma$ thì nhỏ
hơn nó. Nói cách khác: bộ ba $(\mu_j, s, \alpha)$ đó dựng lại Hình 3.8 tới mức không còn
phân biệt được bằng phép đọc.

**Phép thử ngược, và nó phải có** — nếu không thì "khớp" không phân biệt được với "chỉnh
cho vừa". Đổi **một** tham số, giữ hai cái kia:

| Đổi gì | Lệch trung bình | Xấu đi | Lệch $\sigma$ | Xấu đi |
|---|---:|---:|---:|---:|
| $s \times 0{,}5$ | 0,131243 | **39,8 lần** | 0,021889 | 13,9 lần |
| $s \times 2$ | 0,125275 | **38,0 lần** | 0,018679 | 11,8 lần |
| $\alpha \times 0{,}1$ | 0,084896 | 25,8 lần | 0,009640 | 6,1 lần |
| $\alpha \times 10$ | 0,115413 | 35,0 lần | 0,013779 | 8,7 lần |
| 7 hàm cơ sở thay vì 9 | 0,070562 | 21,4 lần | 0,007208 | 4,6 lần |
| 11 hàm cơ sở thay vì 9 | 0,011906 | 3,6 lần | 0,002063 | 1,3 lần |
| $\mu_j$ trên $[0{,}2;\,0{,}8]$ | 0,036579 | 11,1 lần | 0,023292 | 14,7 lần |

Mọi phép đổi đều làm xấu đi, ít nhất 3,6 lần. Dòng "11 hàm cơ sở" là dòng yếu nhất — phép
đo không phân biệt được 9 với 11 mạnh như nó phân biệt được bề rộng — và đó là lý do con
số 9 vẫn phải lấy từ **chú thích của bài báo**, không phải từ phép dò.

**Bốn — kiểm câu "độ bất định giảm khi thêm dữ liệu"** trên chính bốn ô, với bộ tham số
vừa dò:

| Ô | $\sigma$ nhỏ nhất | $\sigma$ lớn nhất | $\sigma$ tại $x=0{,}5$ | $\lvert$trung bình $-\sin\rvert$ lớn nhất |
|---|---:|---:|---:|---:|
| $N=1$ | 0,423078 | **2,151520** | 0,937488 | 1,030636 |
| $N=2$ | 0,417353 | **2,046974** | 0,487906 | 0,859779 |
| $N=4$ | 0,417298 | **1,289321** | 0,479743 | 0,743627 |
| $N=25$ | 0,336586 | **0,374285** | 0,337414 | 0,324064 |

Cột $\sigma$ lớn nhất giảm đơn điệu qua bốn ô — khẳng định của Bishop đúng. Và cột cuối
cho thấy phần *"nhỏ nhất ở vùng quanh các điểm dữ liệu"* cũng đúng: ở $N=25$ thì
$\sigma(0{,}5) = 0{,}337414$ gần sát $\sigma$ nhỏ nhất, còn ở $N=1$ thì $\sigma$ tại
$x=0{,}5$ lớn gấp **2,2159 lần** $\sigma$ nhỏ nhất, vì điểm dữ liệu duy nhất nằm ở
$x = 10/24 = 0{,}4167$.

### 2.10 Hình 3.9 — cùng hậu nghiệm, nhưng nhìn thấy TƯƠNG QUAN

![Hình 3.9 của Bishop 2006](<paper_bishop2006_hinh_3_9.png>)

**Hình 13 — Hình 3.9 nguyên bản: mẫu vẽ từ hậu nghiệm.** Nguồn: Bishop 2006, trang 158,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Plots of the function y(x, w) using samples from the posterior distributions over w
> corresponding to the plots in Figure 3.8."*

**Dịch:** *"Đồ thị của hàm y(x, w) dùng các mẫu rút từ hậu nghiệm trên w, ứng với các ô của
Hình 3.8."*

**Nhận xét của chính tác giả, nguyên văn từ tr.157 và tr.160:**

> *"The plots in Figure 3.8 only show the point-wise predictive variance as a function of x.
> In order to gain insight into the covariance between the predictions at different values
> of x, we can draw samples from the posterior distribution over w, and then plot the
> corresponding functions y(x, w)"*

> *"However, by drawing samples from the posterior distribution over w, and plotting the
> corresponding model functions y(x, w) as in Figure 3.9, we are visualizing the joint
> uncertainty in the posterior distribution between the y values at two (or more) x values,
> as governed by the equivalent kernel."*

**Dịch:** *"Các ô của Hình 3.8 chỉ cho thấy phương sai dự báo TỪNG ĐIỂM như một hàm của x.
Để hiểu được hiệp phương sai giữa các dự báo ở những giá trị x khác nhau, ta có thể rút mẫu
từ hậu nghiệm trên w rồi vẽ các hàm y(x, w) tương ứng"* · *"Tuy nhiên, bằng cách rút mẫu từ
hậu nghiệm trên w và vẽ các hàm mô hình y(x, w) tương ứng như ở Hình 3.9, ta đang hình dung
độ bất định ĐỒNG THỜI trong hậu nghiệm giữa các giá trị y ở hai (hoặc nhiều) giá trị x, do
nhân tương đương chi phối."*

**Cách đọc hình, và vì sao nó không thay được Hình 3.8.** Hai hình dùng **cùng một** hậu
nghiệm và nói **hai** điều khác nhau:

* Hình 3.8 trả lời *"tại $x$ này, $t$ có thể là bao nhiêu"* — một câu về **một** điểm.
* Hình 3.9 trả lời *"nếu đường cong đi cao ở $x_1$ thì nó đi cao hay thấp ở $x_2$"* — một
  câu về **hai** điểm cùng lúc.

Dải hồng của Hình 3.8 không chứa thông tin đó. Một dải rộng có thể đi cùng những đường
cong gần như song song nhau (tương quan cao) hoặc những đường cong cắt nhau loạn xạ (tương
quan thấp), và hai trường hợp đó có nghĩa thực hành khác nhau hẳn: trường hợp đầu nói "ta
không biết mức, nhưng biết hình dạng", trường hợp sau nói "ta không biết gì".

**Điều hình KHÔNG nói ra, và không dựng lại được.** Số mẫu mỗi ô, hạt giống, và cách chọn
chúng thì bài báo không khai. Đường cong trong Hình 3.9 là **mẫu ngẫu nhiên**, nên không
có con số nào của nó đối chiếu được — không như Hình 3.8, nơi dải $\pm\sigma$ là một đại
lượng tất định. Bài này vì thế không trích một con số nào từ Hình 3.9; đại lượng tất định
tương ứng là hiệp phương sai (3.63), và nó được kiểm ở mục 2.14.

### 2.11 Cảnh báo tr.158 — và chỗ nó đúng chỉ khi BỎ hàm cơ sở hằng

![Cảnh báo về ngoại suy, trang 158](<paper_bishop2006_canh_bao_ngoai_suy.png>)

**Hình 14 — cảnh báo về ngoại suy với hàm cơ sở định xứ.** Nguồn: Bishop 2006, trang 158,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"If we used localized basis functions such as Gaussians, then in regions away from the
> basis function centres, the contribution from the second term in the predictive variance
> (3.59) will go to zero, leaving only the noise contribution"*

> *"Thus, the model becomes very confident in its predictions when extrapolating outside
> the region occupied by the basis functions, which is generally an undesirable behaviour."*

**Dịch:** *"Nếu ta dùng hàm cơ sở định xứ như hàm Gauss thì ở những vùng xa tâm các hàm cơ
sở, đóng góp từ số hạng thứ hai trong phương sai dự báo (3.59) sẽ tiến về không, chỉ còn
lại đóng góp của nhiễu"* · *"Vậy mô hình trở nên rất tự tin vào dự báo của nó khi ngoại suy
ra ngoài vùng mà các hàm cơ sở chiếm giữ, và đó nói chung là một hành vi không mong muốn."*

**Cách đọc khung, và một chỗ phải sửa lại.** Chiều của cảnh báo đúng: mô hình **có** tự tin
quá mức khi ngoại suy. Nhưng cận mà Bishop nêu — *"chỉ còn lại đóng góp của nhiễu"*, tức
$\sigma^2 \to \beta^{-1}$ — thì **không đúng** với mô hình mà chính (3.3) định nghĩa. Đo
trên mô hình đã dựng lại ở mục 2.9 ($\mu_j$ = `linspace(0,1,9)`, $s = 0{,}123$,
$\alpha = 0{,}513333$, $\beta = 11{,}1$):

| $x$ | $\sigma_N(x)$ | phần nhiễu $1/\beta$ chiếm | $\boldsymbol\phi^{\!\top}\mathbf S_N\boldsymbol\phi$ |
|---:|---:|---:|---:|
| 0,5 | 0,337414 | 79,13 % | $2{,}376\times10^{-2}$ |
| 1,0 | 0,374285 | 64,31 % | $5{,}000\times10^{-2}$ |
| 1,5 | 0,828610 | 13,12 % | $5{,}965\times10^{-1}$ |
| 2,0 | 0,828742 | 13,12 % | $5{,}967\times10^{-1}$ |
| 3,0 | **0,828742** | 13,12 % | $5{,}967\times10^{-1}$ |
| 5,0 | **0,828742** | 13,12 % | $5{,}967\times10^{-1}$ |

$\sigma$ **không** tụt về $1/\sqrt\beta = 0{,}300150$; nó dừng ở **0,828742** và đứng yên
ở đó, và số hạng thứ hai dừng ở $0{,}596723$ chứ không về 0.

Lý do nằm trong chính (3.3): công thức đó thêm một **hàm cơ sở hằng** $\phi_0(x) \equiv 1$
để $w_0$ làm hệ số chặn — và $\phi_0$ **không tắt** khi $x$ đi xa. Vậy
$\boldsymbol\phi(x) \to (1, 0, \ldots, 0)$ và

$$\boldsymbol\phi^{\!\top}\mathbf S_N\boldsymbol\phi \longrightarrow (\mathbf S_N)_{00} = 0{,}596723 \ne 0$$

Cận đúng là $\sigma^2 \to \beta^{-1} + (\mathbf S_N)_{00} = 0{,}090090 + 0{,}596723 =
0{,}686813$, tức $\sigma$ lớn gấp **2,7611 lần** con số $1/\sqrt\beta$ mà câu của bài báo
gợi ra.

**Phép thử ngược, và nó dứt khoát.** Bỏ $\phi_0$, chỉ giữ 9 hàm Gauss:

| $x$ | $\boldsymbol\phi^{\!\top}\mathbf S_N\boldsymbol\phi$ | $\sigma_N(x)$ | $1/\sqrt\beta$ |
|---:|---:|---:|---:|
| 1,5 | $1{,}445\times10^{-8}$ | 0,300150 | 0,300150 |
| 3,0 | $3{,}330\times10^{-116}$ | 0,300150 | 0,300150 |
| 10,0 | $0{,}000\times10^{0}$ | 0,300150 | 0,300150 |

Không có hàm cơ sở hằng thì câu của Bishop đúng **nguyên văn**, tới từng chữ số. Vậy chỗ
này không phải một sai sót lập luận mà là một **điều kiện không được nêu**: câu đúng khi và
chỉ khi không có hàm cơ sở hằng, và (3.3) thì có. Không kết luận nào khác của chương đổi vì
chỗ này — chiều của cảnh báo vẫn nguyên, chỉ cận là khác.

**Và chiều ngược lại cũng phải đo**, vì nếu không thì người đọc sẽ tưởng hàm cơ sở không
định xứ là cách chữa. Cùng phép ngoại suy với hàm cơ sở **đa thức** bậc 9, cùng $\alpha$,
cùng $\beta$:

| $x$ | $\sigma$ với đa thức | gấp bao nhiêu lần bản Gauss |
|---:|---:|---:|
| 1,0 | 0,385264 | 1,3 |
| 1,5 | 27,334472 | 91,1 |
| 2,0 | 443,216711 | 1 476,7 |
| 3,0 | 19 177,989928 | **63 894,7** |

Đa thức thì $\sigma$ **bùng nổ**; Gauss thì $\sigma$ dừng ở một sàn hữu hạn. Hai lỗi ngược
nhau, và chỉ lỗi thứ hai là lỗi Bishop cảnh báo. Cách chữa thật cho cả hai, Bishop nói
ngay sau đó:

> *"This problem can be avoided by adopting an alternative Bayesian approach"*

**Dịch:** *"Vấn đề này tránh được bằng cách dùng một cách tiếp cận Bayes khác"* — câu của
Bishop chạy tiếp là *"… to re-gression known as a Gaussian process"*, tức **quá trình Gauss**,
và phần còn lại không trích liền được vì lớp text của PDF chèn ghi chú lề *"Section 6.4"* vào
**giữa** chữ `re-gression`. Đó là một chỗ mà phép kiểm trích dẫn tự động không vượt qua được,
nên bài cắt câu ở đây và nói phần sau bằng lời.

Và một nhận xét thứ hai của cùng đoạn, đáng ghi vì nó là chỗ mô hình của cả chương bị bó:

> *"Note that, if both w and β are treated as unknown, then we can introduce a conjugate
> prior distribution p(w, β) that, from the discussion in Section 2.3.6, will be given by a
> Gaussian-gamma distribution (Denison et al., 2002). In this case, the predictive
> distribution is a Student's t-distribution."*

**Dịch:** *"Để ý rằng nếu coi cả w và β là chưa biết thì ta có thể đưa vào một tiên nghiệm
liên hợp p(w, β), và theo phần bàn luận ở mục 2.3.6 của sách, nó sẽ là một phân phối Gauss-gamma
(Denison và cộng sự, 2002). Trong trường hợp đó, phân phối dự báo là một phân phối t của
Student."*

Tức là: **cả chương 3 giả thiết $\beta$ đã biết.** Phân phối dự báo (3.58) là Gauss chỉ vì
thế. Coi $\beta$ là chưa biết thì đuôi dày hơn, và mục 3.5 của sách sẽ chọn một đường khác — ước
lượng điểm cho $\beta$ — thay vì tích nó ra.

### 2.12 (3.60)–(3.62) — nhân tương đương, và Hình 3.10

![Công thức 3.60 đến 3.62 của Bishop 2006](<paper_bishop2006_ct_3_60_3_62.png>)

**Hình 15 — mục 3.3.3 của sách và nhân tương đương (3.60)–(3.62).** Nguồn: Bishop 2006, trang 159,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Thus the mean of the predictive distribution at a point x is given by a linear
> combination of the training set target variables tn"*

> *"We see that they are localized around x, and so the mean of the predictive distribution
> at x, given by y(x, mN ), is obtained by forming a weighted combination of the target
> values in which data points close to x are given higher weight than points further
> removed from x."*

**Dịch:** *"Vậy trung bình của phân phối dự báo tại một điểm x là một tổ hợp tuyến tính của
các giá trị mục tiêu tₙ của tập huấn luyện"* · *"Ta thấy chúng định xứ quanh x, và vì thế
trung bình của phân phối dự báo tại x, cho bởi y(x, mₙ), có được bằng cách lập một tổ hợp
có trọng số của các giá trị mục tiêu, trong đó những điểm dữ liệu gần x nhận trọng số cao
hơn những điểm xa x."*

$$y(x,\mathbf m_N) = \mathbf m_N^{\!\top}\boldsymbol\phi(x) = \beta\boldsymbol\phi(x)^{\!\top}\mathbf S_N\mathbf\Phi^{\!\top}\mathsf t = \sum_{n=1}^{N}\beta\boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x_n)t_n \tag{3.60}$$

$$y(x,\mathbf m_N) = \sum_{n=1}^{N}k(x,x_n)\,t_n \tag{3.61}$$

$$k(x,x') = \beta\boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x') \tag{3.62}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $k(x,x')$ | **nhân tương đương** hay *smoother matrix* — trọng số của $t$ tại $x'$ khi dự báo tại $x$ | vô hướng |
| $\mathbf S_N$ | vẫn là hiệp phương sai hậu nghiệm; **nó phụ thuộc mọi $x_n$ của tập huấn luyện** | $M\times M$ |

**Ý nghĩa trừu tượng.** (3.61) là một cách viết lại không thêm giả thiết nào, nhưng nó đổi
hẳn cách nghĩ. Trước (3.61), mô hình được nhìn như *"học một $\mathbf w$ rồi nhân với
$\boldsymbol\phi(x)$"*. Sau (3.61), nó là *"lấy bình quân có trọng số các $t_n$ đã thấy"* —
và $\mathbf w$ biến mất khỏi phát biểu. Bishop gọi lớp phương pháp như thế là **linear
smoother**.

Hai hệ quả mà cách viết cũ không cho thấy:

* **Trọng số là một hàm của cặp $(x, x_n)$**, và nếu ta định nghĩa thẳng một hàm như thế
  thay vì đi qua $\boldsymbol\phi$, ta được **quá trình Gauss**. Đó là câu cầu nối mà
  Bishop nói rõ: *"Instead of introducing a set of basis functions, which implicitly
  determines an equivalent kernel, we can instead define a localized kernel directly"*.
* **$k$ phụ thuộc toàn bộ tập huấn luyện**, qua $\mathbf S_N$. Nó không phải một nhân cố
  định trước như trong SVM; thêm một điểm dữ liệu là đổi $k$ ở mọi chỗ.

![Hình 3.10 của Bishop 2006](<paper_bishop2006_hinh_3_10.png>)

**Hình 16 — Hình 3.10 nguyên bản: nhân tương đương của hàm cơ sở Gauss.** Nguồn:
Bishop 2006, trang 159, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"The equivalent kernel k(x, x ) for the Gaussian basis functions in Figure 3.1, shown as
> a plot of x versus x , together with three slices through this matrix corresponding to
> three different values of x. The data set used to generate this kernel comprised 200
> values of x equally spaced over the interval"*

**Dịch:** *"Nhân tương đương k(x, x′) của các hàm cơ sở Gauss ở Hình 3.1, vẽ như một đồ thị
x theo x′, kèm ba lát cắt qua ma trận này ứng với ba giá trị khác nhau của x. Tập dữ liệu
dùng để sinh nhân này gồm 200 giá trị x cách đều trên khoảng"* $(-1, 1)$.

**Cách đọc hình.** Ô lớn là ma trận $k(x,x')$ vẽ như một ảnh: hàng là $x$, cột là $x'$,
sáng nghĩa là trọng số lớn. Vệt sáng nằm trên **đường chéo**, nghĩa là trọng số lớn khi
$x' \approx x$. Ba đường cong bên cạnh là ba hàng của ma trận đó ở ba giá trị $x$ khác nhau.

**Điều hình KHÔNG nói ra, và chỗ không đọc được.** Trang 159 là một **ảnh raster**:
`v_bayes.py` mục 3 đếm được 0 đường vectơ trên trang đó. Không đọc được toạ độ, nên hình
này chỉ dựng lại được **thiết lập** (200 điểm cách đều trên $(-1,1)$, hàm cơ sở Gauss của
Hình 3.1 — mà Hình 3.1 thì **không khai $\mu_j$ và $s$**). Dựng lại với $\mu_j$ =
`linspace(-1,1,9)` và $s = 0{,}25$:

| Lát cắt tại | Đỉnh rơi ở $x'$ | Lệch khỏi $x$ | $k(x,x)$ | $\sum_n k(x,x_n)$ |
|---:|---:|---:|---:|---:|
| $x = -0{,}50$ | $-0{,}4975$ | $+0{,}0025$ | 0,038280 | 0,999071 |
| $x = +0{,}00$ | $+0{,}0050$ | $+0{,}0050$ | 0,037453 | 0,999220 |
| $x = +0{,}50$ | $+0{,}4975$ | $-0{,}0025$ | 0,038280 | 0,999071 |

Đỉnh rơi đúng vào chính $x$ trong phạm vi một bước lưới ($2/199 = 0{,}01$), tức tính định
xứ mà Bishop nói là có. Và một nhận xét nữa của tác giả kiểm được ngay:

> *"Note that the kernel function can be negative as well as positive, so although it
> satisfies a summation constraint, the corresponding predictions are not necessarily convex
> combinations of the training set target variables."*

**Dịch:** *"Để ý rằng hàm nhân có thể âm cũng như dương, nên dù nó thoả một ràng buộc về
tổng, các dự báo tương ứng không nhất thiết là tổ hợp lồi của các giá trị mục tiêu trong
tập huấn luyện."*

Đếm: **17 396 trên 40 000 ô của ma trận nhân là số âm — 43,49 %**. Đó là gần một nửa, không
phải một vài ô ngoại lệ. Hệ quả thực hành: dự báo Bayes **có thể nằm ngoài khoảng
$[\min t_n, \max t_n]$**, khác hẳn một phép bình quân có trọng số dương.

### 2.13 Hình 3.11 — hàm cơ sở KHÔNG định xứ vẫn cho nhân định xứ

![Hình 3.11 của Bishop 2006](<paper_bishop2006_hinh_3_11.png>)

**Hình 17 — Hình 3.11 nguyên bản: nhân tương đương của hàm cơ sở đa thức và sigmoid.**
Nguồn: Bishop 2006, trang 160,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Examples of equivalent kernels k(x, x ) for x = 0 plotted as a function of x ,
> corresponding (left) to the polynomial basis functions and (right) to the sigmoidal basis
> functions shown in Figure 3.1. Note that these are localized functions of x even though
> the corresponding basis functions are nonlocal."*

**Dịch:** *"Ví dụ về nhân tương đương k(x, x′) tại x = 0, vẽ như một hàm của x′, ứng với
(trái) các hàm cơ sở đa thức và (phải) các hàm cơ sở sigmoid ở Hình 3.1. Để ý rằng đây là
những hàm định xứ của x′ dù các hàm cơ sở tương ứng là không định xứ."*

**Cách đọc hình.** Hai ô, mỗi ô một đường. Trục ngang là $x'$ chạy từ $-1$ tới $1$; trục
dọc là $k(0, x')$ với vạch chia ở 0 · 0,02 · 0,04. Cả hai đường có một đỉnh ở giữa và tắt
dần ra hai biên — đó là điều chú thích muốn nói.

**Đây là hình vectơ**, nên đọc được số. Kết quả đọc:

| Ô | Số điểm đọc được | Đỉnh | Vị trí đỉnh | Giá trị tại $x'=-1$ | Giá trị tại $x'=+1$ |
|---|---:|---:|---:|---:|---:|
| đa thức | 109 | **0,026316** | $x' = +0{,}0033$ | $+0{,}004606$ | $+0{,}004606$ |
| sigmoid | 199 | **0,042497** | $x' = -0{,}0100$ | $-0{,}003631$ | $-0{,}003073$ |

Hai điều đọc được mà chú thích không nói: đỉnh của bản đa thức **thấp hơn** bản sigmoid
(0,0263 so với 0,0425), và bản sigmoid **âm** ở cả hai biên trong khi bản đa thức thì
dương. Cả hai đỉnh nằm ở $x' \approx 0$ với sai lệch dưới 0,01, tức đúng vị trí $x=0$ mà
chú thích nói.

**Chỗ KHÔNG dựng lại được.** Hình 3.11 dẫn về "các hàm cơ sở ở Hình 3.1", và chú thích Hình
3.1 chỉ nói *"Examples of basis functions, showing polynomials on the left, Gaussians of the
form (3.4) in the centre, and sigmoidal of the form (3.5) on the right"* — **không** một
con số nào về $\mu_j$, $s$, bậc đa thức, $\alpha$, hay $\beta$. Dựng lại với bộ tham số dò
được ở mục 2.9 cho:

| Ô | Đỉnh dựng lại | Vị trí | $\sum_n k$ | Số ô âm trên 200 |
|---|---:|---:|---:|---:|
| đa thức | 0,018972 | $x' = -0{,}0050$ | 0,999122 | 72 |
| sigmoid | 0,020667 | $x' = +0{,}0050$ | 0,999134 | 79 |

Vị trí đỉnh khớp; **độ cao thì không** — 0,0190 so với 0,0263 đọc được, và 0,0207 so với
0,0425. Đó là kết quả đúng phải báo cáo: với bốn tham số không khai, hình này chỉ kiểm được
**hình dạng** (một đỉnh ở $x'\approx 0$, tắt dần ra biên, có ô âm), không kiểm được biên độ.
Ghi ra hạn chế đó tốt hơn hẳn việc chỉnh bốn tham số cho tới lúc độ cao chịu khớp.

### 2.14 (3.63) — hiệp phương sai giữa hai dự báo

![Công thức 3.63 của Bishop 2006](<paper_bishop2006_ct_3_63.png>)

**Hình 18 — hiệp phương sai (3.63).** Nguồn: Bishop 2006, trang 160,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"From the form of the equivalent kernel, we see that the predictive mean at nearby points
> will be highly correlated, whereas for more distant pairs of points the correlation will
> be smaller."*

**Dịch:** *"Từ dạng của nhân tương đương, ta thấy trung bình dự báo ở những điểm gần nhau sẽ
tương quan mạnh, còn với những cặp điểm xa nhau hơn thì tương quan sẽ nhỏ hơn."*

$$\operatorname{cov}\!\left[y(x), y(x')\right] = \operatorname{cov}\!\left[\boldsymbol\phi(x)^{\!\top}\mathbf w,\ \mathbf w^{\!\top}\boldsymbol\phi(x')\right] = \boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x') = \beta^{-1}k(x,x') \tag{3.63}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\operatorname{cov}[\cdot,\cdot]$ | hiệp phương sai theo **hậu nghiệm của $\mathbf w$**, không theo nhiễu | vô hướng |
| $\beta^{-1}k(x,x')$ | cùng một hàm $k$ của (3.62), chỉ chia lại bằng $\beta$ | vô hướng |

**Ý nghĩa trừu tượng.** (3.63) là chỗ nhân tương đương thôi làm một mẹo viết lại và trở
thành **một đại lượng có nghĩa xác suất**: nó chính là hiệp phương sai hậu nghiệm giữa hai
dự báo, sai khác hằng số $\beta^{-1}$. Từ đó Hình 3.9 có một cách đọc định lượng: các đường
cong ở đó song song nhau nhiều tới đâu thì $k$ lớn tới đó.

Kiểm bằng số: dựng cả $\boldsymbol\phi(x)^{\!\top}\mathbf S_N\boldsymbol\phi(x')$ và
$\beta^{-1}k(x,x')$ trên lưới $200\times200$ rồi so từng ô — lệch lớn nhất
$4{,}432\times10^{-16}$, tức bằng nhau tới mức làm tròn `float64`. Và ma trận nhân đối xứng
tới $4{,}774\times10^{-15}$, đúng như (3.62) đòi.

### 2.15 (3.64) — "these weights sum to one". Đúng tới đâu?

![Công thức 3.64 của Bishop 2006](<paper_bishop2006_ct_3_64.png>)

**Hình 19 — ràng buộc tổng (3.64) và phần chứng minh không hình thức.** Nguồn:
Bishop 2006, trang 160, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"We have seen that the effective kernel defines the weights by which the training set
> target values are combined in order to make a prediction at a new value of x, and it can
> be shown that these weights sum to one, in other words"*

$$\sum_{n=1}^{N}k(x,x_n) = 1 \tag{3.64}$$

Đọc thành lời: dự báo Bayes là một phép **bình quân đúng nghĩa** — bộ trọng số cộng lại bằng 1,
nên nếu mọi $t_n$ bằng nhau thì dự báo bằng đúng giá trị chung đó, không co cũng không nở. Đó
là một ràng buộc mạnh, và phần còn lại của mục này đo xem nó đúng tới đâu.

> *"This intuitively pleasing result can easily be proven informally by noting that the
> summation is equivalent to considering the predictive mean y(x) for a set of target data
> in which tn = 1 for all n. Provided the basis functions are linearly independent, that
> there are more data points than basis functions, and that one of the basis functions is
> constant (corresponding to the bias parameter), then it is clear that we can fit the
> training data exactly and hence that the predictive mean will be simply"*

**Dịch:** *"Ta đã thấy nhân hiệu dụng định ra các trọng số dùng để tổ hợp các giá trị mục
tiêu của tập huấn luyện nhằm dự báo tại một giá trị x mới, và có thể chứng minh được rằng
các trọng số này cộng lại bằng một, nói cách khác"* · *"Kết quả dễ chịu về mặt trực giác này
chứng minh không hình thức rất dễ, bằng cách để ý rằng phép tổng đó tương đương với việc xét
trung bình dự báo y(x) cho một bộ dữ liệu mục tiêu trong đó tₙ = 1 với mọi n. Miễn là các
hàm cơ sở độc lập tuyến tính, rằng số điểm dữ liệu nhiều hơn số hàm cơ sở, và rằng một trong
các hàm cơ sở là hằng (ứng với tham số chệch), thì rõ ràng là ta khớp được dữ liệu huấn
luyện ĐÚNG KHÍT và do đó trung bình dự báo đơn giản bằng"* $y(x) = 1$.

**Ý nghĩa trừu tượng.** (3.64) nói rằng phép dự báo Bayes là một phép **bình quân** đúng
nghĩa: các trọng số cộng lại bằng 1, nên nếu mọi $t_n$ bằng nhau thì dự báo bằng đúng giá
trị đó. Cộng với chuyện $k$ có thể âm (mục 2.12), ta được một phát biểu gọn: nó là bình quân
**có trọng số cộng bằng 1** nhưng **không phải tổ hợp lồi**.

**Nhưng chứng minh trên có một chỗ hở, và nó đo được.** Bước quyết định là *"we can fit the
training data exactly"*. Trung bình dự báo (3.53) **không** phải nghiệm khớp đúng khít — nó
là nghiệm **đã co** bởi tiên nghiệm. Với $t_n \equiv 1$ và một hàm cơ sở hằng, tiên nghiệm
(3.52) phạt **cả** $w_0$, nên $w_0$ bị kéo về 0 và đường khớp không nằm đúng ở mức 1.

Đo trên chính thiết lập của Hình 3.10 (200 điểm cách đều, 9 hàm Gauss cộng $\phi_0$):

| $\alpha$ | $\sum_n k(0, x_n)$ | $\sum_n k(0{,}5, x_n)$ | $\max_x\lvert\sum_n k - 1\rvert$ |
|---:|---:|---:|---:|
| $10^{-12}$ | 1,0000000000 | 1,0000000000 | $5{,}647\times10^{-12}$ |
| $10^{-8}$ | 1,0000000005 | 1,0000000005 | $2{,}514\times10^{-9}$ |
| $10^{-6}$ | 1,0000000483 | 1,0000000489 | $2{,}512\times10^{-7}$ |
| $10^{-4}$ | 1,0000048061 | 1,0000048756 | $2{,}505\times10^{-5}$ |
| $10^{-2}$ | 1,0003409750 | 1,0003470078 | $1{,}945\times10^{-3}$ |
| **0,513333** | 0,9992201979 | 0,9990710372 | $\mathbf{1{,}395\times10^{-2}}$ |
| 1 | 0,9986365507 | 0,9981119440 | $1{,}777\times10^{-2}$ |
| 10 | 0,9995008126 | 0,9952108330 | $4{,}170\times10^{-2}$ |
| 100 | 0,9776344549 | 0,9852287984 | $1{,}093\times10^{-1}$ |

Độ lệch **tăng đơn điệu theo $\alpha$**, từ $5{,}6\times10^{-12}$ lên $0{,}109$ — tức hơn
mười bậc độ lớn. Ba điều kiện mà Bishop **có** nêu (hàm cơ sở độc lập tuyến tính · $N > M$ ·
có một hàm cơ sở hằng) đều thoả ở mọi dòng của bảng. Điều kiện ông **không** nêu là
$\alpha \to 0$.

Ở dòng in đậm — $\alpha$ là chính giá trị dựng lại được Hình 3.8 — độ lệch đã là **1,395 %**.
Đó không lớn, nhưng nó không phải 0, và câu *"these weights sum to one"* nên đọc là *"cộng
lại bằng một trong giới hạn tiên nghiệm không ràng buộc"*.

Cách đọc này ăn khớp với một chỗ khác của cùng chương: nếu $\alpha \to 0$ thì theo mục 2.4,
$\mathbf m_N \to \mathbf w_{\text{ML}}$, tức đúng nghiệm khớp khít mà chứng minh của Bishop
giả định. Vậy chứng minh đó đúng cho **ước lượng hợp lẽ cực đại**, và (3.64) là một tính
chất của ước lượng đó chứ không của trung bình hậu nghiệm ở $\alpha$ hữu hạn.

### 2.16 (3.65) — nhân tương đương là một tích trong

![Công thức 3.65 của Bishop 2006](<paper_bishop2006_ct_3_65.png>)

**Hình 20 — (3.65), nhân viết thành tích trong.** Nguồn: Bishop 2006, trang 161,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Finally, we note that the equivalent kernel (3.62) satisfies an important property shared
> by kernel functions in general, namely that it can be expressed in the form an inner
> product with respect to a vector"*

**Dịch:** *"Cuối cùng, ta để ý rằng nhân tương đương (3.62) thoả một tính chất quan trọng mà
các hàm nhân nói chung đều có, cụ thể là nó viết được thành một tích trong theo một vectơ"*
$\boldsymbol\psi(x)$ các hàm phi tuyến.

$$k(x,z) = \boldsymbol\psi(x)^{\!\top}\boldsymbol\psi(z), \qquad \boldsymbol\psi(x) = \beta^{1/2}\mathbf S_N^{1/2}\boldsymbol\phi(x) \tag{3.65}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\boldsymbol\psi(x)$ | ánh xạ đặc trưng mới, bằng $\boldsymbol\phi$ đã đổi thang theo $\mathbf S_N^{1/2}$ | $M$ |
| $\mathbf S_N^{1/2}$ | căn bậc hai ma trận — tồn tại vì $\mathbf S_N$ đối xứng xác định dương | $M\times M$ |

**Ý nghĩa trừu tượng.** Câu này nhỏ về hình thức và lớn về hệ quả: nó nói nhân tương đương
là một **nhân hợp lệ** theo nghĩa của lý thuyết hạt nhân, tức nửa xác định dương. Từ đó
toàn bộ bộ máy của chương 6 dùng được cho nó, và chỗ nối giữa hồi quy tuyến tính Bayes với
quá trình Gauss thành một đẳng thức chứ không chỉ một sự tương tự.

Chi tiết dễ bỏ qua: $\mathbf S_N^{1/2}$ **tồn tại** chỉ vì $\mathbf S_N$ xác định dương, và
điều đó đúng vì (3.54) cộng $\alpha\mathbf I$ với $\alpha > 0$. Nếu $\alpha = 0$ và
$\mathbf\Phi^{\!\top}\mathbf\Phi$ suy biến thì căn bậc hai không còn xác định. Đo trên mô
hình của Hình 3.10: trị riêng nhỏ nhất của $\mathbf S_N$ là $2{,}686\times10^{-4}$, dương,
nên phép căn chạy được. Kiểm đẳng thức (3.65) trên lưới $200\times200$: lệch lớn nhất
$4{,}749\times10^{-14}$.

### 2.17 Mục 3.4 của sách — so sánh mô hình bằng xác suất, không bằng tập giữ lại

![Mở đầu mục 3.4 của sách của Bishop 2006](<paper_bishop2006_mo_dau_3_4.png>)

**Hình 21 — mở đầu mục 3.4.** Nguồn: Bishop 2006, trang 161,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"As we shall see, the over-fitting associated with maximum likelihood can be avoided by
> marginalizing (summing or integrating) over the model parameters instead of making point
> estimates of their values. Models can then be compared directly on the training data,
> without the need for a validation set. This allows all available data to be used for
> training and avoids the multiple training runs for each model associated with
> cross-validation."*

**Dịch:** *"Như ta sẽ thấy, hiện tượng quá khớp gắn với cực đại hoá hợp lẽ tránh được bằng
cách lấy biên (cộng hoặc tích phân) trên các tham số mô hình thay vì ước lượng điểm cho giá
trị của chúng. Khi đó các mô hình so sánh được trực tiếp trên dữ liệu huấn luyện, không cần
một tập thẩm định. Điều này cho phép dùng toàn bộ dữ liệu có được để huấn luyện, và tránh
được nhiều lần huấn luyện cho từng mô hình như kiểm định chéo đòi."*

**Cách đọc khung.** Ba lời hứa trong một đoạn, và mục 1 của bài này đã đo cả ba: *"không cần
tập thẩm định"*, *"dùng toàn bộ dữ liệu"*, *"tránh nhiều lần huấn luyện"* — 34 lần giải hệ
thay cho 1 000, và một giá trị $\lambda$ tất định thay cho một khoảng trải 3,4 lần. Câu cuối
của đoạn cũng đáng ghi vì nó là con đường sang chương 7:

> *"It also allows multiple complexity parameters to be determined simultaneously as part of
> the training process. For example, in Chapter 7 we shall introduce the relevance vector
> machine, which is a Bayesian model having one complexity parameter for every training data
> point."*

**Dịch:** *"Nó cũng cho phép xác định đồng thời nhiều tham số độ phức tạp như một phần của
quá trình huấn luyện. Ví dụ, ở chương 7 ta sẽ giới thiệu relevance vector machine, một mô
hình Bayes có một tham số độ phức tạp cho mỗi điểm dữ liệu huấn luyện."*

Đó chính là `ARDRegression` của `scikit-learn`, và mục 3 đo nó: một $\alpha$ riêng cho từng
hệ số thì **8 trên 10** hệ số bị đẩy về dưới $10^{-3}$. Kiểm định chéo không làm được việc
đó, vì không ai quét được một lưới mười chiều.

### 2.18 (3.66)–(3.67) — hậu nghiệm trên MÔ HÌNH, và dự báo là một hỗn hợp

![Công thức 3.66 của Bishop 2006](<paper_bishop2006_ct_3_66.png>)

**Hình 22 — (3.66) và định nghĩa model evidence.** Nguồn: Bishop 2006, trang 161,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Suppose we wish to compare a set of L models {Mi} where i = 1, . . . , L. Here a model
> refers to a probability distribution over the observed data D."*

> *"We shall suppose that the data is generated from one of these models but we are uncertain
> which one."*

**Dịch:** *"Giả sử ta muốn so sánh một tập L mô hình {Mᵢ} với i = 1, …, L. Ở đây một mô hình
nghĩa là một phân phối xác suất trên dữ liệu quan sát được D."* · *"Ta sẽ giả thiết rằng dữ
liệu được sinh ra từ MỘT trong các mô hình đó, nhưng ta không chắc là mô hình nào."*

$$p(\mathcal M_i \mid \mathcal D) \propto p(\mathcal M_i)\,p(\mathcal D \mid \mathcal M_i) \tag{3.66}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathcal M_i$ | mô hình thứ $i$ — **một phân phối trên dữ liệu**, không phải một hàm | — |
| $p(\mathcal M_i)$ | tiên nghiệm trên mô hình; Bishop đặt bằng nhau cho mọi $i$ | vô hướng |
| $p(\mathcal D\mid\mathcal M_i)$ | **model evidence** hay *marginal likelihood* | vô hướng |

**Ý nghĩa trừu tượng.** (3.66) là quy tắc Bayes áp cho một biến mới: **chính mô hình**.
Chỗ đáng chú ý không phải công thức mà là câu Bishop đặt ngay sau đó — *"We shall suppose
that the data is generated from one of these models"*. Đó là một **giả thiết mạnh và nó có
thể sai**, và Bishop nói thẳng điều đó ở mục 2.20. Nếu mô hình thật không nằm trong họ đang
xét thì evidence vẫn xếp hạng được các mô hình, nhưng bảo đảm "chọn đúng mô hình" mất hiệu
lực.

Bishop cũng nêu một tên gọi và một đại lượng:

> *"The ratio of model evidences p(D|Mi)/p(D|Mj ) for two models is known as a Bayes factor"*

**Dịch:** *"Tỉ số các model evidence p(D|Mᵢ)/p(D|Mⱼ) của hai mô hình được gọi là một Bayes
factor."*

![Công thức 3.67 của Bishop 2006](<paper_bishop2006_ct_3_67.png>)

**Hình 23 — (3.67), dự báo là một hỗn hợp trên các mô hình.** Nguồn: Bishop 2006,
trang 162, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"For instance, if we have two models that are a-posteriori equally likely and one predicts
> a narrow distribution around t = a while the other predicts a narrow distribution around
> t = b, the overall predictive distribution will be a bimodal distribution with modes at
> t = a and t = b, not a single model at t = (a + b)/2."*

> *"A simple approximation to model averaging is to use the single most probable model alone
> to make predictions. This is known as model selection."*

**Dịch:** *"Chẳng hạn, nếu ta có hai mô hình hậu nghiệm ngang nhau và một mô hình dự báo một
phân phối hẹp quanh t = a còn mô hình kia dự báo một phân phối hẹp quanh t = b, thì phân phối
dự báo tổng thể sẽ là một phân phối HAI ĐỈNH với đỉnh ở t = a và t = b, chứ không phải một
đỉnh duy nhất ở t = (a + b)/2."* · *"Một xấp xỉ đơn giản cho phép bình quân mô hình là chỉ
dùng một mô hình khả dĩ nhất để dự báo. Cách đó gọi là CHỌN mô hình."*

$$p(t \mid x, \mathcal D) = \sum_{i=1}^{L} p(t \mid x, \mathcal M_i, \mathcal D)\, p(\mathcal M_i \mid \mathcal D) \tag{3.67}$$

**Ý nghĩa trừu tượng.** Ví dụ hai đỉnh của Bishop là chỗ dễ nhớ nhất của cả mục: **bình
quân các PHÂN PHỐI không phải bình quân các DỰ BÁO**. Trộn hai phân phối hẹp ở $a$ và $b$
cho một phân phối có hai đỉnh, và giá trị $(a+b)/2$ — thứ mà một phép bình quân điểm sẽ trả
về — có thể là chỗ xác suất **thấp nhất**. Đó cũng là lý do câu cuối phải gọi *model
selection* là một **xấp xỉ**, không phải một cách làm tương đương.

### 2.19 (3.68)–(3.72) — evidence là gì, và vì sao nó tự phạt độ phức tạp

![Công thức 3.68 đến 3.70 của Bishop 2006](<paper_bishop2006_ct_3_68_3_70.png>)

**Hình 24 — (3.68)–(3.70).** Nguồn: Bishop 2006, trang 162,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"From a sampling perspective, the marginal likelihood can be viewed as the probability of
> generating the data set D from a model whose parameters are sampled at random from the
> prior."*

> *"It is also interesting to note that the evidence is precisely the normalizing term that
> appears in the denominator in Bayes' theorem when evaluating the posterior distribution
> over parameters"*

**Dịch:** *"Nhìn theo góc độ lấy mẫu, marginal likelihood có thể xem là xác suất sinh ra
được tập dữ liệu D từ một mô hình mà tham số của nó được rút ngẫu nhiên từ tiên nghiệm."* ·
*"Cũng đáng để ý rằng evidence chính là số hạng chuẩn hoá xuất hiện ở mẫu số trong định lý
Bayes khi tính hậu nghiệm trên tham số"*

$$p(\mathcal D \mid \mathcal M_i) = \int p(\mathcal D \mid \mathbf w, \mathcal M_i)\, p(\mathbf w \mid \mathcal M_i)\,\mathrm d\mathbf w \tag{3.68}$$

$$p(\mathbf w \mid \mathcal D, \mathcal M_i) = \frac{p(\mathcal D\mid\mathbf w,\mathcal M_i)\,p(\mathbf w\mid\mathcal M_i)}{p(\mathcal D\mid\mathcal M_i)} \tag{3.69}$$

$$p(\mathcal D) = \int p(\mathcal D\mid w)\,p(w)\,\mathrm dw \simeq p(\mathcal D\mid w_{\text{MAP}})\frac{\Delta w_{\text{posterior}}}{\Delta w_{\text{prior}}} \tag{3.70}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\Delta w_{\text{prior}}$ | bề rộng của tiên nghiệm, coi như phẳng trên khoảng đó | vô hướng |
| $\Delta w_{\text{posterior}}$ | bề rộng của hậu nghiệm quanh $w_{\text{MAP}}$ | vô hướng |

**Ý nghĩa trừu tượng.** Câu *"xác suất sinh ra được tập dữ liệu D từ tham số rút ngẫu nhiên
từ tiên nghiệm"* là cách đọc đúng nhất của evidence, và nó giải thích ngay vì sao mô hình
phức tạp bị phạt: một mô hình phức tạp **rút ra được rất nhiều bộ dữ liệu khác nhau**, nên
xác suất nó rút đúng bộ dữ liệu ta đang có phải nhỏ. Không có ai thêm một số hạng phạt vào
đâu cả — phép chuẩn hoá tự làm việc đó.

(3.69) nói một điều nhỏ mà đáng nhớ: evidence là **mẫu số** của định lý Bayes. Nó là thứ
thường bị bỏ qua khi người ta chỉ cần hậu nghiệm (vì nó không phụ thuộc $\mathbf w$) — và
chính thứ bị bỏ qua đó là thứ dùng để chọn mô hình.

![Hình 3.12 của Bishop 2006](<paper_bishop2006_hinh_3_12.png>)

**Hình 25 — Hình 3.12 nguyên bản: xấp xỉ thô cho evidence.** Nguồn: Bishop 2006,
trang 163, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"We can obtain a rough approximation to the model evidence if we assume that the posterior
> distribution over parameters is sharply peaked around its mode wMAP."*

**Dịch:** *"Ta được một xấp xỉ thô cho model evidence nếu giả thiết rằng hậu nghiệm trên tham
số có một đỉnh nhọn quanh đỉnh wMAP của nó."*

**Cách đọc hình.** Trục ngang là một tham số $w$. Hình vẽ hai bề rộng: $\Delta w_{\text{prior}}$
trải rộng, $\Delta w_{\text{posterior}}$ hẹp, và $w_{\text{MAP}}$ ở giữa chỗ hẹp. Tích phân
(3.68) được xấp xỉ bằng *"giá trị của hàm dưới dấu tích phân tại cực đại nhân bề rộng của
đỉnh"*, và vì tiên nghiệm phẳng với bề rộng $\Delta w_{\text{prior}}$ nên
$p(w) = 1/\Delta w_{\text{prior}}$.

**Điều hình KHÔNG nói ra.** Hình vẽ **một** tham số. Với $M$ tham số thì phép xấp xỉ phải
lặp lại $M$ lần, và Bishop nói rõ điều kiện ông thêm vào: *"Assuming that all parameters
have the same ratio of"* $\Delta w_{\text{posterior}}/\Delta w_{\text{prior}}$ — một giả
thiết mà hình không thể vẽ được, và là chỗ khoản phạt trở thành **tuyến tính theo $M$**.

![Công thức 3.71 và 3.72 của Bishop 2006](<paper_bishop2006_ct_3_71_3_72.png>)

**Hình 26 — (3.71) và (3.72).** Nguồn: Bishop 2006, trang 163,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"The first term represents the fit to the data given by the most probable parameter values,
> and for a flat prior this would correspond to the log likelihood. The second term penalizes
> the model according to its complexity."*

> *"Thus, in this very simple approximation, the size of the complexity penalty increases
> linearly with the number M of adaptive parameters in the model."*

**Dịch:** *"Số hạng thứ nhất biểu diễn mức khớp với dữ liệu ứng với bộ giá trị tham số khả dĩ
nhất, và với tiên nghiệm phẳng thì nó ứng với log hợp lẽ. Số hạng thứ hai phạt mô hình theo
độ phức tạp của nó."* · *"Vậy, trong phép xấp xỉ rất đơn giản này, độ lớn của khoản phạt độ
phức tạp tăng TUYẾN TÍNH theo số tham số điều chỉnh được M của mô hình."*

$$\ln p(\mathcal D) \simeq \ln p(\mathcal D \mid w_{\text{MAP}}) + \ln\!\left(\frac{\Delta w_{\text{posterior}}}{\Delta w_{\text{prior}}}\right) \tag{3.71}$$

$$\ln p(\mathcal D) \simeq \ln p(\mathcal D \mid \mathbf w_{\text{MAP}}) + M\ln\!\left(\frac{\Delta w_{\text{posterior}}}{\Delta w_{\text{prior}}}\right) \tag{3.72}$$

**Ý nghĩa trừu tượng.** Vì $\Delta w_{\text{posterior}} < \Delta w_{\text{prior}}$, logarit
tỉ số là **âm**, nên số hạng thứ hai luôn là một khoản trừ. Và nó lớn dần khi hậu nghiệm hẹp
đi so với tiên nghiệm — tức khi tham số bị *"finely tuned to the data"*, chữ của Bishop. Đó
là định nghĩa định lượng của "mô hình dùng đúng bộ tham số vừa khít dữ liệu này thì đáng ngờ".

**Kiểm bằng số: (3.71) có thật là hai số hạng đó không.** Công thức đóng (3.86) của mục 2.23
tách được thành đúng ba phần, và phần phạt phải khớp mô tả trên. Đo trên 10 điểm của mục 1.1,
$\alpha = 5\times10^{-3}$, $\beta = 11{,}1$, hàm cơ sở đa thức:

| Bậc | (3.86) chính xác | log hợp lẽ tại $\mathbf m_N$ | $-\tfrac12\ln\lvert\mathbf A\rvert$ | $\tfrac M2\ln\alpha - \tfrac12\ln\lvert\mathbf A\rvert$ |
|---:|---:|---:|---:|---:|
| 0 | $-23{,}123666$ | $-18{,}119632$ | $-2{,}354788$ | $-5{,}003946$ |
| 1 | $-17{,}879100$ | $-9{,}007361$ | $-3{,}567711$ | $-8{,}866028$ |
| 2 | $-20{,}306141$ | $-8{,}829851$ | $-3{,}516908$ | $-11{,}464384$ |
| 3 | $-13{,}934076$ | $+0{,}471507$ | $-2{,}268036$ | $-12{,}864670$ |
| 5 | $-14{,}359022$ | $+0{,}602254$ | $+1{,}359124$ | $-14{,}535828$ |
| 9 | $-16{,}290483$ | $+0{,}747268$ | $+9{,}895016$ | $-16{,}596571$ |

Hai điều đọc được, và điều thứ hai là một cái bẫy:

* **Cột log hợp lẽ tăng đơn điệu** theo bậc, từ $-18{,}12$ lên $+0{,}75$ — đúng điều
  Bishop nói, *"a more complex model is better able to fit the data"*.
* **Cột $-\tfrac12\ln\lvert\mathbf A\rvert$ ĐỔI DẤU**, từ $-2{,}354788$ lên
  $+9{,}895016$. Vậy một mình nó **không** phải khoản phạt. Cụm đầy đủ
  $\tfrac M2\ln\alpha - \tfrac12\ln\lvert\mathbf A\rvert$ thì giảm đơn điệu ở cả mười
  bậc: $-5{,}00$ · $-8{,}87$ · $-11{,}46$ · $-12{,}86$ · $-13{,}80$ · $-14{,}54$ ·
  $-15{,}16$ · $-15{,}71$ · $-16{,}18$ · $-16{,}60$. Đó mới là thứ (3.72) nói tới, và nó
  đúng.

Và một phép kiểm cộng đơn giản: ba số hạng cộng lại phải bằng cột đầu ở **cả sáu dòng**,
đúng tới $10^{-9}$.

### 2.20 (3.73) và Hình 3.13 — vì sao evidence thích mô hình ĐỘ PHỨC TẠP TRUNG BÌNH

![Hình 3.13 của Bishop 2006](<paper_bishop2006_hinh_3_13.png>)

**Hình 27 — Hình 3.13 nguyên bản: phân phối của các bộ dữ liệu.** Nguồn: Bishop 2006,
trang 164, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Schematic illustration of the distribution of data sets for three models of different
> complexity, in which M1 is the simplest and M3 is the most complex. Note that the
> distributions are normalized. In this example, for the particular observed data set D0, the
> model M2 with intermediate complexity has the largest evidence."*

**Dịch:** *"Minh hoạ sơ đồ về phân phối của các bộ dữ liệu ứng với ba mô hình có độ phức tạp
khác nhau, trong đó M₁ là đơn giản nhất và M₃ là phức tạp nhất. Để ý rằng các phân phối đều
đã CHUẨN HOÁ. Trong ví dụ này, với bộ dữ liệu quan sát được cụ thể D₀, mô hình M₂ có độ phức
tạp trung bình là mô hình có evidence lớn nhất."*

**Nhận xét của chính tác giả, nguyên văn từ thân bài tr.164:**

> *"A simple model (for example, based on a first order polynomial) has little variability and
> so will generate data sets that are fairly similar to each other. Its distribution p(D) is
> therefore confined to a relatively small region of the horizontal axis."*

> *"Because the distributions p(D|Mi) are normalized, we see that the particular data set D0
> can have the highest value of the evidence for the model of intermediate complexity.
> Essentially, the simpler model cannot fit the data well, whereas the more complex model
> spreads its predictive probability over too broad a range of data sets and so assigns
> relatively small probability to any one of them."*

**Dịch:** *"Một mô hình đơn giản (chẳng hạn dựa trên đa thức bậc một) có ít biến thiên nên sẽ
sinh ra những bộ dữ liệu khá giống nhau. Phân phối p(D) của nó vì thế bị giới hạn trong một
vùng tương đối nhỏ của trục ngang."* · *"Vì các phân phối p(D|Mᵢ) đã được chuẩn hoá, ta thấy
rằng bộ dữ liệu cụ thể D₀ có thể có giá trị evidence cao nhất ở mô hình có độ phức tạp trung
bình. Về bản chất, mô hình đơn giản hơn thì không khớp được dữ liệu, còn mô hình phức tạp hơn
thì rải xác suất dự báo của nó trên một dải bộ dữ liệu quá rộng và do đó gán xác suất khá nhỏ
cho bất kỳ bộ nào."*

**Cách đọc hình.** Trục ngang là **không gian các bộ dữ liệu**, nén xuống một chiều: mỗi điểm
trên trục là một bộ dữ liệu. Ba đường là $p(\mathcal D\mid\mathcal M_i)$ cho ba mô hình. Vạch
dọc $\mathcal D_0$ là bộ dữ liệu ta thật sự có. Giá trị của ba đường **tại vạch đó** là ba
evidence.

**Điều hình KHÔNG nói ra, và nó là điều quan trọng nhất.** Từ *"normalized"* trong chú thích
là bản lề của cả lập luận: ba đường đều có **diện tích bằng 1**, nên đường nào trải rộng hơn
thì bắt buộc phải **thấp hơn** ở mọi chỗ. Không có chỗ nào trong lập luận cần tới "độ phức
tạp" như một khái niệm riêng — chỉ cần phép chuẩn hoá. Đó là lý do phép phạt độ phức tạp của
Bayes **không phải một tham số người dùng chọn**.

Và hình **không** nói điều ngược lại, thứ mà người đọc dễ tự thêm vào: nó không nói mô hình
trung bình **luôn** thắng. Nó nói *"can have the highest value"* — với một $\mathcal D_0$
khác, nằm ở đuôi phải, mô hình $\mathcal M_3$ sẽ thắng.

![Công thức 3.73 của Bishop 2006](<paper_bishop2006_ct_3_73.png>)

**Hình 28 — (3.73) và bảo đảm trung bình của Bayes factor.** Nguồn: Bishop 2006,
trang 164, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Implicit in the Bayesian model comparison framework is the assumption that the true
> distribution from which the data are generated is contained within the set of models under
> consideration."*

> *"For a given finite data set, it is possible for the Bayes factor to be larger for the
> incorrect model. However, if we average the Bayes factor over the distribution of data sets,
> we obtain the expected Bayes factor in the form"*

**Dịch:** *"Ngầm định trong khuôn khổ so sánh mô hình Bayes là giả thiết rằng phân phối thật
sinh ra dữ liệu NẰM TRONG tập các mô hình đang xét."* · *"Với một bộ dữ liệu hữu hạn cho
trước, Bayes factor có thể lớn hơn ở mô hình SAI. Tuy nhiên, nếu lấy bình quân Bayes factor
trên phân phối của các bộ dữ liệu, ta được Bayes factor kỳ vọng ở dạng"*

$$\int p(\mathcal D\mid\mathcal M_1)\ln\frac{p(\mathcal D\mid\mathcal M_1)}{p(\mathcal D\mid\mathcal M_2)}\,\mathrm d\mathcal D \tag{3.73}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\int \cdots \mathrm d\mathcal D$ | bình quân trên phân phối **thật** của các bộ dữ liệu | — |
| cả biểu thức | phân kỳ Kullback–Leibler giữa hai phân phối trên không gian bộ dữ liệu | vô hướng |

**Ý nghĩa trừu tượng.** (3.73) là một phát biểu về **trung bình**, và hai chữ đó phải giữ khi
trích. Vì KL luôn không âm và bằng 0 chỉ khi hai phân phối trùng nhau, Bayes factor kỳ vọng
luôn nghiêng về mô hình đúng. Nhưng Bishop nói rõ câu đi trước: với **một** bộ dữ liệu hữu
hạn thì nó có thể nghiêng về mô hình sai. Đọc (3.73) thành *"Bayes factor luôn chọn đúng"* là
bỏ mất chính điều kiện làm nó thành một định lý.

Và điều kiện thứ hai, nêu trong câu đầu, là điều kiện dễ vi phạm nhất trong thực hành: phân
phối thật phải **nằm trong** họ mô hình đang xét. Với dữ liệu thật thì gần như không bao giờ
đúng.

### 2.21 Cảnh báo tr.165 — evidence rất nhạy với TIÊN NGHIỆM

![Cảnh báo về tiên nghiệm, trang 165](<paper_bishop2006_canh_bao_tien_nghiem.png>)

**Hình 29 — cảnh báo cuối mục 3.4.** Nguồn: Bishop 2006, trang 165,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"a Bayesian approach, like any approach to pattern recognition, needs to make assumptions
> about the form of the model, and if these are invalid then the results can be misleading. In
> particular, we see from Figure 3.12 that the model evidence can be sensitive to many aspects
> of the prior, such as the behaviour in the tails."*

> *"Indeed, the evidence is not defined if the prior is improper, as can be seen by noting that
> an improper prior has an arbitrary scaling factor"*

> *"In a practical application, therefore, it will be wise to keep aside an independent test set
> of data on which to evaluate the overall performance of the final system."*

**Dịch:** *"một cách tiếp cận Bayes, cũng như mọi cách tiếp cận nhận dạng mẫu khác, phải đặt
giả thiết về dạng của mô hình, và nếu các giả thiết đó không đúng thì kết quả có thể gây ngộ
nhận. Cụ thể, từ Hình 3.12 ta thấy model evidence có thể NHẠY với nhiều khía cạnh của tiên
nghiệm, chẳng hạn hành vi ở phần đuôi."* · *"Thật vậy, evidence KHÔNG ĐƯỢC ĐỊNH NGHĨA nếu tiên
nghiệm là không chính quy, điều này thấy được khi để ý rằng một tiên nghiệm không chính quy có
một thừa số tỉ lệ tuỳ ý"* · *"Vì thế trong một ứng dụng thực tế, nên để riêng một tập dữ liệu
kiểm tra độc lập để đánh giá hiệu năng tổng thể của hệ thống cuối cùng."*

**Cách đọc khung, và đây là đoạn quan trọng nhất của khối GIÁ.** Ba câu, ba mức nặng:

1. **Evidence nhạy với tiên nghiệm, kể cả phần đuôi.** Nhìn lại (3.70): evidence tỉ lệ với
   $1/\Delta w_{\text{prior}}$, nên nới bề rộng tiên nghiệm ra gấp đôi là chia evidence cho
   $2^M$. Với $M=10$ đó là **1 024 lần**. Không có chỗ nào trong dữ liệu để chống lại chuyện
   đó.
2. **Tiên nghiệm không chính quy thì evidence KHÔNG tồn tại.** Đây là chỗ kỹ thuật mà người
   dùng hay bỏ qua: một tiên nghiệm "không thông tin" kiểu $p(w) \propto 1$ trên toàn trục
   thực không chuẩn hoá được, nên (3.68) không có giá trị xác định. Hệ quả: **không thể tránh
   việc chọn một tiên nghiệm cụ thể** nếu muốn dùng evidence. Đó là cái giá không trả bằng
   tính toán mà bằng một quyết định người dùng phải đưa ra.
3. **Bishop tự khuyên giữ lại một tập kiểm tra.** Đây là câu phải trích nguyên văn, vì nó
   **giới hạn lại** chính lời hứa ở mục 3.4. Mục 3.4 của sách nói evidence cho phép so sánh mô hình
   *"without the need for a validation set"*; mục này nói vẫn nên giữ một **test set**. Hai
   câu không mâu thuẫn — một tập để *chọn*, một tập để *báo cáo hiệu năng* — nhưng ai chỉ đọc
   một trong hai sẽ hiểu sai.

Mục 2.30 đo đúng khoảng cách giữa hai câu đó: $\alpha$ mà evidence chọn lệch khỏi $\alpha$
tốt nhất trên tập kiểm tra một thừa số **2,9154 lần**, và cái giá của chỗ lệch ấy là
**1,098508 %** sai số căn quân phương.

### 2.22 Mục 3.5 của sách — vì sao KHÔNG tích ra được $\alpha$ và $\beta$

![Mở đầu mục 3.5 của sách của Bishop 2006](<paper_bishop2006_mo_dau_3_5.png>)

**Hình 30 — mở đầu mục 3.5.** Nguồn: Bishop 2006, trang 165,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"In a fully Bayesian treatment of the linear basis function model, we would introduce prior
> distributions over the hyperparameters α and β and make predictions by marginalizing with
> respect to these hyperparameters as well as with respect to the parameters w. However,
> although we can integrate analytically over either w or over the hyperparameters, the complete
> marginalization over all of these variables is analytically intractable."*

> *"This framework is known in the statistics literature as empirical Bayes (Bernardo and Smith,
> 1994; Gelman et al., 2004), or type 2 maximum likelihood (Berger, 1985), or generalized
> maximum likelihood (Wahba, 1975), and in the machine learning literature is also called the
> evidence approximation (Gull, 1989; MacKay, 1992a)."*

**Dịch:** *"Trong một cách xử lý Bayes đầy đủ cho mô hình hàm cơ sở tuyến tính, ta sẽ đưa vào
các tiên nghiệm trên siêu tham số α và β rồi dự báo bằng cách lấy biên theo các siêu tham số đó
lẫn theo tham số w. Tuy nhiên, dù ta tích phân được về mặt giải tích theo w HOẶC theo các siêu
tham số, việc lấy biên đầy đủ trên toàn bộ các biến này là KHÔNG giải tích được."* · *"Khuôn
khổ này được gọi trong tài liệu thống kê là empirical Bayes, hoặc cực đại hợp lẽ loại 2, hoặc
cực đại hợp lẽ tổng quát, và trong tài liệu học máy còn gọi là xấp xỉ evidence."*

**Cách đọc khung.** Đây là chỗ bài báo **thú nhận một xấp xỉ**, và điều đó phải nói thẳng trong
bất kỳ bản trình bày trung thực nào: mục 3.5 của sách **không** phải Bayes đầy đủ. Bayes đầy đủ sẽ tích
ra cả $\alpha$ và $\beta$; mục này thay chúng bằng **hai ước lượng điểm** $\widehat\alpha$,
$\widehat\beta$ và coi như xong. Bốn cái tên mà Bishop liệt kê cho cùng một việc — *empirical
Bayes*, *type 2 maximum likelihood*, *generalized maximum likelihood*, *evidence approximation*
— là bốn cộng đồng gọi khác nhau, và người đọc tài liệu khác sẽ gặp đủ bốn.

![Công thức 3.74 và 3.75 của Bishop 2006](<paper_bishop2006_ct_3_74_3_75.png>)

**Hình 31 — (3.74) và (3.75).** Nguồn: Bishop 2006, trang 165,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

$$p(t\mid\mathsf t) = \iiint p(t\mid\mathbf w,\beta)\,p(\mathbf w\mid\mathsf t,\alpha,\beta)\,p(\alpha,\beta\mid\mathsf t)\,\mathrm d\mathbf w\,\mathrm d\alpha\,\mathrm d\beta \tag{3.74}$$

$$p(t\mid\mathsf t) \simeq p(t\mid\mathsf t,\widehat\alpha,\widehat\beta) = \int p(t\mid\mathbf w,\widehat\beta)\,p(\mathbf w\mid\mathsf t,\widehat\alpha,\widehat\beta)\,\mathrm d\mathbf w \tag{3.75}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $p(\alpha,\beta\mid\mathsf t)$ | hậu nghiệm trên **siêu tham số** | — |
| $\widehat\alpha$, $\widehat\beta$ | hai **ước lượng điểm**, chỗ hậu nghiệm đó có đỉnh | vô hướng |

**Ý nghĩa trừu tượng.** Bước từ (3.74) sang (3.75) là bước xấp xỉ, và điều kiện của nó Bishop
viết rõ: *"If the posterior distribution $p(\alpha,\beta\mid\mathsf t)$ is sharply peaked around
values $\widehat\alpha$ and $\widehat\beta$"*. Tức là: thay một tích phân bằng giá trị tại đỉnh
**chỉ hợp lệ khi đỉnh nhọn**. Với $N$ nhỏ thì hậu nghiệm trên $(\alpha,\beta)$ không nhọn, và
(3.75) khi đó bỏ mất một phần độ bất định — đúng thứ mà cả bài này quảng cáo là điểm mạnh của
Bayes. Đó là một mâu thuẫn nội tại của phương pháp, không phải một lỗi, và nó thuộc khối GIÁ.

![Công thức 3.76 của Bishop 2006](<paper_bishop2006_ct_3_76.png>)

**Hình 32 — (3.76) và điều kiện "tiên nghiệm tương đối phẳng".** Nguồn: Bishop 2006,
trang 166, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"If the prior is relatively flat, then in the evidence framework the values of α and β are
> obtained by maximizing the marginal likelihood function p(t|α, β). We shall proceed by
> evaluating the marginal likelihood for the linear basis function model and then finding its
> maxima. This will allow us to determine values for these hyperparameters from the training
> data alone, without recourse to cross-validation. Recall that the ratio α/β is analogous to a
> regularization parameter."*

**Dịch:** *"Nếu tiên nghiệm tương đối phẳng thì trong khuôn khổ evidence, các giá trị α và β có
được bằng cách cực đại hoá hàm marginal likelihood p(t|α, β). Ta sẽ tiến hành bằng cách tính
marginal likelihood cho mô hình hàm cơ sở tuyến tính rồi tìm cực đại của nó. Việc này cho phép
ta xác định giá trị của các siêu tham số CHỈ TỪ DỮ LIỆU HUẤN LUYỆN, không phải nhờ tới kiểm
định chéo. Nhắc lại rằng tỉ số α/β tương tự một tham số chính quy hoá."*

$$p(\alpha,\beta\mid\mathsf t) \propto p(\mathsf t\mid\alpha,\beta)\,p(\alpha,\beta) \tag{3.76}$$

**Ý nghĩa trừu tượng.** Câu *"only from the training data alone, without recourse to
cross-validation"* là lời hứa trung tâm của cả mục 3.5 của sách, và mục 1 của bài này đã đo nó. Chú ý
điều kiện đi kèm: *"if the prior is relatively flat"* — nếu tiên nghiệm trên $(\alpha,\beta)$
không phẳng thì cực đại của (3.76) không trùng cực đại của marginal likelihood. Chỗ này chính
là chỗ `scikit-learn` làm khác: nó đặt tiên nghiệm Gamma **có tham số** lên $\alpha$ và $\beta$,
nên công thức cập nhật của nó khác (3.92)/(3.95). Mục 3 đo phần chênh.

Bishop còn nói một điều phủ định đáng ghi, về một đường mà người ta dễ nghĩ tới trước:

> *"However, the integrand as a function of w typically has a strongly skewed mode so that the
> Laplace approximation fails to capture the bulk of the probability mass, leading to poorer
> results than those obtained by maximizing the evidence (MacKay, 1999)."*

**Dịch:** *"Tuy nhiên, hàm dưới dấu tích phân, xem như hàm của w, thường có một đỉnh lệch mạnh,
nên xấp xỉ Laplace không bắt được phần lớn khối xác suất, dẫn tới kết quả tệ hơn so với cực đại
hoá evidence."*

### 2.23 (3.77)–(3.86) — dẫn ra công thức evidence

![Công thức 3.77 và 3.78 của Bishop 2006](<paper_bishop2006_ct_3_77_3_78.png>)

**Hình 33 — mục 3.5.1 của sách, (3.77) và (3.78).** Nguồn: Bishop 2006, trang 166,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

$$p(\mathsf t\mid\alpha,\beta) = \int p(\mathsf t\mid\mathbf w,\beta)\,p(\mathbf w\mid\alpha)\,\mathrm d\mathbf w \tag{3.77}$$

$$p(\mathsf t\mid\alpha,\beta) = \left(\frac{\beta}{2\pi}\right)^{N/2}\left(\frac{\alpha}{2\pi}\right)^{M/2}\int\exp\{-E(\mathbf w)\}\,\mathrm d\mathbf w \tag{3.78}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $(\beta/2\pi)^{N/2}$ | hằng chuẩn hoá của $N$ Gauss nhiễu, từ (3.10) | vô hướng |
| $(\alpha/2\pi)^{M/2}$ | hằng chuẩn hoá của tiên nghiệm $M$ chiều, từ (3.52) | vô hướng |

**Ý nghĩa trừu tượng.** (3.78) là chỗ **mọi hằng chuẩn hoá phải được viết ra**, và đó không
phải chi tiết kế toán: (3.77) là một xác suất của dữ liệu, nên nó có **thang**, khác hẳn hậu
nghiệm (3.49) mà ở đó hằng chuẩn hoá bị chia đi. Mục 2.24 cho thấy đúng một hằng trong số
chúng bị bỏ mất khi vẽ Hình 3.14, và cái giá là chỗ đỉnh của đồ thị.

![Công thức 3.79 đến 3.84 của Bishop 2006](<paper_bishop2006_ct_3_79_3_84.png>)

**Hình 34 — (3.79)–(3.84): gom thành bình phương đúng.** Nguồn: Bishop 2006, trang 167,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

$$E(\mathbf w) = \beta E_D(\mathbf w) + \alpha E_W(\mathbf w) = \frac\beta2\lVert\mathsf t - \mathbf\Phi\mathbf w\rVert^2 + \frac\alpha2\mathbf w^{\!\top}\mathbf w \tag{3.79}$$

$$E(\mathbf w) = E(\mathbf m_N) + \tfrac12(\mathbf w-\mathbf m_N)^{\!\top}\mathbf A(\mathbf w-\mathbf m_N) \tag{3.80}$$

$$\mathbf A = \alpha\mathbf I + \beta\mathbf\Phi^{\!\top}\mathbf\Phi \tag{3.81}$$

$$E(\mathbf m_N) = \frac\beta2\lVert\mathsf t-\mathbf\Phi\mathbf m_N\rVert^2 + \frac\alpha2\mathbf m_N^{\!\top}\mathbf m_N \tag{3.82}$$

$$\mathbf A = \nabla\nabla E(\mathbf w) \tag{3.83}$$

$$\mathbf m_N = \beta\mathbf A^{-1}\mathbf\Phi^{\!\top}\mathsf t \tag{3.84}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf A$ | ma trận **Hesse** của $E(\mathbf w)$ — và bằng $\mathbf S_N^{-1}$ | $M\times M$ |
| $\nabla\nabla$ | đạo hàm bậc hai theo $\mathbf w$ | — |
| $E(\mathbf m_N)$ | giá trị của hàm lỗi tại điểm cực tiểu | vô hướng |

**Ý nghĩa trừu tượng.** (3.80) là phép **gom thành bình phương đúng** (*completing the square*),
và nó nói một điều hình học: hàm lỗi $E(\mathbf w)$ là một **paraboloid** hoàn hảo, nên nó bằng
giá trị tại đáy cộng một dạng toàn phương. Chính vì thế tích phân (3.78) làm được bằng công
thức — không phải xấp xỉ, mà chính xác.

Nhận xét Bishop chèn vào đây đáng dừng lại:

> *"Using (3.54), we see that A = S−1 N , and hence (3.84) is equivalent to the previous
> definition (3.53), and therefore represents the mean of the posterior distribution."*

**Dịch:** *"Dùng (3.54), ta thấy A = S_N⁻¹, và do đó (3.84) tương đương với định nghĩa trước đó
(3.53), nên nó biểu diễn trung bình của hậu nghiệm."*

Tức là **ba** thứ trùng nhau: ma trận Hesse của hàm lỗi ridge, nghịch đảo hiệp phương sai hậu
nghiệm, và ma trận cần nghịch đảo để tìm nghiệm. Một ma trận, ba cách gọi, và mỗi cách gọi
thuộc một truyền thống khác nhau — tối ưu hoá, thống kê Bayes, đại số tuyến tính. Ai đọc ba
tài liệu khác nhau sẽ tưởng đó là ba đại lượng.

![Công thức 3.85 và 3.86 của Bishop 2006](<paper_bishop2006_ct_3_85_3_86.png>)

**Hình 35 — (3.85) và (3.86), công thức evidence.** Nguồn: Bishop 2006, trang 167,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"The integral over w can now be evaluated simply by appealing to the standard result for the
> normalization coefficient of a multivariate Gaussian, giving"*

**Dịch:** *"Tích phân theo w giờ tính được đơn giản bằng cách viện tới kết quả chuẩn cho hệ số
chuẩn hoá của một Gauss nhiều biến, cho ra"*

$$\int\exp\{-E(\mathbf w)\}\,\mathrm d\mathbf w = \exp\{-E(\mathbf m_N)\}\,(2\pi)^{M/2}\lvert\mathbf A\rvert^{-1/2} \tag{3.85}$$

$$\ln p(\mathsf t\mid\alpha,\beta) = \frac M2\ln\alpha + \frac N2\ln\beta - E(\mathbf m_N) - \frac12\ln\lvert\mathbf A\rvert - \frac N2\ln(2\pi) \tag{3.86}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $(2\pi)^{M/2}\lvert\mathbf A\rvert^{-1/2}$ | hằng chuẩn hoá của một Gauss $M$ chiều có độ chính xác $\mathbf A$ | vô hướng |
| $\lvert\mathbf A\rvert$ | định thức của Hesse — càng lớn nghĩa là hậu nghiệm càng hẹp | vô hướng |

**Ý nghĩa trừu tượng.** (3.86) là **hàm mục tiêu** của cả phần còn lại của chương, và nó nên
đọc theo năm số hạng:

* $\tfrac M2\ln\alpha$ — thang của tiên nghiệm. Tiên nghiệm hẹp hơn (α lớn) làm số hạng này
  tăng, nhưng nó phải trả bằng $E(\mathbf m_N)$ lớn hơn.
* $\tfrac N2\ln\beta$ — thang của hàm hợp lẽ.
* $-E(\mathbf m_N)$ — phần **khớp dữ liệu**, cộng phần phạt hệ số. Đây là hàm lỗi ridge, đúng
  (3.27) nhân $\beta$.
* $-\tfrac12\ln\lvert\mathbf A\rvert$ — **phần phạt độ phức tạp**, là bản chính xác của
  $\ln(\Delta w_{\text{posterior}}/\Delta w_{\text{prior}})$ ở (3.71). Nó lớn khi hậu nghiệm
  hẹp, tức khi tham số bị điều chỉnh vừa khít dữ liệu.
* $-\tfrac N2\ln(2\pi)$ — hằng số, không phụ thuộc $\alpha$, $\beta$, hay mô hình. Nó không ảnh
  hưởng chỗ cực đại nhưng ảnh hưởng **giá trị** in ra, nên phải có khi so hai mô hình khác $N$.

**Phép kiểm độc lập cho (3.86)**, và nó rẻ: so công thức đóng với **tích phân số** trên mô hình
hai tham số của mục 0.4, cầu phương trên lưới $2001\times2001$:

| $\alpha$ | $\beta$ | (3.86) | Tích phân số | Lệch |
|---:|---:|---:|---:|---:|
| 1 | 2 | $-4{,}4275393987$ | $-4{,}4275393987$ | $7{,}550\times10^{-14}$ |
| 0,3 | 5 | $-4{,}7432769637$ | $-4{,}7432769637$ | $7{,}638\times10^{-14}$ |
| 3 | 1 | $-4{,}9643431084$ | $-4{,}9643431084$ | $7{,}550\times10^{-14}$ |

Công thức đóng đúng. Điều đó quan trọng cho mục sau, vì nó loại một cách giải thích: chỗ lệch
tìm được ở Hình 3.14 **không** phải do (3.86) sai.

### 2.24 Hình 3.14 — hình được vẽ KHÔNG bằng công thức (3.86)

![Hình 3.14 của Bishop 2006](<paper_bishop2006_hinh_3_14.png>)

**Hình 36 — Hình 3.14 nguyên bản: evidence theo bậc đa thức.** Nguồn: Bishop 2006,
trang 168, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Plot of the model evidence versus the order M, for the polynomial regression model, showing
> that the evidence favours the model with M = 3."*

**Dịch:** *"Đồ thị model evidence theo bậc M, cho mô hình hồi quy đa thức, cho thấy evidence
nghiêng về mô hình với M = 3."*

![Bình luận của Bishop về Hình 3.14](<paper_bishop2006_binh_hinh_3_14.png>)

**Hình 37 — đoạn thân bài Bishop dùng để bình luận Hình 3.14.** Nguồn: Bishop 2006,
trang 168, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nhận xét của chính tác giả, nguyên văn:**

> *"Here we have assumed a prior of the form (1.65) with the parameter α fixed at"*

> *"However, in going to M = 2, the data fit is improved only very marginally, due to the fact
> that the underlying sinusoidal function from which the data is generated is an odd function
> and so has no even terms in a polynomial expansion."*

> *"Because this richer model suffers a greater complexity penalty, the evidence actually falls
> in going from M = 1 to M = 2."*

> *"The evidence values, however, show a clear preference for M = 3, since this is the simplest
> model which gives a good explanation for the"*

**Dịch:** *"Ở đây ta đã giả thiết một tiên nghiệm dạng (1.65) với tham số α cố định ở"*
$\alpha = 5\times10^{-3}$ · *"Tuy nhiên, khi đi lên M = 2, mức khớp dữ liệu chỉ cải thiện rất
nhẹ, do việc hàm hình sin nền sinh ra dữ liệu là một hàm LẺ nên không có số hạng bậc chẵn trong
khai triển đa thức."* · *"Vì mô hình giàu hơn này phải chịu một khoản phạt độ phức tạp lớn hơn,
evidence THẬT SỰ GIẢM khi đi từ M = 1 sang M = 2."* · *"Tuy nhiên, các giá trị evidence cho thấy
một sự ưu ái rõ rệt cho M = 3, vì đó là mô hình đơn giản nhất cho một lời giải thích tốt cho"*
dữ liệu quan sát được.

**Cách đọc hình.** Trục ngang là **bậc** đa thức, có nhãn ở 0, 2, 4, 6, 8. Trục dọc là
$\ln p(\mathsf t\mid\alpha,\beta)$, vạch ở $-26$, $-24$, $-22$, $-20$, $-18$. Một đường gấp
khúc nối các giá trị.

**Điều hình KHÔNG nói ra, thứ nhất: nó có MƯỜI điểm, không phải chín.** Đường vẽ là một
polyline toàn đoạn thẳng, nên đọc lớp vectơ cho ra đúng các điểm gãy. Hoành độ đọc được làm
tròn ra $0, 1, 2, \ldots, 9$ — tức bậc chạy tới **9**, trong khi nhãn trục chỉ in tới 8, nên
người đọc dễ tưởng chỉ có chín điểm. Sai lệch giữa hoành độ đọc được và số nguyên lớn nhất là
**0,0025**, con số xác nhận phép quy đổi trục.

| Bậc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\ln p$ **đọc từ hình** | $-24{,}0367$ | $-19{,}7295$ | $-23{,}0751$ | $\mathbf{-17{,}6126}$ | $-18{,}3072$ | $-19{,}8698$ | $-21{,}4123$ | $-22{,}8281$ | $-24{,}1703$ | $-25{,}4791$ |

Đỉnh rơi ở bậc 3 — đúng điều chú thích nói. Và hai chi tiết của bình luận cũng đọc được: bậc 1
cao hơn bậc 0 rất nhiều ($+4{,}307193$), còn bậc 2 **thấp hơn** bậc 1 ($-3{,}345587$) — đúng câu *"the
evidence actually falls in going from M = 1 to M = 2"*, và đúng lý do hàm lẻ mà Bishop đưa ra.

**Điều hình KHÔNG nói ra, thứ hai — và đây là phát hiện đắt nhất của bài này.** Dựng lại mười
giá trị đó bằng **đúng công thức (3.86)**, với 10 điểm dữ liệu của mục 1.1, $\alpha =
5\times10^{-3}$ mà bài báo khai ở tr.167, và $\beta = 11{,}1$ mà bài báo khai ở tr.171:

| Bậc | Hình 3.14 đọc được | (3.86) dựng lại | Lệch | Lệch $/(M{+}1)$ |
|---:|---:|---:|---:|---:|
| 0 | $-24{,}036722$ | $-23{,}123666$ | $+0{,}913057$ | 0,913057 |
| 1 | $-19{,}729529$ | $-17{,}879100$ | $+1{,}850430$ | 0,925215 |
| 2 | $-23{,}075116$ | $-20{,}306141$ | $+2{,}768975$ | 0,922992 |
| 3 | $-17{,}612635$ | $-13{,}934076$ | $+3{,}678558$ | 0,919640 |
| 4 | $-18{,}307154$ | $-13{,}713212$ | $+4{,}593942$ | 0,918788 |
| 5 | $-19{,}869763$ | $-14{,}359022$ | $+5{,}510741$ | 0,918457 |
| 6 | $-21{,}412340$ | $-14{,}981523$ | $+6{,}430817$ | 0,918688 |
| 7 | $-22{,}828063$ | $-15{,}481400$ | $+7{,}346664$ | 0,918333 |
| 8 | $-24{,}170305$ | $-15{,}906108$ | $+8{,}264196$ | 0,918244 |
| 9 | $-25{,}479131$ | $-16{,}290483$ | $+9{,}188648$ | 0,918865 |

Cột lệch không ngẫu nhiên: nó **tuyến tính theo số tham số**, và cột cuối cho thấy hệ số ấy gần
như một hằng. Khớp một đường thẳng qua mười điểm:

$$\text{lệch} \approx +0{,}006646 + 0{,}917810\,(M+1), \qquad \text{phần dư lớn nhất } 0{,}011400$$

Hệ số góc đo được là **0,917810**. Còn $\tfrac12\ln(2\pi) = 0{,}918939$ — chênh **0,001128**.
Hệ số chặn đo được là $0{,}006646$, cùng bậc với sai số đọc vectơ, tức coi như 0.

Nói cách khác: đường vẽ trong Hình 3.14 bằng đúng

$$(3.86) - \frac M2\ln(2\pi)$$

tức **thiếu đúng một thừa số $(2\pi)^{M/2}$** — chính hằng chuẩn hoá của tích phân Gauss ở
(3.85). Đối chiếu từng ô với cách đọc đó:

| Bậc | Hình 3.14 đọc được | Bản thiếu $(2\pi)^{M/2}$ | Lệch |
|---:|---:|---:|---:|
| 0 | $-24{,}036722$ | $-24{,}042604$ | $-0{,}005882$ |
| 1 | $-19{,}729529$ | $-19{,}716977$ | $+0{,}012553$ |
| 2 | $-23{,}075116$ | $-23{,}062957$ | $+0{,}012160$ |
| 3 | $-17{,}612635$ | $-17{,}609830$ | $+0{,}002804$ |
| 4 | $-18{,}307154$ | $-18{,}307904$ | $-0{,}000751$ |
| 5 | $-19{,}869763$ | $-19{,}872653$ | $-0{,}002890$ |
| 6 | $-21{,}412340$ | $-21{,}414093$ | $-0{,}001753$ |
| 7 | $-22{,}828063$ | $-22{,}832908$ | $-0{,}004845$ |
| 8 | $-24{,}170305$ | $-24{,}176555$ | $-0{,}006250$ |
| 9 | $-25{,}479131$ | $-25{,}479868$ | $-0{,}000737$ |

Lệch lớn nhất **0,012553**, lệch trung bình **0,005062**, trên mức đọc vectơ 0,003277 của bài
này. **Mười trên mười ô khớp trong 0,03** — tức trong mức đọc được của một đồ thị in.

**Phân xử hai cách đọc, và nó phải làm bằng số, không bằng lời.** Có thể lập luận rằng chỗ lệch
là do một $\alpha$ hoặc $\beta$ khác. Thử: quét **cả mặt phẳng** $(\alpha,\beta)$ trên lưới
$120\times120$ theo log, cho **từng** cách đọc, rồi lấy lệch lớn nhất nhỏ nhất đạt được:

| Cách đọc | Lệch lớn nhất tốt nhất | $\alpha$ dò ra | $\beta$ dò ra | Đỉnh |
|---|---:|---:|---:|---:|
| **đúng (3.86)** | **1,464193** | $2{,}298\times10^{-4}$ | 10,1273 | $M=3$ |
| **bản thiếu $(2\pi)^{M/2}$** | **0,114826** | $5{,}281\times10^{-3}$ | 11,0704 | $M=3$ |

Bản thiếu $(2\pi)^{M/2}$ khớp hơn **12,75 lần**. Và bằng chứng thứ hai, độc lập với con số đó:
hai tham số mà nó dò ra là $\alpha = 5{,}281\times10^{-3}$ và $\beta = 11{,}0704$ — **khớp hai
con số bài báo tự in**, $5\times10^{-3}$ ở tr.167 và $11{,}1$ ở tr.171. Trong khi đó bản đúng
(3.86) đòi $\alpha = 2{,}298\times10^{-4}$, tức lệch con số của chính bài báo **21,8 lần**.
Hai bằng chứng cùng chỉ một hướng.

**Hệ quả, và nó không nhỏ.** Thừa số thiếu là một khoản phạt $\tfrac12\ln(2\pi) = 0{,}918939$
**mỗi tham số**, nên nó đổi chỗ đỉnh:

| $\beta$ | Đỉnh theo đúng (3.86) | Đỉnh theo bản thiếu $2\pi$ | Biên thắng (3.86) / bản thiếu |
|---:|---:|---:|---|
| 8,0000 | $M=4$ | $M=1$ | 0,1669 / 0,2209 |
| 9,0000 | $M=4$ | $M=3$ | 0,1896 / 0,5121 |
| 10,0000 | $M=4$ | $M=3$ | 0,2069 / 0,7121 |
| **11,1000** | $M=4$ | $M=3$ | 0,2209 / 0,6981 |
| 11,1111 | $M=4$ | $M=3$ | 0,2210 / 0,6980 |
| 13,0000 | $M=4$ | $M=3$ | 0,2358 / 0,6832 |
| 15,0000 | $M=4$ | $M=3$ | 0,2423 / 0,6766 |
| 20,0000 | $M=4$ | $M=3$ | 0,2346 / 0,6844 |

Theo **đúng** (3.86) thì evidence chọn **$M = 4$**, không phải $M=3$ như chú thích Hình 3.14
nói — và điều đó đúng ở cả tám giá trị $\beta$ thử, tức nó không phải hệ quả của một $\beta$
chọn khéo. Biên thắng của $M=4$ là 0,22, nhỏ, nhưng nó đổi dấu.

**Câu phải viết ra, và câu KHÔNG được viết ra.** Không kết luận nào khác của chương đổi vì chỗ
này: mọi lập luận về sau của mục 3.5 của sách dùng (3.86) chứ không dùng Hình 3.14, và câu *"evidence
thích mô hình đơn giản nhất giải thích được dữ liệu"* vẫn đúng ở cả hai cách đọc. Cũng không
có cách nào biết thừa số thiếu nằm ở bản in của công thức hay ở đoạn mã vẽ hình. Thứ **đo
được** là: hình được vẽ và công thức được in **không phải cùng một hàm**, và hai hàm đó không
đồng ý về bậc nào thắng.

### 2.25 (3.87)–(3.92) — cực đại evidence theo $\alpha$, và đại lượng $\gamma$

![Công thức 3.87 của Bishop 2006](<paper_bishop2006_ct_3_87.png>)

**Hình 38 — mục 3.5.2 của sách và phương trình trị riêng (3.87).** Nguồn: Bishop 2006, trang 168,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

$$\left(\beta\mathbf\Phi^{\!\top}\mathbf\Phi\right)\mathbf u_i = \lambda_i\mathbf u_i \tag{3.87}$$

$$\frac{\mathrm d}{\mathrm d\alpha}\ln\lvert\mathbf A\rvert = \frac{\mathrm d}{\mathrm d\alpha}\ln\prod_i(\lambda_i+\alpha) = \sum_i\frac{1}{\lambda_i+\alpha} \tag{3.88}$$

$$0 = \frac{M}{2\alpha} - \frac12\mathbf m_N^{\!\top}\mathbf m_N - \frac12\sum_i\frac{1}{\lambda_i+\alpha} \tag{3.89}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf u_i$ | vectơ riêng thứ $i$ của $\beta\mathbf\Phi^{\!\top}\mathbf\Phi$ | $M$ |
| $\lambda_i$ | **trị riêng** thứ $i$ — đo **độ cong** của hàm hợp lẽ theo hướng $\mathbf u_i$ | vô hướng |

**Ý nghĩa trừu tượng.** (3.87) là chỗ bài toán được **xoay trục**. Trong hệ trục của các
$\mathbf u_i$, mọi thứ tách thành $M$ bài toán một chiều độc lập, và $\mathbf A$ có trị riêng
$\alpha + \lambda_i$ — tức "độ chính xác tiên nghiệm cộng độ chính xác dữ liệu", đúng
(3.51) đọc theo từng hướng. Từ đó (3.88) tính được đạo hàm của $\ln\lvert\mathbf A\rvert$ bằng
một tổng đơn giản, và đó là bước duy nhất còn khó của phép dẫn.

Đây cũng là chỗ chữ $\lambda$ đổi nghĩa so với (3.27): ở (3.27) nó là **mức phạt** do người
dùng chọn; ở (3.87) nó là **trị riêng** do dữ liệu quyết định. Hai chữ cùng hình, cách nhau
24 trang.

![Công thức 3.90 đến 3.92 của Bishop 2006](<paper_bishop2006_ct_3_90_3_92.png>)

**Hình 39 — (3.90)–(3.92) và phép lặp cho $\alpha$.** Nguồn: Bishop 2006, trang 169,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Note that this is an implicit solution for α not only because γ depends on α, but also
> because the mode mN of the posterior distribution itself depends on the choice of α. We
> therefore adopt an iterative procedure in which we make an initial choice for α and use this
> to find mN , which is given by (3.53), and also to evaluate γ, which is given by (3.91).
> These values are then used to re-estimate α using (3.92), and the process repeated until
> convergence."*

> *"It should be emphasized that the value of α has been determined purely by looking at the
> training data. In contrast to maximum likelihood methods, no independent data set is required
> in order to optimize the model complexity."*

**Dịch:** *"Để ý rằng đây là một nghiệm ẨN cho α, không chỉ vì γ phụ thuộc α mà còn vì bản thân
đỉnh mₙ của hậu nghiệm cũng phụ thuộc cách chọn α. Vì thế ta dùng một thủ tục lặp: chọn một giá
trị α ban đầu, dùng nó để tìm mₙ theo (3.53) và để tính γ theo (3.91). Các giá trị đó rồi được
dùng để ước lượng lại α bằng (3.92), và quá trình lặp lại tới khi hội tụ."* · *"Cần nhấn mạnh
rằng giá trị của α đã được xác định hoàn toàn bằng cách nhìn vào dữ liệu huấn luyện. Khác với
các phương pháp cực đại hợp lẽ, không cần một tập dữ liệu độc lập nào để tối ưu độ phức tạp mô
hình."*

$$\alpha\,\mathbf m_N^{\!\top}\mathbf m_N = M - \alpha\sum_i\frac{1}{\lambda_i+\alpha} = \gamma \tag{3.90}$$

$$\gamma = \sum_i\frac{\lambda_i}{\alpha+\lambda_i} \tag{3.91}$$

$$\alpha = \frac{\gamma}{\mathbf m_N^{\!\top}\mathbf m_N} \tag{3.92}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\gamma$ | **số tham số hiệu dụng** — xem mục 2.28 | vô hướng |
| $\lambda_i/(\alpha+\lambda_i)$ | phần của hướng $i$ mà **dữ liệu** quyết định, nằm trong $(0,1)$ | vô hướng |

**Ý nghĩa trừu tượng.** (3.91) là công thức đáng nhớ nhất của mục 3.5 của sách, vì nó là một **phép đếm
có phần lẻ**. Mỗi hướng riêng góp một lượng nằm giữa 0 và 1: gần 1 nếu $\lambda_i \gg \alpha$
(dữ liệu áp đảo tiên nghiệm theo hướng đó), gần 0 nếu $\lambda_i \ll \alpha$ (tiên nghiệm áp
đảo). Cộng lại được một số **thực** trong $[0, M]$ — không phải một số nguyên. Đó là chỗ khái
niệm "số tham số" thôi là một phép đếm và thành một phép đo.

(3.92) thì là một **phương trình bất động**: $\alpha$ ở vế phải qua $\gamma$ và qua
$\mathbf m_N$. Bishop nói rõ nó là nghiệm ẩn và phải lặp.

**Đo (3.91) và miền của nó.** Trên mô hình 9 hàm cơ sở Gauss ($M = 10$, $N = 25$,
$\beta = 11{,}1$):

| $\alpha$ | $\gamma$ | $\gamma/M$ | $\alpha\,\mathbf m_N^{\!\top}\mathbf m_N$ theo (3.90) | Lệch của (3.90) |
|---:|---:|---:|---:|---:|
| $10^{-8}$ | 9,999998 | 1,000000 | 0,000000 | $1{,}000\times10^{1}$ |
| $10^{-4}$ | 9,983036 | 0,998304 | 0,002389 | $9{,}981\times10^{0}$ |
| $10^{-2}$ | 9,268543 | 0,926854 | 0,117511 | $9{,}151\times10^{0}$ |
| $0{,}1$ | 8,303040 | 0,830304 | 0,542332 | $7{,}761\times10^{0}$ |
| $1$ | 6,807265 | 0,680727 | 2,142945 | $4{,}664\times10^{0}$ |
| $10$ | 4,851331 | 0,485133 | 13,218698 | $8{,}367\times10^{0}$ |
| $10^{3}$ | 0,580486 | 0,058049 | 17,431211 | $1{,}685\times10^{1}$ |
| $10^{6}$ | 0,000748 | 0,000075 | 0,022395 | $2{,}165\times10^{-2}$ |
| $10^{10}$ | 0,000000 | 0,000000 | 0,000002 | $2{,}165\times10^{-6}$ |

$\gamma$ nằm trong $[0, M]$ ở **mọi** giá trị $\alpha$ thử, và **giảm đơn điệu** khi $\alpha$
tăng — hai tính chất mà (3.91) đòi.

Cột cuối là cột phải đọc kỹ: nó lớn ở khắp nơi. Đó **không** phải lỗi, mà là ý nghĩa của
(3.90): đẳng thức $\alpha\mathbf m_N^{\!\top}\mathbf m_N = \gamma$ **chỉ đúng tại điểm dừng**
của evidence, không đúng ở mọi $\alpha$. Chính khoảng cách giữa hai vế là thứ (3.92) dùng làm
bước cập nhật, và nó về 0 khi phép lặp hội tụ.

### 2.26 (3.93)–(3.95) — cực đại evidence theo $\beta$, và phép lặp hoàn chỉnh

![Công thức 3.93 đến 3.95 của Bishop 2006](<paper_bishop2006_ct_3_93_3_95.png>)

**Hình 40 — (3.93)–(3.95) và phép lặp cho $\beta$.** Nguồn: Bishop 2006, trang 169,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Again, this is an implicit solution for β and can be solved by choosing an initial value for
> β and then using this to calculate mN and γ and then re-estimate β using (3.95), repeating
> until convergence. If both α and β are to be determined from the data, then their values can
> be re-estimated together after each update of γ."*

**Dịch:** *"Lại nữa, đây là một nghiệm ẨN cho β, và giải được bằng cách chọn một giá trị β ban
đầu rồi dùng nó để tính mₙ và γ, sau đó ước lượng lại β bằng (3.95), lặp tới khi hội tụ. Nếu cả
α và β đều phải xác định từ dữ liệu thì giá trị của chúng có thể được ước lượng lại CÙNG NHAU
sau mỗi lần cập nhật γ."*

$$\frac{\mathrm d}{\mathrm d\beta}\ln\lvert\mathbf A\rvert = \frac{1}{\beta}\sum_i\frac{\lambda_i}{\lambda_i+\alpha} = \frac{\gamma}{\beta} \tag{3.93}$$

$$0 = \frac{N}{2\beta} - \frac12\sum_{n=1}^{N}\left\{t_n-\mathbf m_N^{\!\top}\boldsymbol\phi(x_n)\right\}^2 - \frac{\gamma}{2\beta} \tag{3.94}$$

$$\frac1\beta = \frac{1}{N-\gamma}\sum_{n=1}^{N}\left\{t_n-\mathbf m_N^{\!\top}\boldsymbol\phi(x_n)\right\}^2 \tag{3.95}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $N-\gamma$ | số **bậc tự do còn lại** sau khi $\gamma$ tham số đã bị dữ liệu dùng hết | vô hướng |

**Ý nghĩa trừu tượng.** (3.95) là một công thức quen ở dạng lạ. Ước lượng phương sai nhiễu bằng
tổng bình phương phần dư chia cho một số — và số đó **không** phải $N$, cũng không phải $N-M$,
mà là $N - \gamma$. Mục 2.28 nói vì sao đó là con số đúng.

Điều (3.93) nói thêm, và đó là lý do $\gamma$ xuất hiện cả ở đây: đạo hàm của
$\ln\lvert\mathbf A\rvert$ theo $\beta$ bằng đúng $\gamma/\beta$. Cùng một đại lượng $\gamma$
điều khiển cả hai phương trình cập nhật, nên nó phải được tính lại ở **mỗi** vòng — đúng câu
*"re-estimated together after each update of γ"*.

**Đo phép lặp.** Xuất phát từ $\alpha = \beta = 1$, cập nhật xen kẽ (3.92) và (3.95):

| Vòng | $\alpha$ | $\beta$ | $\gamma$ | $\ln$ evidence |
|---:|---:|---:|---:|---:|
| 1 | 1,00000000 | 1,00000000 | 4,748868 | $-30{,}298258$ |
| 2 | 3,68552988 | 12,20136724 | 5,848988 | $-12{,}604738$ |
| 3 | 3,54059855 | 13,23772352 | 5,951851 | $-12{,}570234$ |
| 5 | 3,51975263 | 13,26825812 | 5,958737 | $-12{,}570161$ |
| 10 | 3,51966163 | 13,26836557 | 5,958765 | $-12{,}570161$ |
| 12 | 3,51966163 | 13,26836557 | 5,958765 | $-12{,}570161$ |

Hội tụ sau **12 vòng** với dung sai tương đối $10^{-12}$. Kết quả:
$\widehat\alpha = 3{,}51966163$, $\widehat\beta = 13{,}26836557$,
$\gamma = 5{,}958765$ trên $M = 10$. Và $1/\sqrt{\widehat\beta} = 0{,}274531$, so với độ lệch
chuẩn nhiễu thật của mục 1.1 là $0{,}3$ — ước lượng hơi **nhỏ hơn** giá trị thật, đúng chiều
người ta chờ đợi khi mô hình hút một phần nhiễu vào hệ số.

$\ln$ evidence **tăng đơn điệu** qua mọi vòng, từ $-30{,}298258$ lên $-12{,}570161$. Bishop
không khẳng định điều đó, nên đây là một quan sát của bài viết, không phải một trích dẫn.

**Evidence có nhiều cực trị không?** Bishop không nói. Thử năm điểm xuất phát cách nhau tới sáu
bậc độ lớn:

| $(\alpha_0, \beta_0)$ | $\widehat\alpha$ | $\widehat\beta$ | $\gamma$ | Số vòng |
|---|---:|---:|---:|---:|
| $(10^{-6},\ 10^{-3})$ | 3,51966163 | 13,26836557 | 5,958765 | 13 |
| $(1,\ 1)$ | 3,51966163 | 13,26836557 | 5,958765 | 12 |
| $(10^{3},\ 10^{3})$ | 3,51966163 | 13,26836557 | 5,958765 | 12 |
| $(10^{-3},\ 10^{3})$ | 3,51966163 | 13,26836557 | 5,958765 | 13 |
| $(10^{3},\ 10^{-3})$ | 3,51966163 | 13,26836557 | 5,958765 | **21** |

Năm điểm xuất phát, **một** nghiệm, giống nhau tới chữ số thứ tám. Số vòng thì đổi — từ 12 lên
21 — nên điểm xuất phát ảnh hưởng **chi phí** chứ không ảnh hưởng **kết quả**, trên bộ dữ liệu
này. Đó là một phép không bác bỏ, không phải một phép chứng minh: nó nói *"không tìm thấy cực
trị thứ hai"*, không nói *"evidence chỉ có một cực trị"*.

Giữ $\beta = 11{,}1$ cố định và chỉ tối ưu $\alpha$ — đúng thiết lập của Hình 3.16 — thì
$\widehat\alpha = 3{,}54894556$, tức $\ln\widehat\alpha = 1{,}266651$, với
$\gamma = 5{,}800382$ sau **10 vòng**. Con số $1{,}266651$ là con số mục 2.30 đem so với hình.

### 2.27 Hình 3.15 — vì sao $\gamma$ đếm được "số hướng dữ liệu quyết định"

![Hình 3.15 của Bishop 2006](<paper_bishop2006_hinh_3_15.png>)

**Hình 41 — Hình 3.15 nguyên bản: đường đồng mức hợp lẽ và tiên nghiệm trong hệ trục riêng.**
Nguồn: Bishop 2006, trang 170,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Contours of the likelihood function (red) and the prior (green) in which the axes in
> parameter space have been rotated to align with the eigenvectors ui of the Hessian. For
> α = 0, the mode of the posterior is given by the maximum likelihood solution wML, whereas
> for nonzero α the mode is at wMAP = mN . In the direction w1 the eigenvalue λ1, defined by
> (3.87), is small compared with α and so the quantity λ1/(λ1 + α) is close to zero, and the
> corresponding MAP value of w1 is also close to zero. By contrast, in the direction w2 the
> eigenvalue λ2 is large compared with α and so the quantity λ2/(λ2+α) is close to unity, and
> the MAP value of w2 is close to its maximum likelihood value."*

**Dịch:** *"Đường đồng mức của hàm hợp lẽ (đỏ) và của tiên nghiệm (xanh lá), trong đó các trục
của không gian tham số đã được XOAY cho trùng với các vectơ riêng uᵢ của ma trận Hesse. Với
α = 0, đỉnh của hậu nghiệm là nghiệm hợp lẽ cực đại wML, còn với α khác 0 thì đỉnh ở
wMAP = mₙ. Theo hướng w₁, trị riêng λ₁ định nghĩa ở (3.87) là NHỎ so với α, nên lượng
λ₁/(λ₁ + α) gần 0, và giá trị MAP tương ứng của w₁ cũng gần 0. Trái lại, theo hướng w₂ thì trị
riêng λ₂ LỚN so với α, nên lượng λ₂/(λ₂ + α) gần 1, và giá trị MAP của w₂ gần giá trị hợp lẽ
cực đại của nó."*

**Cách đọc hình.** Hai bộ đường đồng mức. Bộ **xanh lá** là tiên nghiệm — các đường tròn đồng
tâm quanh gốc, vì (3.52) đẳng hướng. Bộ **đỏ** là hàm hợp lẽ — các ellipse **thẳng trục** vì
trục đã được xoay theo $\mathbf u_i$. Ellipse đỏ **dài** theo $w_1$ (độ cong nhỏ, tức
$\lambda_1$ nhỏ) và **ngắn** theo $w_2$ ($\lambda_2$ lớn). Hai điểm được đánh dấu:
$\mathbf w_{\text{ML}}$ ở tâm ellipse đỏ, $\mathbf w_{\text{MAP}}$ bị kéo về phía gốc — và bị
kéo **nhiều hơn** theo hướng $w_1$.

**Điều hình KHÔNG nói ra.** Hình là **sơ đồ**, không có một con số nào và không có thang. Nó
vẽ $M=2$; ý nghĩa thật của $\gamma$ chỉ hiện ra khi $M$ lớn và phổ trị riêng trải rộng. Và hình
không nói phần lượng hoá: rằng phép co theo mỗi hướng bằng **đúng** hệ số
$\lambda_i/(\lambda_i+\alpha)$, tức đúng số hạng trong (3.91). Đây là chỗ hình và công thức bổ
sung nhau: hình cho trực giác hình học, (3.91) cho con số.

Bảng ở mục 2.25 là bản định lượng của chính hình này: ở $\alpha = 10^{-8}$ thì mọi hướng đều
"dữ liệu quyết định" và $\gamma/M = 1{,}000000$; ở $\alpha = 10^{10}$ thì không hướng nào, và
$\gamma/M = 0{,}000000$.

### 2.28 Mục 3.5.3 của sách và (3.96)–(3.97) — vì sao chia cho $N-\gamma$

![Mục 3.5.3 của sách của Bishop 2006](<paper_bishop2006_gamma_so_tham_so.png>)

**Hình 42 — mục 3.5.3 của sách, số tham số hiệu dụng.** Nguồn: Bishop 2006, trang 170,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"Because βΦTΦ is a positive definite matrix, it will have positive eigenvalues, and so the
> ratio λi/(λi + α) will lie between 0 and 1."*

> *"For directions in which λi"* $\gg \alpha$ *"the corresponding parameter wi will be close to
> its maximum likelihood value, and the ratio λi/(λi + α) will be close to 1. Such parameters
> are called well determined because their values are tightly constrained by the data."*

> *"The quantity γ defined by (3.91) therefore measures the effective total"* number of well
> determined parameters.

**Dịch:** *"Vì βΦᵀΦ là ma trận xác định dương, nó có các trị riêng dương, nên tỉ số λᵢ/(λᵢ + α)
sẽ nằm giữa 0 và 1."* · *"Với những hướng mà λᵢ lớn hơn α nhiều, tham số wᵢ tương ứng sẽ gần
giá trị hợp lẽ cực đại của nó, và tỉ số λᵢ/(λᵢ + α) sẽ gần 1. Những tham số như thế được gọi là
XÁC ĐỊNH TỐT vì giá trị của chúng bị dữ liệu ràng buộc chặt."* · *"Lượng γ định nghĩa ở (3.91)
vì thế ĐO tổng số hiệu dụng các tham số được xác định tốt."*

![Công thức 3.96 và 3.97 của Bishop 2006](<paper_bishop2006_ct_3_96_3_97.png>)

**Hình 43 — (3.96), (3.97) và phép loại suy với ước lượng phương sai.** Nguồn: Bishop 2006,
trang 171, `2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"and that this estimate is biased because the maximum likelihood solution µML for the mean
> has fitted some of the noise on the data. In effect, this has used up one degree of freedom
> in the model."*

> *"The effective number of parameters that are determined by the data is γ, with the remaining
> M − γ parameters set to small values by the prior. This is reflected in the Bayesian result
> for the variance that has a factor N − γ in the denominator, thereby correcting for"* the
> bias of the maximum likelihood result.

**Dịch:** *"và ước lượng này bị CHỆCH, vì nghiệm hợp lẽ cực đại µML cho trung bình đã khớp một
phần nhiễu trên dữ liệu. Trên thực tế, việc đó đã dùng hết một bậc tự do của mô hình."* ·
*"Số tham số hiệu dụng được dữ liệu xác định là γ, còn M − γ tham số còn lại bị tiên nghiệm
đặt về những giá trị nhỏ. Điều đó phản ánh trong kết quả Bayes cho phương sai, nơi mẫu số có
thừa số N − γ, và nhờ thế hiệu chỉnh được độ chệch của kết quả hợp lẽ cực đại."*

$$\sigma_{\text{ML}}^2 = \frac1N\sum_{n=1}^{N}(x_n-\mu_{\text{ML}})^2 \tag{3.96}$$

$$\sigma_{\text{MAP}}^2 = \frac{1}{N-1}\sum_{n=1}^{N}(x_n-\mu_{\text{ML}})^2 \tag{3.97}$$

**Ý nghĩa trừu tượng.** Cặp (3.96)/(3.97) là phép loại suy dạy được nhất của cả chương, và nó
nối một chuyện ai cũng biết với một chuyện ít ai biết. Ai học thống kê cơ bản đều biết phương
sai mẫu chia cho $N-1$ chứ không phải $N$, và lý do là *"một bậc tự do đã dùng để ước lượng
trung bình"*. (3.95) là **cùng một hiệu chỉnh** ở dạng tổng quát: trong hồi quy, không phải
một bậc tự do bị dùng mà là $\gamma$ bậc — và $\gamma$ không nguyên.

Ba con số của cùng một mô hình nói rõ khoảng cách giữa ba cách đếm, trên bộ 25 điểm:
$N = 25$, $M = 10$, và $\gamma = 5{,}958765$. Chia cho $N$ thì ước lượng chệch xuống; chia cho
$N - M = 15$ thì chệch lên, vì mô hình **không** dùng hết mười tham số; chia cho
$N - \gamma = 19{,}041235$ mới là con số (3.95) đòi.

Hệ quả thực hành: **$\gamma$ là thứ đáng báo cáo** cùng một mô hình chính quy hoá, không phải
$M$. Ở mục 1.1 của bài này, một đa thức bậc 9 (tức $M = 10$) sau khi chọn $\alpha$ bằng
evidence có $\gamma = 2{,}9511$ — nói "mô hình mười tham số" là nói sai về độ phức tạp thật của
nó **3,3886 lần**.

### 2.29 (3.98)–(3.99) — xấp xỉ khi $N \gg M$, và nó lệch bao nhiêu

![Công thức 3.98 và 3.99 của Bishop 2006](<paper_bishop2006_ct_3_98_3_99.png>)

**Hình 44 — (3.98) và (3.99).** Nguồn: Bishop 2006, trang 171,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"If we consider the limit N"* $\gg M$ *"in which the number of data points is large in
> relation to the number of parameters, then from (3.87) all of the parameters will be well
> determined by the data because ΦTΦ involves an implicit sum over data points, and so the
> eigenvalues λi increase with the size of the data set."*

> *"These results can be used as an easy-to-compute approximation to the full evidence"*

> *"because they do not require evaluation of the eigenvalue spectrum of the Hessian."*

**Dịch:** *"Nếu ta xét giới hạn N lớn hơn M nhiều, tức số điểm dữ liệu lớn so với số tham số,
thì từ (3.87) mọi tham số sẽ được dữ liệu xác định tốt, vì ΦᵀΦ chứa một phép tổng ngầm trên các
điểm dữ liệu, nên các trị riêng λᵢ tăng theo cỡ của tập dữ liệu."* · *"Các kết quả này dùng được
như một xấp xỉ DỄ TÍNH cho công thức ước lượng lại evidence đầy đủ, vì chúng không đòi phải
tính phổ trị riêng của ma trận Hesse."*

$$\alpha = \frac{M}{2E_W(\mathbf m_N)} \tag{3.98}$$

$$\beta = \frac{N}{2E_D(\mathbf m_N)} \tag{3.99}$$

**Ý nghĩa trừu tượng.** (3.98) và (3.99) là (3.92) và (3.95) với $\gamma$ thay bằng $M$ và
$N-\gamma$ thay bằng $N$. Lý do dùng chúng là **chi phí**: không cần phân rã trị riêng của một
ma trận $M\times M$. Lý do không nên dùng chúng khi $N$ nhỏ thì đo được:

| $N$ | $\gamma$ | $M$ | $\widehat\alpha$ theo (3.92) | $\widehat\alpha$ theo (3.98) | $\widehat\beta$ theo (3.95) | $\widehat\beta$ theo (3.99) |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 6,069365 | 10 | 3,328116 | 5,483465 | 19,128181 | 25,260883 |
| 50 | 6,090114 | 10 | 4,208064 | 6,909665 | 11,662953 | 13,280555 |
| 200 | 7,038450 | 10 | 4,887305 | 6,943723 | 11,738335 | 12,166502 |
| 1 000 | 8,148917 | 10 | 4,103454 | 5,035582 | 10,957859 | 11,047888 |
| 10 000 | 9,119744 | 10 | 4,251189 | 4,661523 | 11,195339 | 11,205558 |

Ở $N = 25$ thì $\widehat\alpha$ của xấp xỉ lệch **1,6476 lần** và $\widehat\beta$ lệch
**1,3206 lần**; ở $N = 10\,000$ thì hai cột gần trùng nhau. Nguyên nhân đọc được ngay ở cột
$\gamma$: nó đi từ 6,07 lên 9,12 trên $M = 10$, tức chỉ khi $N$ rất lớn thì $\gamma$ mới tiến
về $M$ và xấp xỉ mới đúng.

Một chỗ dễ đọc nhầm ở bảng này: cột $\gamma$ **không** đơn điệu tuyệt đối theo $N$ nếu so hai
dòng đầu (6,069365 rồi 6,090114) — hai dòng đó dùng hai bộ dữ liệu khác nhau nên chênh lệch nhỏ
đó là nhiễu lấy mẫu, không phải một hiện tượng. Xu hướng chỉ đọc được trên cả năm dòng.

### 2.30 Hình 3.16 — hình không có trục dọc, và ba hoành độ đọc được

![Thiết lập của Hình 3.16](<paper_bishop2006_thiet_lap_hinh_3_16.png>)

**Hình 45 — đoạn khai thiết lập của Hình 3.16 và 3.17.** Nguồn: Bishop 2006, trang 171,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"We can illustrate the evidence framework for setting hyperparameters using the sinusoidal
> synthetic data set from Section 1.1, together with the Gaussian basis function model
> comprising 9 basis functions, so that the total number of parameters in the model is given by
> M = 10 including the bias. Here, for simplicity of illustration, we have set β to its true
> value of 11.1 and then used the evidence framework to determine α"*

**Dịch:** *"Ta có thể minh hoạ khuôn khổ evidence cho việc đặt siêu tham số bằng tập dữ liệu
hình sin tổng hợp của mục 1.1, cùng với mô hình hàm cơ sở Gauss gồm 9 hàm cơ sở, sao cho tổng số
tham số của mô hình là M = 10 kể cả hệ số chệch. Ở đây, cho đơn giản trong minh hoạ, ta đã đặt β
bằng giá trị thật của nó là 11,1 rồi dùng khuôn khổ evidence để xác định α"*

**Đây là chỗ con số 11,1 xuất hiện**, và nó là con số mà mục 2.9 và mục 2.24 đem dùng.
$1/0{,}3^2 = 11{,}111111$, đúng với độ lệch chuẩn nhiễu 0,3 của mục 1.1.

![Hình 3.16 của Bishop 2006](<paper_bishop2006_hinh_3_16.png>)

**Hình 46 — Hình 3.16 nguyên bản.** Nguồn: Bishop 2006, trang 172,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"The left plot shows γ (red curve) and 2αEW (mN ) (blue curve) versus ln α for the sinusoidal
> synthetic data set. It is the intersection of these two curves that defines the optimum value
> for α given by the evidence procedure. The right plot shows the corresponding graph of log
> evidence ln p(t|α, β) versus ln α (red curve) showing that the peak coincides with the crossing
> point of the curves in the left plot. Also shown is the test set error (blue curve) showing that
> the evidence maximum occurs close to the point of best generalization."*

**Dịch:** *"Ô trái vẽ γ (đường đỏ) và 2αE_W(mₙ) (đường xanh) theo ln α cho tập dữ liệu hình sin
tổng hợp. Chính GIAO ĐIỂM của hai đường này định ra giá trị tối ưu cho α mà thủ tục evidence cho.
Ô phải vẽ đồ thị tương ứng của log evidence ln p(t|α, β) theo ln α (đường đỏ), cho thấy đỉnh của
nó TRÙNG với điểm giao của hai đường ở ô trái. Cũng vẽ thêm sai số tập kiểm tra (đường xanh), cho
thấy cực đại evidence xảy ra GẦN điểm tổng quát hoá tốt nhất."*

**Cách đọc hình, và một hạn chế phải nói trước.** Trục ngang của cả hai ô là $\ln\alpha$ với
nhãn $-5$, $0$, $5$. **Trục dọc không có một con số nào** — lớp text của trang 172 chỉ chứa
`ln α` và `−5 0 5`. Vậy hình này chỉ cho đọc **hoành độ**, và đúng ba hoành độ có nghĩa:

| Đại lượng đọc từ hình | $\ln\alpha$ | $\alpha$ |
|---|---:|---:|
| giao điểm của $\gamma$ và $2\alpha E_W$ ở ô trái | $+1{,}281827$ | 3,603217 |
| đỉnh của log evidence ở ô phải | $+1{,}247189$ | 3,480547 |
| cực tiểu sai số tập kiểm tra ở ô phải | $+2{,}516238$ | 12,381926 |

Hai dòng đầu **lệch nhau 0,034638** trong $\ln\alpha$. Chú thích nói chúng *"coincides"*, và
con số đó là mức "trùng" thật sự đọc được từ một đồ thị in.

**Dựng lại, và phân xử xem hình dùng bộ dữ liệu nào.** Bishop nói *"the sinusoidal synthetic
data set from Section 1.1"* mà không nói bao nhiêu điểm. Hai ứng cử: 25 điểm của Hình 3.8, hoặc
10 điểm của Hình 1.4. Thử cả hai, với $\mu_j$ = `linspace(0,1,9)`, $s = 0{,}123$,
$\beta = 11{,}1$:

| Bộ dữ liệu | giao $\gamma = 2\alpha E_W$ | đỉnh evidence | cực tiểu kiểm tra | $\lvert w\rvert$ lớn nhất tại $\gamma\to M$ |
|---|---:|---:|---:|---:|
| **25 điểm (Hình 3.8)** | $+1{,}267500$ | $+1{,}267500$ | $+2{,}337500$ | 3,301856 |
| 10 điểm (mục 1.1) | $+1{,}895000$ | $+1{,}895000$ | $-4{,}282500$ | 2,101671 |
| **hình đọc được** | $+1{,}281827$ | $+1{,}247189$ | $+2{,}516238$ | 2,317659 |

Bộ 25 điểm lệch **0,0143**; bộ 10 điểm lệch **0,6132** — khớp hơn **42,8 lần**. Vậy Hình 3.16
dùng chính 25 điểm của Hình 3.8. Lệch còn lại 0,0143 trong $\ln\alpha$ tương đương **1,42 %**
trong $\alpha$, nằm trong mức đọc được của một **giao điểm** hai đường.

Và một khẳng định của chú thích được xác nhận **chính xác** trong bản dựng lại: giao điểm
$\gamma = 2\alpha E_W$ và đỉnh log evidence rơi vào **cùng một điểm lưới**, lệch
$0{,}000\times10^{0}$. Đó không phải trùng hợp mà là nội dung (3.90) — hai vế của nó bằng nhau
đúng tại điểm dừng của evidence.

**Đo phần "close to" của câu cuối chú thích.** Đây là chỗ một tính từ phải được đổi thành số:

| | $\alpha$ theo evidence | $\alpha$ tốt nhất trên tập kiểm tra | Tỉ số |
|---|---:|---:|---:|
| bản dựng lại (tập kiểm tra 2 000 điểm) | 3,551962 | 10,355316 | **2,9154 lần** |
| đọc từ hình | 3,480547 | 12,381926 | **3,5575 lần** |

Hai bản cho cùng bậc: *"close to"* nghĩa là **cùng bậc độ lớn**, chênh một thừa số khoảng ba
lần trong $\alpha$ — không phải trùng. Và cái giá của chỗ chênh đó nhỏ hơn hẳn con số 3 lần gợi
ra:

| Chọn $\alpha$ thế nào | $\alpha$ | RMSE trên 2 000 điểm kiểm tra |
|---|---:|---:|
| theo evidence | 3,551962 | 0,319721 |
| tốt nhất có thể trên tập kiểm tra | 10,355316 | 0,316247 |

Chênh **1,098508 %** sai số căn quân phương. Đó là con số đúng để trích khi nói về lời hứa của
mục 3.5 của sách: evidence không cho $\alpha$ tối ưu, nó cho một $\alpha$ lệch khoảng ba lần và tốn
khoảng một phần trăm chất lượng — mà **không dùng một điểm dữ liệu kiểm tra nào**.

### 2.31 Hình 3.17 — và phép đo sắc nhất cho bề rộng hàm cơ sở

![Hình 3.17 của Bishop 2006](<paper_bishop2006_hinh_3_17.png>)

**Hình 47 — Hình 3.17 nguyên bản: mười hệ số theo $\gamma$.** Nguồn: Bishop 2006, trang 172,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Chú thích gốc, nguyên văn:**

> *"Plot of the 10 parameters wi from the Gaussian basis function model versus the effective
> number of parameters γ, in which the hyperparameter α is varied in the range"*
> $0 \le \alpha \le \infty$ *"causing γ to vary in the range"* $0 \le \gamma \le M$.

**Dịch:** *"Đồ thị 10 tham số wᵢ của mô hình hàm cơ sở Gauss theo số tham số hiệu dụng γ, trong
đó siêu tham số α được cho biến thiên trong khoảng 0 ≤ α ≤ ∞, làm γ biến thiên trong khoảng
0 ≤ γ ≤ M."*

**Cách đọc hình.** Trục ngang là $\gamma$ với nhãn 0, 2, 4, 6, 8, 10. Trục dọc là $w_i$ với
nhãn $-2$ tới $2$. Mười đường, mỗi đường một hệ số, có nhãn số ở đầu phải. Bên trái mọi đường
bắt đầu ở gần 0 (khi $\alpha$ lớn, tiên nghiệm nén hết) và toả ra khi $\gamma$ tiến về 10.

Đây là hình vectơ, nên đọc được cả hai trục. Mười đường đọc được:

| Object trong PDF | Số điểm | $\gamma$ nhỏ nhất | $\gamma$ lớn nhất | $w$ tại $\gamma$ lớn nhất | $\lvert w\rvert$ lớn nhất |
|---:|---:|---:|---:|---:|---:|
| 97 | 146 | 0,0377 | 9,9944 | $-1{,}977711$ | 1,977711 |
| 99 | 112 | 0,0377 | 9,9944 | $-1{,}358599$ | 1,358837 |
| 101 | 135 | 0,0377 | 9,9944 | $-1{,}262362$ | 1,262362 |
| 103 | 154 | 0,0377 | 9,9944 | $+0{,}271004$ | 0,347020 |
| 105 | 153 | 0,0377 | 9,9944 | $+0{,}877298$ | 0,877298 |
| 107 | 163 | 0,0377 | 9,9944 | $+0{,}979963$ | 0,979963 |
| 109 | 159 | 0,0377 | 9,9944 | $+1{,}095448$ | 1,101310 |
| 111 | 125 | 0,0377 | 9,9944 | $+1{,}162814$ | 1,162814 |
| 113 | 141 | 0,0377 | 9,9944 | $+1{,}891021$ | 1,891021 |
| 115 | 124 | 0,0377 | 9,9944 | $+2{,}317659$ | **2,317659** |

Đúng **mười** đường, và cả mười chạy trên cùng khoảng $\gamma \in [0{,}0377;\ 9{,}9944]$ —
tức hình dừng ở $\gamma = 9{,}9944$ chứ không tới đúng $M = 10$.

**Điều hình KHÔNG nói ra, và chỗ dựng lại được.** Chiều thì đúng: quét $\alpha$ từ lớn về nhỏ
cho thấy $\gamma$ tăng về $M$ và biên độ hệ số nở ra:

| $\alpha$ | $\gamma$ | $\lvert w\rvert$ lớn nhất |
|---:|---:|---:|
| $10^{3}$ | 0,580486 | 0,064436 |
| $10^{2}$ | 2,398830 | 0,327285 |
| $10$ | 4,851331 | 0,643570 |
| $1$ | 6,807265 | 0,995827 |
| $0{,}1$ | 8,303040 | 1,647544 |
| $10^{-2}$ | 9,268543 | 2,406796 |
| $10^{-4}$ | 9,983036 | 3,279845 |
| $10^{-8}$ | 9,999998 | 3,301854 |

Nhưng ở $\gamma$ mà hình dừng, biên độ dựng lại là **3,28** còn hình đọc được **2,318** — lệch
**1,4152 lần**. Đó không phải một chỗ bỏ qua được, vì $\lvert w\rvert$ lớn nhất **cực nhạy** với bề
rộng $s$, và chính vì thế nó thành **phép đo sắc nhất** cho $s$ trong cả bài:

| $s$ | giao $\ln\alpha$ | Lệch với hình | $\lvert w\rvert$ lớn nhất | Lệch với hình |
|---:|---:|---:|---:|---:|
| 0,100 | $+1{,}2100$ | 0,0718 | 1,4811 | 0,8365 |
| 0,105 | $+1{,}2300$ | 0,0518 | 1,6802 | 0,6375 |
| 0,110 | $+1{,}2500$ | 0,0318 | 1,9605 | 0,3572 |
| **0,115** | $+1{,}2600$ | **0,0218** | **2,3482** | **0,0306** |
| 0,120 | $+1{,}2600$ | 0,0218 | 2,8819 | 0,5643 |
| 0,125 | $+1{,}2700$ | 0,0118 | 3,6174 | 1,2998 |
| 0,130 | $+1{,}2700$ | 0,0118 | 4,6354 | 2,3178 |
| 0,140 | $+1{,}2500$ | 0,0318 | 8,0364 | 5,7188 |
| 0,150 | $+1{,}2200$ | 0,0618 | 14,7798 | 12,4622 |

Cột $\lvert w\rvert$ đi từ 1,48 ở $s = 0{,}10$ tới **14,78** ở $s = 0{,}15$ — hơn mười lần,
trên một khoảng $s$ rất hẹp. Bộ tốt nhất cho **cả hai** hình là $s = 0{,}1150$: giao điểm lệch
0,0218 và biên độ lệch 0,0306.

**Ba hình độc lập cùng chỉ về một bề rộng.** Ba phép đo dùng ba đại lượng hoàn toàn khác nhau:

| Hình | Đại lượng dùng để dò | $s$ dò ra |
|---|---|---:|
| 3.8 | dải $\pm\sigma$ và đường trung bình | **0,1230** |
| 3.16 | hoành độ một GIAO ĐIỂM | ≈ 0,115–0,125 |
| 3.17 | biên độ lớn nhất của hệ số | **0,1150** |

Ba con số nằm trong khoảng $[0{,}115;\ 0{,}123]$, lệch nhau **6,5 %**. Đó là bằng chứng mạnh hơn
hẳn một hình — nhưng cũng phải nói rõ chỗ không sạch: phép đo của Hình 3.17 **sắc hơn** hai phép
kia (nó phân biệt được $s = 0{,}115$ với $s = 0{,}120$), và theo thước đo sắc đó thì con số
0,1230 dò từ Hình 3.8 đã **ngoài** biên. Vậy hai hình không dùng đúng cùng một bộ tham số, hoặc
một trong hai phép đọc có một độ lệch hệ thống. Bài này ghi ra cả hai con số kèm nguồn thay vì
chọn một.

### 2.32 Mục 3.6 của sách — giới hạn của hàm cơ sở CỐ ĐỊNH

![Mục 3.6 của sách của Bishop 2006](<paper_bishop2006_gioi_han_3_6.png>)

**Hình 48 — mở đầu mục 3.6.** Nguồn: Bishop 2006, trang 172,
`2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

**Nguyên văn Bishop viết:**

> *"The difficulty stems from the assumption that the basis functions φj (x) are fixed before the
> training data set is observed and is a manifestation of the curse of dimensionality discussed
> in Section 1.4. As a consequence, the number of basis functions needs to grow rapidly, often
> exponentially, with the dimensionality D of the input space."*

> *"First of all, the data vectors {xn} typically lie close to a nonlinear manifold whose
> intrinsic dimensionality is smaller than that of the input space as a result of strong
> correlations between the input variables."*

> *"The second property is that target variables may have significant dependence on only a small
> number of possible directions within the data manifold."*

**Dịch:** *"Khó khăn bắt nguồn từ giả thiết rằng các hàm cơ sở φⱼ(x) được CỐ ĐỊNH TRƯỚC KHI quan
sát tập dữ liệu huấn luyện, và đó là một biểu hiện của lời nguyền số chiều đã bàn ở mục 1.4. Hệ
quả là số hàm cơ sở phải tăng rất nhanh, thường là theo hàm mũ, theo số chiều D của không gian
đầu vào."* · *"Trước hết, các vectơ dữ liệu {xₙ} thường nằm gần một đa tạp phi tuyến có số chiều
nội tại nhỏ hơn số chiều của không gian đầu vào, do các biến đầu vào tương quan mạnh với nhau."*
· *"Tính chất thứ hai là biến mục tiêu có thể phụ thuộc đáng kể vào chỉ một số ít hướng trong đa
tạp dữ liệu."*

**Cách đọc khung.** Mục 3.6 của sách dài chưa tới một trang và nó là mục quan trọng nhất cho câu "khi nào
đừng dùng". Ba điều nó nói:

1. **Giả thiết chết người là "cố định trước".** Cả chương 3 không có chỗ nào cho dữ liệu nói
   $\phi_j$ nên là gì. Với $D$ chiều đầu vào và một lưới hàm cơ sở định xứ, số hàm cơ sở cần
   tăng theo $\text{const}^D$ — với $D = 10$ và 5 mốc mỗi chiều thì đã là gần mười triệu hàm.
2. **Hai tính chất của dữ liệu thật cứu được phần nào**: dữ liệu nằm trên một đa tạp chiều
   thấp, và mục tiêu chỉ phụ thuộc vài hướng. Bishop nói rõ ai khai thác gì: *"This approach is
   used in radial basis function networks and also in support vector and relevance vector
   machines"* cho tính chất thứ nhất, *"Neural networks can exploit this property by choosing
   the directions in input space to which the basis functions respond"* cho tính chất thứ hai.
3. **Mạng nơ-ron là câu trả lời mà chương này chỉ ra.** Điều mạng nơ-ron làm khác là **học
   $\boldsymbol\phi$**, và đó chính là thứ làm mất công thức đóng của cả chương 3. Đánh đổi:
   một bên có hậu nghiệm đóng và một tiên nghiệm cố định; bên kia học được biểu diễn nhưng mất
   mọi thứ Bayes ở trên.

---

## 3. Cầu sang `scikit-learn` — cùng thuật toán, ba chỗ khác phải nói ra

`sklearn.linear_model.BayesianRidge` cài đúng phép lặp của mục 2.26. Kiểm trên cùng 25 điểm
và cùng ma trận thiết kế:

| Đại lượng | `v_bayes.py` mục 17 | `BayesianRidge` | Lệch tương đối |
|---|---:|---:|---:|
| $\alpha$ của Bishop = `lambda_` của sklearn | 3,51966163 | 3,51965874 | $8{,}225\times10^{-7}$ |
| $\beta$ của Bishop = `alpha_` của sklearn | 13,26836557 | 13,26834837 | $1{,}296\times10^{-6}$ |
| $\lvert\mathbf m_N\rvert$ lớn nhất | 0,80200687 | 0,80200680 | $8{,}721\times10^{-8}$ |

Ba chỗ khác, và cả ba đều gây lỗi im lặng nếu không biết:

**Chỗ khác 1 — TÊN bị đảo.** `alpha_` của `BayesianRidge` **chính là $\beta$** của Bishop, tức
độ chính xác **nhiễu**; còn `lambda_` mới là $\alpha$ của Bishop, tức độ chính xác **tiên
nghiệm**. Ai đọc `alpha_` thành $\alpha$ sẽ lấy một con số lớn hơn **3,7698 lần** con số mình
cần (13,26836557 so với 3,51966163 ở ví dụ trên), và không có gì báo lỗi. Đây là chỗ trùng tên đắt nhất của
cả bài, và nó tệ hơn một chỗ trùng tên bình thường vì `Ridge` và `Lasso` của **cùng thư viện**
dùng `alpha` cho **mức phạt** — tức nghĩa thứ ba của cùng một chữ trong cùng một gói.

**Chỗ khác 2 — sklearn đặt tiên nghiệm lên chính $\alpha$ và $\beta$.** `BayesianRidge` có bốn
tham số `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, mặc định $10^{-6}$, là tham số của hai
tiên nghiệm Gamma. Vậy công thức cập nhật của nó **không** đúng bằng (3.92)/(3.95) mà có thêm số
hạng; với mặc định $10^{-6}$ thì phần thêm nhỏ, và bảng trên đo được nó: lệch cỡ $10^{-6}$ đến
$10^{-7}$. Đổi bốn tham số đó lên $10^{-2}$ thì `lambda_` đi từ 3,51965874 xuống **3,49101286**,
tức lệch **0,81 %**. Điều đó khớp đúng điều kiện Bishop nêu ở (3.76) — *"if the prior is
relatively flat"*.

**Chỗ khác 3 — hệ số chặn không bị phạt.** `fit_intercept=True` (mặc định) **tâm hoá** dữ liệu
và **không** phạt hệ số chặn. Bishop thì phạt cả $w_0$, vì $\phi_0 = 1$ nằm trong tiên nghiệm
(3.52). Đo: cùng dữ liệu, `lambda_` ra **2,92539716** thay vì 3,51965874, và dự báo tại
$x = 0{,}5$ lệch **0,002225** giữa hai quy ước ($-0{,}028552$ theo Bishop, $-0{,}030777$ theo
mặc định sklearn).

Chênh dự báo nhỏ, nhưng **hệ quả thì không nhỏ**, và mục 2.11 đã đo nó: chính việc $w_0$ có bị
phạt hay không quyết định cận dưới của phương sai dự báo khi ngoại suy. Với quy ước của Bishop,
$\sigma$ dừng ở $0{,}828742$; bỏ hàm cơ sở hằng ra khỏi tiên nghiệm thì nó tụt về $0{,}300150$.
Hai câu trả lời khác nhau **2,7611 lần** cho câu hỏi "ngoài vùng dữ liệu thì ta biết gì".

**Và một họ hàng gần, `ARDRegression`.** Đây là *relevance vector machine* mà Bishop hẹn ở
chương 7: **một $\alpha$ riêng cho mỗi hệ số** thay vì một $\alpha$ chung. Chạy trên cùng dữ
liệu: **8 trên 10** hệ số bị đẩy về dưới $10^{-3}$, và `lambda_` lớn nhất lên tới
$8{,}363\times10^{4}$ — tức tám hướng bị tiên nghiệm khoá cứng về 0. Đó là phép chọn đặc trưng
tự động mà kiểm định chéo không làm được, vì không ai quét được một lưới mười chiều.

Ba dòng đủ để chạy lại:

```python
from sklearn.linear_model import BayesianRidge

br = BayesianRidge(fit_intercept=False, tol=1e-12, max_iter=2000).fit(Phi, t)
print(br.lambda_, br.alpha_)   # lambda_ là alpha của Bishop; alpha_ là beta của Bishop
```

---

## 4. Cơ chế — vì sao đổi cái đó lại ăn thua

Ba cơ chế, và mỗi cơ chế đi kèm một phép đo — rồi một mục thứ tư về hai chỗ mà một
khẳng định của nguồn chỉ đúng ở một giới hạn.

### 4.1 Cơ chế thứ nhất: độ chính xác CỘNG được, nên thông tin tích luỹ được

(3.51) nói $\mathbf S_N^{-1} = \mathbf S_0^{-1} + \beta\mathbf\Phi^{\!\top}\mathbf\Phi$. Phép
cộng đó là toàn bộ lý do mô hình này học trực tuyến được mà không mất gì: nạp 20 điểm một lượt
và nạp từng điểm một cho $\mathbf m_N$ giống nhau tới $5{,}551\times10^{-17}$ và
$\mathbf S_N$ giống nhau tới $5{,}421\times10^{-20}$ (mục 2.4).

Và nó giải thích vì sao độ rộng hậu nghiệm **chỉ có thể giảm**: mỗi điểm dữ liệu cộng một ma
trận nửa xác định dương vào $\mathbf S_N^{-1}$, nên không hướng nào của hậu nghiệm rộng ra
được. Đó là bản đại số của khẳng định $\sigma_{N+1}^2(x) \le \sigma_N^2(x)$ mà mục 2.8 kiểm
được **0 chỗ vi phạm trên 15 336 phép so**.

Nhưng phải nói rõ chỗ cơ chế này **không** bảo đảm: nó nói về **độ rộng**, không nói về **sai
số**. Mục 2.6 đo được sai số của $\mathbf m_N$ đi $0{,}500 \to 0{,}545 \to 0{,}424 \to 0{,}063$
— tăng ở bước đầu. Một hậu nghiệm hẹp hơn có thể hẹp quanh một chỗ sai.

### 4.2 Cơ chế thứ hai: hai nguồn bất định, và chỉ MỘT nguồn giảm được

(3.59) tách phương sai dự báo thành $1/\beta$ cộng $\boldsymbol\phi^{\!\top}\mathbf S_N
\boldsymbol\phi$. Hình 3.8 của Bishop vẽ **tổng** của hai số hạng đó, nên nhìn nó không biết
phần nào là phần nào. Hình dưới tách chúng ra.

```python
# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [A] in ra; khối này chỉ VẼ.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

XA = np.array([-0.20000000, -0.16666667, -0.13333333, -0.10000000, -0.06666667,
               -0.03333333, 0.00000000, 0.03333333, 0.06666667, 0.10000000,
               0.13333333, 0.16666667, 0.20000000, 0.23333333, 0.26666667,
               0.30000000, 0.33333333, 0.36666667, 0.40000000, 0.43333333,
               0.46666667, 0.50000000, 0.53333333, 0.56666667, 0.60000000,
               0.63333333, 0.66666667, 0.70000000, 0.73333333, 0.76666667,
               0.80000000, 0.83333333, 0.86666667, 0.90000000, 0.93333333,
               0.96666667, 1.00000000, 1.03333333, 1.06666667, 1.10000000,
               1.13333333, 1.16666667, 1.20000000])
# phần BẤT ĐỊNH THAM SỐ, tức phi^T S_N phi, ở N = 4 và N = 25
Q4 = np.array([0.44834905, 0.37499148, 0.30387827, 0.23794115, 0.17335376,
               0.11330627, 0.08777928, 0.15302167, 0.35614508, 0.68913412,
               1.07526870, 1.40076911, 1.56421824, 1.51422887, 1.26712384,
               0.90236729, 0.53170773, 0.25242669, 0.10982662, 0.08638062,
               0.11926351, 0.14006367, 0.11926351, 0.08638062, 0.10982662,
               0.25242669, 0.53170773, 0.90236729, 1.26712384, 1.51422887,
               1.56421824, 1.40076911, 1.07526870, 0.68913412, 0.35614508,
               0.15302167, 0.08777928, 0.11330627, 0.17335376, 0.23794115,
               0.30387827, 0.37499148, 0.44834905])
Q25 = np.array([0.39618272, 0.31616900, 0.23707402, 0.16945754, 0.11751071,
                0.07879148, 0.04999943, 0.03140507, 0.02393613, 0.02402522,
                0.02521367, 0.02449884, 0.02350580, 0.02361636, 0.02382469,
                0.02350078, 0.02343058, 0.02368906, 0.02352067, 0.02320255,
                0.02345499, 0.02375827, 0.02345499, 0.02320255, 0.02352067,
                0.02368906, 0.02343058, 0.02350078, 0.02382469, 0.02361636,
                0.02350580, 0.02449884, 0.02521367, 0.02402522, 0.02393613,
                0.03140507, 0.04999943, 0.07879148, 0.11751071, 0.16945754,
                0.23707402, 0.31616900, 0.39618272])
NHIEU = 0.09009009          # 1/beta voi beta = 11,1

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True)
for a, Q, ten in ((ax[0], Q4, "N = 4"), (ax[1], Q25, "N = 25")):
    a.fill_between(XA, 0, NHIEU, color="#94a3b8", alpha=0.65,
                   label=r"$1/\beta$ — nhiễu, không giảm được")
    a.fill_between(XA, NHIEU, NHIEU + Q, color="#7c3aed", alpha=0.45,
                   label=r"$\phi^\top S_N\phi$ — bất định tham số")
    a.plot(XA, NHIEU + Q, color="#4c1d95", lw=2.0)
    a.axvspan(0.0, 1.0, color="#16a34a", alpha=0.07)
    a.axvline(0.0, color="#16a34a", lw=1.0, ls=(0, (4, 3)))
    a.axvline(1.0, color="#16a34a", lw=1.0, ls=(0, (4, 3)))
    a.set_title("%s — vùng xanh là nơi có dữ liệu" % ten)
    a.set_xlabel("x")
    a.set_yscale("log")
ax[0].set_ylabel(r"$\sigma_N^2(x)$, hai thành phần cộng lại")
ax[0].legend(loc="upper center")
print("phần bất định tham số tại x = 0,5: N=4 -> %.8f ; N=25 -> %.8f"
      % (Q4[21], Q25[21]))
print("tỉ số hai giá trị đó: %.6f lần" % (Q4[21] / Q25[21]))
print("so với nhiễu 1/beta tại x = 0,5: N=4 -> %.4f lần ; N=25 -> %.4f phần trăm"
      % (Q4[21] / NHIEU, 100 * Q25[21] / NHIEU))
print("tại x = 1,2 (ngoài dữ liệu): N=4 -> %.8f ; N=25 -> %.8f"
      % (Q4[-1], Q25[-1]))
fig.savefig("bayes_hai_nguon_bat_dinh.png", dpi=130, bbox_inches="tight",
            facecolor="white")
```

![Hai nguồn của phương sai dự báo, tách rời](<bayes_hai_nguon_bat_dinh.png>)

**Hình 49 — hai số hạng của (3.59) tách rời, trục dọc thang log.** Hình tự vẽ; mọi con số do
`tools/v_bayes.py` mục 23 [A] in ra, khối mã ngay trên sinh ra hình này.

**Cách đọc hình.** Dải xám là $1/\beta = 0{,}09009009$, một hằng số — **cùng độ dày ở mọi $x$**.
Dải tím là $\boldsymbol\phi^{\!\top}\mathbf S_N\boldsymbol\phi$. Vùng xanh nhạt là
$[0,1]$, nơi có dữ liệu.

Ba điều đọc được mà Hình 3.8 của Bishop không cho thấy:

* **Ở $N=25$, trong vùng dữ liệu, dải tím mỏng hẳn đi**: tại $x=0{,}5$ nó bằng
  $0{,}02375827$ so với $0{,}09009009$ của nhiễu, tức chỉ **26,3717 %**. Nói cách khác: với 25
  điểm thì độ bất định gần như **hoàn toàn** là nhiễu, và thu thêm dữ liệu không giúp gì nữa.
* **Ở $N=4$ thì ngược lại**: tại $x=0{,}5$ dải tím bằng $0{,}14006367$, tức **1,5547 lần** dày
  hơn nhiễu. Ở đó thu thêm dữ liệu **là** việc đáng làm. Tỉ số giữa hai giá trị là
  **5,895365 lần** — đó là phần mà 21 điểm dữ liệu mua được.
* **Ra ngoài $[0,1]$ thì dải tím phình lên ở cả hai bảng** và hai bảng tiến về **cùng một**
  giá trị: tại $x = 1{,}2$ nó là $0{,}44834905$ ở $N=4$ và $0{,}39618272$ ở $N=25$. Thêm 21
  điểm gần như **không** giúp gì cho vùng ngoại suy, vì các hàm cơ sở không phủ tới đó.

Đó là hệ quả dùng được ngay của việc tách hai số hạng: câu hỏi *"thu thêm dữ liệu có giúp
không"* có câu trả lời **khác nhau ở từng $x$**, và (3.59) là chỗ đọc ra câu trả lời đó.

### 4.3 Cơ chế thứ ba: evidence phạt độ phức tạp bằng phép CHUẨN HOÁ, không bằng một tham số

Đây là cơ chế khó tin nhất và cũng là cơ chế đáng học nhất. Không ai thêm một số hạng phạt vào
(3.86); khoản phạt $-\tfrac12\ln\lvert\mathbf A\rvert$ **xuất hiện từ hằng chuẩn hoá** của tích
phân (3.85). Lập luận của Hình 3.13 nói đúng điều đó bằng lời: các phân phối
$p(\mathcal D\mid\mathcal M_i)$ đều **chuẩn hoá**, nên mô hình rải rộng hơn bắt buộc phải thấp
hơn ở mọi chỗ.

Bảng ở mục 2.19 là bản định lượng: cột log hợp lẽ **tăng** đơn điệu theo bậc ($-18{,}12$ lên
$+0{,}75$) trong khi cột phạt **giảm** đơn điệu ($-5{,}00$ xuống $-16{,}60$), và tổng của chúng
có một đỉnh ở giữa.

**Phép đo cho thấy cơ chế này TINH TẾ tới mức nào.** Mục 2.24 tìm ra rằng Hình 3.14 được vẽ
bằng một công thức lệch $\tfrac12\ln(2\pi) = 0{,}918939$ **mỗi tham số** so với (3.86) — một
lượng nhỏ nếu so với dải giá trị $7{,}866496$ đơn vị của trục dọc, nhưng đủ để đổi chỗ đỉnh từ
$M=4$ sang $M=3$. Nghĩa là: **hằng chuẩn hoá không phải chi tiết kế toán**. Nó là chính phần
phạt, và bỏ một thừa số của nó là đổi mô hình được chọn.

### 4.4 Chỗ một khẳng định của nguồn chỉ đúng ở một giới hạn

Hai chỗ như thế đã đo được, và cả hai cùng một khuôn: **một điều kiện không được nêu**.

Chỗ thứ nhất là (3.64), *"these weights sum to one"*. Chứng minh không hình thức của Bishop
dùng bước *"we can fit the training data exactly"*, mà trung bình dự báo (3.53) là nghiệm **đã
co**. Độ lệch tăng đơn điệu theo $\alpha$:

```python
# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [C] in ra; khối này chỉ VẼ.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

ALC = np.array([1.000000e-12, 4.216965e-12, 1.778279e-11, 7.498942e-11, 3.162278e-10,
                1.333521e-09, 5.623413e-09, 2.371374e-08, 1.000000e-07, 4.216965e-07,
                1.778279e-06, 7.498942e-06, 3.162278e-05, 1.333521e-04, 5.623413e-04,
                2.371374e-03, 1.000000e-02, 4.216965e-02, 1.778279e-01, 7.498942e-01,
                3.162278e+00, 1.333521e+01, 5.623413e+01, 2.371374e+02, 1.000000e+03])
DC = np.array([5.646816e-12, 7.835954e-12, 9.912515e-12, 1.483169e-11, 8.092427e-11,
               3.404190e-10, 1.410850e-09, 5.958219e-09, 2.511531e-08, 1.059338e-07,
               4.466998e-07, 1.883383e-06, 7.936416e-06, 3.336588e-05, 1.389257e-04,
               5.563985e-04, 1.944724e-03, 4.959254e-03, 9.310149e-03, 1.603030e-02,
               2.676825e-02, 4.679721e-02, 8.473261e-02, 1.632411e-01, 3.301435e-01])
AL_H38 = 0.513333       # alpha dung lai duoc Hinh 3.8 (v_bayes.py muc 9)
DC_H38 = 1.395e-02      # do lech tai chinh alpha do (v_bayes.py muc 13)

fig, ax = plt.subplots(figsize=(7.8, 4.3))
ax.loglog(ALC, DC, color="#2563eb", lw=2.0, marker="o", ms=3.4,
          label=r"$\max_x\,|\sum_n k(x,x_n)-1|$")
ax.axvline(AL_H38, color="#dc2626", lw=1.8, ls=(0, (5, 3)),
           label=r"$\alpha$ dựng lại được Hình 3.8")
ax.plot([AL_H38], [DC_H38], "*", color="#dc2626", ms=17, zorder=6,
        label="độ lệch tại đúng $\\alpha$ đó")
ax.axhline(1e-2, color="#64748b", lw=1.0, ls=":")
ax.text(2.0e2, 1.25e-2, "mức 1 %", color="#64748b", fontsize=8, ha="right")
ax.set_xlabel(r"$\alpha$ — độ chính xác tiên nghiệm")
ax.set_ylabel(r"độ lệch khỏi ràng buộc (3.64)")
ax.set_title(r"(3.64) $\sum_n k(x,x_n)=1$ chỉ đúng trong giới hạn $\alpha\to 0$")
ax.legend(loc="upper left", fontsize=8)
print("do lech tai alpha nho nhat (%.3e): %.6e" % (ALC[0], DC[0]))
print("do lech tai alpha lon nhat (%.3e): %.6e" % (ALC[-1], DC[-1]))
print("so bac do lon giua hai dau: %.4f" % np.log10(DC[-1] / DC[0]))
print("do lech tai alpha = %.6f la %.6e, tuc %.4f phan tram"
      % (AL_H38, DC_H38, 100 * DC_H38))
fig.savefig("bayes_tong_nhan.png", dpi=130, bbox_inches="tight", facecolor="white")
```

![Độ lệch khỏi ràng buộc tổng của nhân tương đương](<bayes_tong_nhan.png>)

**Hình 50 — độ lệch khỏi (3.64) theo $\alpha$, cả hai trục thang log.** Hình tự vẽ; mọi con số
do `tools/v_bayes.py` mục 23 [C] in ra.

**Cách đọc hình.** Trục ngang là $\alpha$, trục dọc là $\max_x\lvert\sum_n k(x,x_n)-1\rvert$.
Đường thẳng trên giấy log–log nghĩa là quan hệ luỹ thừa. Vạch đỏ là $\alpha = 0{,}513333$, giá
trị dựng lại được Hình 3.8, và dấu sao là độ lệch tại **đúng** giá trị đó,
$1{,}395000\times10^{-2}$ — tức **1,3950 %**, con số của mục 2.15. Hai đầu cách nhau **10,7669 bậc độ lớn** — từ
$5{,}646816\times10^{-12}$ tới $3{,}301435\times10^{-1}$.

**Điều hình KHÔNG nói ra:** nó không nói (3.64) sai. Nó nói (3.64) là một tính chất của
**nghiệm hợp lẽ cực đại** ($\alpha\to0$), và mọi $\alpha$ hữu hạn đều rời khỏi nó một lượng
đo được. Ba điều kiện mà Bishop **có** nêu đều thoả ở mọi điểm của hình.

Chỗ thứ hai là câu ở tr.158 về ngoại suy, *"leaving only the noise contribution"*. Mục 2.11 đo
được: với (3.3) như bài báo viết — tức có hàm cơ sở hằng — thì $\sigma$ dừng ở $0{,}828742$ chứ
không ở $0{,}300150$; bỏ hàm cơ sở hằng thì câu đúng tới từng chữ số. Cùng khuôn, cùng cách
chữa: **nói ra điều kiện bị bỏ**, không nói bài báo sai.

---

## 5. Kết quả — bằng chứng đâu

Chương 3 của Bishop có **0 bảng số**, nên không có một ô bảng nào để trích. Mọi con số của
nguồn nằm trong hình, và bảng dưới đây là toàn bộ những gì dựng lại được, kèm mức sai lệch.

### 5.1 Định lượng — bảy phép dựng lại, và mức sai của từng phép

| # | Đối tượng của nguồn | Dựng lại được gì | Sai lệch | Mốc so sánh |
|---|---|---|---|---|
| 1 | **25 điểm dữ liệu** của Hình 3.8 tr.157 | cả 25 toạ độ $(x_n, t_n)$ | $x$ lệch khỏi `linspace(0,1,25)` nhiều nhất **0,000348** | — |
| 2 | **đường $\sin(2\pi x)$** của Hình 3.8 | đọc lại và so với hàm thật | **0,003277** trên 74 điểm | đây LÀ mốc sai số đọc vectơ |
| 3 | **dải $\pm\sigma$** của ô $N=25$, Hình 3.8 | $\sigma(x)$ tại 49 hoành độ | lệch $\sigma$ **0,001580** | 0,003277 — TRONG mốc |
| 4 | **đường trung bình đỏ** của ô $N=25$ | cả đường | lệch **0,003296** | 0,003277 — NGANG mốc (tỉ số 1,01) |
| 5 | **10 giá trị evidence** của Hình 3.14 tr.168 | cả mười, theo bản thiếu $(2\pi)^{M/2}$ | lệch lớn nhất **0,012553**, trung bình **0,005062** | 10/10 ô trong 0,03 |
| 6 | **giao điểm** của Hình 3.16 tr.172 | hoành độ $\ln\alpha$ | **0,0143** trong $\ln\alpha$, tức **1,42 %** trong $\alpha$ | mốc của một giao điểm |
| 7 | **biên độ hệ số** của Hình 3.17 tr.172 | $\lvert w\rvert$ lớn nhất | **0,0306** ở $s = 0{,}1150$ | phép đo sắc nhất cho $s$ |

Và bảy phép kiểm **toán học** không cần tới hình, nên không có sai số đọc:

| Khẳng định của nguồn | Cách kiểm | Kết quả |
|---|---|---|
| (3.55) MAP $=$ nghiệm ridge với $\lambda = \alpha/\beta$ | phân số chính xác trên 3 điểm | **bằng nhau**: $\left[\tfrac{38}{41},\tfrac{24}{41}\right]$ ở cả hai đường |
| (3.49) cập nhật tuần tự $=$ cập nhật một lượt | 20 điểm, hai đường | $\mathbf m_N$ lệch $5{,}551\times10^{-17}$ · $\mathbf S_N$ lệch $5{,}421\times10^{-20}$ |
| (3.86) công thức đóng $=$ tích phân thật | cầu phương $2001\times2001$, ba cặp $(\alpha,\beta)$ | lệch $\le 7{,}638\times10^{-14}$ |
| (3.59) $\sigma_{N+1}^2 \le \sigma_N^2$ (Qazaz 1997) | 15 336 phép so, kể cả ngoại suy | **0 chỗ vi phạm** |
| (3.63) $\operatorname{cov} = \beta^{-1}k$ | lưới $200\times200$ | lệch $4{,}432\times10^{-16}$ |
| (3.65) $k = \boldsymbol\psi^{\!\top}\boldsymbol\psi$ | lưới $200\times200$ | lệch $4{,}749\times10^{-14}$ |
| (3.91) $0 \le \gamma \le M$ | 9 giá trị $\alpha$ trải 18 bậc | đúng ở cả 9, và $\gamma$ giảm đơn điệu |

### 5.2 Hình dựng lại Hình 3.14 — phát hiện chính, vẽ ra

```python
# -*- coding: utf-8 -*-
# Mọi con số trong khối này do tools/v_bayes.py mục 23 [B] in ra; khối này chỉ VẼ.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

M = np.arange(10)
# ba day so, doc tu Hinh 3.14 va dung lai bang hai cach doc cong thuc
DOC = np.array([-24.036722, -19.729529, -23.075116, -17.612635, -18.307154,
                -19.869763, -21.412340, -22.828063, -24.170305, -25.479131])
CT386 = np.array([-23.123666, -17.879100, -20.306141, -13.934076, -13.713212,
                  -14.359022, -14.981523, -15.481400, -15.906108, -16.290483])
THIEU = np.array([-24.042604, -19.716977, -23.062957, -17.609830, -18.307904,
                  -19.872653, -21.414093, -22.832908, -24.176555, -25.479868])

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))

ax[0].plot(M, DOC, "o-", color="#111827", lw=2.4, ms=6, zorder=6,
           label="đọc từ Hình 3.14 (lớp vectơ)")
ax[0].plot(M, THIEU, "s--", color="#dc2626", lw=1.8, ms=5, zorder=5,
           label=r"(3.86) $-\ (M/2)\ln 2\pi$")
ax[0].plot(M, CT386, "^-", color="#2563eb", lw=1.8, ms=5, zorder=4,
           dashes=(5, 3), label="đúng công thức (3.86)")
ax[0].plot([int(np.argmax(DOC))], [DOC.max()], "*", color="#111827", ms=17, zorder=7)
ax[0].plot([int(np.argmax(CT386))], [CT386.max()], "*", color="#2563eb", ms=17, zorder=7)
ax[0].set_xlabel("bậc đa thức M")
ax[0].set_ylabel(r"$\ln p(\mathsf{t}\mid\alpha,\beta)$")
ax[0].set_title("Hình 3.14 và hai cách đọc công thức")
ax[0].legend(loc="lower left")

le = CT386 - DOC
ax[1].plot(M + 1, le, "o", color="#7c3aed", ms=7, zorder=5, label="lệch đo được")
A = np.c_[np.ones(10), M + 1]
ab = np.linalg.lstsq(A, le, rcond=None)[0]
xx = np.linspace(0.6, 10.4, 50)
ax[1].plot(xx, ab[0] + ab[1] * xx, color="#0891b2", lw=2.0,
           label="đường khớp: %+.6f %+.6f(M+1)" % (ab[0], ab[1]))
ax[1].plot(xx, 0.5 * np.log(2 * np.pi) * xx, color="#dc2626", lw=1.6,
           dashes=(6, 3), label=r"$\frac{1}{2}\ln(2\pi)\,(M+1)$")
ax[1].set_xlabel("số tham số M + 1")
ax[1].set_ylabel("(3.86) trừ giá trị đọc được")
ax[1].set_title("lệch TUYẾN TÍNH theo số tham số")
ax[1].legend(loc="upper left")

print("he so goc do duoc      : %.6f" % ab[1])
print("mot nua ln(2 pi)       : %.6f" % (0.5 * np.log(2 * np.pi)))
print("chenh hai con so       : %.6f" % abs(ab[1] - 0.5 * np.log(2 * np.pi)))
print("he so chan do duoc     : %+.6f" % ab[0])
print("phan du lon nhat       : %.6f" % np.abs(le - A @ ab).max())
print("dinh cua duong doc duoc: M = %d" % int(np.argmax(DOC)))
print("dinh theo dung (3.86)  : M = %d" % int(np.argmax(CT386)))
print("lech lon nhat ban thieu 2pi: %.6f" % np.abs(THIEU - DOC).max())
fig.savefig("bayes_dung_lai_hinh_3_14.png", dpi=130, bbox_inches="tight",
            facecolor="white")
```

![Dựng lại Hình 3.14 theo hai cách đọc công thức evidence](<bayes_dung_lai_hinh_3_14.png>)

**Hình 51 — dựng lại Hình 3.14, hai cách đọc, và cấu trúc của phần lệch.** Hình tự vẽ; mọi con
số do `tools/v_bayes.py` mục 23 [B] in ra.

**Cách đọc hình.** Ô trái: ba đường. Đường **đen** là mười giá trị đọc từ lớp vectơ của Hình
3.14. Đường **đỏ nét gạch** là $(3.86) - \tfrac M2\ln2\pi$ — nó **nằm lên đúng** đường đen.
Đường **xanh** là đúng (3.86) — nó nằm cao hơn và **xa dần**. Hai dấu sao đánh hai đỉnh: đỉnh
đen ở $M=3$, đỉnh xanh ở $M=4$.

Ô phải là chỗ kết luận được rút ra. Trục ngang là **số tham số** $M+1$, trục dọc là phần lệch
giữa (3.86) và giá trị đọc được. Mười điểm tím nằm gần như đúng trên một **đường thẳng qua gốc**,
và đường đỏ nét gạch — hệ số góc $\tfrac12\ln(2\pi)$, không khớp gì cả, vẽ thẳng từ hằng số —
đi trùng nó.

Khối mã in ra: hệ số góc đo được **0,917810**, $\tfrac12\ln(2\pi) = 0{,}918939$, chênh
**0,001128**; hệ số chặn $+0{,}006645$; phần dư lớn nhất **0,011400**; và lệch lớn nhất của
bản thiếu $2\pi$ so với đường đọc được là **0,012552**. Ba con số cuối lệch chữ số thứ sáu so
với mục 2.24, vì khối mã tính lại từ mảng **đã làm tròn** sáu chữ số thập phân chứ không từ
đủ chữ số — đó là phép làm tròn hai lần, và nó được ghi ra thay vì làm tròn cho khớp.

**Điều hình KHÔNG nói ra.** Nó không nói thừa số thiếu nằm ở bản in của (3.85)/(3.86) hay ở
đoạn mã vẽ hình — không có cách nào biết. Nó cũng không nói kết luận nào của chương sai: mọi
lập luận về sau dùng (3.86), và bài này đã kiểm (3.86) bằng tích phân số ở mục 2.23. Thứ hình
nói là: **hình được in và công thức được in không phải cùng một hàm, và hai hàm đó không đồng ý
về bậc nào thắng.**

### 5.3 Định tính — một ví dụ vào–ra thật, chạy tay được

Đầu vào: ba điểm $(0,1)$, $(1,2)$, $(2,2)$; mô hình $y = w_0 + w_1x$; $\alpha = 1$, $\beta = 2$.

Đầu ra:

```
m_N = [38/41, 24/41] = [0,926829 ; 0,585366]
S_N = [[11/41, -6/41], [-6/41, 7/41]]

dự báo tại x* = 3 : 110/41 = 2,682927 ± 1,194500     (95 %: ± 2,341219)
   trong đó nhiễu 1/2 chiếm 35,04 % · bất định tham số 38/41 chiếm 64,96 %
dự báo tại x* = 1 : phần bất định tham số chỉ 6/41 = 0,146341
   -> nhỏ hơn 6,3333 lần so với tại x* = 3
```

Mọi con số ở trên là **số hữu tỉ chính xác**, `v_bayes.py` mục 5 in ra bằng
`fractions.Fraction` — nên chúng kiểm được bằng mắt, không phụ thuộc thư viện hay phiên bản.

---

## 6. Giá phải trả

| Mặt được | Đo bao nhiêu | Trả giá bằng gì | Nguồn số |
|---|---|---|---|
| Có độ bất định cho từng $x$ | $\sigma$ ở $x=1{,}3$ so với $x=0{,}5$: **1,0652 lần** | phải giữ và nghịch đảo một ma trận $M\times M$; bình phương tối thiểu chỉ cần giải một hệ | mục 1.4 · `v_bayes.py` mục 21 |
| Chọn $\lambda$ không cần tập giữ lại | **34** lần giải hệ thay cho **1 000**, rẻ **29,4 lần** | phải chọn **dạng** tiên nghiệm; và evidence **không tồn tại** nếu tiên nghiệm không chính quy | mục 1.2 · mục 2.21 |
| $\lambda$ tất định, không phụ thuộc cách chia mảnh | kiểm định chéo cho $\lambda$ trải **3,4 lần** trên 60 cách chia; evidence cho **một** giá trị | $\alpha$ evidence chọn lệch $\alpha$ tốt nhất trên tập kiểm tra **2,9154 lần**, tốn **1,098508 %** RMSE | mục 1.3 · mục 2.30 |
| Chống quá khớp bằng chính tiên nghiệm | bậc 9 trên 10 điểm: RMSE kiểm tra **1,291709 → 0,501726**, hệ số lớn nhất $1{,}062\times10^{6} \to 0{,}9596$ | $\gamma = 2{,}9511$ — mô hình mất phần lớn năng lực biểu diễn để đổi lấy ổn định | mục 1.1 |
| Học trực tuyến không mất gì | tuần tự bằng một lượt tới $5{,}551\times10^{-17}$ | chỉ đúng vì tiên nghiệm Gauss liên hợp; đổi sang $q \ne 2$ ở (3.56) là mất hết | mục 2.4 · mục 2.7 |
| Báo được số tham số thật | $\gamma = 5{,}958765$ trên $M = 10$ | phải phân rã trị riêng một ma trận $M\times M$ mỗi vòng lặp; (3.98)/(3.99) rẻ hơn nhưng lệch **1,65 lần** ở $N=25$ | mục 2.26 · mục 2.29 |

**Sáu chỗ hỏng, xếp theo mức nguy hiểm:**

1. **Tự tin sai khi mô hình sai.** $\sigma_N(x)$ của (3.59) là độ bất định **trong** mô hình đã
   chọn, không phải độ bất định về việc mô hình có đúng hay không. Nếu $\phi_j$ chọn sai thì
   $\sigma$ vẫn nhỏ và vẫn sai. Bishop nói thẳng ở tr.165: *"a Bayesian approach ... needs to
   make assumptions about the form of the model, and if these are invalid then the results can
   be misleading."*
2. **Ngoại suy với hàm cơ sở định xứ thì tự tin quá mức — và nó hỏng ĐO ĐƯỢC.** Ở
   $x = 1{,}3$, khoảng 95 % là $[-0{,}586046;\ +0{,}873930]$ trong khi giá trị thật là
   $+0{,}951057$: khoảng tin cậy **không chứa** giá trị thật, vượt cận trên **0,077127** đơn
   vị. Và $\sigma$ ở đó chỉ lớn hơn $\sigma$ giữa dữ liệu **1,0652 lần**, dù sai số dự báo
   lớn hơn **4,1528 lần** so với bình phương tối thiểu tại chính điểm đó. Với hàm cơ sở đa
   thức thì lỗi **ngược lại** và cũng tệ: $\sigma$ tại $x=3$ lớn gấp **63 894,7 lần**.
3. **Evidence rất nhạy với bề rộng tiên nghiệm.** Từ (3.70): nới bề rộng tiên nghiệm gấp đôi
   là chia evidence cho $2^M$, tức **1 024 lần** với $M = 10$. Không có gì trong dữ liệu chống
   lại chuyện đó.
4. **Tiên nghiệm không chính quy thì evidence KHÔNG ĐƯỢC ĐỊNH NGHĨA.** Vậy không tránh được
   việc phải chọn một tiên nghiệm cụ thể. Một tiên nghiệm "không thông tin" kiểu
   $p(w)\propto 1$ làm cả mục 3.5 của sách vô nghĩa.
5. **Mục 3.5 của sách KHÔNG phải Bayes đầy đủ.** Nó thay $\alpha$, $\beta$ bằng hai ước lượng điểm, và
   điều kiện để làm thế là hậu nghiệm trên $(\alpha,\beta)$ phải nhọn — điều không đúng khi $N$
   nhỏ. Đúng thứ mà phương pháp quảng cáo là điểm mạnh, ở đây bị bỏ đi một phần.
6. **Chi phí $O(M^3)$ cho mỗi lần cập nhật.** Với $M$ lớn thì phép nghịch đảo $\mathbf S_N$ và
   phân rã trị riêng của (3.87) thành chỗ đắt nhất. Bishop đề nghị (3.98)/(3.99) làm đường
   rẻ, và bảng ở mục 2.29 đo được cái giá của đường rẻ đó.

**Và một giới hạn thuộc về cả chương, nêu ở mục 3.6 của sách:** hàm cơ sở phải **cố định trước khi xem
dữ liệu**. Với $D$ chiều đầu vào, số hàm cơ sở định xứ cần thiết tăng theo hàm mũ của $D$.

---

## 7. Họ hàng

| Phương pháp | Quan hệ với bài này |
|---|---|
| [Linear Regression](<../Linear Regression/Linear Regression.md>) | trường hợp $\alpha \to 0$: hậu nghiệm rộng vô hạn, $\mathbf m_N \to \mathbf w_{\text{ML}}$ |
| [Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) | **chính là** điểm MAP của bài này, với $\lambda = \alpha/\beta$ — xem (3.55) và mục 0.4 |
| [Lasso Regression](<../Lasso Regression/Lasso Regression.md>) | MAP với tiên nghiệm (3.56) ở $q = 1$; mất liên hợp, mất công thức đóng, và **trung bình hậu nghiệm không còn thưa** |
| [Elastic Net](<../Elastic Net/Elastic Net.md>) | MAP với tiên nghiệm trộn $q=1$ và $q=2$ |
| [Polynomial Regression](<../Polynomial Regression/Polynomial Regression.md>) | cùng 10 điểm dữ liệu của mục 1.1 sách Bishop; bài đó dựng lại dữ liệu, bài này dùng nó để dựng lại Hình 3.14 |
| **Gaussian Processes** — database chưa có bài | định nghĩa thẳng nhân $k$ thay vì đi qua $\boldsymbol\phi$ — Bishop dẫn sang nó ngay ở (3.62) và ở cảnh báo tr.158, chương 6 của sách |
| **Bayesian Optimization** — database chưa có bài | dùng chính phân phối dự báo (3.58) làm căn cứ chọn điểm thử tiếp theo |
| [Naive Bayes](<../Naive Bayes Classifier/Naive Bayes Classifier.md>) | cũng mang chữ "Bayes" nhưng khác hẳn: nó giả định độc lập giữa đặc trưng, **không** đặt tiên nghiệm lên tham số |
| [Variational Inference](<../../Statistical Learning/Variational Inference/Variational Inference.md>) | đường thay thế khi hậu nghiệm không có công thức đóng — tức khi $q\ne2$ hoặc mô hình không tuyến tính theo tham số |

**Ba thứ nổi tiếng gắn với bài này mà KHÔNG nằm trong chương 3:**

* **Relevance vector machine** — một $\alpha$ cho **mỗi** hệ số. Bishop hẹn ở chương 7; bài này
  đo nó qua `ARDRegression` ở mục 3: **8/10** hệ số bị đẩy về 0.
* **Phân phối dự báo Student-t** — khi coi cả $\beta$ là chưa biết và dùng tiên nghiệm
  Gauss-gamma. Bishop nhắc ở tr.158 nhưng **không** làm.
* **Thuật toán EM cho evidence** — Bishop nói ở tr.166 rằng nó hội tụ về cùng nghiệm với
  (3.92)/(3.95), nhưng để sang mục 9.3.4 của sách.

---

## 8. Khi nào dùng, khi nào đừng

**Dùng khi:**

* **Cần một thanh sai số cho từng dự báo**, không chỉ một điểm. Đây là lý do số một, và không
  có cách nào lấy nó từ bình phương tối thiểu.
* **Dữ liệu ít so với số tham số.** Ở $N = 10$ với $M = 10$, kiểm định chéo phải huấn luyện
  trên 8 điểm mỗi lần và cho $\lambda$ trải **3,4 lần**; evidence dùng cả 10 điểm và cho một
  giá trị.
* **Có nhiều hơn một siêu tham số phải chọn.** Với hai siêu tham số thì lưới kiểm định chéo đã
  là bình phương; với một $\alpha$ cho mỗi hệ số thì không quét được. Evidence thì chỉ là một
  phép lặp.
* **Dữ liệu đến dần và phải cập nhật trực tuyến.** (3.51) cho phép cập nhật chính xác, không
  xấp xỉ.
* **Cần chọn giữa vài mô hình có sẵn** mà không muốn tách tập thẩm định — (3.66) và Bayes
  factor làm được việc đó.

**Đừng dùng khi:**

* **Không có ý kiến gì về tiên nghiệm và không muốn có.** Evidence **không tồn tại** với tiên
  nghiệm không chính quy, và nó nhạy với bề rộng tiên nghiệm tới mức $2^M$. Nếu không có cơ sở
  nào để chọn bề rộng đó, kiểm định chéo trung thực hơn.
* **Phải ngoại suy.** Cả hai loại hàm cơ sở đều hỏng, chỉ hỏng theo hai chiều ngược nhau: định
  xứ thì $\sigma$ tụt về một sàn ($1{,}0652$ lần so với trong vùng dữ liệu), đa thức thì
  $\sigma$ bùng nổ ($63\,894{,}7$ lần). Chỗ này dùng quá trình Gauss, đúng như Bishop chỉ ở tr.158.
* **Nghi mô hình sai dạng.** $\sigma_N(x)$ đo độ bất định **trong** mô hình, không đo khả năng
  mô hình sai. Bishop tự khuyên giữ một tập kiểm tra độc lập; đó là lời khuyên phải nghe.
* **Đầu vào nhiều chiều và không có cấu trúc.** Mục 3.6 của sách: số hàm cơ sở cần thiết tăng theo hàm
  mũ của $D$. Chỗ này dùng mạng nơ-ron, tức **học** $\boldsymbol\phi$ — xem [Multi‑Layer Perceptron (MLP)](<../../Deep Learning/Multi‑Layer Perceptron (MLP)/Multi‑Layer Perceptron (MLP).md>).
* **Cần nghiệm THƯA.** Tiên nghiệm Gauss co mọi hệ số nhưng không đặt hệ số nào về đúng 0.
  Tiên nghiệm Laplace cho MAP thưa, nhưng khi đó mất công thức đóng — và **trung bình hậu
  nghiệm vẫn không thưa**. Chỗ này dùng [Lasso](<../Lasso Regression/Lasso Regression.md>) hoặc RVM.
* **$M$ rất lớn.** $O(M^3)$ cho mỗi vòng lặp, cộng một phân rã trị riêng cho (3.87).

**Quy trình dùng thật, sáu bước:**

1. Chuẩn hoá đầu vào, rồi chọn họ hàm cơ sở và **số** hàm cơ sở. Đây là quyết định lớn nhất
   và evidence **không** chọn được nó nếu $\mu_j$, $s$ chưa cố định.
2. Khởi tạo $\alpha = \beta = 1$, chạy (3.92) + (3.95) tới khi hội tụ. Trên bộ 25 điểm việc đó
   mất **12 vòng**; thử vài điểm xuất phát để chắc không có cực trị thứ hai.
3. **Báo cáo $\gamma$**, không chỉ $M$. $\gamma = 5{,}958765$ trên $M = 10$ nói nhiều hơn hẳn
   con số 10.
4. Kiểm $1/\sqrt{\widehat\beta}$ với hiểu biết của mình về nhiễu phép đo. Nếu nó nhỏ hơn hẳn
   mức nhiễu ta biết là có thật, mô hình đang hút nhiễu vào hệ số.
5. Vẽ $\sigma_N(x)$ **tách làm hai số hạng** của (3.59) trên miền quan tâm — xem Hình 49. Nếu
   phần tham số áp đảo thì thu thêm dữ liệu; nếu phần nhiễu áp đảo thì đổi phép đo.
6. **Vẫn giữ một tập kiểm tra độc lập.** Đây là câu của chính Bishop ở tr.165, và mục 2.30 đo
   được lý do: $\alpha$ evidence chọn lệch $\alpha$ tốt nhất **2,9154 lần**.

---

## 9. Đối chiếu với nguồn

Ba bảng, ba câu hỏi khác nhau: *đã lấy hết HÌNH chưa* · *đã lấy hết CÔNG THỨC đánh số chưa* ·
*đã lấy hết NỘI DUNG chưa*. Một bài có thể lấy đủ 100 % hình mà vẫn bỏ sót cả một mục, vì phần
lớn lập luận của một nguồn nằm trong văn xuôi.

### 9.1 Theo HÌNH và BẢNG — cả 17 hình của chương 3

Chương 3 có **17 hình và 0 bảng**, đếm bằng `v_bayes.py` mục 2 trên lớp text của PDF.

| Hình | Trang | Nhóm | Bài dùng ở đâu | Nếu không lấy thì vì sao |
|---|---:|---|---|---|
| 3.1 ba họ hàm cơ sở | 140 | **RUỘT** (bị chú thích 3.10 và 3.11 trỏ vào) | mục 2.1 | — |
| 3.2 hình học của bình phương tối thiểu | 143 | NGOÀI | — | thuộc mục 3.1.2 của sách, phần ruột của bài [Linear Regression](<../Linear Regression/Linear Regression.md>); không công thức nào của (3.48)–(3.99) dùng nó |
| 3.3 đường đồng mức của chính quy hoá $q$ | 145 | NGOÀI | nhắc ở mục 2.7 | hình của (3.29); bài [Lasso Regression](<../Lasso Regression/Lasso Regression.md>) dùng nó làm ruột |
| 3.4 ràng buộc ridge so với lasso | 146 | NGOÀI | nhắc ở mục 2.7 | như trên |
| 3.5 phân rã chệch–phương sai | 149 | NGOÀI | — | thuộc mục 3.2 của sách; Bishop tự nói nó *"is of limited practical value"* và mục 3.3 của sách mở ra để thay nó |
| 3.6 chệch–phương sai theo $\ln\lambda$ | 150 | NGOÀI | — | như trên |
| **3.7 học Bayes tuần tự** | 155 | **RUỘT** | mục 2.6 | — |
| **3.8 phân phối dự báo** | 157 | **RUỘT** | mục 2.9 | — |
| **3.9 mẫu từ hậu nghiệm** | 158 | **RUỘT** | mục 2.10 | — |
| **3.10 nhân tương đương Gauss** | 159 | **RUỘT** | mục 2.12 | — |
| **3.11 nhân tương đương đa thức và sigmoid** | 160 | **RUỘT** | mục 2.13 | — |
| **3.12 xấp xỉ thô cho evidence** | 163 | **RUỘT** | mục 2.19 | — |
| **3.13 phân phối của các bộ dữ liệu** | 164 | **RUỘT** | mục 2.20 | — |
| **3.14 evidence theo bậc** | 168 | **RUỘT** | mục 2.24 | — |
| **3.15 hợp lẽ và tiên nghiệm trong hệ trục riêng** | 170 | **RUỘT** | mục 2.27 | — |
| **3.16 $\gamma$, evidence và sai số kiểm tra theo $\ln\alpha$** | 172 | **RUỘT** | mục 2.30 | — |
| **3.17 mười hệ số theo $\gamma$** | 172 | **RUỘT** | mục 2.31 | — |
| **Bảng số** | — | — | — | **chương 3 KHÔNG có bảng nào** — xem mục 0.5 |

Cộng: **11 hình nhóm ruột, và bài dùng 11 — 100 %.** Sáu hình nhóm ngoài đều đã ghi lý do.

### 9.2 Theo CÔNG THỨC đánh số — cả 52 công thức của mục 3.3–3.5 của sách

| Công thức | Nội dung | Bài dùng ở đâu |
|---|---|---|
| (3.48) | tiên nghiệm liên hợp tổng quát | mục 2.2 |
| (3.49) (3.50) (3.51) | hậu nghiệm: dạng, trung bình, hiệp phương sai | mục 2.3 |
| (3.52) (3.53) (3.54) (3.55) | tiên nghiệm đẳng hướng và cầu nối sang ridge | mục 2.5 |
| (3.56) | tiên nghiệm tổng quát bậc $q$ | mục 2.7 |
| (3.57) (3.58) (3.59) | phân phối dự báo và phương sai của nó | mục 2.8 |
| (3.60) (3.61) (3.62) | nhân tương đương | mục 2.12 |
| (3.63) | hiệp phương sai giữa hai dự báo | mục 2.14 |
| (3.64) | ràng buộc tổng của nhân | mục 2.15 |
| (3.65) | nhân viết thành tích trong | mục 2.16 |
| (3.66) | hậu nghiệm trên mô hình | mục 2.18 |
| (3.67) | dự báo là hỗn hợp trên mô hình | mục 2.18 |
| (3.68) (3.69) (3.70) | evidence, định lý Bayes, xấp xỉ bề rộng | mục 2.19 |
| (3.71) (3.72) | xấp xỉ thô, một tham số và $M$ tham số | mục 2.19 |
| (3.73) | Bayes factor kỳ vọng là một phân kỳ KL | mục 2.20 |
| (3.74) (3.75) | lấy biên đầy đủ, và xấp xỉ bằng ước lượng điểm | mục 2.22 |
| (3.76) | hậu nghiệm trên siêu tham số | mục 2.22 |
| (3.77) (3.78) | marginal likelihood và dạng viết ra đủ hằng chuẩn hoá | mục 2.23 |
| (3.79) (3.80) (3.81) (3.82) (3.83) (3.84) | hàm lỗi, gom bình phương đúng, Hesse | mục 2.23 |
| (3.85) (3.86) | tích phân Gauss và công thức evidence | mục 2.23 |
| (3.87) (3.88) (3.89) | phương trình trị riêng và điểm dừng theo $\alpha$ | mục 2.25 |
| (3.90) (3.91) (3.92) | $\gamma$ và phép lặp cho $\alpha$ | mục 2.25 |
| (3.93) (3.94) (3.95) | điểm dừng và phép lặp cho $\beta$ | mục 2.26 |
| (3.96) (3.97) | phép loại suy với ước lượng phương sai một biến | mục 2.28 |
| (3.98) (3.99) | xấp xỉ khi $N \gg M$ | mục 2.29 |

**52 / 52.** Và bảy công thức nền của mục 3.1 của sách được nhắc lại ở mục 2.1 theo Luật 1: (3.3),
(3.4), (3.8), (3.10), (3.12), (3.25), (3.27).

Hai mươi công thức (3.100)–(3.119) nằm ở **phần bài tập** của chương, không phải thân bài; bài
này không lấy chúng, và (3.100)/(3.101)/(3.102) thuộc bài tập 3.1 về quan hệ giữa `tanh` và
sigmoid, tức về hàm cơ sở chứ không về hồi quy Bayes.

### 9.3 Theo MỤC — cả 17 mục đánh số của chương 3

| Mục của Bishop | Tên | Bài dùng ở đâu | Nếu không lấy thì vì sao |
|---|---|---|---|
| 3.1 | Linear Basis Function Models | mục 2.1 (nhắc lại (3.3), (3.4)) | phần ruột của bài [Linear Regression](<../Linear Regression/Linear Regression.md>) |
| 3.1.1 | Maximum likelihood and least squares | mục 2.1 (nhắc lại (3.8), (3.10), (3.12)) | như trên |
| 3.1.2 | Geometry of least squares | — | hình học của phép chiếu; không công thức nào của phần ruột dùng nó |
| 3.1.3 | Sequential learning | nhắc ở mục 2.4 | đó là gradient trực tuyến cho **ước lượng điểm**; bản Bayes tuần tự của mục 2.4 mạnh hơn và chính xác |
| 3.1.4 | Regularized least squares | mục 2.1 ((3.25), (3.27)) và mục 2.7 ((3.56)) | phần ruột của bài [Ridge](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) và [Lasso](<../Lasso Regression/Lasso Regression.md>) |
| 3.1.5 | Multiple outputs | — | Bishop tự nói phần mở rộng sang nhiều biến mục tiêu là *"straightforward"*; mục 3.3 của sách tuyên bố chỉ xét một biến mục tiêu |
| 3.2 | The Bias-Variance Decomposition | nhắc ở mục 2.17 | góc nhìn tần suất; Bishop kết mục đó bằng câu nó *"is of limited practical value"* rồi mở mục 3.3 của sách để thay |
| **3.3** | **Bayesian Linear Regression** | **mục 2.2 – 2.5** | — |
| **3.3.1** | **Parameter distribution** | **mục 2.2 – 2.7** | — |
| **3.3.2** | **Predictive distribution** | **mục 2.8 – 2.11** | — |
| **3.3.3** | **Equivalent kernel** | **mục 2.12 – 2.16** | — |
| **3.4** | **Bayesian Model Comparison** | **mục 2.17 – 2.21** | — |
| **3.5** | **The Evidence Approximation** | **mục 2.22** | — |
| **3.5.1** | **Evaluation of the evidence function** | **mục 2.23 – 2.24** | — |
| **3.5.2** | **Maximizing the evidence function** | **mục 2.25 – 2.27** | — |
| **3.5.3** | **Effective number of parameters** | **mục 2.28 – 2.31** | — |
| **3.6** | **Limitations of Fixed Basis Functions** | **mục 2.32** | — |

Cộng: **10 mục thuộc phần ruột (3.3 tới 3.6), và bài lấy cả 10 — 100 %.** Bảy mục của 3.1–3.2
đều có lý do, và bốn trong bảy vẫn được nhắc lại ở mục 2.1 vì Luật 1.

### 9.4 Năm thứ KHÔNG kiểm được, và vì sao

Phần này để ra ngoài mọi bảng trên, vì một hạn chế không viết ra thì lần sau sẽ bị đọc thành
lời hứa.

1. **Hình 3.7 và Hình 3.10 là ẢNH RASTER.** `v_bayes.py` mục 3 đếm được 0 đường vectơ trên
   trang 155 và trang 159. Với hai hình đó bài chỉ dựng lại được **thiết lập** — và với Hình
   3.7 thì thiết lập đầy đủ nên dựng lại được có nghĩa; với Hình 3.10 thì $\mu_j$ và $s$ không
   khai nên chỉ kiểm được **tính định xứ**, không kiểm được biên độ.
2. **Hình 3.16 không có TRỤC DỌC.** Lớp text của trang 172 chỉ chứa `ln α` và `−5 0 5`. Ba
   hoành độ đọc được đã dùng hết ở mục 2.30; giá trị của $\gamma$, của $2\alpha E_W$, của log
   evidence và của sai số kiểm tra thì **không** đọc được. Đây là hạn chế không vá được: không
   có thang thì không có số.
3. **Hình 3.9 vẽ MẪU NGẪU NHIÊN.** Số mẫu, hạt giống, cách chọn — bài báo không khai, và ngay
   cả khi khai thì mẫu rút ra vẫn phụ thuộc bộ sinh số. Bài này không trích một con số nào từ
   Hình 3.9; đại lượng tất định tương ứng là (3.63), và nó được kiểm ở mục 2.14.
4. **Hình 3.11, 3.12, 3.13, 3.15 không có con số nào để đối chiếu.** Ba hình sau là **sơ đồ**,
   không có thang; Hình 3.11 có thang nhưng bốn tham số của mô hình vẽ nó ($\mu_j$, $s$,
   $\alpha$, $\beta$) đều không khai, nên chỉ kiểm được hình dạng.
5. **Hai phép đọc cho bề rộng $s$ không trùng nhau.** Hình 3.8 cho $s = 0{,}1230$, Hình 3.17
   cho $s = 0{,}1150$, lệch **6,5 %** — mà phép đo của Hình 3.17 sắc tới mức phân biệt được
   $0{,}115$ với $0{,}120$. Vậy hoặc hai hình dùng hai thiết lập hơi khác, hoặc một trong hai
   phép đọc có một độ lệch hệ thống. Bài này ghi cả hai con số kèm nguồn thay vì chọn một.

Và một hạn chế về **cách trích dẫn**, phải nói vì nó áp cho mọi câu nguyên văn trong bài: các
đoạn tiếng Anh được `tools/kiemtrich.py` đối chiếu với lớp text của PDF. PDF này có lớp text đầy
đủ, nên phép kiểm đó chạy được — khác với bài [Huber Regression](<../Huber Regression/Huber Regression.md>) và
[Quantile Regression](<../Quantile Regression/Quantile Regression.md>), nơi nguồn là bản quét và không đoạn trích nào
kiểm được tự động.

---

## 10. Tài liệu tham khảo

```
[1] C. M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006.
    -> NGUỒN RUỘT. Chương 3 "Linear Models for Regression", sách tr.137-177:
       muc 3.3 tr.152-161 · muc 3.4 tr.161-165 · muc 3.5 tr.165-172 · muc 3.6 tr.172-173.
       52 cong thuc danh so (3.48)-(3.99); 17 hinh; 0 bang.
    AI algorithms/_papers/2006 - Bishop - Pattern Recognition and Machine Learning.pdf

[2] D. J. C. MacKay, "Bayesian Interpolation", Neural Computation 4(3):415-447, 1992.
    -> Nguon cua khuon kho evidence, Bishop dan o tr.165 va tr.170 (MacKay, 1992a).
       KHONG tai duoc ban truy cap mo hop phap; bai nay khong trich mot con so nao tu no,
       va moi cong thuc evidence deu lay tu chuong 3 cua [1].

[3] S. F. Gull, "Bayesian data analysis: straight-line fitting", trong Maximum Entropy and
    Bayesian Methods, Kluwer, 1989.
    -> Nguon thu hai cua khuon kho evidence, Bishop dan o tr.165 (Gull, 1989).
       KHONG tai duoc ban truy cap mo hop phap.

[4] C. S. Qazaz, C. K. I. Williams, C. M. Bishop, "An upper bound on the Bayesian error bars
    in generalized linear regression", trong Mathematics of Neural Networks, 1997.
    -> Nguon cua khang dinh sigma^2_{N+1}(x) <= sigma^2_N(x) ma Bishop dan o tr.156.
       KHONG tai duoc ban truy cap mo hop phap; bai nay kiem khang dinh do bang so
       (v_bayes.py muc 10: 15 336 phep so, 0 cho vi pham) chu khong trich chung minh.

[5] R. Tibshirani, "Regression shrinkage and selection via the lasso", J. R. Statist. Soc. B
    58(1):267-288, 1996.
    -> Bishop dan o tr.145 cho truong hop q = 1 cua cong thuc (3.56).
       Ban PDF khong truy cap mo; xem bai Lasso Regression, muc tai lieu tham khao.

[6] Tai lieu scikit-learn 1.9.1, sklearn.linear_model.BayesianRidge va ARDRegression.
    -> Doi chieu o muc 3. Ba cho khac so voi (3.92)/(3.95) deu do bang so.
```

**Hình gốc.** 48 khung được cắt từ `2006 - Bishop - Pattern Recognition and Machine
Learning.pdf` bằng `tools/pdffig.py` với cờ `--sat`, ở 300 DPI. Khung được suy ra **từ lớp
text** của PDF chứ không gõ tay: `tools/_work_bayes/cat.py` tìm vị trí hai chuỗi neo trên trang
rồi lấy dải $y$ giữa chúng, và `--sat` nới ra tới dải trắng gần nhất để không cắt ngang thân
chữ. Danh sách đầy đủ 48 khung nằm trong `tools/_work_bayes/catall.py`.

**Hình tự vẽ.** Ba hình — Hình 49, Hình 50, Hình 51 — do ba khối mã in ngay trong bài sinh ra,
mỗi khối tự chứa, dán sẵn mảng số và kết thúc bằng **đúng một** `savefig()`. Bản trong `tools/`
là `tools/fig_bayes.py`, sinh ra **từ chính bài** bằng
`python extract_fig.py "Bayesian Linear Regression" fig_bayes.py`, nên hai bản giống nhau theo
cấu tạo.

**Số liệu.** Mọi con số của bài do `tools/v_bayes.py` in ra — 24 mục, chạy dưới nửa phút, chỉ
cần `numpy` và `pypdfium2` (cộng `scikit-learn` cho mục 22). Script ghim số luồng BLAS bằng biến
môi trường **trước** khi `import numpy`, vì thứ tự cộng dồn dấu phẩy động đổi theo số luồng.
Thời gian chạy **không** được ghi vào bài: nó là con số đồng hồ, chỉ tái lập trong đúng một lần
chạy.

---

## Kiểm tra trước khi đóng bài

- [x] **Bảng ký hiệu có cột "Đọc là gì"** — mục 0.2, 27 dòng, kèm **bốn chỗ trùng tên NGAY
      TRONG chương 3** ($\lambda$ phạt so với $\lambda_i$ trị riêng, cách nhau 24 trang ·
      $\sigma$ sigmoid so với $\sigma_N^2$ phương sai dự báo · $M$ số chiều so với $M$ bậc đa
      thức, cách nhau 1 trang · $\phi_j$ hàm so với $\boldsymbol\phi(x_n)$ hàng so với
      $\boldsymbol\varphi_j$ cột, cùng trang 143) và **hai chỗ trùng với bài khác** ($\beta$ là
      vectơ hệ số ở Ridge/Lasso/Quantile · `alpha_` của `BayesianRidge` **là $\beta$** của
      Bishop)
- [x] **Cả 52 công thức đánh số (3.48)–(3.99) đều có mặt** — bảng đối chiếu mục 9.2; mỗi công
      thức có công thức kèm số hiệu, bảng ký hiệu kèm chiều, và một đoạn **ý nghĩa trừu tượng**
- [x] **Cả 11 hình nhóm ruột đều được cắt vào bài** — bảng mục 9.1; 6 hình nhóm ngoài ghi rõ lý
      do; mỗi hình ruột có caption gốc nguyên văn, **nhận xét nguyên văn của chính tác giả** kèm
      số trang, bản dịch, và phần "Cách đọc hình" kèm điều hình KHÔNG nói ra
- [x] **Cả 10 mục đánh số của phần ruột (3.3–3.6) đều có mặt** — bảng mục 9.3; 7 mục của
      3.1–3.2 ghi rõ lý do, và 4 trong 7 vẫn được nhắc lại ở mục 2.1 theo Luật 1
- [x] **Ví dụ chạy tay bằng PHÂN SỐ ĐÚNG** — mục 0.4, năm bước, $\mathbf m_N =
      \left[\tfrac{38}{41},\tfrac{24}{41}\right]$, và đẳng thức MAP $=$ ridge kiểm bằng phân số
      chứ không bằng số thực
- [x] **Dựng lại 25 điểm dữ liệu mà nguồn KHÔNG in** — mục 2.9, đọc lớp vectơ của Hình 3.8;
      hoành độ lệch khỏi `linspace(0,1,25)` nhiều nhất 0,000348, và chính phép so đó xác nhận
      phép quy đổi trục
- [x] **Ba tham số nguồn không khai được dò ra, kèm PHÉP THỬ NGƯỢC** — mục 2.9: đổi một tham số
      làm sai số xấu đi từ 3,6 tới 39,8 lần
- [x] **Ba hình độc lập cùng chỉ một bề rộng $s$** — mục 2.31; và chỗ hai phép đọc **không**
      trùng nhau (0,1230 so với 0,1150, lệch 6,5 %) được ghi ra ở mục 9.4 thay vì chọn một
- [x] **Phát hiện chính được phân xử bằng SỐ, hai chiều** — mục 2.24: bản thiếu $(2\pi)^{M/2}$
      khớp hơn 12,75 lần, **và** hai tham số nó dò ra khớp đúng hai con số bài báo tự in; bản
      đúng (3.86) đòi một $\alpha$ lệch con số của bài báo 21,8 lần
- [x] **Hai chỗ khẳng định của nguồn chỉ đúng ở một giới hạn, cả hai có phép thử ngược** —
      (3.64) chỉ đúng khi $\alpha\to0$ (mục 2.15); câu tr.158 về ngoại suy chỉ đúng khi bỏ hàm
      cơ sở hằng (mục 2.11, phép thử ngược cho $\sigma = 0{,}300150$ đúng tới từng chữ số)
- [x] **Khối SS nêu bế tắc bằng SỐ, không bằng tính từ** — mục 1: ba tính từ của chính Bishop
      (*excessively complex* · *computationally expensive* · *wasteful*) đổi thành ba phép đo
- [x] **Khối GIÁ nằm ở thân bài** — mục 6, bảng bốn cột kèm nguồn số, và sáu chỗ hỏng xếp theo
      mức nguy hiểm, đứng đầu là "tự tin sai khi mô hình sai"
- [x] **Nói rõ mục 3.5 của sách KHÔNG phải Bayes đầy đủ** — mục 2.22 và mục 6 chỗ hỏng số 5
- [x] **Ba hình tự vẽ, mỗi hình một khối mã tự chứa, mỗi khối đúng một `savefig()`** — Hình 49,
      49, 50; mọi mảng số dán từ `v_bayes.py` mục 23, in đủ chữ số
- [x] **Không con số nào của bài nằm ngoài `stdout` của `v_bayes.py` hoặc của ba khối mã** —
      kiểm bằng `tools/_work_bayes/ngat.py`
- [x] **Thời gian chạy script KHÔNG ghi vào bài** — mục 0.5 và mục 10 nói rõ vì sao
- [x] Mọi đoạn tiếng Anh nguyên văn đối chiếu được với lớp text của PDF — `kiemtrich.py`
- [ ] **Chưa kiểm được: bốn tham số của Hình 3.1, 3.10, 3.11.** Chú thích Hình 3.1 không cho
      $\mu_j$, $s$, bậc đa thức hay số hàm cơ sở, nên nhân tương đương ở mục 2.13 chỉ đối chiếu
      được hình dạng. Muốn đóng ô này phải tìm được mã nguồn sinh hình của sách, và nó không
      công bố
- [ ] **Chưa kiểm được: trục dọc của Hình 3.16.** Hình không in một con số nào trên trục đứng,
      nên bốn đường của nó chỉ đọc được ba hoành độ. Đây là hạn chế **không vá được**
      — xem mục 9.4 điểm 2
