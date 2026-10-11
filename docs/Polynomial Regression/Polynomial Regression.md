# Polynomial Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 370
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Joseph Diez Gergonne
Date: January 1, 1815
url: http://www.numdam.org/item/AMPA_1815-1816__6__242_0/

> **Polynomial Regression — Hồi quy đa thức.** Cho một cột số đo $x$ và một cột giá trị $y$,
> tìm đa thức $\hat y = \beta_0 + \beta_1 x + \dots + \beta_d x^d$ sao cho tổng bình phương
> phần dư nhỏ nhất. Về máy móc, nó **không phải thuật toán mới**: nó là hồi quy tuyến tính chạy
> trên một bảng đã thêm cột $x^2, x^3, \dots$ Thứ đáng học nằm ở chỗ khác — **vì sao lại dùng ÍT
> số hạng hơn số điểm dữ liệu**, và **chọn bao nhiêu số hạng**.
>
> Bài này theo hai nguồn gốc. Nguồn thứ nhất là bài báo 11 trang của **J. D. Gergonne năm 1815**
> — bài đầu tiên đem phương pháp bình phương tối thiểu vào bài toán khớp đa thức, và nó để lại
> đúng câu hỏi mà cả ngành học máy vẫn đang trả lời: *bậc nào?* Nguồn thứ hai là **mục 1.1 sách
> của C. M. Bishop năm 2006**, nơi hồi quy đa thức được dùng làm ví dụ mở đầu cho khái niệm khớp
> quá mức. Mọi ví dụ số của hai nguồn đều được dựng lại bằng
> [`tools/v_poly.py`](<../../../tools/v_poly.py>), và việc dựng lại tìm ra **bảy** chỗ in sai, nói
> quá hoặc tự bất đồng trong hai nguồn (bảng đầu mục 3).

---

## 0. Bắt đầu từ đây

### 0.1 Nhận gì, trả gì

Hồi quy đa thức nhận **đúng những gì hồi quy tuyến tính nhận**, cộng thêm một con số do người
dùng chọn: bậc $d$. Toàn bộ chỗ khác nhau nằm ở một bước biến đổi bảng dữ liệu.

| | Cách hiểu thông thường | Ký hiệu | Kích thước ở ví dụ dưới |
|---|---|---|---|
| **Đầu vào** | một cột số đo | $x\in\mathbb{R}^{n}$ | $4$ số |
| **Siêu tham số** | đường cong được phép "cong tới mức nào" | $d\in\{0,1,2,\dots\}$ | $d=2$ |
| **Bước biến đổi** | mỗi số $x_i$ nở thành một hàng $[1,x_i,\dots,x_i^{d}]$ | $V\in\mathbb{R}^{n\times(d+1)}$ | $4\times3$ |
| **Đáp án khi huấn luyện** | một giá trị số cho mỗi mẫu | $y\in\mathbb{R}^{n}$ | $4$ số |
| **Đầu ra của việc học** | một hệ số cho mỗi cột của $V$ | $\hat\beta\in\mathbb{R}^{d+1}$ | $3$ số |
| **Dự đoán** | nhân từng hàng của $V$ với các hệ số | $\hat y=V\hat\beta$ | $4$ số |

Một ví dụ chạy được, copy ra là chạy:

```python
import numpy as np

x = np.array([-1.0, 0.0, 1.0, 2.0])
y = np.array([ 3.0, 1.0, 2.0, 9.0])
d = 2

V = np.vander(x, d + 1, increasing=True)   # cột 1, x, x²
beta = np.linalg.lstsq(V, y, rcond=None)[0]
y_hat = V @ beta

print("beta =", np.round(beta, 2))
print("y_hat =", np.round(y_hat, 2))
```

```text
beta = [ 0.55 -0.35  2.25]
y_hat = [3.15 0.55 2.45 8.85]
```

`increasing=True` xếp cột theo thứ tự $1, x, x^2$, nên `beta[0]` là hệ số chặn. Nhìn bằng số cho
chắc:

```
x  = [-1,  0,  1,  2]            shape (4,)
V  = [[1, -1,  1],               shape (4, 3)   ← cột 1, cột x, cột x²
      [1,  0,  0],
      [1,  1,  1],
      [1,  2,  4]]
β̂  = [0.55, -0.35, 2.25]         shape (3,)
ŷ  = V @ β̂ = [3.15, 0.55, 2.45, 8.85]
```

Ba điều dễ hiểu sai, nói rõ ngay:

- **Đầu vào vẫn là $x$, không phải $V$.** Người dùng đưa vào một cột số; $V$ là thứ thuật toán
  tự dựng. Lúc dự đoán cho một $x$ mới, phải dựng lại $V$ theo **đúng** bậc và đúng cách chuẩn hoá
  đã dùng lúc huấn luyện.
- **$d$ không phải thứ học được.** Nó do người dùng chọn, và chọn sai là nguồn của cả hai kiểu hỏng
  ở mục 6. Cách chọn ở mục 7.2.
- **Số cột là $d+1$, không phải $d$** — cột đầu toàn số 1 dùng cho hệ số chặn.

### 0.2 Từ mới trong bài này

| Từ | Trong bài này | Nghĩa |
|---|---|---|
| **Mẫu** (sample) | một cặp $(x_i,y_i)$ | một dòng dữ liệu |
| **Đặc trưng** (feature) | các cột $1,x,x^2,\dots$ | thông tin dùng để dự đoán |
| **Bậc** $d$ (degree, order) | $d=2$ ở ví dụ trên; Bishop gọi là $M$ | số mũ lớn nhất được đưa vào |
| **Ma trận thiết kế** (design matrix) | $V$ | bảng đã biến đổi mà phép khớp thật sự chạy trên đó |
| **Vandermonde** | chính $V$ | tên riêng của ma trận có các cột là luỹ thừa của cùng một biến |
| **Phần dư** (residual) | $e_i=y_i-\hat y_i$ | giá trị thật trừ dự đoán |
| **Bình phương tối thiểu** (least squares) | *moindres quarrés* ở Gergonne | chọn hệ số sao cho tổng bình phương phần dư nhỏ nhất |
| **Phương trình chuẩn** (normal equations) | hệ (1) tr.250 của Gergonne | hệ phương trình tuyến tính mà nghiệm bình phương tối thiểu phải thoả: $V^\top V\beta = V^\top y$ |
| **Nội suy** (interpolation) | *interpolation des suites* | tìm đường cong đi qua **đúng** mọi điểm cho trước — số hệ số bằng số điểm |
| **Hoành độ, tung độ** (abscissa, ordinate) | *abscisse*, *ordonnée* | giá trị trên trục ngang ($x$) và trục dọc ($y$) của một điểm |
| **Cách đều** (equidistant) | *progression par différences* | các giá trị $x$ cách nhau một khoảng không đổi |
| **Hệ số vi phân** (differential coefficient) | *coefficient différentiel* | tên thế kỷ 19 của đạo hàm: $dy/dx$ là hệ số vi phân bậc một, $d^2y/dx^2$ là bậc hai |
| **RMSE** | $\sqrt{\frac1n\sum e_i^2}$; Bishop gọi là $E_{\text{RMS}}$ | sai số trung bình, cùng đơn vị với $y$ — càng nhỏ càng khớp |
| **Tập huấn luyện / kiểm định / kiểm tra** | *training / validation (hold-out) / test* | dữ liệu dùng để tìm hệ số · để chọn bậc · để báo cáo một lần ở cuối |
| **Khớp thiếu** (underfit) | đường thẳng trên dữ liệu cong | mô hình quá cứng, bỏ sót cấu trúc thật |
| **Khớp quá mức** (over-fitting) | bậc 9 qua 10 điểm ở Hình 15 | mô hình học cả dao động ngẫu nhiên và dự đoán dữ liệu mới kém |
| **Chệch** (bias) | sai số còn lại khi không có nhiễu | phần sai do mô hình **thiếu hình dạng** — thêm dữ liệu không chữa được |
| **Phương sai** (variance) | sai số do nhiễu trong dữ liệu | phần sai do mô hình **quá nhạy** — thêm dữ liệu chữa được |
| **Chuẩn hoá / phạt Ridge** (regularization) | số hạng $\frac{\lambda}{2}\lVert w\rVert^2$ ở (1.4) | cộng thêm vào hàm lỗi một phần phạt hệ số lớn, để hệ số không vọt lên hàng triệu |
| **Số điều kiện** $\kappa$ (condition number) | $\kappa(V)$ ở mục 6.2 | con số đo mức khuếch đại sai số làm tròn; càng lớn càng ít chữ số đáng tin |
| **BIC** | mục 7.2 | một điểm số chọn mô hình: khớp tốt thì điểm giảm, thêm hệ số thì bị cộng phạt $\ln n$ mỗi hệ số |
| **Kiểm tra chéo** (cross-validation) | 5 mảnh ở mục 7.2 | chia tập huấn luyện làm $K$ phần, lần lượt giữ một phần ra để đo, rồi lấy trung bình |

### 0.3 Bảng ký hiệu — và chỗ trùng tên giữa ba nguồn

Bài dùng ký hiệu hiện đại ($x, y, \beta, d, n$). Gergonne 1815 và Bishop 2006 dùng hai hệ khác,
và phần 3 trích nguyên văn cả hai, nên bảng này có ba cột nguồn.

| Đại lượng | Bài này | Đọc là gì | Gergonne 1815 | Bishop 2006 | Chiều / kiểu |
|---|---|---|---|---|---|
| số điểm dữ liệu | $n$ | "en" | $2n+1$ (khi cách đều) | $N$ | số nguyên |
| bậc đa thức | $d$ | "đê" | số số hạng trừ 1 | $M$ | số nguyên $\ge0$ |
| giá trị đầu vào | $x_i$ | "ích chỉ số i" | $a, a', a'', \dots$ | $x_n$ | vô hướng |
| giá trị đầu ra | $y_i$ | "y chỉ số i" | $b, b', b'', \dots$ | $t_n$ ("target") | vô hướng |
| hệ số | $\beta_0,\beta_1,\dots$ | "bê-ta" | $A, B, C, D, \dots$ | $w_0, w_1, \dots$ | vô hướng |
| vector hệ số ước lượng | $\hat\beta$ | "bê-ta mũ" | — | $\mathbf w^\star$ | $\mathbb{R}^{d+1}$ |
| ma trận thiết kế | $V$ | "vê" | — (viết từng dòng) | — | $\mathbb{R}^{n\times(d+1)}$ |
| sai số của một tung độ | $\varepsilon_i$ | "ép-xi-lon" | $\beta$ | — | vô hướng |
| sai số mà một đại lượng nhận được | — | — | $\mathrm{E}.\,\frac{dy}{dx}$ | — | toán tử viết tắt |
| hàm lỗi | $\frac12\sum e_i^2$ | — | "somme de leurs quarrés" | $E(\mathbf w)$ | vô hướng $\ge0$ |
| khoảng cách giữa hai $x$ liên tiếp | $h$ | "hát" | $a$ | — | vô hướng $>0$ |
| hệ số phạt | $\lambda$ | "lam-đa" | — | $\lambda$ | vô hướng $\ge0$ |
| độ lệch chuẩn nhiễu | $\sigma$ | "xích-ma" | — | $0{,}3$ ở Phụ lục A | vô hướng $>0$ |
| số điều kiện | $\kappa$ | "kap-pa" | — | — | vô hướng $\ge1$ |

**Toán tử và hằng số:**

| Ký hiệu | Đọc là gì | Nghĩa |
|---|---|---|
| $V^\top$ | "vê chuyển vị" | lật ma trận qua đường chéo; $V$ cỡ $n\times(d+1)$ thì $V^\top$ cỡ $(d+1)\times n$ |
| $x^{\,j}$ | "ích mũ j" | luỹ thừa; quy ước $x^0=1$, nên cột đầu của $V$ toàn số 1 |
| $\Sigma a^m$ | "xích-ma a mũ m" | ký hiệu của Gergonne tr.250: $a^m + a'^m + a''^m + \dots$, tổng luỹ thừa bậc $m$ của mọi hoành độ |
| $\Sigma n^m$ | "xích-ma en mũ m" | ký hiệu của Gergonne tr.251: $1^m + 2^m + \dots + n^m$ |
| $\lVert\mathbf w\rVert^2$ | "chuẩn bình phương của w" | $w_0^2 + w_1^2 + \dots + w_M^2$ |
| $\ln\lambda$ | "lốc tự nhiên của lam-đa" | Bishop ghi $\lambda$ theo thang log: $\ln\lambda=-18$ nghĩa là $\lambda = e^{-18} = 1{,}52\times10^{-8}$ |
| $\varepsilon_{\text{float64}}$ | "ép-xi-lon máy" | $2{,}22\times10^{-16}$; nghịch đảo $4{,}50\times10^{15}$ là vạch đỏ ở Hình 25 |
| $\binom{p+d}{d}$ | "tổ hợp chập d của p cộng d" | số số hạng của đa thức bậc $d$ trên $p$ biến — mục 9.1 |

> **Năm chỗ trùng tên phải nhớ trước khi đọc phần 3.** Chúng gây hiểu sai thật, không chỉ khó
> đọc:
>
> 1. **$\beta$ của Gergonne là SAI SỐ**, $\beta$ của bài này là **HỆ SỐ**. Câu *"l'ordonnée $b'$
>    seule soit en erreur d'une quantité $\beta$"* (tr.246) nói về một lỗi đo, không phải một tham
>    số của mô hình.
> 2. **$a$ của Gergonne mang HAI nghĩa trong cùng một bài**: ở tr.249–251 là **hoành độ**
>    ($a, a', a''$), còn ở tr.247–248 là **khoảng cách** giữa hai hoành độ liên tiếp. Legendre tr.iv
>    lại dùng $a, a', a''$ cho **kinh độ quan sát** — tức cho $y$, không cho $x$.
> 3. **$n$ của Gergonne không phải số điểm**: số điểm là $2n+1$, và $n$ là số điểm ở MỖI BÊN
>    điểm giữa. Bảy quan sát là $n=3$.
> 4. **$t$ của Bishop là giá trị đích** ($t$ = *target*), còn $t$ của Legendre tr.iv là **thời
>    gian**.
> 5. **$E$ mang ba nghĩa**: $\mathrm{E}.\,\frac{dy}{dx}$ của Gergonne là *"sai số của"*; $E(\mathbf
>    w)$ của Bishop là **hàm lỗi**; $E_{\text{RMS}}$ của Bishop là **căn trung bình bình phương**. Và
>    $\sigma$ của bài này (độ lệch chuẩn nhiễu) khác $\sigma_{\min}$, $\sigma_{\max}$ ở mục 6.3
>    (giá trị kỳ dị của ma trận).

### 0.4 Luồng xử lý

```
x (n số) ──► dựng V: cột x⁰, x¹, …, x^d ──► giải bình phương tối thiểu ──► β̂ (d+1 số)
                 │                                │
                 │ (chuẩn hoá x trước,            │ (dùng QR / SVD / lstsq,
                 │  hoặc dùng cơ sở trực giao)    │  KHÔNG nghịch đảo VᵀV)
                 ▼                                ▼
x mới ─────► dựng V mới cùng bậc, cùng chuẩn hoá ─► ŷ = V_mới · β̂
```

Hai mũi tên ngang là toàn bộ thuật toán. Hai nhánh chú thích là hai chỗ người mới hay làm sai:
dựng $V$ mới bằng cách chuẩn hoá khác với lúc huấn luyện, và giải bằng nghịch đảo $V^\top V$ (mục
6.3).

### 0.5 Một vòng tính tay, bằng số thật

Bốn điểm: $x = (-1, 0, 1, 2)$, $y = (3, 1, 2, 9)$. Khớp bậc 2, tức 3 hệ số trên 4 điểm — ít hệ số
hơn số điểm, đúng tinh thần của Gergonne, nên có phần dư thật.

**Bước 1 — lập ma trận Vandermonde và hai tích.**

$$V = \begin{pmatrix}1&-1&1\\1&0&0\\1&1&1\\1&2&4\end{pmatrix}, \qquad
V^\top V = \begin{pmatrix}4&2&6\\2&6&8\\6&8&18\end{pmatrix}, \qquad
V^\top y = \begin{pmatrix}15\\17\\41\end{pmatrix}$$

Mọi ô của $V^\top V$ là một tổng luỹ thừa của $x$ — đúng các $\Sigma a^m$ của Gergonne tr.250 —
nên kiểm được từng ô:

| Ô | Công thức | Tính |
|---|---|---|
| $(1,1)$ | $n$ | 4 |
| $(1,2)$ | $\sum x_i$ | $-1+0+1+2 = 2$ |
| $(1,3) = (2,2)$ | $\sum x_i^2$ | $1+0+1+4 = 6$ |
| $(2,3)$ | $\sum x_i^3$ | $-1+0+1+8 = 8$ |
| $(3,3)$ | $\sum x_i^4$ | $1+0+1+16 = 18$ |

$V^\top y$: $\sum y_i = 15$; $\sum x_iy_i = -3+0+2+18 = 17$; $\sum x_i^2y_i = 3+0+2+36 = 41$.

Mọi phần tử trên cùng một đường chéo phụ của $V^\top V$ đều bằng nhau, vì
$(V^\top V)_{jk} = \sum_i x_i^{\,j+k}$ chỉ phụ thuộc $j+k$. Đó là một phép kiểm nhanh: nếu ô
$(1,3)$ khác ô $(2,2)$ thì phép tính đã sai ở đâu đó.

**Bước 2 — giải hệ $3\times3$ bằng phân số chính xác** (`tools/v_poly.py` mục 10):

$$\hat\beta = \left(\tfrac{11}{20},\ -\tfrac{7}{20},\ \tfrac{9}{4}\right)^\top = (0{,}55;\ -0{,}35;\ 2{,}25)^\top
\;\Longrightarrow\; \hat y = 0{,}55 - 0{,}35x + 2{,}25x^2$$

**Ý nghĩa.** Hệ số của $x^2$ dương và lớn (2,25): đường cong mở lên, khớp với chuyện $y$ giảm rồi
tăng vọt ở $x=2$. Hệ số chặn 0,55 là giá trị dự đoán tại $x=0$, thấp hơn giá trị đo 1 — mô hình
chấp nhận sai ở điểm giữa để khớp tốt hơn toàn bộ bốn điểm.

**Bước 3 — kiểm ba điều kiện trực giao.**

| $x_i$ | $y_i$ | $\hat y_i$ | $e_i$ |
|---|---|---|---|
| −1 | 3 | 3,15 | −0,15 |
| 0 | 1 | 0,55 | +0,45 |
| 1 | 2 | 2,45 | −0,45 |
| 2 | 9 | 8,85 | +0,15 |

- $\sum e_i = -0{,}15+0{,}45-0{,}45+0{,}15 = 0$ (cột $x^0$)
- $\sum e_ix_i = 0{,}15+0-0{,}45+0{,}30 = 0$ (cột $x^1$)
- $\sum e_ix_i^2 = -0{,}15+0-0{,}45+0{,}60 = 0$ (cột $x^2$)

Phần dư trực giao với **cả ba** cột. Đó chính là ba dòng của hệ phương trình chuẩn: Gergonne tr.250
viết chúng ra bằng cách đạo hàm tổng bình phương theo $A, B, C$ rồi cho bằng 0 (mục 3.6).

$\text{RSS} = \tfrac{9}{20} = 0{,}45$, $R^2 = 0{,}9884$, $\kappa(V) = 4{,}54$ — điều kiện rất tốt vì
bậc thấp và $x$ nằm quanh 0.

### 0.6 Điểm dễ nhầm

| Nhầm | Đúng |
|---|---|
| "Đa thức là mô hình phi tuyến nên cần thuật toán tối ưu phi tuyến" | mô hình **tuyến tính theo hệ số**; Bishop tr.5: *"it is a linear function of the coefficients w"*. Nghiệm có dạng đóng |
| "Đi qua đúng mọi điểm là khớp tốt nhất" | đó là **nội suy**, và Gergonne tr.244–246 đo được nó khuếch đại sai số đo; Bishop Hình 15 cho thấy nó dao động dữ dội |
| "Thêm bậc thì luôn tốt hơn" | sai số **huấn luyện** không bao giờ tăng; sai số **dữ liệu mới** thì có đáy rồi tăng (Hình 17, Hình 25) |
| "Hệ số khổng lồ xen kẽ dấu là mô hình học được cấu trúc tinh vi" | Bảng 1.1 của Bishop: bậc 9 có hệ số tới $10^6$ trên dữ liệu nằm trong $[-1, 1]$ — đó là dấu hiệu khớp nhiễu |
| "Chuẩn hoá $x$ là tuỳ chọn" | với $x \approx 2000$ thì $x^6 = 6{,}4\times10^{19}$ — tràn ý nghĩa ngay ở bậc 6 |
| "Chọn bậc bằng tập kiểm tra" | tập kiểm tra chỉ dùng **một lần ở cuối**; chọn bậc bằng tập kiểm định hoặc kiểm tra chéo (Bishop tr.11) |

### 0.7 Đọc bài theo thứ tự nào

| Nếu cần | Đọc |
|---|---|
| chạy được ngay | mục 0.1 → 7.1 |
| hiểu vì sao lại dùng ít hệ số hơn số điểm | mục 1 → 3.1 → 3.5 |
| hiểu khớp quá mức và cách chọn bậc | mục 3.11 → 3.13 → 6 → 7.2 |
| kiểm lại nguồn gốc | toàn bộ mục 3, rồi mục 11 |
| dùng cho bài toán thật | mục 7.3 → 9 |

### 0.8 Khi KHÔNG nên dùng

- **Cần ngoại suy** ra ngoài khoảng $x$ đã thấy: số hạng bậc cao chi phối, dự đoán đổi rất nhanh.
  Chính Gergonne tr.244 đã dặn chỉ tin công thức *"comprise entre celles des points donnés, et même
  ne se rapprochant pas trop de la plus grande ni de la plus petite"*.
- **Nhiều biến**: đa thức bậc 3 trên 100 biến có **176 851** số hạng (mục 9.1).
- **Quan hệ thay đổi cục bộ** — một đoạn phẳng rồi một đoạn dốc: đa thức là hàm toàn cục, đổi một
  điểm ở đầu này làm đổi đường khớp ở đầu kia. Dùng spline hoặc cây quyết định (mục 9.3).
- **Bậc cao trên cơ sở đơn thức mà không chuẩn hoá**: bài toán số hỏng trước cả khi bài toán thống
  kê hỏng (mục 6.2).

---

## 1. Bài toán mà nó giải — và cách cũ tắc ở đâu

### 1.1 Bài toán của năm 1815: biết giá trị ở vài điểm, cần đạo hàm ở một điểm

Hồi quy đa thức không ra đời từ câu hỏi "vẽ đường cong qua đám điểm". Nó ra đời từ một bài toán
thiên văn rất cụ thể. Phương pháp tính quỹ đạo sao chổi của Laplace (1780) cần **vận tốc và gia
tốc** của kinh độ sao chổi tại một thời điểm — tức $dx/dt$ và $d^2x/dt^2$ — trong khi nhà thiên văn
chỉ có **vài lần đo kinh độ** ở vài đêm khác nhau. Legendre tr.iv nêu đúng nhu cầu đó: phải
*"déterminer avec précision les coëfficiens différentiels du premier et du second ordre, tant de la
longitude que de la latitude"* (xác định chính xác các hệ số vi phân bậc một và bậc hai, của cả kinh
độ lẫn vĩ độ).

Gergonne tr.242 phát biểu bài toán tổng quát, gọi là *interpolation des suites* — nội suy dãy số:
biết giá trị của một hàm tại một số điểm, hãy tính giá trị của hàm **và các hệ số vi phân của nó**
tại một điểm khác. Cách làm quen thuộc thời đó, theo Gergonne tr.244:

> *"on suppose que l'ordonnée de la courbe cherchée est une fonction complète, rationnelle et entière
> de l'abscisse, dans laquelle on admet autant de termes qu'il y a de systèmes de valeur donnés ; les
> coefficiens de ces termes sont inconnus, et on les détermine en exprimant que la courbe passe par
> les points donnés."*
>
> (người ta giả sử tung độ của đường cần tìm là một đa thức đầy đủ của hoành độ, **lấy số số hạng
> bằng đúng số cặp giá trị đã cho**; các hệ số của những số hạng ấy chưa biết, và người ta xác định
> chúng bằng cách đòi đường cong **đi qua đúng** các điểm đã cho.)

Đó là **nội suy đa thức**: bảy điểm thì bảy hệ số, đa thức bậc 6, phần dư bằng 0 ở mọi điểm.

### 1.2 Cách cũ tắc ở đâu, đo bằng số

Cách nội suy tắc ở một chỗ mà cả Legendre lẫn Gergonne đều chỉ ra: **dữ liệu có sai số đo, và bắt
đường cong đi qua đúng từng điểm là bắt nó đi qua đúng từng sai số**. Legendre tr.iv:

> *"Comme toutes les observations sont affectées de quelqu'erreur, et que cette erreur ne suit aucune
> loi d'une observation à l'autre, il s'ensuit que plus on combinera d'observations, et plus l'erreur
> des coëfficiens différentiels qui en sont déduits pourra devenir sensible."*
>
> (Vì mọi quan sát đều mắc sai số, và sai số ấy không theo quy luật nào từ quan sát này sang quan sát
> khác, nên **càng kết hợp nhiều quan sát, sai số của các hệ số vi phân suy ra càng có thể lớn**.)

Câu đó nghe ngược trực giác — thêm dữ liệu mà tệ đi — nên đo nó ra. Ba phép đo, mỗi phép một góc.

**Phép đo thứ nhất — giữ bước giữa hai quan sát bằng 1, thêm quan sát.** Nội suy qua $2n+1$ điểm
cách đều, lấy đạo hàm tại điểm giữa. Mỗi tung độ mang sai số ngẫu nhiên độc lập độ lệch chuẩn
$\sigma$. Phương sai của đạo hàm ước lượng là $\sigma^2$ nhân tổng bình phương các hệ số của công
thức nội suy (`tools/v_poly.py` mục 1):

| Số quan sát | $\mathrm{Var}(\hat y')/\sigma^2$ | $\mathrm{Var}(\hat y'')/\sigma^2$ |
|---:|---:|---:|
| 3 | 0,500000 | 6,000000 |
| 5 | 0,902778 | 9,819444 |
| 7 | 1,170556 | 11,955741 |
| 9 | 1,362928 | 13,307971 |
| 13 | 1,626132 | 14,923770 |

Từ 3 lên 13 quan sát, phương sai của đạo hàm bậc một tăng **3,25 lần**. Legendre đúng — và đúng
ngay cả khi sai số ngẫu nhiên có cơ hội bù trừ nhau, một điều Gergonne tr.247 cho rằng Legendre đã
loại trừ (mục 3.3).

**Phép đo thứ hai — đoạn đo cố định, và đây là tình huống thật.** Có 21 quan sát cách đều trên
$[-1, 1]$. Cách cũ buộc phải chọn: nội suy qua ít điểm (bỏ phần còn lại), hoặc qua nhiều điểm (bước
nhỏ, sai số khuếch đại). Đo trên $\sigma=1$, và đo chệch trên hàm $f(x)=e^x$ không nhiễu, nơi
$f'(0)=f''(0)=1$ (`tools/v_poly.py` mục 3):

| Cách | sd($\hat y'$) | sd($\hat y''$) | chệch $\hat y'$ | chệch $\hat y''$ |
|---|---:|---:|---:|---:|
| nội suy 3 quan sát, bước 1, **bỏ 18** | 0,707107 | 2,449490 | $+1{,}752\times10^{-1}$ | $+8{,}616\times10^{-2}$ |
| nội suy 5 quan sát, bước 1/2, **bỏ 16** | 1,900292 | 12,534397 | $-2{,}146\times10^{-3}$ | $-7{,}101\times10^{-4}$ |
| nội suy 11 quan sát, bước 1/5, **bỏ 10** | 6,143191 | 94,342339 | $+3{,}747\times10^{-11}$ | $+6{,}231\times10^{-12}$ |
| nội suy **cả 21** quan sát, bước 1/10 | **13,894448** | **405,832301** | $+1{,}110\times10^{-15}$ | $+7{,}105\times10^{-15}$ |
| **bình phương tối thiểu bậc 2 trên cả 21** | **0,360375** | **1,335333** | $+1{,}140\times10^{-1}$ | $+8{,}037\times10^{-2}$ |
| bình phương tối thiểu bậc 4 trên cả 21 | 0,908213 | 4,755677 | $-2{,}405\times10^{-3}$ | $-1{,}499\times10^{-3}$ |

Bốn dòng đầu là bế tắc của cách cũ, đọc thành số: dùng hết dữ liệu thì độ lệch chuẩn của đạo hàm
bậc hai là **405,8** lần sai số của một phép đo; muốn bớt thì phải **vứt** dữ liệu. Không dòng nội suy
nào vừa ít nhiễu vừa ít chệch.

**Phép đo thứ ba — cách cũ của thời hiện đại: đường thẳng.** Ngày nay "cách cũ" trước hồi quy đa
thức thường là hồi quy tuyến tính. Đo trên 200 điểm từ $y=\sin(2{,}5x)+\varepsilon$, $x\in[-1,1]$,
nhiễu độ lệch chuẩn **0,10** (`tools/v_gd1ss.py` mục `370  Polynomial Regression`):

| Bậc | RMSE trên chính dữ liệu đó |
|---|---:|
| 1 (đường thẳng) | 0,2962 |
| 2 | 0,2962 |
| **3** | **0,0937** |
| 5 | 0,0920 |

| Số điểm $n$ | RMSE của bậc 1 |
|---:|---:|
| 50 | 0,3013 |
| 200 | 0,2924 |
| 2 000 | 0,2933 |
| 20 000 | 0,2925 |
| 200 000 | 0,2938 |

Đường thẳng sai gấp **2,9 lần** mức nhiễu 0,10, và tăng dữ liệu **4 000 lần** thì sai số vẫn đứng ở
0,29. Đổi sang bậc 3 kéo RMSE từ 0,2962 xuống 0,0937 — giảm **3,2 lần** — và đã chạm sàn: bậc 5 gần như
không cải thiện thêm. Đó là **chệch** — mô hình thiếu hình dạng — nên thêm dữ liệu không chữa được. Bậc 2 không giúp
gì (0,2962, y hệt bậc 1 tới bốn chữ số) vì $\sin$ là hàm lẻ còn $x^2$ là hàm chẵn trên dữ liệu đối
xứng quanh 0: cột $x^2$ gần như trực giao với phần đường thẳng chưa giải thích được. Thêm bậc không
tự động tốt hơn; phải thêm **đúng** bậc mà cấu trúc thật cần.

### 1.3 Nó đổi đúng cái gì

Gergonne đổi **một** thứ: số hệ số không còn phải bằng số điểm. Ta giữ số hệ số nhỏ như khi chỉ dùng
vài điểm, nhưng đưa **mọi** điểm vào, và vì không đa thức nào đi qua đúng tất cả được nữa, ta chọn
đa thức làm **tổng bình phương phần dư nhỏ nhất**.

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số ở đâu ra |
|---|---|---|---|
| dùng hết dữ liệu mà không khuếch đại sai số | sd($\hat y''$) từ 405,8 (nội suy 21 điểm) xuống 1,34 (bậc 2 trên 21 điểm) | **chệch**: $+0{,}080$ trên $f''(0)$ của $e^x$, trong khi nội suy 5 điểm chỉ chệch $-0{,}0007$ | mục 1.2, phép đo thứ hai |
| giá trị khớp được **chính xác hơn chính dữ liệu** ở giữa khoảng | sd 0,3280 lần sai số một phép đo, bậc 2 trên 21 điểm | ở **mép** khoảng chỉ còn 0,5969 — gần gấp đôi | mục 3.6 |
| biểu diễn được quan hệ cong | RMSE 0,2962 → 0,0937 khi đổi bậc 1 sang bậc 3 | thêm **một siêu tham số** $d$, chọn sai hỏng theo hai kiểu | `v_gd1ss.py` mục 370 · mục 6 |
| vẫn là bình phương tối thiểu tuyến tính, nghiệm dạng đóng | 0 dòng thuật toán mới | ma trận thiết kế xấu đi rất nhanh: $\kappa(V^\top V)$ từ 3,73 ở bậc 1 lên $1{,}96\times10^{13}$ ở bậc 12 | mục 6.2 |
| — | — | **ngoại suy tệ hơn đường thẳng** — số hạng bậc cao chi phối ngoài vùng dữ liệu | mục 9.1 |

Dòng cuối là dòng THUA: đây là một cái đổi, không phải một cải tiến thuần tuý. Và dòng thứ ba để lại
đúng câu hỏi Gergonne không trả lời được năm 1815 — **bậc nào** — mà mục 3.9, 3.13 và 7.2 lần lượt
đo.

---

## 2. Ý tưởng trung tâm — bằng lời, chưa công thức

Hãy hình dung 21 điểm đo nằm gần một đường cong trơn, mỗi điểm lệch lên hoặc xuống một chút vì sai
số đo.

- **Nội suy** bắt đường cong chui qua cả 21 điểm. Nó có 21 hệ số, đủ tự do để chui qua mọi vết lệch,
  nên giữa hai điểm nó uốn lượn theo nhiễu. Đạo hàm — độ dốc và độ cong — nhạy với những uốn lượn đó
  hơn hẳn giá trị, nên sai lệch lớn nhất nằm ở đó.
- **Hồi quy đa thức** chỉ cho đường cong 3 hệ số. Ba hệ số không đủ để chui qua 21 vết lệch, nên
  đường cong buộc phải đi **giữa** các điểm. Vết lệch lên của điểm này bù cho vết lệch xuống của điểm
  kia — Gergonne gọi là *"compensations d'erreurs"* (tr.248).
- **Cái giá**: nếu đường cong thật cần nhiều hơn 3 hệ số thì 3 hệ số không đủ để theo nó, và sai số
  còn lại không phải nhiễu mà là **chệch**.

Bishop tr.5 nói thêm một điều làm cho toàn bộ chuyện này **tính được bằng tay**: đa thức là hàm
phi tuyến của $x$ nhưng là hàm **tuyến tính của các hệ số**.

> *"Note that, although the polynomial function y(x, w) is a nonlinear function of x, it is a linear
> function of the coefficients w."*
>
> (Lưu ý rằng dù hàm đa thức y(x, w) là hàm phi tuyến của x, nó là hàm tuyến tính của các hệ số w.)

Và tr.6 rút ra hệ quả:

> *"Because the error function is a quadratic function of the coefficients w, its derivatives with
> respect to the coefficients will be linear in the elements of w, and so the minimization of the
> error function has a unique solution"*
>
> (Vì hàm lỗi là hàm bậc hai của các hệ số w, đạo hàm của nó theo các hệ số tuyến tính theo các phần
> tử của w, nên bài toán cực tiểu hoá hàm lỗi có một nghiệm duy nhất)

Tức là: đạo hàm tổng bình phương theo từng hệ số, cho bằng 0, được một hệ phương trình tuyến tính.
Đó chính là việc Gergonne làm ở tr.250, 191 năm trước Bishop.

| Số hệ số so với số điểm | Tên | Phần dư | Hỏng thế nào |
|---|---|---|---|
| bằng số điểm | nội suy | 0 ở mọi điểm | khuếch đại sai số đo (mục 1.2) |
| **ít hơn** số điểm, đủ để theo hình dạng thật | **hồi quy đa thức, bậc vừa** | khác 0, cỡ nhiễu | — |
| ít hơn quá nhiều | hồi quy bậc quá thấp | lớn hơn nhiễu | chệch (mục 1.2, phép đo thứ ba) |
| gần bằng hoặc bằng số điểm | hồi quy bậc quá cao | gần 0 | khớp quá mức (Bishop Hình 15, $M=9$ trên $N=10$) |

---

## 3. Phần ruột — theo đúng hai nguồn gốc

Phần này đi qua **toàn bộ** phần phương pháp của hai nguồn, theo đúng thứ tự in:

| Nguồn | Phần đưa vào | Vì sao là phần ruột |
|---|---|---|
| Legendre 1805, lời nói đầu tr.iv–v | công thức (a), (b) và tỉ số 4 : 3 | đây là lời cảnh báo mà cả bài Gergonne viết ra để trả lời |
| **Gergonne 1815, tr.242–252 — cả bài** | mọi công thức, mọi ví dụ số, mọi câu tự đánh giá | bài đầu tiên đem bình phương tối thiểu vào bài toán khớp đa thức |
| **Bishop 2006, mục 1.1, tr.4–11** | công thức (1.1)–(1.4), Hình 1.2–1.8, Bảng 1.1–1.2 | nguồn hiện đại có dữ liệu và có câu trả lời thực hành cho câu hỏi "bậc nào" |

**Cách trích.** Bài của Gergonne và sách của Legendre là bản in cũ; lớp text của cả hai là OCR có
lỗi (bản Numdam của Gergonne đọc *"non consécutifs"* thành *"lion consécutifs"*, bản Google của
Legendre gần như không đọc được). Mọi câu tiếng Pháp dưới đây **chép từ ảnh trang**, giữ nguyên chính tả
thời đó (*coefficiens*, *quarrés*, *étoient*). Gergonne và Legendre **không có hình hay bảng nào**, nên
"Hình" ở mục 3.1–3.9 là **khung cắt từ trang in** — không có caption gốc; ô "caption" được thay bằng
dòng nói khung đó chứa gì. Câu tiếng Anh của Bishop lấy từ lớp text của PDF.

**Bảy chỗ việc dựng lại tìm ra.** Không chỗ nào làm sai một kết luận chính của nguồn; mỗi chỗ đổi
cách một con số hoặc một câu phải được đọc.

| # | Nguồn · chỗ | Nguồn in / nói | Dựng lại cho thấy | Mục |
|---|---|---|---|---|
| 1 | Gergonne tr.252, công thức đổi biến | mẫu của $dy/dx$ là $d\lambda/dx$; mẫu của $d^2y/dx^2$ là $(dx/dz)^{2}$ | mẫu đúng là $dx/dz$ và $(dx/dz)^{3}$; với $x=z^2$, $y=z^3$, $z=2$ bản in cho $3/2$, đúng là $3/8$ | 3.8 |
| 2 | Gergonne tr.247 | tung độ lỗi $b'$ là tung độ *"exerce l'influence la plus notable sur nos deux coefficiens"* | đúng cho $dy/dx$; với $d^2y/dx^2$ tung độ **giữa** $b$ nặng hơn $b'$ ở mọi số tung độ đã thử (2 so với 1 khi có 3 tung độ) | 3.3 |
| 3 | Gergonne tr.249–250 | hai thứ khác nhau cùng mang số (1) | đa thức ở tr.249 và hệ phương trình chuẩn ở tr.250; tr.251 *"les équations (1)"* chỉ hệ ở tr.250 | 3.6 |
| 4 | Bishop Bảng 1.1 tr.8 | cột nhãn "$M = 6$" | đó là $M = 3$: bốn hệ số khớp $M = 3$ dựng lại tới ba chữ số | 3.11 |
| 5 | Bishop Hình 1.5 tr.8 | trục dọc là $E_{\text{RMS}}$ theo (1.3) | đường huấn luyện là $\sqrt{E/N}$, nhỏ hơn (1.3) một hệ số $\sqrt2$: tỉ số số hoá trung bình 0,716 | 3.11 |
| 6 | Bishop Bảng 1.2 tr.11 | nhãn $\ln\lambda = 0$ và $-18$ | cột "0" chỉ khớp (1.4) tại $\ln\lambda = 2{,}28 \approx \ln N$; cột "−18" không khớp $\lambda$ nào trong mức làm tròn | 3.13 |
| 7 | Bishop Hình 1.8 tr.11 so với Bảng 1.2 | sai số huấn luyện ở $\ln\lambda \approx -20$ là 0,4178 | hệ số in của cột $\ln\lambda = -18$ cho 0,1265 trên cùng thang; sai số huấn luyện Ridge không giảm khi $\lambda$ tăng, nên hai thứ không cùng dữ liệu hoặc không cùng trục $\lambda$ | 3.13 |

### 3.1 Legendre 1805, tr.iv–v — lời cảnh báo mà Gergonne trả lời

![Legendre 1805, lời nói đầu tr.iv — công thức (a) và (b)](<paper_poly_legendre1805_triv_noi_suy.png>)

**Hình 1 — Legendre 1805, *Nouvelles méthodes pour la détermination des orbites des comètes*, lời nói
đầu tr.iv (PDF trang 8), `_papers/1805 - Legendre - Nouvelles methodes pour la determination des
orbites des cometes.pdf`.** Khung cắt từ trang văn bản, không có caption gốc: đoạn nêu vấn đề và hai
hệ công thức nội suy (a), (b).

**Legendre nói gì.** Tr.iv:

> *"Il semble, au premier coup-d'œil, qu'en liant ensemble plusieurs observations par la méthode des
> interpolations, on en déduira les coëfficiens dont il s'agit avec d'autant plus d'exactitude qu'il y
> a plus d'observations combinées ; et cela auroit lieu en effet, si les observations étoient exemptes
> d'erreur, ou si les lieux de la comète étoient calculés d'après une formule exacte ; mais il en est
> autrement dans l'état réel des choses."*
>
> (Thoạt nhìn, tưởng rằng khi nối nhiều quan sát bằng phương pháp nội suy, ta sẽ suy ra các hệ số
> [vi phân] càng chính xác khi càng kết hợp nhiều quan sát; và điều đó quả sẽ xảy ra nếu các quan sát
> không có sai số, hoặc nếu vị trí sao chổi được tính từ một công thức chính xác; nhưng trong thực tế thì
> khác.)

Hai hệ công thức ngay sau, cho quan sát cách đều một đơn vị thời gian. Với ba kinh độ $a', a'', a'''$ ở
thời điểm $-1, 0, +1$:

$$\frac{dx}{dt} = \tfrac12\,(a''' - a'), \qquad \frac{ddx}{dt^2} = a''' - 2a'' + a' \tag{a}$$

Với năm kinh độ $a, a', a'', a''', a^{\text{iv}}$ ở thời điểm $-2, -1, 0, +1, +2$, và
$x = a'' + At + Bt^2 + Ct^3 + \dots$:

$$A = \frac{dx}{dt} = \tfrac23\,(a''' - a') - \tfrac1{12}\,(a^{\text{iv}} - a), \qquad
2B = \frac{ddx}{dt^2} = \tfrac43\,(a''' - 2a'' + a') - \tfrac1{12}\,(a^{\text{iv}} - 2a'' + a) \tag{b}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $a, a', a'', a''', a^{\text{iv}}$ | kinh độ quan sát được — **đầu ra $y$**, không phải đầu vào | góc |
| $t$ | thời gian, lấy khoảng giữa hai lần quan sát làm đơn vị | vô hướng |
| $x$ | kinh độ tại thời điểm $t$ theo đa thức nội suy | góc |
| $A$, $2B$ | đạo hàm bậc một và bậc hai tại $t=0$ | góc / đơn vị thời gian, góc / (đơn vị thời gian)² |

**Ý nghĩa.** (a) là đa thức bậc hai đi qua đúng 3 điểm, (b) là đa thức bậc bốn đi qua đúng 5 điểm. Cả
hai lấy **tổ hợp tuyến tính của các quan sát** làm ước lượng đạo hàm, nên sai số của đạo hàm cũng là
tổ hợp tuyến tính của sai số quan sát — với chính các hệ số đó. Hệ số càng lớn, sai số càng khuếch đại.
Viết (b) theo từng kinh độ thì hệ số của $a''$ ở đạo hàm bậc hai là $\tfrac43\cdot(-2) -
\tfrac1{12}\cdot(-2) = -\tfrac52$, trùng công thức 5 tung độ của Gergonne ở mục 3.3
(`tools/v_poly.py` mục 1 dựng lại cả hai bằng phân số chính xác).

![Legendre 1805, lời nói đầu tr.v — tỉ số 4 : 3](<paper_poly_legendre1805_trv_ti_so_4_3.png>)

**Hình 2 — Legendre 1805, lời nói đầu tr.v (PDF trang 9).** Khung cắt từ trang văn bản: phép tính
sai số khi riêng kinh độ $a'$ lệch $\delta a'$, và kết luận.

**Legendre nói gì.** Theo (a), $\delta\frac{dx}{dt} = -\tfrac12\delta a'$ và $\delta\frac{ddx}{dt^2} =
\delta a'$; theo (b), $-\tfrac23\delta a'$ và $\tfrac43\delta a'$. Tr.v:

> *"d'où l'on voit que dans le cas de cinq observations combinées, l'erreur des coëfficiens
> différentiels, due à la cause mentionnée, est plus grande dans le rapport de 4 à 3, que celle qui a
> lieu dans le cas de trois observations. Elle augmenteroit encore si on combinoit ensemble plus de
> cinq observations."*
>
> (từ đó thấy rằng khi kết hợp năm quan sát, sai số của các hệ số vi phân do nguyên nhân nói trên lớn
> hơn **theo tỉ số 4 : 3** so với khi dùng ba quan sát. Nó còn tăng nữa nếu kết hợp nhiều hơn năm quan
> sát.)

Và kết luận của Legendre, tr.v: *"D'après ces réflexions, j'ai pensé que ce qu'il y avoit de mieux à
faire dans le problème des comètes, étoit de partir des données immédiates de l'observation"* (theo
những suy nghĩ đó, [Legendre] cho rằng việc tốt nhất trong bài toán sao chổi là **đi thẳng từ số liệu
quan sát**, bỏ bước nội suy).

**Dựng lại** (`v_poly.py` mục 1): $\tfrac23 : \tfrac12 = \tfrac43$ và $\tfrac43 : 1 = \tfrac43$ — **đúng
4 : 3 cho cả hai hệ số**. Với 7 quan sát tỉ số là $\tfrac34 : \tfrac12 = \tfrac32$, đúng câu "còn tăng
nữa".

**Cách đọc hình.** Dấu $\delta$ là "sai số của". Dấu trừ ở $-\tfrac12\delta a'$ là vì $a'$ đứng
**trước** thời điểm giữa: kinh độ đo sớm bị đẩy lên thì độ dốc ước lượng giảm xuống.

**Điều hình không nói.** Legendre chỉ xét **một** quan sát lỗi, và không xét cách **bỏ** quan sát
hoặc **giãn** khoảng cách. Gergonne tr.247 chỉ ra đúng hai chỗ đó.

### 3.2 Gergonne tr.242–245 — bài toán, và vì sao chọn đa thức

Gergonne mở bài bằng một nhận xét hình học. Tr.242–243: qua các điểm cho trước — *"non consécutifs,
même en nombre infini"* — luôn vẽ được vô số đường cong khác nhau. Nếu các điểm đủ gần nhau, mọi đường
cong ấy cho **tung độ** gần như nhau, nhưng tr.243:

> *"mais il n'en sera plus de même des coefficiens différentiels successifs qui, d'une fonction à
> l'autre, pourront différer de plus en plus, à mesure que l'ordre en sera plus élevé."*
>
> (nhưng các hệ số vi phân liên tiếp thì không như thế: từ hàm này sang hàm khác chúng có thể khác
> nhau ngày càng nhiều, khi bậc đạo hàm càng cao.)

Đó là lý do chọn **đa thức** — không vì nó đúng, mà vì nó đơn giản nhất. Tr.244:

> *"La courbe et la fonction pouvant ainsi être choisies d'une infinité de manières différentes, il
> sera convenable de s'arrêter aux plus simples, c'est-a-dire, à la courbe parabolique et à la fonction
> rationnelle et entière qu'elle représente graphiquement."*
>
> (Vì đường cong và hàm số có thể chọn theo vô số cách, nên dừng ở cách đơn giản nhất: đường
> *parabolique* và hàm hữu tỉ nguyên — tức đa thức — mà nó biểu diễn.)

*"Courbe parabolique"* ở đây là thuật ngữ thế kỷ 19 cho **đồ thị của một đa thức bất kỳ bậc**, không
riêng parabol bậc hai. Gergonne tự đánh giá lựa chọn đó ngay tr.244: phương pháp Laplace dùng nó
*"renferme une source d'erreur, dans la supposition, tout à fait gratuite, d'une courbe du genre
parabolique"* (chứa một nguồn sai, ở giả thiết **hoàn toàn tuỳ tiện** rằng đường cong là đa thức). Và
dặn giới hạn dùng: chỉ tin công thức cho hoành độ *"comprise entre celles des points donnés, et même ne
se rapprochant pas trop de la plus grande ni de la plus petite"* — nằm giữa các điểm đã cho, và đừng
sát điểm lớn nhất hay nhỏ nhất quá. Mục 3.6 đo đúng cảnh báo này.

Nguồn sai thứ hai, tr.245, là sai số đo, và Gergonne ghi công cho Legendre:

> *"les valeurs discontinues de la fonction, sur lesquelles on s'appuie pour construire la formule,
> sont communément déduites d'expériences ou d'observations susceptibles d'une exactitude assez bornée ;
> et il arrive alors, comme M. Legendre l'a fort bien observé (*), que les erreurs qui les affectent
> peuvent avoir d'autant plus d'influence sur la formule finale et sur les résultats qu'on en déduit,
> que ces valeurs sont en plus grand nombre."*
>
> (các giá trị rời rạc của hàm dùng để dựng công thức thường lấy từ thí nghiệm hay quan sát có độ chính
> xác khá hạn chế; và khi đó, như Legendre đã nhận xét rất đúng, sai số của chúng có thể ảnh hưởng tới
> công thức cuối và kết quả càng nhiều khi các giá trị ấy càng nhiều.)

Chú thích (*) dẫn *"Nouvelles méthodes pour la détermination des orbites des comètes. Paris, 1806,
(pag. iv)"* — chính trang ở Hình 1. Trang bìa bản trong `_papers/` ghi *"AN XIII – 1805"*; Gergonne ghi
1806.

### 3.3 Gergonne tr.246 — công thức đạo hàm 3/5/7 tung độ, và sai số khi một tung độ lệch

![Gergonne 1815 tr.246 — công thức đạo hàm nội suy 3, 5, 7 tung độ](<paper_poly_gergonne1815_tr246_cong_thuc_dao_ham.png>)

**Hình 3 — Gergonne 1815, *Application de la méthode des moindres quarrés à l'interpolation des
suites*, Annales de Mathématiques pures et appliquées 6, tr.246 (PDF trang 6),
`_papers/1815 - Gergonne - Application de la methode des moindres quarres a l'interpolation des
suites.pdf`.** Khung cắt từ trang văn bản: ba cặp công thức đạo hàm tại tung độ giữa.

Thiết lập, tr.245: $2n+1$ tung độ cách đều một đơn vị; tung độ giữa là $b$ tại hoành độ 0; các tung
độ **sau** là $b', b'', b''', \dots$ tại $1, 2, 3, \dots$; các tung độ **trước** là $b_{\prime},
b_{\prime\prime}, b_{\prime\prime\prime}, \dots$ tại $-1, -2, -3, \dots$ (dấu phẩy **dưới**). Gergonne
in, cho 3, 5 và 7 tung độ:

$$\frac{dy}{dx} = \frac{b'-b_{\prime}}{2}, \qquad \frac{d^2y}{dx^2} = (b'+b_{\prime}) - 2b$$

$$\frac{dy}{dx} = \frac{8(b'-b_{\prime}) - (b''-b_{\prime\prime})}{12}, \qquad
\frac{d^2y}{dx^2} = -\frac{30b - 16(b'+b_{\prime}) + (b''+b_{\prime\prime})}{12}$$

$$\frac{dy}{dx} = \frac{45(b'-b_{\prime}) - 9(b''-b_{\prime\prime}) + (b'''-b_{\prime\prime\prime})}{60}, \qquad
\frac{d^2y}{dx^2} = -\frac{490b - 270(b'+b_{\prime}) + 27(b''+b_{\prime\prime}) - 2(b'''+b_{\prime\prime\prime})}{180}$$

**Ý nghĩa.** Mỗi công thức là đạo hàm tại 0 của **đa thức nội suy** bậc $2n$ đi qua đúng $2n+1$ điểm.
Đạo hàm bậc một chỉ dùng **hiệu** hai tung độ đối xứng (tung độ giữa có hệ số 0); đạo hàm bậc hai chỉ
dùng **tổng** của chúng. Đó là hệ quả của đối xứng: đạo hàm bậc một đổi dấu khi lật trục $x$, bậc hai
thì không.

**Dựng lại** (`v_poly.py` mục 1): giải hệ Vandermonde bằng phân số chính xác cho từng tung độ đơn vị,
lấy đạo hàm tại 0. Ba dòng đầu **khớp trang in tuyệt đối**; ba dòng sau kéo dài *"et ainsi de suite"*:

| Tung độ | $dy/dx$: hệ số của $b', b'', \dots$ | $d^2y/dx^2$: hệ số của $b, b', b'', \dots$ |
|---:|---|---|
| 3 | 1/2 | −2 · 1 |
| 5 | 2/3 · −1/12 | −5/2 · 4/3 · −1/12 |
| 7 | 3/4 · −3/20 · 1/60 | −49/18 · 3/2 · −3/20 · 1/90 |
| 9 | 4/5 · −1/5 · 4/105 · −1/280 | −205/72 · 8/5 · −1/5 · 8/315 · −1/560 |
| 11 | 5/6 · −5/21 · 5/84 · −5/504 · 1/1260 | −5269/1800 · 5/3 · −5/21 · 5/126 · −5/1008 · 1/3150 |
| 13 | 6/7 · −15/56 · 5/63 · −1/56 · 1/385 · −1/5544 | −5369/1800 · 12/7 · −15/56 · 10/189 · −1/112 · 2/1925 · −1/16632 |

(Dòng 7 tung độ: $45/60 = 3/4$, $9/60 = 3/20$, $490/180 = 49/18$ — cùng số, rút gọn.)

![Gergonne 1815 tr.246 — sai số khi riêng tung độ b' lệch β](<paper_poly_gergonne1815_tr246_sai_so_mot_tung_do.png>)

**Hình 4 — Gergonne 1815, tr.246 (PDF trang 6).** Khung cắt từ trang văn bản: phép tính sai số và
kết luận "croissent comme".

**Gergonne nói gì.** Tr.246–247:

> *"Or, supposons que, toutes les autres ordonnées étant d'ailleurs exactes, l'ordonnée b' seule soit
> en erreur d'une quantité β, et désignons par E. dy/dx, E. d²y/dx² les erreurs qui en résulteront sur
> les coefficiens différentiels"* […] *"de sorte que les erreurs sur le coefficient différentiel du
> premier ordre croissent comme les nombres 1/2, 2/3, 3/4, 4/5, .... et tendent ainsi sans cesse à
> devenir égales à l'erreur même commise sur l'ordonnée b' ; et que l'erreur commise sur le coefficient
> différentiel du second ordre est constamment double de celle-là."*
>
> (Giả sử mọi tung độ khác đều đúng, riêng $b'$ sai một lượng $\beta$, và gọi E.dy/dx, E.d²y/dx² là
> sai số gây ra trên các hệ số vi phân […] thì sai số trên hệ số vi phân bậc một **tăng như các số 1/2,
> 2/3, 3/4, 4/5** và dần tiến tới đúng bằng sai số của chính tung độ $b'$; còn sai số trên hệ số vi
> phân bậc hai **luôn gấp đôi** con số đó.)

Trang in cho: 3 tung độ $\tfrac12\beta$ và $\tfrac22\beta$; 5 tung độ $\tfrac23\beta$ và $\tfrac43\beta$;
7 tung độ $\tfrac34\beta$ và $\tfrac64\beta$. Nhắc lại: **$\beta$ ở đây là sai số đo**, không phải hệ
số hồi quy.

**Dựng lại** (`v_poly.py` mục 1):

| Tung độ | E.dy/dx ÷ β | E.d²y/dx² ÷ β | tỉ số | $n/(n+1)$ | tung độ nặng nhất trong $d^2y/dx^2$ |
|---:|---|---|---:|---|---|
| 3 | 1/2 | 1 | 2 | 1/2 | **giữa $b$**, hệ số 2 — còn $b'$ là 1 |
| 5 | 2/3 | 4/3 | 2 | 2/3 | **giữa $b$**, 5/2 — còn $b'$ là 4/3 |
| 7 | 3/4 | 3/2 | 2 | 3/4 | **giữa $b$**, 49/18 — còn $b'$ là 3/2 |
| 9 | 4/5 | 8/5 | 2 | 4/5 | **giữa $b$**, 205/72 — còn $b'$ là 8/5 |
| 11 | 5/6 | 5/3 | 2 | 5/6 | **giữa $b$**, 5269/1800 — còn $b'$ là 5/3 |
| 13 | 6/7 | 12/7 | 2 | 6/7 | **giữa $b$**, 5369/1800 — còn $b'$ là 12/7 |

Ba câu định lượng của Gergonne **đúng**, khớp 3/3 dòng in: sai số bậc một bằng $n/(n+1)$ nên tiến tới
$\beta$; bậc hai luôn gấp đôi. Nhưng câu ở tr.247 thì chỉ đúng một nửa:

> *"et en supposant de plus que l'ordonnée fautive est précisément celle dont la valeur, exacte ou non,
> exerce l'influence la plus notable sur nos deux coefficiens différentiels."*
>
> (và giả thiết thêm rằng tung độ sai chính là tung độ có ảnh hưởng lớn nhất lên **cả hai** hệ số vi
> phân của ta.)

Với $dy/dx$, $b'$ đúng là tung độ nặng nhất. Với $d^2y/dx^2$ thì **không**: cột cuối cho thấy tung độ
giữa $b$ có hệ số lớn hơn $b'$ ở mọi số tung độ đã thử. Gergonne dùng câu này để nói Legendre đã chọn
trường hợp xấu nhất; với đạo hàm bậc hai, trường hợp xấu nhất còn xấu hơn thế.

**Phép đo Gergonne không làm.** Ông cũng nói Legendre *"exclut toute possibilité de compensation
d'erreurs"* — chỉ một tung độ sai thì không có bù trừ. Cho **mọi** tung độ cùng mang sai số ngẫu nhiên
độc lập thì bù trừ có cơ hội xảy ra, và phương sai là $\sigma^2$ nhân tổng bình phương hệ số. Bảng phép
đo thứ nhất ở mục 1.2 cho kết quả: từ 3 lên 13 tung độ, phương sai của $dy/dx$ **vẫn tăng 3,25 lần**
(`v_poly.py` mục 1 in 3,2523). Kết luận của Legendre đứng vững cả khi có bù trừ.

**Cách đọc hình.** Chữ E. đứng trước một đại lượng nghĩa là "sai số của đại lượng đó". Các phân số in
rất nhỏ — $\tfrac22\beta$ ở dòng 3 tung độ trông giống $\tfrac12\beta$ nếu nhìn ở độ phân giải thấp;
bài này đọc ở ảnh phóng to 6 lần và kiểm lại bằng phân số.

### 3.4 Gergonne tr.247–248 — trả lời Laplace: giãn khoảng cách theo số quan sát

![Gergonne 1815 tr.247 — trích Laplace, và sai số theo khoảng cách a](<paper_poly_gergonne1815_tr247_laplace_khoang_cach.png>)

**Hình 5 — Gergonne 1815, tr.247 (PDF trang 7).** Khung cắt từ trang văn bản: nhận xét về Legendre,
đoạn trích *Mécanique céleste*, và sai số theo khoảng cách $a$.

**Gergonne nói gì.** Ông trích Laplace (*Mécanique céleste*, t.I, tr.201): *"pour diminuer
l'influence de ces erreurs, il faut augmenter l'intervalle des observations extrêmes, à mesure que l'on
emploie plus d'observations"* (để giảm ảnh hưởng của sai số, phải **nới rộng khoảng giữa hai quan sát
ngoài cùng** khi dùng nhiều quan sát hơn), rồi sửa lại cho chính xác hơn:

> *"Il serait peut-être plus exact de dire qu'il faut employer des observations de plus en plus
> distantes entre elles, à mesure qu'on en emploie un plus grand nombre ; et nous allons voir, en effet,
> qu'avec cette attention, on peut, à volonté, atténuer les erreurs."*
>
> (Có lẽ chính xác hơn nếu nói phải dùng những quan sát **ngày càng cách xa nhau**, khi dùng số lượng
> ngày càng lớn; và quả thật, với chú ý đó, có thể giảm sai số tuỳ ý.)

Gọi $a$ là khoảng cách giữa hai hoành độ liên tiếp — **chữ $a$ lần này là khoảng cách, không phải
hoành độ**. Chia công thức đạo hàm cho $a$ (bậc một) và $a^2$ (bậc hai), Gergonne in:
$\tfrac12\frac{\beta}{a}$ và $\tfrac22\frac{\beta}{a^2}$ cho 3 quan sát, $\tfrac23\frac{\beta}{a}$ và
$\tfrac43\frac{\beta}{a^2}$ cho 5, $\tfrac34\frac{\beta}{a}$ và $\tfrac64\frac{\beta}{a^2}$ cho 7.

![Gergonne 1815 tr.248 — a tăng theo số tự nhiên; và cái giá](<paper_poly_gergonne1815_tr248_so_tu_nhien.png>)

**Hình 6 — Gergonne 1815, tr.248 (PDF trang 8).** Khung cắt từ trang văn bản: ví dụ $a$ tăng theo số
tự nhiên, rồi đoạn nêu cái giá và câu mở lối sang bình phương tối thiểu.

**Gergonne nói gì.** Cho $a$ tăng theo số tự nhiên, $a = 1$ khi có 3 quan sát, ông in 3 quan sát
$\tfrac12\beta$, $\tfrac11\beta$; 5 quan sát $\tfrac13\beta$, $\tfrac13\beta$; 7 quan sát $\tfrac14\beta$,
$\tfrac16\beta$, rồi kết luận tr.248:

> *"d'où l'on voit qu'alors les erreurs sur les coefficiens différentiels du premier ordre décroîtront
> comme les inverses des nombres naturels, et que celles qui affecteront les coefficiens différentiels du
> second ordre décroîtront suivant la progression, plus rapide encore, des inverses des nombres
> triangulaires. La méthode de M. Laplace est donc, du moins de ce côté, tout à fait à l'abri du
> reproche."*
>
> (khi đó sai số trên hệ số vi phân bậc một giảm như **nghịch đảo của các số tự nhiên**, còn sai số trên
> bậc hai giảm theo cấp số còn nhanh hơn — **nghịch đảo của các số tam giác**. Vậy ít nhất ở mặt này,
> phương pháp của Laplace hoàn toàn không đáng chê.)

**Dựng lại** (`v_poly.py` mục 2) — cột "trang in" và cột phân số chính xác trùng nhau ở 3/3 dòng; ba
dòng sau kéo dài, kèm độ lệch chuẩn khi **mọi** quan sát có nhiễu độc lập:

| Quan sát | $a$ | E.dy/dx ÷ β | E.d²y/dx² ÷ β | $2/(n(n+1))$ | sd($\hat y'$)/σ | sd($\hat y''$)/σ |
|---:|---:|---|---|---|---:|---:|
| 3 | 1 | 1/2 | 1 | 1 | 0,707107 | 2,449490 |
| 5 | 2 | 1/3 | 1/3 | 1/3 | 0,475073 | 0,783400 |
| 7 | 3 | 1/4 | 1/6 | 1/6 | 0,360641 | 0,384190 |
| 9 | 4 | 1/5 | 1/10 | 1/10 | 0,291861 | 0,228001 |
| 11 | 5 | 1/6 | 1/15 | 1/15 | 0,245728 | 0,150948 |
| 13 | 6 | 1/7 | 1/21 | 1/21 | 0,212533 | 0,107309 |

Cả hai câu **đúng**: $1/(n+1)$ là nghịch đảo số tự nhiên, $2/(n(n+1))$ là nghịch đảo số tam giác
$1, 3, 6, 10, 15, 21$.

**Điều bảng không nói — và Gergonne nói ngay sau.** Với $2n+1$ quan sát và $a = n$, các quan sát trải
từ $-n^2$ tới $n^2$: 7 quan sát đã phủ đoạn dài 18, gấp 9 lần đoạn dài 2 của 3 quan sát. Nếu dữ liệu
chỉ có trong một **đoạn cố định** thì không giãn được. Tr.248:

> *"Mais, supposons qu'on ait, entre deux limites fixes données, des observations assez nombreuses pour
> rendre très-petite la différence entre les valeurs consécutives de x. Suivant ce qui vient d'être dit,
> on devra rejeter un d'autant plus grand nombre de ces observations qu'on en voudra employer davantage
> dans la recherche de dy/dx et d²y/dx². Or, c'est là un inconvénient assez grave, sur-tout si l'on n'a
> aucun motif de suspecter plutôt les données que l'on rejette que celles dont on se propose de faire
> exclusivement usage ; puisqu'on se prive ainsi des compensations d'erreurs sur lesquelles on pourrait
> compter en les employant toutes."*
>
> (Nhưng giả sử **giữa hai giới hạn cố định** có rất nhiều quan sát, đủ để khoảng cách giữa hai giá trị
> $x$ liên tiếp rất nhỏ. Theo điều vừa nói, muốn dùng càng nhiều quan sát cho dy/dx và d²y/dx² thì càng
> phải **vứt bỏ** nhiều quan sát. Đó là một bất tiện khá nặng, nhất là khi không có lý do gì nghi ngờ số
> liệu bị vứt hơn số liệu được giữ; vì như thế là tự tước đi sự **bù trừ sai số** mà ta có thể trông
> cậy nếu dùng hết.)

Bảng phép đo thứ hai ở mục 1.2 đo đúng tình huống này trên 21 quan sát trong $[-1, 1]$.

### 3.5 Gergonne tr.248–249 — phát biểu phương pháp

Đây là đoạn **phát biểu phương pháp** — đoạn quan trọng nhất của bài. Tr.248–249:

> *"En réfléchissant sur ce sujet, il m'a paru qu'il était possible de tout concilier, au moyen de la
> méthode des moindres quarrés (*), et d'arriver par elle à toute la précision qu'il est possible
> d'espérer dans la recherche qui nous occupe."*
>
> (Suy nghĩ về vấn đề này, [Gergonne] thấy có thể **dung hoà được tất cả** bằng phương pháp bình
> phương tối thiểu, và nhờ nó đạt tới toàn bộ độ chính xác có thể hy vọng trong bài toán đang xét.)

Chú thích (*) tr.248–249 là một đoạn lịch sử ngắn, viết sáu năm sau Gauss 1809:

> *"Le premier ouvrage imprimé dans lequel il ait été fait mention de cette méthode est le mémoire de
> M. Legendre, déjà cité dans une précédente note (1806). Dans un ouvrage publié eu 1809, M. Gauss a
> déclaré faire usage d'une semblable méthode depuis 1795 ; et M. Laplace a démontré postérieurement que
> cette méthode est rigoureusement conforme à la doctrine des probabilités."*
>
> (Tác phẩm in đầu tiên nhắc tới phương pháp này là khảo luận của Legendre (1806). Trong một tác phẩm in
> năm 1809, Gauss tuyên bố đã dùng một phương pháp tương tự từ 1795; và Laplace về sau chứng minh phương
> pháp này hoàn toàn phù hợp với lý thuyết xác suất.)

(*"eu 1809"* là chữ in trên trang, lỗi sắp chữ của *"en 1809"*.)

### 3.6 Gergonne tr.249–250 — đa thức (1) và hệ phương trình chuẩn (1)

![Gergonne 1815 tr.249 — đặt bài toán, công thức (1)](<paper_poly_gergonne1815_tr249_dat_bai_toan.png>)

**Hình 7 — Gergonne 1815, tr.249 (PDF trang 9).** Khung cắt từ trang văn bản: dữ liệu, đa thức (1),
và danh sách sai số.

**Gergonne nói gì.** Tr.249:

> *"Soient a, a', a'', .... des valeurs de x, en nombre quelconque, et soient b, b', b'', .... les
> valeurs données et correspondantes de y. Soit posé"* [đa thức (1) ở dưới] *"en prenant dans cette
> fonction autant de termes seulement qu'on en admettrait si, suivant ce qui vient d'être dit ci-dessus,
> on ne se proposait d'employer qu'une partie des valeurs correspondantes de x et de y ; il s'agira de
> déterminer les valeurs des coefficiens A, B, C, D, ..... Si leur nombre était égal à celui des
> observations, on pourrait leur assigner des valeurs qui rendissent les erreurs tout à fait nulles ;
> mais la chose sera impossible dans le cas actuel, et il faudra se contenter de rendre minimum la somme
> de leurs quarrés."*
>
> (Gọi $a, a', a''$ là các giá trị của $x$, số lượng tuỳ ý, và $b, b', b''$ là các giá trị tương ứng đã
> cho của $y$. Đặt [đa thức (1)], lấy trong hàm này **chỉ bấy nhiêu số hạng như khi chỉ định dùng một
> phần** các cặp giá trị $x, y$; việc còn lại là xác định các hệ số A, B, C, D. Nếu số hệ số bằng số
> quan sát, có thể chọn chúng để sai số triệt tiêu hoàn toàn; nhưng ở đây điều đó không thể, và phải bằng
> lòng với việc **làm cho tổng bình phương của chúng nhỏ nhất**.)

$$y = A + Bx + Cx^2 + Dx^3 + \dots \tag{1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $a, a', a'', \dots$ | hoành độ — **đầu vào** $x_i$ (ở tr.247–248 chữ $a$ là khoảng cách) | vô hướng |
| $b, b', b'', \dots$ | tung độ quan sát — **đầu ra** $y_i$ | vô hướng |
| $A, B, C, D, \dots$ | hệ số đa thức — $\beta_0, \beta_1, \beta_2, \beta_3$ của bài này | vô hướng |

**Ý nghĩa.** Câu này chứa **toàn bộ ý tưởng** của hồi quy đa thức, và có một chi tiết thường bị bỏ
khi kể lại: số số hạng **không được chọn tự do** mà lấy bằng số số hạng *"qu'on en admettrait si […]
on ne se proposait d'employer qu'une partie"* — bằng số hạng của công thức nội suy qua **vài** điểm mà
nhà thiên văn vẫn dùng. Nghĩa là Gergonne giữ nguyên **bậc** của phương pháp cũ, chỉ đổi **dữ liệu**
từ vài điểm thành tất cả. Câu hỏi "bậc nào mới đúng" ông để lại ở tr.252 (mục 3.9).

![Gergonne 1815 tr.250 — hệ phương trình chuẩn (1)](<paper_poly_gergonne1815_tr250_phuong_trinh_chuan.png>)

**Hình 8 — Gergonne 1815, tr.250 (PDF trang 10).** Khung cắt từ trang văn bản: tổng bình phương, đạo
hàm bằng 0, ký hiệu $\Sigma a^m$, hệ (1), và câu tự đánh giá độ chính xác.

Sai số của từng quan sát là $A + Ba + Ca^2 + Da^3 + \dots - b$. Làm tổng bình phương nhỏ nhất, đạo hàm
theo $A, B, C, D$, rồi *"À cause de l'indépendance entre A, B, C, ..... les multiplicateurs de dA, dB,
dC, .... devront séparément être nuls"* (vì A, B, C độc lập, hệ số của dA, dB, dC phải bằng 0 riêng
từng cái). Đặt $\Sigma a^m = a^m + a'^m + a''^m + \dots$ và $\Sigma a^mb = a^mb + a'^mb' + a''^mb'' +
\dots$, Gergonne được:

$$\begin{aligned}
\Sigma a^0\cdot A + \Sigma a\cdot B + \Sigma a^2\cdot C + \Sigma a^3\cdot D + \dots &= \Sigma a^0b\\
\Sigma a\cdot A + \Sigma a^2\cdot B + \Sigma a^3\cdot C + \Sigma a^4\cdot D + \dots &= \Sigma ab\\
\Sigma a^2\cdot A + \Sigma a^3\cdot B + \Sigma a^4\cdot C + \Sigma a^5\cdot D + \dots &= \Sigma a^2b\\
&\ \,\vdots
\end{aligned} \tag{1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\Sigma a^m$ | tổng luỹ thừa bậc $m$ của mọi hoành độ; $\Sigma a^0$ là số quan sát | vô hướng |
| $\Sigma a^m b$ | tổng của (hoành độ mũ $m$ × tung độ tương ứng) | vô hướng |

**Ý nghĩa.** Viết bằng ma trận, hệ này chính là $V^\top V\beta = V^\top y$: phần tử hàng $j$ cột $k$ của
$V^\top V$ là $\Sigma a^{j+k}$, phần tử hàng $j$ của $V^\top y$ là $\Sigma a^jb$ — đúng bảng kiểm ô ở mục
0.5. Nó nói rằng tại nghiệm, **phần dư không còn tương quan với cột nào** của $V$. Và vì chỉ phụ thuộc
$j+k$, cả hệ chỉ cần tính $2d+1$ tổng luỹ thừa, chứ không phải $(d+1)^2$ tích.

**Hai thứ cùng mang số (1).** Đa thức ở tr.249 và hệ phương trình ở tr.250 đều in số (1). Câu tr.251
*"les équations (1) deviendront"* (các phương trình (1) trở thành) chỉ **hệ ở tr.250**. Bài này ghi
$\text{(1)}_{\text{tr.249}}$ và $\text{(1)}_{\text{tr.250}}$ khi cần phân biệt.

**Gergonne tự đánh giá.** Tr.250, ngay sau hệ (1):

> *"et, tandis que les méthodes ordinaires donnent pour y et ses coefficiens différentiels des valeurs
> d'une précision toujours un peu inférieure à celle des données d'après lesquelles on les calcule, on
> pourra le plus souvent espérer ici de l'emporter en précision sur ces données elles-mêmes."*
>
> (và trong khi các phương pháp thường cho y và các hệ số vi phân những giá trị có độ chính xác luôn
> kém hơn một chút so với số liệu dùng để tính, thì ở đây phần lớn trường hợp có thể hy vọng **vượt cả
> độ chính xác của chính số liệu**.)

**Dựng lại** (`v_poly.py` mục 4) — độ lệch chuẩn của đa thức khớp tại **giữa** ($x=0$) và tại **mép**
($x=n$), tính theo độ lệch chuẩn $\sigma$ của **một** quan sát, với $2n+1$ quan sát cách đều:

| Quan sát | bậc 2: giữa | bậc 2: mép | bậc 4: giữa | bậc 4: mép |
|---:|---:|---:|---:|---:|
| 3 | 1,000000 | 1,000000 | — | — |
| 5 | 0,696932 | 0,941124 | 1,000000 | 1,000000 |
| 7 | 0,577350 | 0,872872 | 0,753060 | 0,993485 |
| 11 | 0,455477 | 0,761853 | 0,577350 | 0,957123 |
| 21 | 0,327950 | 0,596905 | 0,411379 | 0,841539 |
| 41 | 0,234377 | 0,446561 | 0,293235 | 0,679283 |

Câu của Gergonne **đúng ở giữa khoảng**: ngay khi số quan sát vượt số hệ số, giá trị khớp chính xác hơn
một quan sát (sd dưới 1), và 21 quan sát bậc 2 cho 0,3280. Dòng "3 quan sát, bậc 2" bằng đúng 1 vì đó là
nội suy — không có gì để bù trừ. Hai điều câu đó không nói, và bảng nói:

- **Ở mép khoảng thì chậm hơn hẳn**: 21 quan sát bậc 2 còn 0,5969 ở mép, gần gấp đôi ở giữa — khớp với
  lời dặn tr.244 đừng dùng công thức sát điểm lớn nhất hay nhỏ nhất.
- **Bậc cao hơn thì trả nhiều hơn**: cùng 21 quan sát, bậc 4 cho 0,4114 ở giữa. Mỗi hệ số thêm vào ăn
  bớt một phần bù trừ.

### 3.7 Gergonne tr.251 — $x$ cách đều: hệ số chẵn và lẻ tách được

![Gergonne 1815 tr.251 — trường hợp cách đều](<paper_poly_gergonne1815_tr251_cach_deu.png>)

**Hình 9 — Gergonne 1815, tr.251 (PDF trang 11).** Khung cắt từ trang văn bản: đánh số $-n \dots n$,
tổng luỹ thừa, hệ (1) tách đôi, và mở đầu trường hợp không cách đều.

**Gergonne nói gì.** Khi $x$ cách đều, đánh số lại thành $-n, \dots, -1, 0, 1, \dots, n$ với tổng
$2n+1$ quan sát. Gọi $\Sigma n^m = 1^m + 2^m + \dots + n^m$ thì

$$\Sigma a^0 = 2n+1,\quad \Sigma a = 0,\quad \Sigma a^2 = 2\Sigma n^2,\quad \Sigma a^3 = 0,\quad \Sigma a^4 = 2\Sigma n^4, \dots$$

**Ý nghĩa.** Hoành độ đối xứng quanh 0 thì mỗi số $k$ có một số $-k$ đi kèm; luỹ thừa lẻ của hai số
triệt tiêu nhau, luỹ thừa chẵn cộng dồn gấp đôi. Nên chỉ còn nửa số tổng cần tính, và mọi tổng ấy là tổng
luỹ thừa của $1, 2, \dots, n$ — thứ đã có công thức đóng.

Thế vào, hệ (1) tách thành hai hệ độc lập:

$$\begin{aligned}
(2n+1)A + 2\Sigma n^2\cdot C + \dots &= \Sigma b, & 2\Sigma n^2\cdot B + 2\Sigma n^4\cdot D + \dots &= \Sigma ab,\\
2\Sigma n^2\cdot A + 2\Sigma n^4\cdot C + \dots &= \Sigma a^2b, & 2\Sigma n^4\cdot B + 2\Sigma n^6\cdot D + \dots &= \Sigma a^3b,\\
2\Sigma n^4\cdot A + 2\Sigma n^6\cdot C + \dots &= \Sigma a^4b, & 2\Sigma n^6\cdot B + 2\Sigma n^8\cdot D + \dots &= \Sigma a^5b.
\end{aligned}$$

**Ý nghĩa.** Cột trái chỉ chứa hệ số hạng chẵn $A, C, \dots$; cột phải chỉ chứa hệ số hạng lẻ $B, D,
\dots$. Không phương trình nào trộn hai loại, vì mọi ô nối một hệ số chẵn với một hệ số lẻ là một tổng
luỹ thừa lẻ, đã bằng 0. Ma trận $V^\top V$ vì thế có **số 0 xen kẽ như bàn cờ**.

Gergonne rút ra lợi ích, tr.251:

> *"ainsi, outre que les sommes de puissances semblables des nombres naturels sont données par des
> formules connues et générales, on aura ici l'avantage de pouvoir calculer séparément les coefficiens de
> rangs pairs et ceux de rangs impairs, ce qui simplifiera le travail d'une manière notable."*
>
> (như vậy, ngoài chuyện tổng luỹ thừa của các số tự nhiên đã có công thức tổng quát, ta còn được lợi là
> **tính riêng các hệ số hạng chẵn và hạng lẻ**, giảm công việc đáng kể.)

**Hệ quả thực hành.** Một hệ $4\times4$ thành hai hệ $2\times2$ — với người tính bằng tay, số phép khử
giảm hẳn. Cách làm hiện đại tương ứng là **trừ trung bình của $x$ trước khi khớp** (bước 1–2 của mục 5):
với $x$ cách đều, sau bước đó hoành độ đối xứng quanh 0 đúng như Gergonne đánh số.

**Dựng lại** (`v_poly.py` mục 5): $\Sigma a^{\text{lẻ}} = 0$, $\Sigma a^2 = 2\cdot\frac{n(n+1)(2n+1)}6$,
$\Sigma a^4 = 2\cdot\frac{n(n+1)(2n+1)(3n^2+3n-1)}{30}$ đúng với $n = 1..7$. Một ví dụ số: 7 điểm $x =
-3..3$, $y = (4, 2, 1, 1, 3, 6, 12)$, bậc 3. Ma trận chuẩn:

```
     7      0     28      0
     0     28      0    196
    28      0    196      0
     0    196      0   1588
```

Hệ đủ $4\times4$ cho $A = \tfrac{23}{21}$, $B = \tfrac{52}{63}$, $C = \tfrac{16}{21}$, $D = \tfrac1{18}$;
hai hệ $2\times2$ cho **đúng cùng bốn phân số**. Ví dụ này của bài, không của Gergonne — bài báo không
có ví dụ số nào có dữ liệu.

### 3.8 Gergonne tr.251–252 — $x$ không cách đều: đổi biến, và hai chỗ in sai

![Gergonne 1815 tr.252 — công thức đổi biến, tự đánh giá, và câu hỏi bậc](<paper_poly_gergonne1815_tr252_doi_bien_bac.png>)

**Hình 10 — Gergonne 1815, tr.252 (PDF trang 12).** Khung cắt từ trang văn bản: công thức đổi biến,
lời tự đánh giá, và bài toán để ngỏ.

**Gergonne nói gì.** Tr.251: khi cả $x$ lẫn $y$ không cách đều, coi cả hai là hàm của một biến thứ ba
$z$ đánh số $-n..n$, dùng hệ tách đôi để tìm $\frac{dx}{dz}, \frac{dy}{dz}, \frac{d^2x}{dz^2},
\frac{d^2y}{dz^2}$, rồi đổi biến. Trang 252 in:

$$\frac{dy}{dx} = \frac{\dfrac{dy}{dz}}{\dfrac{d\lambda}{dx}}, \qquad
\frac{d^2y}{dx^2} = \frac{\dfrac{dx}{dz}\dfrac{d^2y}{dz^2} - \dfrac{dy}{dz}\dfrac{d^2x}{dz^2}}{\left(\dfrac{dx}{dz}\right)^{2}}
\qquad \text{(bản in)}$$

Và lý do chọn cách này, tr.252: *"cette méthode me semblerait préférable à celle qui consiste à
interpoler entre les observations, afin de les rendre équidistantes ; attendu qu'il peut être dangereux,
dans un problème d'une nature aussi délicate, de dénaturer les données avant d'en faire usage."* (cách này
có vẻ hơn cách nội suy giữa các quan sát cho chúng cách đều, vì trong một bài toán tinh tế như vậy, **làm
biến dạng số liệu trước khi dùng** có thể nguy hiểm.)

**Dựng lại** (`v_poly.py` mục 6) — hai phép đổi biến có đáp án đóng:

| Phép đổi | $dy/dx$ thật | $\frac{dy/dz}{dx/dz}$ | $d^2y/dx^2$ thật | mẫu mũ 2 (bản in) | mẫu mũ 3 |
|---|---|---|---|---|---|
| $x = z^2$, $y = z^3$, tại $z = 2$ | 3 | 3 | 3/8 | **3/2** | 3/8 |
| $x = 3z+1$, $y = z^4$, tại $z = 1$ | 4/3 | 4/3 | 4/3 | **4** | 4/3 |

**Hai chỗ in sai trong một dòng.** Mẫu của $dy/dx$ in là $d\lambda/dx$ — không có biến $\lambda$ nào
trong bài; đúng là $dx/dz$. Mẫu của $d^2y/dx^2$ in số mũ 2; đúng là **3**:

$$\frac{d^2y}{dx^2} = \frac{\dfrac{dx}{dz}\dfrac{d^2y}{dz^2} - \dfrac{dy}{dz}\dfrac{d^2x}{dz^2}}{\left(\dfrac{dx}{dz}\right)^{3}}$$

**Ý nghĩa.** Công thức nói độ cong theo $x$ bằng độ cong theo $z$ trừ phần cong **do chính phép đổi
thang** $x(z)$ gây ra, rồi chia cho $(dx/dz)^3$: một luỹ thừa vì $dy/dx$ đã chia một lần cho $dx/dz$,
thêm hai luỹ thừa vì đạo hàm lần nữa theo $x$ lại chia cho $dx/dz$. Mũ 2 chỉ cho đúng khi $dx/dz = 1$,
tức khi $z$ **chính là** $x$ — đúng trường hợp không cần đổi biến. Dòng 2 của bảng là chỗ dễ bị lừa
nhất: phép đổi tuyến tính $x = 3z+1$ có $d^2x/dz^2 = 0$, nên người kiểm bằng một phép đổi tuyến tính
sẽ thấy tử số "đúng" và dễ bỏ qua mẫu số — mà mẫu mũ 2 vẫn cho 4 thay vì 4/3.

### 3.9 Gergonne tr.252 — tự đánh giá, và câu hỏi "bậc nào"

**Gergonne nói gì về phương pháp Laplace sau khi thêm bình phương tối thiểu.** Tr.252:

> *"Il nous paraît que l'introduction des procédés que nous venons d'indiquer, dans la méthode de M.
> Laplace, pour la détermination des orbites des comètes, ne peut qu'ajouter beaucoup à sa précision, du
> moins dans le cas où l'on peut disposer d'un grand nombre d'observations ; mais cette méthode, comme
> beaucoup d'autres, aura toujours le grave inconvénient de n'être, au fond, qu'un tatonnement bien
> dirigé."*
>
> (Có vẻ việc đưa các cách làm vừa nêu vào phương pháp xác định quỹ đạo sao chổi của Laplace chỉ có
> thể tăng thêm nhiều độ chính xác, ít nhất khi có nhiều quan sát; nhưng **phương pháp đó**, như nhiều
> phương pháp khác, sẽ luôn mang một bất tiện nặng là về bản chất chỉ là **một phép mò mẫm có định
> hướng**.)

*"Cette méthode"* ở đây là **phương pháp Laplace**, không phải bình phương tối thiểu — lời chê nhắm vào
việc tính quỹ đạo bằng đạo hàm của kinh độ, thứ Legendre tr.v cũng chọn bỏ.

**Và bài toán Gergonne để ngỏ.** Tr.252, câu cuối bài:

> *"Il resterait ici au surplus un autre problème à résoudre, lequel pourrait être énoncé comme il
> suit : On sait que des points donnés, en nombre quelconque, sont à peu près situés sur une courbe
> parabolique d'un degré déterminé, mais inconnu ; et l'on demande de découvrir quel est le plus
> probablement le degré de cette courbe ? La solution de ce problème lèverait complètement l'incertitude
> du calculateur qui, voulant appliquer la méthode de M. Laplace, se trouve pouvoir disposer d'un grand
> nombre d'observations."*
>
> (Còn lại một bài toán khác cần giải, phát biểu như sau: biết rằng các điểm cho trước, số lượng bất
> kỳ, nằm gần trên một đường đa thức có **bậc xác định nhưng chưa biết**; hãy tìm xem **bậc nào có khả
> năng nhất**? Lời giải sẽ xoá hẳn nỗi lưỡng lự của người tính toán khi có trong tay nhiều quan sát.)

**Ý nghĩa.** Đây là lần đầu bài toán **chọn mô hình** được phát biểu cho hồi quy đa thức, và nó được
phát biểu bằng từ *"le plus probablement"* — tức như một bài toán **xác suất**, không phải một bài toán
khớp. Mục 1.2 cho thấy vì sao nó không tránh được: bậc thấp thì chệch, bậc cao thì phương sai lớn, và
không có bậc nào tốt nhất mà không nhìn vào dữ liệu. Kirstine Smith (1918) nghiên cứu cách **đặt các
điểm đo** cho hồi quy đa thức; Bishop 2006 trả lời bằng tập kiểm định (mục 3.13); BIC — một xấp xỉ của
đúng câu "bậc nào có xác suất hậu nghiệm lớn nhất" — ở mục 7.2.

### 3.10 Bishop mục 1.1 — mô hình (1.1), hàm lỗi (1.2), và bộ dữ liệu 10 điểm (tr.4–6)

Gergonne để lại hai thứ chưa làm: ông không đưa ra **một ví dụ số nào có dữ liệu**, và ông không trả
lời được câu hỏi **bậc nào**. Bishop mục 1.1 làm cả hai trên một bộ dữ liệu nhân tạo 10 điểm, và dùng
nó làm ví dụ mở đầu cho cả cuốn sách. Mục này là nguồn hiện đại của bài; mọi hình và bảng của mục 1.1
đều ở dưới, theo đúng thứ tự sách.

![Công thức (1.1) của Bishop 2006, tr.5](<paper_bishop2006_cong_thuc_1_1.png>)

**Hình 11 — Công thức (1.1).** Bishop 2006, *Pattern Recognition and Machine Learning*, tr.5 (PDF
trang 25), `_papers/2006 - Bishop - Pattern Recognition and Machine Learning.pdf`.

$$y(x,\mathbf w) = w_0 + w_1x + w_2x^2 + \dots + w_Mx^M = \sum_{j=0}^{M} w_jx^j \tag{1.1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $x$ | giá trị đầu vào | vô hướng |
| $M$ | bậc (*order*) của đa thức — bài này gọi là $d$ | số nguyên $\ge 0$ |
| $w_j$ | hệ số của $x^j$ — bài này gọi là $\beta_j$ | vô hướng |
| $\mathbf w$ | vector gom $w_0, \dots, w_M$ | $\mathbb{R}^{M+1}$ |
| $y(x, \mathbf w)$ | giá trị đa thức tại $x$ | vô hướng |

Bishop tr.5 bình luận ngay sau công thức: *"Note that, although the polynomial function y(x, w) is
a nonlinear function of x, it is a linear function of the coefficients w."*

**Ý nghĩa.** Công thức nói rằng mọi đa thức bậc $M$ là **một tổ hợp tuyến tính của $M+1$ hàm cố
định** $1, x, \dots, x^M$. Người học chỉ chọn trọng số của tổ hợp, không chọn hình dạng của các hàm.
Đó là lý do toàn bộ máy móc của hồi quy tuyến tính dùng được nguyên xi — và cũng là lý do mô hình bị
giới hạn bởi đúng bộ hàm đã chọn trước.

![Công thức (1.2) của Bishop 2006, tr.5](<paper_bishop2006_cong_thuc_1_2.png>)

**Hình 12 — Công thức (1.2).** Bishop 2006, tr.5 (PDF trang 25).

$$E(\mathbf w) = \frac12 \sum_{n=1}^{N} \{y(x_n, \mathbf w) - t_n\}^2 \tag{1.2}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $N$ | số điểm huấn luyện | số nguyên |
| $x_n$, $t_n$ | đầu vào và **giá trị đích** (*target*) của điểm thứ $n$ — bài này gọi là $x_i$, $y_i$ | vô hướng |
| $E(\mathbf w)$ | hàm lỗi: một nửa tổng bình phương phần dư | vô hướng $\ge 0$ |

Bishop tr.5 giải thích hệ số $\tfrac12$: *"where the factor of 1/2 is included for later
convenience"*.

**Ý nghĩa.** Đây chính là *"rendre minimum la somme de leurs quarrés"* của Gergonne tr.249, viết lại
191 năm sau. Hệ số $\tfrac12$ không đổi nghiệm — nhân một hàm với hằng số dương không dời điểm cực
tiểu của nó — mà chỉ để đạo hàm gọn hơn. Nhưng nó **đổi giá trị** của hàm lỗi, và chính chỗ đó làm
Hình 1.5 lệch khỏi định nghĩa (1.3) một hệ số $\sqrt2$ (Hình 17).

![Hình 1.2 của Bishop 2006 — 10 điểm huấn luyện và đường sin(2πx)](<paper_bishop2006_fig1_2_du_lieu.png>)

**Hình 13 — Hình 1.2 của Bishop 2006, tr.4 (PDF trang 24).**

**Caption gốc.** *"Plot of a training data set of N = 10 points, shown as blue circles, each
comprising an observation of the input variable x along with the corresponding target variable t.
The green curve shows the function sin(2πx) used to generate the data. Our goal is to predict the
value of t for some new value of x, without knowledge of the green curve."*

(Đồ thị một tập huấn luyện gồm N = 10 điểm, vẽ bằng vòng tròn xanh, mỗi điểm gồm một quan sát của
biến đầu vào x cùng biến đích t tương ứng. Đường xanh lá là hàm sin(2πx) dùng để sinh dữ liệu. Mục
tiêu là dự đoán t cho một giá trị x mới, mà không biết đường xanh lá.)

**Tác giả nói gì về hình.** Tr.4: *"The input data set x in Figure 1.2 was generated by choosing
values of xn, for n = 1,...,N, spaced uniformly in range [0, 1], and the target data set t was
obtained by first computing the corresponding values of the function"* — rồi sang tr.5: *"sin(2πx)
and then adding a small level of random noise having a Gaussian distribution"*. Phụ lục A tr.682 cho
độ lệch chuẩn: *"then adding random noise with a Gaussian distribution having standard deviation
0.3."*

(Tập đầu vào x ở Hình 1.2 được sinh bằng cách chọn các giá trị xn, n = 1,…,N, **cách đều trong đoạn
[0, 1]**, và tập đích t có được bằng cách tính giá trị sin(2πx) tương ứng rồi cộng một lượng nhỏ nhiễu
Gauss — độ lệch chuẩn 0,3.)

**Cách đọc hình.** Mười vòng tròn nằm ở mười hoành độ đều nhau, điểm đầu ở $x=0$ và điểm cuối ở
$x=1$. Đường xanh lá là hàm thật, mà trong bài toán thật thì không ai thấy. Khoảng cách dọc từ vòng
tròn tới đường xanh lá là nhiễu.

**Điều hình không nói.** Sách **không in 10 giá trị $t_n$**. Nhưng chúng dựng lại được, và việc dựng
lại là phép kiểm mạnh nhất của cả mục 3.11 (Hình 19). Và có một chỗ lệch nhỏ giữa hai câu mô tả: mục
1.1 nói $x$ *"spaced uniformly in range [0, 1]"* (cách đều, **có** hai đầu mút), Phụ lục A nói *"The
input values {xn} are generated uniformly in range (0, 1)"* — cách viết đó thường hiểu là rút ngẫu
nhiên trên khoảng **mở**. Phép dựng lại ở mục 3.11 đứng về phía mục 1.1: chỉ lưới
`linspace(0, 1, 10)` — có $x=0$ và $x=1$ — khớp được Bảng 1.1.

![Hình 1.3 của Bishop 2006 — hàm lỗi là tổng bình phương các đoạn dọc](<paper_bishop2006_fig1_3_ham_loi.png>)

**Hình 14 — Hình 1.3 của Bishop 2006, tr.6 (PDF trang 26).**

**Caption gốc.** *"The error function (1.2) corresponds to (one half of) the sum of the squares of
the displacements (shown by the vertical green bars) of each data point from the function y(x, w)."*

(Hàm lỗi (1.2) bằng (một nửa) tổng bình phương các độ dời — các vạch xanh lá thẳng đứng — của mỗi
điểm dữ liệu so với hàm y(x, w).)

**Tác giả nói gì về hình.** Tr.6: *"The geometrical interpretation of the sum-of-squares error
function is illustrated in Figure 1.3."* Và ngay sau: *"We can solve the curve fitting problem by
choosing the value of w for which E(w) is as small as possible."*

**Cách đọc hình.** Đường đỏ là một đa thức đang thử. Mỗi vạch xanh lá nối một điểm dữ liệu với
đường đỏ **theo phương thẳng đứng** — tức đo sai lệch theo $t$, không theo khoảng cách ngắn nhất tới
đường cong.

**Điều hình không nói.** Đo theo phương thẳng đứng là một **giả định**: sai số chỉ nằm ở $t$, còn $x$
đo chính xác. Gergonne tr.251–252 gặp đúng giới hạn này khi $x$ không cách đều và phải đổi biến (mục
3.8). Khi $x$ cũng có sai số đo, bình phương tối thiểu thường cho hệ số góc chệch về 0.

### 3.11 Bishop — bốn bậc, hàm $E_{\text{RMS}}$ (1.3), Hình 1.4–1.5 và Bảng 1.1 (tr.6–9)

![Hình 1.4 của Bishop 2006 — đa thức bậc 0, 1, 3, 9 khớp cùng một bộ 10 điểm](<paper_bishop2006_fig1_4_bac_da_thuc.png>)

**Hình 15 — Hình 1.4 của Bishop 2006, tr.7 (PDF trang 27).**

**Caption gốc.** *"Plots of polynomials having various orders M, shown as red curves, fitted to the
data set shown in Figure 1.2."*

(Đồ thị các đa thức với nhiều bậc M khác nhau, vẽ bằng đường đỏ, khớp vào bộ dữ liệu của Hình 1.2.)

**Tác giả nói gì về hình.** Tr.6: *"The third order (M = 3) polynomial seems to give the best fit to
the function sin(2πx) of the examples shown in Figure 1.4."* Rồi: *"we obtain an excellent fit to
the training data"* — *"However, the fitted curve oscillates wildly and gives a very poor
representation of the function sin(2πx). This latter behaviour is known as over-fitting."*

(Đa thức bậc ba có vẻ khớp hàm sin(2πx) tốt nhất trong các ví dụ ở Hình 1.4. [Với M = 9] ta được một
đường khớp dữ liệu huấn luyện rất tốt — nhưng đường khớp dao động dữ dội và biểu diễn hàm sin(2πx)
rất kém. Hành vi sau cùng này gọi là khớp quá mức.)

**Cách đọc hình.** Cùng 10 điểm ở cả bốn ô.

| Ô | $M$ | Đường đỏ | Tên |
|---|---|---|---|
| trên trái | 0 | đường ngang ở $0{,}19$ — trung bình của 10 giá trị | khớp thiếu |
| trên phải | 1 | đường thẳng dốc xuống | khớp thiếu |
| dưới trái | 3 | bám sát đường xanh lá dù chưa từng nhìn thấy nó | vừa |
| dưới phải | 9 | đi qua **đúng** cả 10 điểm, vọt ra khỏi khung ở hai đầu | khớp quá mức |

Ô dưới phải là hình quan trọng nhất: 10 hệ số trên 10 điểm là **nội suy** — đúng cái "phương pháp
thường" mà Gergonne tr.244 phê phán. Gergonne đo nó khuếch đại sai số của **đạo hàm**; Bishop cho thấy
nó khuếch đại sai số của **chính giá trị dự đoán** giữa các điểm.

**Điều hình không nói.** Hình không nói đường đỏ ở ô $M=9$ ra khỏi khung bao xa. Bảng 1.1 trả lời
gián tiếp: hệ số lớn nhất là $-1\,061\,800{,}52$, tức cỡ $10^{6}$ (`v_poly.py` mục 11 in
$\log_{10} = 6{,}0260$).

![Công thức (1.3) của Bishop 2006, tr.7](<paper_bishop2006_cong_thuc_1_3.png>)

**Hình 16 — Công thức (1.3).** Bishop 2006, tr.7 (PDF trang 27).

$$E_{\text{RMS}} = \sqrt{2E(\mathbf w^\star)/N} \tag{1.3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf w^\star$ | hệ số tìm được — điểm cực tiểu của $E$ | $\mathbb{R}^{M+1}$ |
| $E(\mathbf w^\star)$ | giá trị hàm lỗi (1.2) tại đó | vô hướng |
| $E_{\text{RMS}}$ | căn trung bình bình phương phần dư | cùng đơn vị với $t$ |

Bishop tr.7 giải thích: *"in which the division by N allows us to compare different sizes of data
sets on an equal footing, and the square root ensures that ERMS is measured on the same scale (and in
the same units) as the target variable t."*

**Ý nghĩa.** Thay $E = \tfrac12\sum e_n^2$ vào thì $2E/N = \tfrac1N\sum e_n^2$, nên $E_{\text{RMS}}$
**đúng bằng RMSE thường**. Số 2 trong căn là để bù đúng hệ số $\tfrac12$ của (1.2). Bỏ số 2 đi thì
con số nhỏ đi $\sqrt2 = 1{,}4142$ lần — và đó chính là con số Hình 1.5 vẽ.

![Hình 1.5 của Bishop 2006 — E_RMS huấn luyện và kiểm tra theo M](<paper_bishop2006_fig1_5_erms_theo_M.png>)

**Hình 17 — Hình 1.5 của Bishop 2006, tr.8 (PDF trang 28).**

**Caption gốc.** *"Graphs of the root-mean-square error, defined by (1.3), evaluated on the training
set and on an independent test set for various values of M."*

(Đồ thị sai số căn trung bình bình phương, **định nghĩa theo (1.3)**, tính trên tập huấn luyện và
trên một tập kiểm tra độc lập, cho nhiều giá trị M.)

**Tác giả nói gì về hình.** Tr.6 mô tả tập kiểm tra: *"a separate test set comprising 100 data points
generated using exactly the same procedure used to generate the training set points but with new
choices for the random noise values included in the target values."* Tr.8: *"For M = 9, the training
set error goes to zero, as we might expect because this polynomial contains 10 degrees of freedom
corresponding to the 10 coefficients"* và *"However, the test set error has become very large"*. Rồi
Bishop tự nêu nghịch lý: *"This may seem paradoxical because a polynomial of given order contains all
lower order polynomials as special cases."*

**Cách đọc hình.** Trục ngang là bậc $M$, trục dọc là sai số. Đường xanh (huấn luyện) chỉ đi xuống;
đường đỏ (kiểm tra) có đáy bằng từ $M=3$ tới $M=8$ rồi vọt lên ở $M=9$.

**Điều hình không nói — và đo được.** Caption nói trục dọc là $E_{\text{RMS}}$ **theo (1.3)**. Số hoá
điểm ảnh 10 vòng tròn xanh (`v_poly.py` mục 9) rồi đặt cạnh RMSE của phép khớp trên 10 điểm dựng lại
ở Hình 19:

| $M$ | Hình 1.5, đường huấn luyện (số hoá) | RMSE dựng lại = (1.3) | RMSE/$\sqrt2$ = $\sqrt{E/N}$ |
|---:|---:|---:|---:|
| 0 | 0,4363 | 0,6146 | 0,4346 |
| 1 | 0,3281 | 0,4621 | 0,3268 |
| 2 | 0,3253 | 0,4587 | 0,3243 |
| 3 | 0,1342 | 0,1870 | 0,1322 |
| 4 | 0,1342 | 0,1868 | 0,1321 |
| 5 | 0,1329 | 0,1850 | 0,1308 |
| 6 | 0,1301 | 0,1817 | 0,1285 |
| 7 | 0,1295 | 0,1797 | 0,1271 |
| 8 | 0,1212 | 0,1682 | 0,1189 |
| 9 | 0,0034 | 0,0000 | 0,0000 |

Tỉ số "hình / RMSE" ở $M = 0..8$ nằm trong **0,7093–0,7208**, trung bình **0,716**, trong khi
$1/\sqrt2 = 0{,}7071$. Phần dư 1 % là độ dày nét vẽ và bán kính vòng tròn khi đọc điểm ảnh. Kết luận:
**đường huấn luyện của Hình 1.5 là $\sqrt{E/N}$, không phải $\sqrt{2E/N}$ như caption viết.** Không
kết luận nào của Bishop đổi vì chỗ này — thứ hạng giữa các $M$ giữ nguyên dưới mọi hằng số nhân —
nhưng ai đem đường ở $M=3$ (khoảng 0,13) so với độ lệch chuẩn nhiễu 0,3 sẽ tưởng mô hình khớp **dưới
cả mức nhiễu** hơn hai lần, trong khi RMSE thật là 0,1870.

Đường kiểm tra không kiểm được theo cùng cách, vì 100 điểm kiểm tra không được in. Tập kiểm tra tự
sinh 100 điểm cho $\sqrt{E/N}$ ở $M=0$ là 0,5576 so với 0,5644 đọc từ hình — phù hợp với cùng hệ số
$\sqrt2$, nhưng đó chỉ là phù hợp, không phải phép kiểm.

![Bảng 1.1 của Bishop 2006 — hệ số w* theo bậc](<paper_bishop2006_bang1_1_he_so.png>)

**Hình 18 — Bảng 1.1 của Bishop 2006, tr.8 (PDF trang 28).**

**Caption gốc.** *"Table of the coefficients w⋆ for polynomials of various order. Observe how the
typical magnitude of the coefficients increases dramatically as the order of the polynomial
increases."*

(Bảng hệ số w⋆ của các đa thức nhiều bậc. Hãy để ý độ lớn điển hình của hệ số tăng vọt khi bậc
tăng.)

**Tác giả nói gì về bảng.** Tr.8: *"We see that, as M increases, the magnitude of the coefficients
typically gets larger. In particular for the M = 9 polynomial, the coefficients have become finely
tuned to the data by developing large positive and negative values"*.

**Cách đọc bảng.** Mỗi cột là một đa thức; mỗi dòng là một hệ số. Cột $M=0$ có một ô, cột $M=1$ có
hai ô. Cột thứ ba **mang nhãn "$M = 6$" nhưng chỉ có bốn ô**.

**Điều bảng không nói — và dựng lại được.** Đa thức $M=9$ đi qua **đúng** 10 điểm (tr.6 viết
*"the polynomial passes exactly through each data point"*). Vậy lấy 10 hệ số của cột $M=9$, tính đa
thức tại 10 hoành độ, là **dựng lại được chính 10 điểm dữ liệu** mà sách không in. Rồi khớp lại các
bậc thấp trên 10 điểm đó và so với các cột còn lại. Thử bốn cách đặt hoành độ
(`tools/v_poly.py` mục 7):

| Lưới $x$ | lệch max cột $M=0$ | lệch max cột $M=1$ | cột "$M=6$" so với $M=3$ | so với $M=6$ thật |
|---|---:|---:|---:|---:|
| **`linspace(0,1,10)`** | **0,0035** | **0,0013** | **0,0042** | 223,6263 |
| `(i+0,5)/10` | 0,0906 | 3,6461 | 80,4421 | 1054,2913 |
| `i/10`, $i = 0..9$ | 0,0209 | 0,8531 | 5,9051 | 1229,3224 |
| `(i+1)/11` | 0,0470 | 1,7839 | 29,4599 | 167,7656 |

Chỉ lưới cách đều có hai đầu mút khớp, và nó khớp **trong mức làm tròn** 0,005 của bảng in. Mười điểm
dựng lại:

```
x = 0.000000 0.111111 0.222222 0.333333 0.444444 0.555556 0.666667 0.777778 0.888889 1.000000
t = 0.350000 0.831508 1.008099 0.972291 0.133766 0.167337 -0.848070 -0.445806 -0.564177 0.260000
```

| Cột | In | Dựng lại | Khoảng khi rút 4 000 lần sai số làm tròn $\pm0{,}005$ cho 10 hệ số của cột $M=9$ |
|---|---|---|---|
| $M=0$, $w_0$ | 0,19 | 0,1865 | [0,1757 ; 0,1970] |
| $M=1$, $w_0$ | 0,82 | 0,8213 | [0,8120 ; 0,8308] |
| $M=1$, $w_1$ | −1,27 | −1,2696 | [−1,2920 ; −1,2376] |
| "$M=6$", $w_0$ | 0,31 | **$M=3$: 0,3142** | [0,3089 ; 0,3197] |
| "$M=6$", $w_1$ | 7,99 | **$M=3$: 7,9866** | [7,9664 ; 8,0117] |
| "$M=6$", $w_2$ | −25,43 | **$M=3$: −25,4270** | [−25,5121 ; −25,3489] |
| "$M=6$", $w_3$ | 17,37 | **$M=3$: 17,3718** | [17,2871 ; 17,4647] |

**Nhãn "$M = 6$" là lỗi in của "$M = 3$".** Ba bằng chứng độc lập cùng chỉ về đó: cột chỉ có bốn hệ
số — đúng số hệ số của bậc 3; bốn hệ số ấy khớp $M=3$ dựng lại tới ba chữ số; và $M=6$ thật trên cùng
dữ liệu có bảy hệ số $0{,}35 \cdot 2{,}62 \cdot 32{,}09 \cdot -206{,}26 \cdot 398{,}95 \cdot
-332{,}64 \cdot 105{,}14$. Chính văn bản tr.6 cũng nói bậc được vẽ là *"M = 0, 1, 3, and 9"*.

Phép dựng lại còn cho hai con số kiểm chéo: phần dư của $M=9$ trên dữ liệu dựng lại là
$1{,}746\times10^{-10}$ (khớp câu $E(\mathbf w^\star)=0$ của tr.6), và $\kappa(V) = 1{,}5193\times10^{7}$
với $M=9$ trên lưới này — tức bài toán này còn trong vùng `float64` giải tốt, chưa tới vùng hỏng số học
của mục 6.

![Hình tự vẽ — dựng lại Bảng 1.1, Hình 1.5 và Bảng 1.2 từ chính số in của Bishop](<poly_bishop_dung_lai.png>)

**Hình 19 — Dựng lại mục 1.1 của Bishop từ chính các con số sách in (hình tự vẽ).** Hình này bù cho
đúng chỗ ba hình/bảng gốc không làm được: sách không in dữ liệu, nên người đọc không kiểm được bảng
nào khớp hình nào.

**Cách đọc hình.**

- **Ô trái** — 10 vòng tròn xanh là 10 điểm suy ra từ cột $M=9$ của Bảng 1.1. Dải đỏ nhạt là đa thức
  vẽ bằng bốn hệ số của cột in nhãn "$M = 6$"; nét đứt đen là $M=3$ khớp lại trên 10 điểm. Hai đường
  trùng nhau — so với Hình 15 ô dưới trái thì cũng là cùng một đường.
- **Ô giữa** — dấu × đỏ là 10 điểm số hoá từ Hình 1.5. Đường xám là RMSE theo đúng (1.3); đường xanh là
  RMSE/$\sqrt2$. Dấu × nằm trên đường xanh.
- **Ô phải** — với mỗi $\ln\lambda$ đưa vào (1.4), giải Ridge rồi đo độ lệch lớn nhất so với một cột
  của Bảng 1.2 (trục dọc log). Vạch chấm là nhãn in trên bảng; đáy đường cong là $\lambda$ thật sự cho
  ra cột đó. Vạch ngang đứt là mức làm tròn 0,005. Đọc ở mục 3.13.

**Điều hình không nói.** Ô giữa chỉ kiểm được đường **huấn luyện**; đường kiểm tra cần 100 điểm kiểm
tra mà sách không in.

<details>
<summary><b>Code sinh Hình 19</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Bishop 2006, Bảng 1.1 tr.8, cột M = 9 (in 2 chữ số thập phân)
W9 = np.array([0.35, 232.37, -5321.83, 48568.31, -231639.30, 640042.26,
               -1061800.52, 1042400.18, -557682.99, 125201.43])
W3_IN = np.array([0.31, 7.99, -25.43, 17.37])          # cột in nhãn "M = 6"
R18_IN = np.array([0.35, 4.74, -0.77, -31.97, -3.89, 55.28, 41.32, -45.95, -91.53, 72.68])
R0_IN = np.array([0.13, -0.05, -0.06, -0.05, -0.03, -0.02, -0.01, -0.00, 0.00, 0.01])
# Hình 1.5 tr.8 số hoá bằng điểm ảnh (tools/v_poly.py mục 9)
HINH15_HUAN = np.array([0.4363, 0.3281, 0.3253, 0.1342, 0.1342, 0.1329, 0.1301, 0.1295, 0.1212, 0.0034])

x = np.linspace(0, 1, 10)
V9 = np.vander(x, 10, increasing=True)
t = V9 @ W9                                            # E(w*) = 0 nên đa thức M = 9 đi qua đúng 10 điểm


def khop(M, lam=0.0):
    V = np.vander(x, M + 1, increasing=True)
    A = np.vstack([V, np.sqrt(lam) * np.eye(M + 1)])
    return np.linalg.lstsq(A, np.r_[t, np.zeros(M + 1)], rcond=None)[0]


g = np.linspace(0, 1, 400)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

# --- trái: 10 điểm dựng lại, M = 3 dựng lại chồng lên cột in "M = 6" ---
w3 = khop(3)
ax[0].plot(g, np.sin(2 * np.pi * g), color="#16a34a", lw=1.2, label="sin(2πx)")
ax[0].plot(g, np.vander(g, 4, increasing=True) @ W3_IN, color="#dc2626", lw=2.4, alpha=.45,
           label='cột in "M = 6"')
ax[0].plot(g, np.vander(g, 4, increasing=True) @ w3, "--", color="#111827", lw=1.1, label="M = 3 dựng lại")
ax[0].scatter(x, t, s=28, facecolors="none", edgecolors="#2563eb", lw=1.4, zorder=4, label="10 điểm dựng lại")
ax[0].set_ylim(-1.5, 1.5); ax[0].set_xlabel("x"); ax[0].set_ylabel("t"); ax[0].legend(fontsize=7)
ax[0].set_title('Từ cột M = 9 suy ra 10 điểm;\ncột in "M = 6" trùng M = 3 dựng lại', fontsize=9)

# --- giữa: Hình 1.5 số hoá so với RMSE và RMSE/√2 ---
rm = np.array([np.sqrt(((np.vander(x, M + 1, increasing=True) @ khop(M) - t) ** 2).mean()) for M in range(10)])
Ms = np.arange(10)
ax[1].plot(Ms, rm, "s-", color="#9ca3af", ms=4, lw=1.1, label="RMSE = √(2E/N), công thức (1.3)")
ax[1].plot(Ms, rm / np.sqrt(2), "o-", color="#2563eb", ms=4, lw=1.1, label="√(E/N) = RMSE/√2")
ax[1].plot(Ms, HINH15_HUAN, "x", color="#dc2626", ms=8, mew=1.8, label="Hình 1.5, đường huấn luyện (số hoá)")
ax[1].set_xlabel("M"); ax[1].set_ylabel("sai số huấn luyện"); ax[1].legend(fontsize=7)
ti = HINH15_HUAN[:9] / rm[:9]
ax[1].set_title("Hình 1.5 vẽ √(E/N), không phải (1.3)\ntỉ số hình / RMSE trung bình %.4f" % ti.mean(), fontsize=9)

# --- phải: quét λ cho hai cột của Bảng 1.2 ---
LL = np.arange(-30, 10.0001, 0.02)
for inb, l_in, mau, ten in ((R18_IN, -18, "#7c3aed", "cột ln λ = −18"), (R0_IN, 0, "#d97706", "cột ln λ = 0")):
    lech = np.array([np.abs(khop(9, np.exp(l)) - inb).max() for l in LL])
    i = int(lech.argmin())
    ax[2].semilogy(LL, lech, color=mau, lw=1.2, label="%s — tốt nhất ln λ = %.2f" % (ten, LL[i]))
    ax[2].axvline(l_in, color=mau, ls=":", lw=1)
    print("%s: ln λ tốt nhất %.2f, lệch max %.4f, cách nhãn in %+.2f" % (ten, LL[i], lech[i], LL[i] - l_in))
ax[2].axhline(0.005, color="#111827", ls="--", lw=.8)
ax[2].text(-29.5, 0.0065, "±0,005 (mức làm tròn)", fontsize=7)
ax[2].set_xlabel("ln λ đưa vào (1.4)"); ax[2].set_ylabel("max |hệ số dựng − hệ số in|"); ax[2].legend(fontsize=7)
ax[2].set_title("Bảng 1.2: nhãn in lệch khỏi λ của (1.4)\nvạch chấm = nhãn in, đáy = λ khớp nhất", fontsize=9)

fig.savefig("poly_bishop_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("M = 3 dựng lại:", " ".join("%.6f" % v for v in w3))
print("tỉ số Hình 1.5 / RMSE, M = 0..8:", " ".join("%.4f" % v for v in ti), "· trung bình %.4f" % ti.mean())
print("RMSE dựng lại M = 0..9:", " ".join("%.6f" % v for v in rm))
```

Đầu ra trên máy dùng để viết bài (numpy 2.4.4, Linux x86-64):

```text
cột ln λ = −18: ln λ tốt nhất -16.08, lệch max 0.0644, cách nhãn in +1.92
cột ln λ = 0: ln λ tốt nhất 2.28, lệch max 0.0034, cách nhãn in +2.28
M = 3 dựng lại: 0.314235 7.986641 -25.427033 17.371757
tỉ số Hình 1.5 / RMSE, M = 0..8: 0.7099 0.7100 0.7093 0.7177 0.7183 0.7182 0.7161 0.7205 0.7206 · trung bình 0.7156
RMSE dựng lại M = 0..9: 0.614611 0.462128 0.458652 0.186989 0.186823 0.185036 0.181683 0.179736 0.168183 0.000000
```

Mảng `HINH15_HUAN` là 10 giá trị số hoá điểm ảnh từ PDF, in ra bởi `tools/v_poly.py` mục 9; mọi con
số khác trong khối là số in của Bishop. Bản trong `tools/` là
[`tools/fig_poly.py`](<../../../tools/fig_poly.py>), sinh từ chính bài bằng `extract_fig.py`.

</details>

### 3.12 Bishop — tăng số điểm, Hình 1.6 (tr.9)

![Hình 1.6 của Bishop 2006 — đa thức bậc 9 trên 15 và 100 điểm](<paper_bishop2006_fig1_6_tang_N.png>)

**Hình 20 — Hình 1.6 của Bishop 2006, tr.9 (PDF trang 29).**

**Caption gốc.** *"Plots of the solutions obtained by minimizing the sum-of-squares error function
using the M = 9 polynomial for N = 15 data points (left plot) and N = 100 data points (right plot).
We see that increasing the size of the data set reduces the over-fitting problem."*

(Đồ thị nghiệm cực tiểu hoá tổng bình phương với đa thức M = 9, trên N = 15 điểm (trái) và N = 100
điểm (phải). Tăng cỡ dữ liệu làm giảm vấn đề khớp quá mức.)

**Tác giả nói gì về hình.** Tr.9: *"We see that, for a given model complexity, the over-fitting
problem become less severe as the size of the data set increases."* Và ngay sau, một quy tắc ngón
tay cái kèm lời tự bác: *"One rough heuristic that is sometimes advocated is that the number of data
points should be no less than some multiple (say 5 or 10) of the number of adaptive parameters in the
model. However, as we shall see in Chapter 3, the number of parameters is not necessarily the most
appropriate measure of model complexity."*

(Với cùng độ phức tạp mô hình, khớp quá mức nhẹ đi khi dữ liệu nhiều lên. Một quy tắc thô đôi khi
được khuyên là số điểm không nên ít hơn một bội — chẳng hạn 5 hay 10 — của số tham số. Tuy nhiên số
tham số không nhất thiết là thước đo độ phức tạp thích hợp nhất.)

**Cách đọc hình.** Cùng bậc 9, cùng hàm sinh. Ô trái — 15 điểm, đường đỏ còn uốn lượn ở hai đầu. Ô
phải — 100 điểm, đường đỏ gần trùng đường xanh lá.

**Điều hình không nói — và đo được.** Hai ô là hai lần rút ngẫu nhiên, nên không nói được mức cải
thiện là bao nhiêu. Đo trung vị $E_{\text{RMS}}$ (RMSE thật, theo (1.3)) trên 1 000 điểm kiểm tra,
qua 200 hạt giống, cùng quy trình Phụ lục A (`v_poly.py` mục 9):

| $N$ | trung vị RMSE kiểm tra của $M = 9$ |
|---:|---:|
| 10 | 1065,681841 |
| 15 | 3,935222 |
| 30 | 0,457234 |
| 100 | 0,315789 |

Từ 10 lên 15 điểm, sai số kiểm tra giảm hơn hai bậc; ở 100 điểm nó đã sát sàn nhiễu 0,3. Quy tắc "5
đến 10 lần số tham số" của tr.9 ứng với 50–100 điểm cho 10 hệ số: 30 điểm (3 lần) đã cho 0,4572, chưa
tới sàn; 100 điểm (10 lần) thì tới.

### 3.13 Bishop — phạt Ridge (1.4), Hình 1.7, Bảng 1.2, Hình 1.8, và tập kiểm định (tr.10–11)

![Công thức (1.4) của Bishop 2006, tr.10](<paper_bishop2006_cong_thuc_1_4.png>)

**Hình 21 — Công thức (1.4).** Bishop 2006, tr.10 (PDF trang 30).

$$\widetilde E(\mathbf w) = \frac12\sum_{n=1}^{N}\{y(x_n,\mathbf w) - t_n\}^2 + \frac{\lambda}{2}\lVert\mathbf w\rVert^2 \tag{1.4}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\widetilde E(\mathbf w)$ | hàm lỗi đã chuẩn hoá | vô hướng |
| $\lambda$ | hệ số phạt — càng lớn càng ép hệ số về 0 | vô hướng $\ge 0$ |
| $\lVert\mathbf w\rVert^2$ | $w_0^2 + w_1^2 + \dots + w_M^2$ — **tính cả $w_0$** | vô hướng |

Bishop tr.10 bình luận về $w_0$: *"Note that often the coefficient w0 is omitted from the regularizer
because its inclusion causes the results to depend on the choice of origin for the target variable
(Hastie et al., 2001), or it may be included but with its own regularization coefficient"*. Và gọi
tên: *"The particular case of a quadratic regularizer is called ridge regression (Hoerl and Kennard,
1970)."*

**Ý nghĩa.** (1.4) đặt hai mục tiêu lên một cán cân: khớp dữ liệu (số hạng đầu) và giữ hệ số nhỏ (số
hạng sau). Ở Bảng 1.1, bậc 9 khớp được 10 điểm **chỉ bằng cách** dùng hệ số hàng triệu triệt tiêu
nhau; số hạng sau làm cách đó trở nên đắt. Đặt đạo hàm bằng 0 được nghiệm dạng đóng
$\mathbf w^\star = (V^\top V + \lambda I)^{-1}V^\top \mathbf t$ — cùng hệ phương trình chuẩn của Gergonne,
chỉ cộng thêm $\lambda$ vào đường chéo.

![Hình 1.7 của Bishop 2006 — M = 9 với ln λ = −18 và ln λ = 0](<paper_bishop2006_fig1_7_ridge.png>)

**Hình 22 — Hình 1.7 của Bishop 2006, tr.10 (PDF trang 30).**

**Caption gốc.** *"Plots of M = 9 polynomials fitted to the data set shown in Figure 1.2 using the
regularized error function (1.4) for two values of the regularization parameter λ corresponding to ln
λ = −18 and ln λ = 0. The case of no regularizer, i.e., λ = 0, corresponding to ln λ = −∞, is shown
at the bottom right of Figure 1.4."*

**Tác giả nói gì về hình.** Tr.10: *"We see that, for a value of ln λ = −18, the over-fitting has
been suppressed and we now obtain a much closer representation of the underlying function sin(2πx).
If, however, we use too large a value for λ then we again obtain a poor fit, as shown in Figure 1.7
for ln λ = 0."*

(Với ln λ = −18, khớp quá mức đã bị chặn và đường khớp gần hàm sin(2πx) hơn hẳn. Nhưng nếu λ quá lớn
thì lại khớp kém, như Hình 1.7 với ln λ = 0.)

**Cách đọc hình.** Ô trái: đường đỏ đi gần đường xanh lá, không còn vọt ra khỏi khung. Ô phải: đường
đỏ gần như nằm ngang quanh 0,13 — phạt quá mạnh, mô hình bậc 9 bị ép thành gần hằng số.

![Bảng 1.2 của Bishop 2006 — hệ số M = 9 theo ln λ](<paper_bishop2006_bang1_2_he_so_ridge.png>)

**Hình 23 — Bảng 1.2 của Bishop 2006, tr.11 (PDF trang 31).**

**Caption gốc.** *"Table of the coefficients w⋆ for M = 9 polynomials with various values for the
regularization parameter λ. Note that ln λ = −∞ corresponds to a model with no regularization, i.e.,
to the graph at the bottom right in Figure 1.4. We see that, as the value of λ increases, the typical
magnitude of the coefficients gets smaller."*

**Cách đọc bảng.** Cột đầu trùng cột $M=9$ của Bảng 1.1. Cột giữa: hệ số lớn nhất còn 91,53. Cột cuối:
mọi hệ số dưới 0,14.

**Điều bảng không nói — và dựng lại được.** Trên 10 điểm đã dựng lại ở Hình 19, giải (1.4) với đúng
$\lambda$ in trên bảng thì **9/10 hệ số của mỗi cột nằm ngoài khoảng làm tròn**: lệch lớn nhất 181,65
ở cột $\ln\lambda=-18$ và 0,3302 ở cột $\ln\lambda=0$. Quét $\ln\lambda$ từ −45 tới 15 (bước 0,01) tìm
$\lambda$ khớp nhất cho từng cột (`v_poly.py` mục 8):

| Cột in | Cách phạt | $\ln\lambda$ khớp nhất | lệch max | hệ số ngoài khoảng làm tròn | $\ln\lambda$ khớp − nhãn in |
|---|---|---:|---:|---:|---:|
| $\ln\lambda=-18$ | cả $w_0$, đúng (1.4) | −16,08 | 0,0644 | 5 / 10 | +1,92 |
| $\ln\lambda=0$ | cả $w_0$, đúng (1.4) | **2,28** | **0,0034** | **0 / 10** | **+2,28** |
| $\ln\lambda=0$ | không phạt $w_0$ | 6,81 | 0,0586 | 8 / 10 | +6,81 |

Và thử thẳng giả thuyết **mã sinh bảng nhân $\lambda$ với $N$** — tức cực tiểu hoá $E/N +
\frac{\lambda}{2}\lVert\mathbf w\rVert^2$ thay vì (1.4):

| Cột in | $\lambda$ theo (1.4) | $\lambda\cdot(N-1)$ | $\lambda\cdot N$ |
|---|---|---|---|
| $\ln\lambda=-18$ | lệch 181,6506 · ngoài 9/10 | lệch 12,7610 · ngoài 9/10 | lệch 16,9731 · ngoài 9/10 |
| $\ln\lambda=0$ | lệch 0,3302 · ngoài 9/10 | lệch 0,0076 · **ngoài 0/10** | lệch 0,0041 · **ngoài 0/10** |

Đọc ra ba điều, theo mức chắc chắn giảm dần:

1. **Cột "$\ln\lambda = 0$" không phải nghiệm của (1.4) với $\lambda = 1$.** Nó là nghiệm của (1.4)
   với $\ln\lambda \approx 2{,}28$, rất gần $\ln N = 2{,}3026$ — và giả thuyết nhân $N$ (cũng như nhân
   $N-1$) đưa cả 10 hệ số vào khoảng làm tròn. Phạt cả $w_0$ khớp; không phạt $w_0$ thì không.
2. **Cột "$\ln\lambda = -18$" không dựng lại được với bất kỳ $\lambda$ nào** trong mức làm tròn: tốt
   nhất vẫn có 5/10 hệ số lệch. Ở vùng $\lambda \approx 10^{-7}$ này, hệ số rất nhạy với chữ số cuối
   của dữ liệu và của phép giải — ô phải Hình 19 cho thấy đáy của đường tím hẹp và sâu, tức đổi
   $\ln\lambda$ một phần trăm là đổi hệ số hàng chục đơn vị. Nên ở cột này không phân biệt được "nhãn
   lệch" với "bộ giải khác".
3. Cả hai cột đều cho $\lambda$ **thật lớn hơn nhãn in**, khoảng 2 đơn vị $\ln$. Không có gì trong
   mục 1.1 nói tới hệ số nhân này.

Không câu định tính nào của Bishop đổi vì chỗ này: $\lambda$ lớn thì hệ số nhỏ, $\lambda$ quá lớn thì
khớp kém. Nhưng ai lấy nhãn $\ln\lambda = -18$ đem vào (1.4) trên 10 điểm này sẽ **không** ra đường ở
Hình 1.7.

![Hình 1.8 của Bishop 2006 — E_RMS theo ln λ](<paper_bishop2006_fig1_8_erms_theo_lambda.png>)

**Hình 24 — Hình 1.8 của Bishop 2006, tr.11 (PDF trang 31).**

**Caption gốc.** *"Graph of the root-mean-square error (1.3) versus ln λ for the M = 9 polynomial."*

**Tác giả nói gì về hình.** Tr.11: *"We see that in effect λ now controls the effective complexity of
the model and hence determines the degree of over-fitting."* Và câu kết mục 1.1, là câu trả lời thực
hành cho câu hỏi của Gergonne: *"The results above suggest a simple way of achieving this, namely by
taking the available data and partitioning it into a training set, used to determine the coefficients
w, and a separate validation set, also called a hold-out set, used to optimize the model complexity
(either M or λ). In many cases, however, this will prove to be too wasteful of valuable training data,
and we have to seek more sophisticated approaches."*

(Thực chất λ giờ điều khiển độ phức tạp hiệu dụng của mô hình, và do đó quyết định mức khớp quá mức.
[Cách chọn:] chia dữ liệu thành tập huấn luyện để tìm hệ số w, và một tập kiểm định riêng — còn gọi là
tập giữ lại — để tối ưu độ phức tạp (M hoặc λ). Nhưng trong nhiều trường hợp cách đó quá phí dữ liệu
huấn luyện quý giá, và phải tìm cách tinh vi hơn.)

**Cách đọc hình.** Trục ngang $\ln\lambda$ từ khoảng −38 tới −20. Bên trái (phạt yếu): huấn luyện gần
0, kiểm tra cao — khớp quá mức. Giữa: cả hai phẳng. Bên phải: cả hai tăng — khớp thiếu.

**Điều hình không nói — và không khớp với Bảng 1.2.** Số hoá đầu mút của hai đường (`v_poly.py` mục 9):
đường huấn luyện 0,0103 ở đầu trái và **0,4178 ở đầu phải** ($\ln\lambda \approx -20$); đường kiểm tra
0,8610 và 0,5308. Trong khi đó, **chính các hệ số in** của cột $\ln\lambda = -18$ ở Bảng 1.2, đem đo
trên 10 điểm dựng lại, cho RMSE huấn luyện 0,1790 — tức 0,1265 theo thang $\sqrt{E/N}$ mà Hình 1.5 dùng.

Sai số huấn luyện của Ridge **không giảm khi $\lambda$ tăng** — thêm phạt chỉ thêm ràng buộc. Nên nếu
Hình 1.8 và Bảng 1.2 cùng một bộ dữ liệu và cùng một thang $\lambda$, sai số huấn luyện tại $\ln\lambda
= -18$ phải **lớn hơn hoặc bằng** tại $-20$, tức ít nhất 0,4178. Nó là 0,1265 (hoặc 0,1790 theo
(1.3)). Hai con số không cùng đứng được. Đáng để ý hơn: 0,4178 gần đúng mức huấn luyện của cột
**$\ln\lambda = 0$** — hệ số in của cột đó cho 0,4198 theo thang $\sqrt{E/N}$. Hình 1.8 vì thế nhiều khả
năng dùng một trục $\lambda$ khác với nhãn của Bảng 1.2 và Hình 1.7; sách không nói trục nào đúng, và
bài này không đoán thêm.

Quét $\lambda$ trên 10 điểm dựng lại, theo đúng (1.4), với tập kiểm tra tự sinh cho sai số kiểm tra
thấp nhất ở $\ln\lambda = -16$ (0,391805 RMSE), và dải phẳng từ khoảng −26 tới −8 — rộng hơn hẳn dải
phẳng của Hình 1.8.

**Nối về Gergonne.** Câu hỏi tr.252 — *"quel est le plus probablement le degré de cette courbe ?"* —
có hai câu trả lời trong Bishop mục 1.1: chọn $M$ bằng tập kiểm định, hoặc giữ $M$ lớn và chọn $\lambda$
bằng tập kiểm định. Mục 7.2 chạy cả hai cách hiện đại (kiểm tra chéo và BIC) trên cùng một bộ dữ liệu.

---

---

## 4. Công thức và ký hiệu hiện đại

### 4.1 Mô hình và ma trận thiết kế

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \dots + \beta_d x_i^d + \varepsilon_i, \qquad i = 1, \dots, n$$

**Ý nghĩa.** Mỗi quan sát bằng một đa thức bậc $d$ của đầu vào cộng một sai số. Cả $n$ phương trình
dùng **chung** bộ hệ số — đó là chỗ "học" nằm; và mọi $\beta_j$ đứng ở bậc một, nên đây là mô hình
tuyến tính theo tham số, dù cong theo $x$.

Gom $n$ phương trình vào một dòng:

$$y = V\beta + \varepsilon, \qquad V = \begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^d \\ \vdots & & & & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^d \end{pmatrix}, \qquad V_{ij} = x_i^{\,j}$$

| Ký hiệu | Chiều |
|---|---|
| $y$, $\varepsilon$ | $\mathbb{R}^{n}$ |
| $V$ | $\mathbb{R}^{n \times (d+1)}$ |
| $\beta$ | $\mathbb{R}^{d+1}$ |

**Ý nghĩa.** Mọi thứ riêng của hồi quy đa thức nằm trong **cách dựng $V$**. Sau dòng này, bài toán
không còn biết $V$ là luỹ thừa của một biến hay là $d+1$ biến đo độc lập.

### 4.2 Bài toán và nghiệm

$$\hat\beta = \arg\min_{\beta}\ \lVert y - V\beta\rVert^2 \quad\Longrightarrow\quad V^\top V\,\hat\beta = V^\top y$$

**Ý nghĩa.** Vế trái là câu *"rendre minimum la somme de leurs quarrés"* của Gergonne tr.249. Vế phải
là hệ (1) tr.250: đạo hàm theo từng hệ số bằng 0. Nếu các cột của $V$ độc lập tuyến tính — với một
biến, điều đó xảy ra khi có **ít nhất $d+1$ giá trị $x$ khác nhau** — thì $V^\top V$ khả nghịch và
nghiệm duy nhất. Với $d+1 = n$ thì $V$ vuông, phần dư bằng 0: nội suy.

Phần tử của $V^\top V$:

$$(V^\top V)_{jk} = \sum_{i=1}^{n} x_i^{\,j+k} = \Sigma a^{j+k} \ \text{(ký hiệu Gergonne)}$$

**Ý nghĩa.** Phần tử chỉ phụ thuộc **tổng** $j+k$, nên cả ma trận $(d+1)\times(d+1)$ chỉ có $2d+1$ giá
trị khác nhau. Khi $x$ đối xứng quanh 0, mọi tổng luỹ thừa lẻ bằng 0 và ma trận có số 0 xen kẽ như bàn
cờ (mục 3.7).

### 4.3 Phạt Ridge

$$\hat\beta_\lambda = \arg\min_\beta\ \tfrac12\lVert y - V\beta\rVert^2 + \tfrac{\lambda}{2}\lVert\beta\rVert^2 \quad\Longrightarrow\quad (V^\top V + \lambda I)\,\hat\beta_\lambda = V^\top y$$

**Ý nghĩa.** Đây là (1.4) của Bishop viết bằng ma trận. Cộng $\lambda$ vào đường chéo làm ma trận khả
nghịch với **mọi** $\lambda > 0$, kể cả khi $d+1 > n$, và kéo mọi hệ số về phía 0. Bảng 1.2 của Bishop
cho thấy cái giá khi kéo quá tay: ở cột lớn nhất, đường khớp gần như phẳng (Hình 22 ô phải).

### 4.4 Số điều kiện

$$\kappa(V) = \frac{\sigma_{\max}(V)}{\sigma_{\min}(V)}, \qquad \kappa(V^\top V) = \kappa(V)^2$$

| Ký hiệu | Nghĩa |
|---|---|
| $\sigma_{\max}(V)$, $\sigma_{\min}(V)$ | giá trị kỳ dị lớn nhất và nhỏ nhất của $V$ — **không** phải độ lệch chuẩn nhiễu |
| $\kappa$ | tỉ số khuếch đại sai số tương đối từ dữ liệu sang nghiệm, trong trường hợp xấu nhất |

**Ý nghĩa.** Các cột $x^{10}$ và $x^{11}$ trên $[0, 1]$ gần như trùng nhau (tương quan 0,998966 trên
lưới 1 000 điểm, `v_poly.py` mục 11), nên $V$ gần suy biến và $\kappa$ lớn. Giải qua $V^\top V$ thì
**bình phương** con số đó: $\kappa = 10^{8}$ trên $V$ thành $10^{16}$ trên $V^\top V$, vượt
$1/\varepsilon_{\text{float64}} = 4{,}50\times10^{15}$. Đó là lý do mọi thư viện tốt giải bằng QR hoặc SVD
trên $V$, không lập $V^\top V$.

---

## 5. Thuật toán từng bước

```
Vào : x ∈ ℝⁿ, y ∈ ℝⁿ, bậc d (d + 1 ≤ số giá trị x khác nhau), tuỳ chọn λ ≥ 0
Ra  : β̂ ∈ ℝ^(d+1), cùng hai hằng số chuẩn hoá (m, s)

1. m ← trung bình(x);  s ← độ lệch chuẩn(x)            # chỉ tính trên TẬP HUẤN LUYỆN
2. z ← (x − m) / s                                        # đưa x về quanh 0, bề rộng cỡ 1
3. V ← ma trận n × (d+1), V[i, j] = z[i]^j                # cột 0 toàn số 1
4. nếu λ = 0:  β̂ ← nghiệm bình phương tối thiểu của V β ≈ y   (QR hoặc SVD, KHÔNG nghịch đảo VᵀV)
   nếu λ > 0:  β̂ ← nghiệm bình phương tối thiểu của [V; √λ·I] β ≈ [y; 0]
5. trả về β̂, m, s

Dự đoán tại x_mới:
6. z_mới ← (x_mới − m) / s                                # CÙNG m, s của bước 1
7. ŷ ← Σⱼ β̂[j] · z_mới^j

Chọn d (và λ): lặp bước 1–7 trên từng mảnh kiểm tra chéo, lấy d có sai số trung bình nhỏ nhất (mục 7.2)
```

| Bước | Vì sao | Bỏ đi thì hỏng chỗ nào |
|---|---|---|
| 1–2 | đưa $x$ về quanh 0 | $x \approx 2000$: $x^6 = 6{,}4\times10^{19}$ — tràn ý nghĩa ở bậc 6; và mất bàn cờ số 0 của mục 3.7 |
| 4, dùng QR/SVD | tránh bình phương $\kappa$ | nghiệm mất chữ số từ bậc 14 trên dữ liệu Hình 25 (mục 6.2) |
| 4, dạng $[V; \sqrt\lambda I]$ | Ridge mà không lập $V^\top V + \lambda I$ | cùng lý do trên |
| 6, cùng $m, s$ | đa thức học được là đa thức **của $z$** | dự đoán sai hoàn toàn, không báo lỗi |

---

## 6. Hai kiểu hỏng, đo bằng số

![Bậc đa thức và sai số](<poly_degree.png>)

**Hình 25 — Hai kiểu hỏng, đo trên 30 điểm (hình tự vẽ).** Hình 15 (Bishop 1.4) cho thấy khớp quá mức
trên **bốn bậc rời rạc**, và Hình 17 (Bishop 1.5) quét bậc 0–9. Hình này bù hai chỗ: nó quét bậc tới
25, qua khỏi vùng mà Bishop dừng, và nó thêm **kiểu hỏng thứ hai** mà Bishop không vẽ — mất ổn định số
học.

**Cách đọc hình.**

- **Ô trái** — bốn bậc 1 / 3 / 9 / 24 vẽ chồng; đường xanh lá đứt là hàm thật
  $f(x) = \sin(1{,}6x) + 0{,}35x$. Bậc 24 (đỏ) đi qua gần hết 30 điểm và vô dụng.
- **Ô giữa** — RMSE theo bậc, **trục dọc log**. Đường xanh (huấn luyện) giảm theo bậc; đường đỏ trên tập
  giữ riêng 400 điểm có đáy rồi tăng. Vạch xanh lá ngang là độ lệch chuẩn nhiễu $\sigma = 0{,}35$.
- **Ô phải** — $\kappa(V^\top V)$ theo bậc, trục log. Vạch đỏ đứt là $1/\varepsilon_{\text{float64}}$.

<details>
<summary><b>Code sinh Hình 25</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import sys
import numpy as np, matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def f_true(t):
    return np.sin(1.6 * t) + 0.35 * t


rng = np.random.default_rng(0)
ntr = 30
xtr = np.sort(rng.uniform(-3, 3, ntr))
ytr = f_true(xtr) + rng.normal(0, .35, ntr)
xte = np.sort(rng.uniform(-3, 3, 400))          # tap kiem tra tu CUNG phan phoi
yte = f_true(xte) + rng.normal(0, .35, 400)
grid = np.linspace(-3.2, 3.2, 400)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

# --- bang trai: bon bac ve chong len nhau ---
for d, col in [(1, "#6b7280"), (3, "#2563eb"), (9, "#d97706"), (24, "#dc2626")]:
    cf = np.polyfit(xtr, ytr, d)
    axes[0].plot(grid, np.polyval(cf, grid), color=col, lw=1.4, label="bậc %d" % d)
axes[0].scatter(xtr, ytr, s=20, color="#111827", zorder=4)
axes[0].plot(grid, f_true(grid), "--", color="#059669", lw=1.2, label="hàm thật")
axes[0].set_ylim(-3, 3.2); axes[0].legend(fontsize=7.2, ncol=2)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("30 điểm huấn luyện\nbậc 24 đi qua gần hết — và vô dụng", fontsize=9)

# --- bang giua: RMSE huan luyen vs kiem tra theo bac ---
ds = range(1, 26)
tr_e, te_e = [], []
for d in ds:
    cf = np.polyfit(xtr, ytr, d)
    tr_e.append(rmse(np.polyval(cf, xtr), ytr))
    te_e.append(rmse(np.polyval(cf, xte), yte))
axes[1].semilogy(list(ds), tr_e, "o-", color="#2563eb", ms=3, lw=1.2, label="huấn luyện")
axes[1].semilogy(list(ds), te_e, "o-", color="#dc2626", ms=3, lw=1.2, label="kiểm tra")
axes[1].axhline(.35, color="#059669", ls="--", lw=1.0, label="mức nhiễu σ = 0.35")
best = int(np.argmin(te_e)) + 1
axes[1].axvline(best, color="#111827", lw=.8)
axes[1].legend(fontsize=7.2); axes[1].set_xlabel("bậc đa thức"); axes[1].set_ylabel("RMSE")
axes[1].set_title("Bậc tốt nhất trên tập kiểm tra: %d\n"
                  "sai số huấn luyện giảm mãi, kiểm tra thì không" % best, fontsize=9)

# --- bang phai: so dieu kien theo bac, doc lap voi du lieu ---
# Ba cach tinh cung mot dai luong, de doi chieu (xem canh bao duoi hinh).
conds, conds_direct, conds_sv = [], [], []
for d in ds:
    V = np.vander(xtr, d + 1, increasing=True)
    G = V.T @ V
    conds.append(np.linalg.cond(G))                       # (a) qua bien tam G
    conds_direct.append(np.linalg.cond(np.vander(xtr, d + 1, increasing=True).T
                                       @ np.vander(xtr, d + 1, increasing=True)))  # (b)
    sv = np.linalg.svd(V, compute_uv=False)
    conds_sv.append((sv[0] / sv[-1]) ** 2)                # (c) qua kappa(V)^2
axes[2].semilogy(list(ds), conds, "o-", color="#7c3aed", ms=3, lw=1.2)
axes[2].axhline(1 / np.finfo(float).eps, color="#dc2626", ls="--", lw=1.1)
axes[2].text(2, 1 / np.finfo(float).eps * 1.6,
             "1/ε của float64 — quá đây thì\nnghiệm không còn chữ số nào đúng",
             fontsize=7, color="#dc2626")
axes[2].set_xlabel("bậc đa thức"); axes[2].set_ylabel(r"$\kappa(V^\top V)$")
axes[2].set_title("Ma trận Vandermonde hỏng rất nhanh\n"
                  "Đây mới là lý do không dùng bậc cao", fontsize=9)

fig.savefig("poly_degree.png", dpi=130, bbox_inches="tight", facecolor="white")

d_break = next(d for d, c in zip(ds, conds) if c > 1 / np.finfo(float).eps)
print("bậc tốt nhất trên kiểm tra: %d (RMSE %.4f)" % (best, min(te_e)))
print("RMSE bậc 1  : huấn luyện %.4f  kiểm tra %.4f" % (tr_e[0], te_e[0]))
print("RMSE bậc 24 : huấn luyện %.4f  kiểm tra %.4f" % (tr_e[23], te_e[23]))
print("tỉ số kiểm tra / huấn luyện ở bậc 24: %.1f" % (te_e[23] / tr_e[23]))
print("κ(VᵀV) bậc 1 = %.2e  bậc 12 = %.2e  bậc 24 = %.2e" % (conds[0], conds[11], conds[23]))
print("bậc đầu tiên κ vượt 1/ε: %d" % d_break)
print("\nBa cách tính CÙNG một κ(VᵀV) — chỉ khớp khi κ còn dưới 1/ε:")
for d in (12, 24):
    i = d - 1
    print("   bậc %2d: (a) %.3e   (b) %.3e   (c) κ(V)² = %.3e"
          % (d, conds[i], conds_direct[i], conds_sv[i]))
```

Bảng phải **không dùng `ytr` chút nào** — $\kappa$ chỉ phụ thuộc vào $x$ và bậc. Đó là lý do
mục 6.2 nói kiểu hỏng thứ hai độc lập với dữ liệu và với nhiễu.

</details>

Dữ liệu: 30 điểm huấn luyện từ $f(x) = \sin(1{,}6x) + 0{,}35x$ cộng nhiễu $\sigma = 0{,}35$, $x$ rút đều
trong $[-3, 3]$; 400 điểm kiểm tra từ cùng phân phối. Đầu ra trên máy dùng để viết bài (numpy 2.4.4,
matplotlib 3.10.9, Linux x86-64):

```
bậc tốt nhất trên kiểm tra: 9 (RMSE 0.3881)
RMSE bậc 1  : huấn luyện 0.8645  kiểm tra 0.8434
RMSE bậc 24 : huấn luyện 0.1185  kiểm tra 21.0232
tỉ số kiểm tra / huấn luyện ở bậc 24: 177.3
κ(VᵀV) bậc 1 = 3.73e+00  bậc 12 = 1.96e+13  bậc 24 = 6.08e+26
bậc đầu tiên κ vượt 1/ε: 14

Ba cách tính CÙNG một κ(VᵀV) — chỉ khớp khi κ còn dưới 1/ε:
   bậc 12: (a) 1.965e+13   (b) 1.965e+13   (c) κ(V)² = 1.965e+13
   bậc 24: (a) 6.077e+26   (b) 2.545e+26   (c) κ(V)² = 1.785e+31
```

### 6.1 Khớp quá mức

Bậc 24 có RMSE huấn luyện **0,1185**, thấp hơn độ lệch chuẩn nhiễu 0,35, nhưng RMSE trên tập giữ riêng
là **21,0232** — gấp **177,3 lần**. Bậc 9 cho 0,3881, sát mức 0,35 mà một mô hình biết đúng hàm trung
bình đạt được theo kỳ vọng. Riêng việc RMSE huấn luyện thấp hơn $\sigma$ chưa đủ kết luận, vì trên một
mẫu hữu hạn thước đo cũng dao động; khoảng cách 177 lần tới tập giữ riêng mới là bằng chứng.

Đường huấn luyện **không bao giờ tăng** theo bậc: mọi đa thức bậc $d$ cũng là đa thức bậc $d+1$ với hệ
số đầu bằng 0, nên tăng bậc chỉ mở rộng tập tìm kiếm. Đó chính là nghịch lý Bishop tr.8 tự nêu (*"a
polynomial of given order contains all lower order polynomials as special cases"*), và lời giải là:
tập tìm kiếm rộng hơn thì khớp **nhiễu** tốt hơn, không phải khớp **hàm** tốt hơn.

Nhìn lại mục 3.6 để thấy vì sao — với 30 điểm và bậc 24, 25 hệ số chỉ còn 5 bậc tự do cho bù trừ sai
số; phương sai của giá trị khớp tiến về phương sai của chính dữ liệu, rồi vượt nó giữa các điểm.

### 6.2 Mất ổn định số học

Ô phải là kiểu hỏng thứ hai. Nó **không dùng $y$** — $\kappa$ chỉ phụ thuộc $x$ và bậc — nên nó độc lập
với nhiễu nhưng **không** độc lập với cách đặt $x$. Từ **bậc 14**, $\kappa(V^\top V)$ của ví dụ vượt
$1/\varepsilon_{\text{float64}} = 4{,}50\times10^{15}$. Nghĩa là bài toán theo cơ sở đơn thức này nhạy
tới mức sai số làm tròn có thể xoá hết chữ số của nghiệm trong trường hợp xấu nhất — một cảnh báo, không
phải định luật "bậc 14 luôn sai". Đổi miền $x$, chuẩn hoá, hoặc dùng cơ sở trực giao đều đổi con số này.

Hai kiểu hỏng khác nhau về bản chất: trên tập giữ riêng này, sai số dự báo nhỏ nhất ở bậc 9 còn cảnh báo
số học xuất hiện ở bậc 14. Khớp quá mức là lỗi **thống kê** — có ở mọi máy; điều kiện xấu là lỗi **số
học** — chữa được bằng cách giải khác đi mà không đổi mô hình.

> **Điều cần rút ra.** Khi hồi quy đa thức cho hệ số khổng lồ xen kẽ dấu — như cột $M=9$ của Bảng 1.1,
> $232{,}37 \cdot -5\,321{,}83 \cdot 48\,568{,}31 \cdots$ — thì có hai khả năng: mô hình đang khớp nhiễu
> (kiểm bằng tập giữ riêng), hoặc $\kappa$ đã nổ (kiểm bằng phổ giá trị kỳ dị). Ở Bảng 1.1 là khả năng
> thứ nhất: $\kappa(V) = 1{,}5193\times10^{7}$, còn xa ngưỡng.

### 6.3 Chính con số $\kappa$ ở bậc 24 không tái lập được qua máy

Khối mã tính $\kappa(V^\top V)$ ở bậc 24 theo ba cách **tương đương về toán**:

- **(a)** tính `G = V.T @ V` vào một biến rồi gọi `cond(G)`;
- **(b)** gọi `cond(V.T @ V)` viết thẳng một dòng — cùng phép tính, chỉ khác chỗ lưu kết quả trung gian;
- **(c)** lấy $\kappa(V)^2$ từ SVD của chính $V$.

Ở **bậc 12** ($\kappa \approx 2\times10^{13}$, dưới $1/\varepsilon$) cả ba **khớp tới bốn chữ số** trên
mọi máy đã chạy. Ở **bậc 24** thì không, và chúng **đổi theo máy**:

| Máy | (a) | (b) | (c) |
|---|---:|---:|---:|
| bản trước của bài (Windows, ghi trong lịch sử git) | $3{,}955\times10^{26}$ | $7{,}263\times10^{26}$ | $1{,}789\times10^{31}$ |
| máy viết bài này (numpy 2.4.4, Linux) | $6{,}077\times10^{26}$ | $2{,}545\times10^{26}$ | $1{,}785\times10^{31}$ |

Bản trước viết *"(a) và (b) chênh 1,8 lần"*; trên máy này (b) **nhỏ hơn** (a), và thứ tự đảo. Chỉ (c)
đứng yên tới ba chữ số, vì SVD làm việc thẳng trên $V$ và không bình phương sai số. Con số
$6{,}08\times10^{26}$ ở ô phải Hình 25 vì thế chỉ đọc được là **"lớn hơn $1/\varepsilon$ rất nhiều"**,
không phải một phép đo; và hình PNG trong repo đổi md5 khi sinh lại trên máy khác.

Bài học thực hành: đừng chẩn đoán bằng $V^\top V$ khi có thể làm trên $V$. Xem cả phổ giá trị kỳ dị, đặc
biệt tỉ số $\sigma_{\min}/\sigma_{\max}$ so với ngưỡng hạng mà bộ giải dùng.

---

## 7. Cài đặt và số đo thật

### 7.1 Bản tối thiểu

Khối `numpy` ở mục 0.1 là đủ cho dữ liệu nhỏ, $x$ quanh 0 và bậc thấp. Bản dùng thật cần thêm hai thứ
mà khối đó không có: chuẩn hoá $x$ theo tập huấn luyện, và cách chọn bậc.

### 7.2 Chọn bậc: câu hỏi của Gergonne, trả lời bằng ba cách

Cùng 30 điểm của Hình 25. Mỗi bậc chạy qua một `Pipeline` gồm chuẩn hoá, sinh cột luỹ thừa, và hồi quy
tuyến tính; chấm bằng kiểm tra chéo 5 mảnh và BIC, **chỉ trên 30 điểm huấn luyện**. Cột kiểm tra 400 điểm
chỉ để đối chiếu, không dùng để chọn.

```python
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.stdout.reconfigure(encoding="utf-8")


def f_true(t):
    return np.sin(1.6 * t) + 0.35 * t


rng = np.random.default_rng(0)                      # cùng bộ dữ liệu với Hình 25
xtr = np.sort(rng.uniform(-3, 3, 30))
ytr = f_true(xtr) + rng.normal(0, .35, 30)
xte = np.sort(rng.uniform(-3, 3, 400))
yte = f_true(xte) + rng.normal(0, .35, 400)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
print("bậc  RMSE-CV-5-mảnh  BIC        RMSE kiểm tra (chỉ để đối chiếu)")
bang = []
for d in range(1, 13):
    mo_hinh = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression())
    diem = -cross_val_score(mo_hinh, xtr[:, None], ytr, cv=cv, scoring="neg_root_mean_squared_error")
    mo_hinh.fit(xtr[:, None], ytr)
    rss = float(((mo_hinh.predict(xtr[:, None]) - ytr) ** 2).sum())
    bic = 30 * np.log(rss / 30) + (d + 1) * np.log(30)
    te = float(np.sqrt(((mo_hinh.predict(xte[:, None]) - yte) ** 2).mean()))
    bang.append((d, diem.mean(), bic, te))
    print("%3d  %14.4f  %9.4f  %8.4f" % (d, diem.mean(), bic, te))

d_cv = min(bang, key=lambda r: r[1])[0]
d_bic = min(bang, key=lambda r: r[2])[0]
d_te = min(bang, key=lambda r: r[3])[0]
print("bậc chọn bởi CV: %d · bởi BIC: %d · tốt nhất trên tập kiểm tra: %d" % (d_cv, d_bic, d_te))
print("RMSE kiểm tra tại bậc CV chọn: %.4f · tại bậc BIC chọn: %.4f"
      % (bang[d_cv - 1][3], bang[d_bic - 1][3]))
```

```text
bậc  RMSE-CV-5-mảnh  BIC        RMSE kiểm tra (chỉ để đối chiếu)
  1          0.9025    -1.9347    0.8434
  2          0.8579    -2.4133    0.8955
  3          0.5522   -31.0351    0.5555
  4          0.5399   -29.4146    0.5766
  5          0.3457   -52.6946    0.4009
  6          0.3542   -52.1955    0.4202
  7          0.3950   -50.9894    0.4013
  8          0.4127   -49.4807    0.3959
  9          0.4428   -47.2712    0.3881
 10          0.4398   -45.2600    0.3964
 11          0.5357   -42.1567    0.4023
 12          0.6076   -39.2455    0.4146
bậc chọn bởi CV: 5 · bởi BIC: 5 · tốt nhất trên tập kiểm tra: 9
RMSE kiểm tra tại bậc CV chọn: 0.4009 · tại bậc BIC chọn: 0.4009
```

$$\text{BIC}(d) = n\ln\frac{\text{RSS}(d)}{n} + (d+1)\ln n$$

**Ý nghĩa.** Số hạng đầu thưởng cho khớp tốt; số hạng sau phạt mỗi hệ số thêm một lượng $\ln n$. Với
nhiễu Gauss, BIC xấp xỉ $-2$ lần log xác suất hậu nghiệm của mô hình — tức nó trả lời gần đúng nguyên văn
câu *"quel est le plus probablement le degré"* của Gergonne tr.252, nếu mọi bậc có xác suất tiên nghiệm
như nhau.

Đọc bảng:

- **Kiểm tra chéo và BIC cùng chọn bậc 5**, dù hai thước đo xuất phát từ hai lý lẽ khác nhau. RMSE kiểm
  tra tại bậc đó là 0,4009.
- **Tập kiểm tra chọn bậc 9** (0,3881). Chênh 0,0128 — nhỏ hơn nhiều so với độ dao động giữa các bậc lân
  cận (bậc 7: 0,4013, bậc 8: 0,3959). Chọn bậc bằng tập kiểm tra sẽ **báo cáo lạc quan** đúng phần chênh
  đó, vì cùng một tập vừa chọn vừa chấm.
- Đường RMSE kiểm tra chéo **không đơn điệu** sau đáy (bậc 10: 0,4398 thấp hơn bậc 9: 0,4428). Với 30
  điểm chia 5 mảnh, mỗi mảnh 6 điểm, thước đo này nhiễu; đừng đọc thứ hạng giữa hai bậc lân cận từ nó.

| Cách chọn | Ưu | Trả giá |
|---|---|---|
| tập kiểm định giữ riêng (Bishop tr.11) | đơn giản, không giả định phân phối | *"too wasteful of valuable training data"* — với 30 điểm, giữ 10 là mất một phần ba |
| kiểm tra chéo $K$ mảnh | dùng hết dữ liệu | chạy $K$ lần; nhiễu khi mỗi mảnh nhỏ |
| BIC | chạy một lần, có nghĩa xác suất | cần giả định nhiễu Gauss và mô hình đúng nằm trong họ đang xét |
| giữ bậc cao + Ridge, chọn $\lambda$ | $\lambda$ liên tục nên đường sai số trơn hơn | vẫn phải chọn $\lambda$ bằng một trong ba cách trên; không chữa điều kiện xấu của cơ sở đơn thức |

### 7.3 Quy trình dùng thật

1. Tách huấn luyện / kiểm định (hoặc kiểm tra chéo) / kiểm tra **trước** khi chọn bậc.
2. Đặt chuẩn hoá và sinh cột luỹ thừa **trong cùng một `Pipeline`**, để mỗi mảnh kiểm tra chéo tự tính
   $m, s$ trên phần huấn luyện của nó — nếu không, thông tin từ mảnh đang chấm rò vào bước chuẩn hoá.
3. Bắt đầu từ bậc 1–3. Chỉ tăng bậc khi kiểm tra chéo **và** biểu đồ phần dư cho thấy lợi ích.
4. Giải bằng `lstsq`, QR hoặc SVD; không tự nghịch đảo $V^\top V$.
5. Theo dõi cả sai số ngoài mẫu lẫn phổ giá trị kỳ dị. Hệ số lớn xen kẽ dấu là tín hiệu cần kiểm tra, không
   tự động là "quan hệ phức tạp".
6. Ghi lại miền $x$ lúc huấn luyện, và cảnh báo khi dự đoán ra ngoài miền đó.

> **Lỗi API hay gặp.** `PolynomialFeatures` mặc định **đã thêm** cột toàn số 1, trong khi
> `LinearRegression` mặc định **tự khớp** hệ số chặn. Giữ cả hai là có hai cột trùng hoàn toàn. Khối mã
> ở mục 7.2 đặt `include_bias=False` vì lý do đó.

---

## 8. Vì sao ít hệ số hơn số điểm lại ăn thua

**Cơ chế.** Hệ số của đa thức khớp là một tổ hợp tuyến tính của mọi quan sát, $\hat\beta = (V^\top
V)^{-1}V^\top y$. Sai số của các quan sát đi vào $\hat\beta$ theo đúng tổ hợp đó. Khi số quan sát lớn hơn
số hệ số, mỗi hệ số được tính từ **nhiều** quan sát với trọng số nhỏ, nên sai số lên–xuống của chúng
**triệt tiêu một phần** — đúng *"compensations d'erreurs"* Gergonne tr.248 nói. Khi số hệ số bằng số
quan sát (nội suy), không còn gì để triệt tiêu, và các trọng số có thể lớn hơn 1 (mục 3.3: 45/60, 490/180).

**Phép đo đỡ cơ chế đó** — ba bảng đã có ở trên, đọc cùng nhau:

| Hỏi | Đo | Ở đâu |
|---|---|---|
| thêm quan sát mà giữ số hệ số bằng số quan sát (nội suy) thì sao? | sd của $\hat y''$ trên $[-1, 1]$: 3 quan sát 2,4495 → 21 quan sát **405,8323** | mục 1.2, phép đo thứ hai |
| thêm quan sát mà **giữ** số hệ số cố định (bậc 2) thì sao? | sd của giá trị khớp tại giữa: 5 quan sát 0,6969 → 21 → **0,3280** → 41 → 0,2344 | mục 3.6 |
| cái giá | chệch của $\hat y''$ trên $e^x$: bậc 2 là $+0{,}0804$, bậc 4 là $-0{,}0015$ | mục 1.2, phép đo thứ hai |
| khi số hệ số tiến sát số quan sát | RMSE kiểm tra: bậc 9 là 0,3881, bậc 24 trên 30 điểm là 21,0232 | mục 6.1 |

Hai dòng đầu là ablation một-thay-đổi: cùng 21 quan sát, cùng nhiễu, **chỉ đổi số hệ số** từ 21 xuống 3
thì độ lệch chuẩn của đạo hàm bậc hai giảm $405{,}8323 / 1{,}3353 \approx 304$ lần (`v_poly.py` mục 3 in
hai số hạng; tỉ số tính từ hai ô cùng cột, cùng bảng).

**Vì sao Ridge cũng ăn thua, bằng cùng một lý lẽ.** Ridge không bớt số hệ số mà **bớt độ lớn** của chúng.
Bảng 1.1 cho thấy nội suy bậc 9 khớp 10 điểm bằng hệ số tới $10^6$ triệt tiêu nhau; hệ số lớn nghĩa là
trọng số lớn trên từng quan sát, tức khuếch đại sai số. Phạt $\lVert\mathbf w\rVert^2$ buộc trọng số nhỏ
lại. Bảng 1.2: hệ số lớn nhất từ $1\,061\,800{,}52$ xuống $91{,}53$ ở cột giữa.

**Điều cơ chế này KHÔNG nói.** Nó không nói bậc thấp luôn tốt: phép đo thứ ba ở mục 1.2 cho thấy bậc 1
chệch gấp 2,9 lần mức nhiễu và không chữa được bằng dữ liệu. Ít hệ số chỉ ăn thua khi **đủ** hệ số để
theo hình dạng thật — và "đủ là bao nhiêu" là câu hỏi mục 7.2 đo, không phải câu hỏi cơ chế trả lời.

---

## 9. Giá phải trả — nó hỏng khi nào, và khi nào đừng dùng

### 9.1 Ba giới hạn

1. **Ngoại suy.** Ngoài khoảng dữ liệu, số hạng bậc cao nhất chi phối và dự đoán đổi rất nhanh; mô hình
   vẫn trả một con số mà không tự cảnh báo. Gergonne tr.244 đã dặn chỉ tin công thức giữa các điểm đã cho,
   và mục 3.6 đo được ngay **trong** khoảng: ở mép, độ lệch chuẩn của giá trị khớp gần gấp đôi ở giữa.
2. **Bùng nổ số hạng với nhiều biến.** Đa thức bậc $d$ trên $p$ biến có $\binom{p+d}{d}$ số hạng. Với
   $p = 10$, $d = 3$: **286** cột. Với $p = 100$, $d = 3$: **176 851** cột (`v_poly.py` mục 11).
3. **Toàn cục, không cục bộ.** Mọi hệ số nhận trọng số từ mọi quan sát (mục 8), nên đổi một điểm ở đầu
   này làm đổi đường khớp ở đầu kia. Spline không có tính chất đó.

### 9.2 Bảng giá

| Mặt được | Đo bao nhiêu | Trả giá bằng gì | Nguồn số |
|---|---|---|---|
| dùng hết dữ liệu, bù trừ sai số | sd giá trị khớp tại giữa 0,3280 lần sd một quan sát (21 điểm, bậc 2) | chệch khi bậc quá thấp: $+0{,}0804$ trên $f''(0)$ của $e^x$ | `v_poly.py` mục 3–4 |
| biểu diễn được quan hệ cong | RMSE 0,2962 → 0,0937 khi bậc 1 → bậc 3 | một siêu tham số $d$; chọn bằng tập kiểm tra thì báo cáo lạc quan 0,0128 RMSE | `v_gd1ss.py` mục 370 · mục 7.2 |
| nghiệm dạng đóng, không lặp | một phép QR/SVD | $\kappa(V^\top V)$ vượt $1/\varepsilon$ ở bậc 14 trên 30 điểm | mục 6.2 |
| hệ số đọc được bằng mắt ở bậc thấp | 3 hệ số ở mục 0.5 có nghĩa trực tiếp | ở bậc 9 hệ số tới $10^6$, không còn nghĩa gì | Bishop Bảng 1.1 |
| khớp quá mức kiểm soát được bằng Ridge | hệ số lớn nhất $1\,061\,800{,}52$ → $91{,}53$ | thêm $\lambda$ phải chọn; nhãn $\lambda$ trong nguồn có thể không khớp công thức in cạnh nó | Bishop Bảng 1.2 · mục 3.13 |
| — | — | **ngoại suy tệ hơn đường thẳng** | mục 9.1 |

### 9.3 Cái gì nên dùng thay

| Thay thế | Ưu thế | Đánh đổi |
|---|---|---|
| **Spline bậc ba tự nhiên** — nhiều đa thức bậc ba ghép nối trơn tại các "nút" | cục bộ; ngoại suy tuyến tính thay vì nổ; điều kiện số học tốt | phải chọn số và vị trí nút |
| **Spline làm trơn** | điều khiển độ trơn bằng một tham số phạt | phải chọn mức làm trơn |
| [Cây quyết định](<../Decision Tree (CART)/Decision Tree (CART).md>) | bắt được ngưỡng và tương tác tự động | đường khớp bậc thang, không trơn, không ngoại suy |
| [Kernel / SVR](<../Support Vector Regression (SVR)/Support Vector Regression (SVR).md>) | phi tuyến trơn, không cần chọn bậc | phải chọn hạt nhân và tham số của nó |
| **GAM** — tổng các hàm trơn, mỗi biến một hàm | cộng tính, vẫn đọc được | không bắt tương tác nếu không khai báo |
| **Cơ sở trực giao** (Legendre, Chebyshev) thay cho $1, x, x^2$ | cùng mô hình, điều kiện số học tốt hơn hẳn | hệ số không còn là hệ số của $x^j$ |

### 9.4 Khi nào đừng dùng

- Cần dự đoán **ngoài** khoảng $x$ đã thấy.
- Có nhiều hơn vài biến đầu vào.
- Quan hệ thật thay đổi cục bộ (một đoạn phẳng rồi một đoạn dốc).
- Muốn bậc cao mà không chuẩn hoá hay dùng cơ sở trực giao.

---

## 10. Ứng dụng thật

- **Nguồn gốc: thiên văn.** Gergonne viết bài để cải thiện phương pháp tính quỹ đạo sao chổi của Laplace,
  nơi cần đạo hàm bậc một và hai của kinh độ từ nhiều lần quan sát (mục 3.1, 3.9).
- **Hiệu chuẩn và vật lý.** Khi lý thuyết nói rõ quan hệ là bậc hai (rơi tự do, một số đường hiệu chuẩn
  cảm biến trong dải làm việc), bậc được chọn từ cấu trúc vật lý thay vì dò bằng dữ liệu.
- **Bề mặt đáp ứng trong thiết kế thí nghiệm.** Bậc 2 với đủ số hạng tương tác là mô hình chuẩn để tìm
  cực trị của một quá trình. Chính câu hỏi *đặt điểm đo ở đâu* cho hồi quy đa thức là chủ đề của Kirstine
  Smith 1918.
- **Đặc trưng tương tác trong mô hình tuyến tính.** Thêm $x_1x_2$ là hồi quy đa thức bậc 2 có chọn lọc.
- **Xấp xỉ cục bộ.** LOESS khớp đa thức bậc 1–2 trên cửa sổ trượt: bậc thấp, phạm vi hẹp — cách dùng đa
  thức tránh được cả ba giới hạn ở mục 9.1.

---

## 11. Đối chiếu với nguồn gốc — đã lấy gì, bỏ gì, vì sao

### 11.1 Legendre 1805 — theo đoạn và phần

| Phần | Trang | Nội dung | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|---|
| lời nói đầu, đoạn nêu nhu cầu hệ số vi phân | tr.iv (PDF 8) | cần $dx/dt$, $ddx/dt^2$ của kinh độ | có | mục 1.1 |
| lời nói đầu, *"Il semble, au premier coup-d'œil"* và hệ (a), (b) | tr.iv (PDF 8) | nội suy 3 và 5 quan sát | **có** | Hình 1, mục 3.1 |
| lời nói đầu, sai số $\delta a'$ và tỉ số 4 : 3 | tr.v (PDF 9) | ví dụ số của Legendre | **có, dựng lại** | Hình 2, mục 3.1 |
| lời nói đầu, phần còn lại | tr.v–viii (PDF 9–12) | bố cục cuốn sách, phương pháp quỹ đạo | không | nói về bài toán sao chổi, không về khớp đa thức |
| Phần I và Phần II | PDF 13 trở đi | giải tích quỹ đạo, ví dụ các sao chổi | không | bài toán khác |
| Phụ lục về bình phương tối thiểu | cuối sách | phát biểu phương pháp | không | là nguồn gốc của hồi quy tuyến tính, không riêng đa thức; Gergonne chỉ dẫn tr.iv |

### 11.2 Gergonne 1815 — theo hình và bảng

Bài báo **không có hình và không có bảng nào** (`pdfscan.py` báo 0 dòng khớp caption; đọc cả 11 trang
xác nhận). Bảng này thay bằng các **khối công thức** — thứ đóng vai phần ruột của một bài toán học không
có hình.

| Khối | Trang | Lấy? | Ở đâu |
|---|---|---|---|
| đạo hàm nội suy 3, 5, 7 tung độ | tr.246 | **có, dựng lại 3/3** | Hình 3, mục 3.3 |
| sai số E. khi riêng $b'$ lệch $\beta$ | tr.246 | **có, dựng lại 3/3** | Hình 4, mục 3.3 |
| sai số theo khoảng cách $a$ | tr.247 | **có** | Hình 5, mục 3.4 |
| sai số với $a$ theo số tự nhiên | tr.248 | **có, dựng lại 3/3** | Hình 6, mục 3.4 |
| đa thức (1) | tr.249 | **có** | Hình 7, mục 3.6 |
| danh sách sai số từng quan sát | tr.249 | có | Hình 7, mục 3.6 |
| tổng bình phương = minimum, đạo hàm bằng 0 | tr.250 | có | Hình 8, mục 3.6 |
| định nghĩa $\Sigma a^m$, $\Sigma a^m b$ | tr.250 | có | Hình 8, mục 3.6, bảng ký hiệu 0.3 |
| hệ phương trình chuẩn (1) | tr.250 | **có** | Hình 8, mục 3.6, mục 4.2 |
| đánh số $-n..n$ và tổng luỹ thừa | tr.251 | **có, dựng lại** | Hình 9, mục 3.7 |
| hệ (1) tách chẵn / lẻ | tr.251 | **có, dựng lại trên ví dụ 7 điểm** | Hình 9, mục 3.7 |
| công thức đổi biến | tr.252 | **có — hai chỗ in sai** | Hình 10, mục 3.8 |

### 11.3 Gergonne 1815 — theo trang

Bài báo không đánh số mục; đơn vị chia nhỏ nhất của nó là trang và đoạn.

| Trang | Nội dung | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|
| tr.242 | phát biểu bài toán nội suy dãy | có | mục 1.1 |
| tr.243 | vô số đường cong qua các điểm; hệ số vi phân khác nhau càng nhiều khi bậc càng cao | có | mục 3.2 |
| tr.243, chú thích | dẫn một luận văn ở tr.252 tập V cùng tạp chí | không | không có trong repo; không cần cho lập luận |
| tr.244 | chọn đa thức vì đơn giản; cách nội suy thường dùng; giới hạn dùng giữa các điểm | **có** | mục 1.1, 3.2, 0.8 |
| tr.244, chú thích | Laplace, Mémoires de l'Académie 1780 | không | không có trong repo; chỉ là dẫn nguồn |
| tr.245 | sai số đo; ghi công Legendre; thiết lập $b, b', b_\prime$ | **có** | mục 3.2, 3.3 |
| tr.245, chú thích | dẫn Legendre, *Paris 1806, pag. iv* | có | mục 3.2 (ghi chỗ lệch năm 1805/1806) |
| tr.246 | công thức và sai số | **có** | mục 3.3 |
| tr.247 | nhận xét về Legendre; trích Laplace *Mécanique céleste* t.I tr.201; khoảng cách $a$ | **có** | mục 3.3, 3.4 |
| tr.248 | $a$ theo số tự nhiên; cái giá trên đoạn cố định; câu mở lối sang bình phương tối thiểu | **có** | mục 3.4, 3.5 |
| tr.248–249, chú thích | lịch sử Legendre–Gauss–Laplace | **có** | mục 3.5 |
| tr.249 | phát biểu phương pháp; đa thức (1) | **có** | mục 3.6 |
| tr.250 | hệ (1); tự đánh giá độ chính xác | **có, đo lại** | mục 3.6 |
| tr.251 | trường hợp cách đều; mở đầu đổi biến | **có** | mục 3.7, 3.8 |
| tr.252 | đổi biến; lý do không nội suy cho cách đều; tự đánh giá; câu hỏi bậc nào | **có** | mục 3.8, 3.9 |

### 11.4 Bishop 2006 mục 1.1 — theo hình và bảng

| Hình / bảng | Trang | Nhóm | Lấy? | Ở đâu |
|---|---|---|---|---|
| Hình 1.1 — chữ số viết tay | tr.2 | NGOÀI | không | thuộc phần mở đầu chương 1, trước mục 1.1 |
| Hình 1.2 — 10 điểm và sin(2πx) | tr.4 | RUỘT | **có** | Hình 13, mục 3.10 |
| Hình 1.3 — hàm lỗi hình học | tr.6 | RUỘT | **có** | Hình 14, mục 3.10 |
| Hình 1.4 — bậc 0, 1, 3, 9 | tr.7 | RUỘT | **có** | Hình 15, mục 3.11 |
| Hình 1.5 — $E_{\text{RMS}}$ theo $M$ | tr.8 | RUỘT | **có, số hoá** — trục lệch $\sqrt2$ | Hình 17, mục 3.11 |
| Bảng 1.1 — hệ số theo bậc | tr.8 | RUỘT | **có, dựng lại** — nhãn "$M=6$" là $M=3$ | Hình 18, 19, mục 3.11 |
| Hình 1.6 — $M=9$ với $N=15$, $100$ | tr.9 | RUỘT | **có, đo lại trên dữ liệu tự sinh** | Hình 20, mục 3.12 |
| Hình 1.7 — Ridge, $\ln\lambda = -18$ và $0$ | tr.10 | RUỘT | **có** | Hình 22, mục 3.13 |
| Bảng 1.2 — hệ số theo $\lambda$ | tr.11 | RUỘT | **có, dựng lại** — nhãn $\lambda$ lệch | Hình 23, 19, mục 3.13 |
| Hình 1.8 — $E_{\text{RMS}}$ theo $\ln\lambda$ | tr.11 | RUỘT | **có, số hoá đầu mút** — không khớp Bảng 1.2 | Hình 24, mục 3.13 |

### 11.5 Bishop 2006 mục 1.1 — theo công thức và đoạn

| Phần | Trang | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|
| công thức (1.1) và câu "tuyến tính theo w" | tr.5 | **có** | Hình 11, mục 2, 3.10 |
| công thức (1.2) và hệ số 1/2 | tr.5 | **có** | Hình 12, mục 3.10 |
| đoạn nghiệm dạng đóng, *model selection* | tr.6 | có | mục 2, 3.10 |
| công thức (1.3) | tr.7 | **có** | Hình 16, mục 3.11 |
| đoạn "3 ⩽ M ⩽ 8 cho sai số kiểm tra nhỏ" | tr.7 | có | mục 3.11 (cách đọc Hình 17) |
| nghịch lý "bậc cao chứa bậc thấp" | tr.8 | có | mục 3.11, 6.1 |
| quy tắc "5 hoặc 10 lần số tham số" | tr.9 | **có, đo lại** | mục 3.12 |
| đoạn hợp lý cực đại và Bayes tránh khớp quá mức | tr.9 | không | dẫn sang mục 1.2.5 và 3.4 của sách — phương pháp khác, thuộc bài *Bayesian Linear Regression* |
| công thức (1.4), ghi chú về $w_0$, tên ridge / weight decay | tr.10 | **có, dựng lại** | Hình 21, mục 3.13, 4.3 |
| đoạn tập kiểm định / hold-out | tr.11 | **có** | mục 3.13, 7.2 |
| Bài tập 1.1 và 1.2 (dẫn ở lề tr.6 và tr.10) | cuối chương | không | bài tập tự giải của sách; hệ phương trình chuẩn đã có đầy đủ ở mục 3.6 và 4.2 |
| Phụ lục A, dữ liệu tổng hợp, $\sigma = 0{,}3$ | tr.682 | có | mục 3.10 |
| các mục dẫn tới: 1.2.4, 1.2.5, 1.2.6, 1.3, 1.5, 3.4, 5.5.1 | — | không | mở rộng bằng xác suất, lý thuyết quyết định, Bayes — ngoài phạm vi phương pháp khớp đa thức |

---

## 12. Tài liệu tham khảo và điều kiện tái lập

```
[1] J. D. Gergonne, "Application de la méthode des moindres quarrés à l'interpolation des
    suites", Annales de Mathématiques pures et appliquées 6:242–252, 1815–1816.
    → Nguồn ruột. Bản số hoá Numdam, truy cập mở.
    AI algorithms/_papers/1815 - Gergonne - Application de la methode des moindres quarres a l'interpolation des suites.pdf

[2] A.-M. Legendre, Nouvelles méthodes pour la détermination des orbites des comètes,
    Paris, An XIII (1805). Lời nói đầu tr.iv–v.
    AI algorithms/_papers/1805 - Legendre - Nouvelles methodes pour la determination des orbites des cometes.pdf

[3] C. M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006, mục 1.1 và Phụ lục A.
    AI algorithms/_papers/2006 - Bishop - Pattern Recognition and Machine Learning.pdf

[4] S. M. Stigler, "Gergonne's 1815 paper on the design and analysis of polynomial regression
    experiments", Historia Mathematica 1(4):431–439, 1974 — kèm bản dịch tiếng Anh của [1].
    → KHÔNG có trong repo: ScienceDirect chặn tải tự động. Bài này đọc thẳng bản tiếng Pháp [1]
      và không dùng câu nào của Stigler.

[5] K. Smith, "On the standard deviations of adjusted and interpolated values of an observed
    polynomial function and its constants and the guidance they give towards a proper choice of
    the distribution of observations", Biometrika 12(1/2):1–85, 1918.
    → KHÔNG có trong repo (Oxford Academic, JSTOR, sau tường phí). Chỉ nêu tên ở mục 3.9 và 10;
      không trích số nào.

[6] P.-S. Laplace, Traité de mécanique céleste, t.I, tr.201.
    → KHÔNG có trong repo. Câu trích ở mục 3.4 là câu Gergonne trích, lấy nguyên từ [1] tr.247.

[7] G. James, D. Witten, T. Hastie, R. Tibshirani, An Introduction to Statistical Learning,
    Springer, 2013, chương 7.
    → Spline, spline làm trơn, GAM ở mục 9.3.
    AI algorithms/_papers/2013 - James et al - An Introduction to Statistical Learning.pdf
```

**Script.**

| Script | Sinh ra |
|---|---|
| [`tools/v_poly.py`](<../../../tools/v_poly.py>) | mọi phép dựng lại Gergonne, Legendre, Bishop (mục 1–9), ví dụ tay (mục 10), số lẻ (mục 11). Đầu ra lưu ở `tools/_out_poly.txt` |
| [`tools/v_gd1ss.py`](<../../../tools/v_gd1ss.py>) mục 370 | bảng đường thẳng trên dữ liệu cong, mục 1.2 phép đo thứ ba |
| [`tools/fig_poly.py`](<../../../tools/fig_poly.py>) | Hình 19 và Hình 25 — sinh từ chính bài bằng `extract_fig.py` |

**Điều kiện tái lập.** numpy 2.4.4 · scikit-learn 1.9.1 · matplotlib 3.10.9 · pypdfium2 · Linux x86-64.

- **Tái lập tuyệt đối** (phân số chính xác, không phụ thuộc máy): mọi con số của mục 3.1–3.8, ví dụ tay
  mục 0.5.
- **Tái lập tới chữ số in** trên mọi máy đã chạy: Bảng 1.1 dựng lại, quét $\lambda$ cho Bảng 1.2, RMSE
  của Hình 25 và bảng mục 7.2.
- **Phụ thuộc cách render PDF**: 10 giá trị số hoá Hình 1.5 và 4 đầu mút Hình 1.8 — đổi thư viện render
  hoặc tỉ lệ phóng thì chữ số thứ ba có thể đổi.
- **KHÔNG tái lập qua máy**: $\kappa(V^\top V)$ ở bậc 24 theo cách (a) và (b), và md5 của
  `poly_degree.png` (mục 6.3).

---

## 13. Kiểm tra trước khi đóng bài

- [x] **IO** — mục 0.1: bảng đầu vào/đầu ra kèm kích thước và khối mã chạy được
- [x] **KN** — mục 0.2: bảng thuật ngữ định nghĩa tại chỗ, gồm cả từ vựng thế kỷ 19 của nguồn
- [x] **Bảng ký hiệu** — mục 0.3: ba hệ ký hiệu (bài · Gergonne · Bishop) và năm chỗ trùng tên
- [x] **SS** — mục 1.2: bế tắc của nội suy nêu bằng số (sd 405,8 trên đạo hàm bậc hai khi dùng hết 21
      điểm) và bế tắc của đường thẳng (RMSE đứng ở 0,29 khi $n$ tăng 4 000 lần)
- [x] **CT** — mục 3–5: toàn bộ phương pháp của Gergonne theo trang, mục 1.1 của Bishop theo hình, công
      thức hiện đại, thuật toán từng bước kèm "bỏ đi thì hỏng chỗ nào"
- [x] **VS** — mục 8: cơ chế bù trừ sai số kèm phép đo một-thay-đổi (chỉ đổi số hệ số, sd giảm khoảng
      304 lần)
- [x] **KQ** — mục 3.11–3.13: kết quả của Bishop kèm số hiệu Bảng/Hình và trang, đã dựng lại
- [x] **GIÁ** — mục 9.2: bảng bốn cột có dòng thua (ngoại suy)
- [x] **Luật 6** — 24 khung cắt từ nguồn gốc, mỗi khung có dòng caption, lời tác giả nguyên văn kèm trang,
      và cách đọc; mọi công thức đánh số — (a), (b) của Legendre, (1) và (1) của Gergonne, (1.1)–(1.4) của
      Bishop — có ký hiệu và đoạn ý nghĩa; bốn bảng đối chiếu ở mục 11
- [x] **E.1d** — mọi ví dụ số tác giả tự làm đã dựng lại bằng script: Legendre 4 : 3, Gergonne tr.246 và
      tr.248, Bishop Bảng 1.1 và 1.2; mọi câu tác giả tự đánh giá (Gergonne tr.244, tr.247, tr.250, tr.252)
      đã được đo hoặc ghi rõ phạm vi
- [x] Hình tự vẽ kèm khối mã trong bài — Hình 19 và Hình 25
- [x] Kết quả âm và chỗ nguồn sai giữ nguyên, kèm chẩn đoán — bảng bảy chỗ đầu mục 3

**Còn mở, đã biết:**

1. **Cột $\ln\lambda = -18$ của Bảng 1.2** không dựng lại được trong mức làm tròn với bất kỳ $\lambda$
   nào; không phân biệt được nhãn lệch với bộ giải khác (mục 3.13).
2. **Trục $\lambda$ của Hình 1.8** không khớp Bảng 1.2; bài không đoán trục nào đúng.
3. **Bản dịch Stigler 1974** chưa đọc; mọi câu dịch trong bài là dịch trực tiếp từ tiếng Pháp, chưa đối
   chiếu với bản dịch chuẩn.
4. **Câu trích tiếng Pháp chép từ ảnh trang**: `kiemtrich.py` đối chiếu với lớp OCR của Numdam, nên chỗ
   OCR sai (*"lion consécutifs"*) sẽ bị báo lệch dù câu trong bài đúng với ảnh.
