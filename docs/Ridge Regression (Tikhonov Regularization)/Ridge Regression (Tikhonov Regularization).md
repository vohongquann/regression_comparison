# Ridge Regression (Tikhonov Regularization)

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 380
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Arthur E. Hoerl, Robert W. Kennard
Date: February 1, 1970
url: https://doi.org/10.1080/00401706.1970.10488634

> **Ridge Regression — Hồi quy sườn, còn gọi là chính quy hoá Tikhonov.** Khi các cột dữ liệu mang
> thông tin gần trùng nhau, cách khớp bình phương tối thiểu thông thường cho ra hệ số khổng lồ, đổi
> dấu tuỳ mẫu, vô nghĩa về vật lý. Ridge cộng thêm vào hàm mục tiêu **một khoản phạt tỉ lệ với tổng
> bình phương các hệ số**. Nghiệm vì thế bị kéo về gần 0: sai một chút có hệ thống, đổi lấy việc ổn
> định hơn rất nhiều.
>
> Bài này theo **ba nguồn gốc**, cả ba nằm trong `_papers/`:
>
> 1. **A. N. Tikhonov 1963**, *Doklady* 151(3), 4 trang tiếng Nga — phương pháp chính quy hoá cho phương
>    trình tích phân loại một. Cùng một phép cộng khoản phạt, bảy năm trước ridge và trong một bài toán
>    khác hẳn.
> 2. **A. E. Hoerl & R. W. Kennard 1970**, *Technometrics* 12(1), 13 trang — bài báo đặt tên *ridge
>    regression* và chứng minh tồn tại một mức phạt làm sai số bình phương trung bình **nhỏ hơn** bình
>    phương tối thiểu.
> 3. **T. Hastie, R. Tibshirani & J. Friedman, *The Elements of Statistical Learning*, bản 2, §3.4.1**
>    — nguồn hiện đại, có dữ liệu thật (ung thư tuyến tiền liệt, 97 bệnh nhân) để dựng lại từng con số.
>
> Mọi công thức đánh số của cả ba nguồn đều được kiểm lại bằng
> [`tools/v_ridge.py`](<../../../tools/v_ridge.py>). Việc kiểm tìm ra **mười một chỗ** nguồn in sai, tự
> mâu thuẫn hoặc không dựng lại được (bảng đầu mục 3). Không chỗ nào lật một kết luận chính.

---

## 0. Bắt đầu từ đây

### 0.1 Nhận gì, trả gì

Ridge nhận **đúng những gì hồi quy tuyến tính thông thường nhận**, cộng một con số do người dùng
chọn: mức phạt $\lambda$. Đầu ra cũng cùng hình dạng. Không có kiểu dữ liệu mới nào.

| | Cách hiểu thông thường | Ký hiệu | Kích thước ở ví dụ mục 0.5 |
|---|---|---|---|
| **Đầu vào** | một bảng gồm $n$ mẫu và $p$ cột số | $X\in\mathbb{R}^{n\times p}$ | $3\times2$ |
| **Đáp án khi huấn luyện** | một giá trị số cho mỗi mẫu | $y\in\mathbb{R}^{n}$ | $3$ số |
| **Siêu tham số** | "phạt hệ số lớn mạnh tới mức nào" | $\lambda\ge0$ | $\lambda=1$ |
| **Đầu ra của việc học** | một hệ số cho mỗi cột | $\hat\beta\in\mathbb{R}^{p}$ | $2$ số |
| **Dự đoán** | nhân từng dòng dữ liệu với các hệ số | $\hat y=X\hat\beta$ | $3$ số |

```python
import numpy as np

X = np.array([[1., 1.],
              [2., 2.],
              [3., 3.]])
y = np.array([2., 4., 6.])

ols, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
lam = 1.0
ridge = np.linalg.solve(X.T @ X + lam*np.eye(2), X.T @ y)

print("rank =", rank)
print("OLS chuẩn nhỏ nhất =", np.round(ols, 4))
print("ridge =", np.round(ridge, 4))
```

```text
rank = 1
OLS chuẩn nhỏ nhất = [1. 1.]
ridge = [0.9655 0.9655]
```

Vì hai cột giống hệt nhau, mọi cặp có tổng bằng 2 như $(2,0)$, $(1,1)$ hay $(0,2)$ đều cho **cùng
một dự đoán**. Bình phương tối thiểu không có đáp án duy nhất; `lstsq` chọn nghiệm có chuẩn Euclid
nhỏ nhất, và đó là **quy ước của hàm giải**, không phải của phương pháp. Ridge thì có đáp án duy
nhất: hai hệ số bằng nhau, hơi co về 0. Mục 0.5 tính tay đúng con số $0{,}9655$ này.

Bốn điều dễ hiểu sai, nói rõ ngay:

- **$\lambda$ không học được từ dữ liệu huấn luyện.** Chọn $\lambda$ bằng sai số huấn luyện thì luôn
  ra $\lambda=0$, tức quay về đúng cách cũ. Mục 7 nói cách chọn.
- **Kết quả phụ thuộc ĐƠN VỊ của từng cột.** Đổi một cột từ mét sang xentimét làm hệ số của nó nhỏ
  đi 100 lần, nên khoản phạt tác động lên nó khác hẳn. Bình phương tối thiểu không có tính chất này.
  Mục 3.17 cho thấy ngay cả một cuốn sách giáo khoa cũng dùng **hai** cách chuẩn hoá khác nhau cho
  một bảng và một hình cạnh nhau.
- **Hệ số chặn thường KHÔNG bị phạt.** Nếu phạt, cộng một hằng số vào mọi $y$ sẽ không còn chỉ dời
  dự đoán đi đúng hằng số đó (ESL tr.64, mục 3.16).
- **Đầu ra không có số 0 đúng nghĩa.** Ridge co hệ số nhỏ lại chứ không xoá biến. Muốn xoá hẳn biến
  thì cần [Lasso](<../Lasso Regression/Lasso Regression.md>) — mục 8.3.

### 0.2 Từ mới trong bài này

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Trong bài này | Nghĩa |
|---|---|---|
| **Bình phương tối thiểu** (OLS, *least squares*) | trường hợp $\lambda=0$ | chọn hệ số làm **tổng bình phương phần dư** nhỏ nhất |
| **Phần dư** (*residual*) | $e_i=y_i-\hat y_i$ | giá trị thật trừ dự đoán |
| **Đa cộng tuyến** / **không trực giao** (*nonorthogonal*) | ví dụ mục 0.5; tên bài báo 1970 | các cột của $X$ mang thông tin gần trùng nhau; ma trận $X^\top X$ có trị riêng rất nhỏ |
| **Trị riêng** (*eigenvalue*) | $\lambda_i$ của Hoerl–Kennard | số $\lambda$ sao cho $A v=\lambda v$ với một vector $v\ne0$. Với $X^\top X$, trị riêng nhỏ nghĩa là có một hướng mà dữ liệu gần như không trải ra |
| **Co hệ số** (*shrinkage*) | tác dụng của $\lambda$ | kéo độ lớn hệ số về gần 0 |
| **Chính quy hoá** (*regularization*) | tên Tikhonov đặt năm 1963 | thêm một điều kiện ưu tiên nghiệm "hiền" để bài toán có nghiệm ổn định |
| **Bài toán đặt không chỉnh** (*некорректно поставленная задача*, *ill-posed*) | Tikhonov tr.501 | bài toán mà một thay đổi rất nhỏ ở dữ liệu có thể gây thay đổi rất lớn ở nghiệm |
| **Chệch** (*bias*) và **phương sai** (*variance*) | Hoerl–Kennard §4 | *chệch*: sai lệch có hệ thống giữa trung bình của ước lượng và giá trị thật; *phương sai*: mức nhảy của ước lượng khi đổi mẫu dữ liệu |
| **Sai số bình phương trung bình** (MSE) | $E[L_1^2]$ của Hoerl–Kennard | trung bình của bình phương khoảng cách từ ước lượng tới giá trị thật; bằng chệch² cộng phương sai |
| **Vết ridge** (*ridge trace*) | Hoerl–Kennard §3 | đồ thị từng hệ số theo mức phạt $k$ — cái tên "ridge" đi kèm với nó |
| **Phân rã giá trị kỳ dị** (SVD) | ESL (3.45) | viết $X=UDV^\top$: hai ma trận cột trực chuẩn và một ma trận đường chéo chứa **giá trị kỳ dị** $d_j\ge0$ |
| **Bậc tự do hiệu dụng** $\operatorname{df}(\lambda)$ | ESL (3.50) | mô hình ridge "linh hoạt bằng bao nhiêu tham số"; là số thực từ $p$ (khi $\lambda=0$) giảm về 0 |
| **Kiểm định chéo 10 mảnh** (*10-fold cross-validation*) | ESL tr.61 | chia tập huấn luyện làm 10 phần; lần lượt khớp trên 9 phần và đo sai số trên phần còn lại; lấy trung bình 10 lần |
| **Quy tắc một sai số chuẩn** (*one-standard-error rule*) | ESL tr.61 | trong các mô hình có sai số kiểm định chéo không vượt quá "cực tiểu + một sai số chuẩn", chọn mô hình **đơn giản nhất** |
| **Chuẩn hoá** (*standardize*) | ESL tr.50, tr.63 | trừ trung bình rồi chia độ lệch chuẩn cho từng cột, để mọi cột có cùng thang |
| **Hậu nghiệm** (*posterior*) | Hoerl–Kennard §6, ESL tr.64 | phân phối của tham số sau khi đã thấy dữ liệu, trong cách nhìn xác suất Bayes |

### 0.3 Bảng ký hiệu — và chỗ trùng tên giữa ba nguồn

Ba nguồn dùng **ba hệ ký hiệu**, và chúng va nhau ở gần như mọi chữ cái. Bảng đầu là ký hiệu của
chính bài này.

| Ký hiệu | Đọc là gì | Nghĩa | Chiều / kiểu |
|---|---|---|---|
| $n$, $p$ | "en", "pê" | số mẫu, số cột của $X$ | số nguyên |
| $X$, $y$ | "ích lớn", "y" | bảng đầu vào, cột đáp án | $\mathbb{R}^{n\times p}$, $\mathbb{R}^{n}$ |
| $\beta$, $\hat\beta$ | "bê-ta", "bê-ta mũ" | hệ số thật (không quan sát được), hệ số ước lượng | $\mathbb{R}^{p}$ |
| $\lambda$ | "lam-đa" | **mức phạt** — núm điều khiển của cả bài | vô hướng $\ge0$ |
| $\sigma^2$ | "xích-ma bình phương" | phương sai của nhiễu | vô hướng |
| $I$ | "ma trận đơn vị" | đường chéo toàn 1 | $\mathbb{R}^{p\times p}$ |
| $d_j$ | "đê chỉ số j" | giá trị kỳ dị thứ $j$ của $X$ đã trừ trung bình | vô hướng $\ge0$ |
| $\operatorname{df}(\lambda)$ | "đê-ép của lam-đa" | bậc tự do hiệu dụng | số thực trong $[0,p]$ |
| $\lVert\beta\rVert_2^2$ | "chuẩn hai của bê-ta, bình phương" | $\beta_1^2+\dots+\beta_p^2$ — **thứ ridge phạt** | vô hướng |
| $\lVert\beta\rVert_1$ | "chuẩn một của bê-ta" | $\lvert\beta_1\rvert+\dots+\lvert\beta_p\rvert$ — thứ Lasso phạt | vô hướng |
| $\operatorname{tr}(A)$ | "vết của a" | tổng các phần tử đường chéo | vô hướng |
| $\arg\min_\beta$ | "ác-min theo bê-ta" | giá trị $\beta$ làm biểu thức nhỏ nhất | $\mathbb{R}^{p}$ |

**Ba hệ ký hiệu đặt cạnh nhau.**

| Đại lượng | Bài này | Tikhonov 1963 | Hoerl–Kennard 1970 | ESL 2009 | scikit-learn |
|---|---|---|---|---|---|
| mức phạt | $\lambda$ | $\alpha$ | $k$ | $\lambda$ | `alpha` |
| nghiệm | $\hat\beta$ | $z^\alpha(s)$ | $\hat\beta^*$ | $\hat\beta^{\text{ridge}}$ | `coef_` |
| dữ liệu đầu vào | $X$ | nhân $K(x,s)$ | $X$ | $\mathbf X$ | `X` |
| đáp án | $y$ | $\bar u(x)$ hoặc $\tilde u_\delta(x)$ | $Y$ | $\mathbf y$ | `y` |
| trị riêng của $X^\top X$ | $d_j^2$ | — | $\lambda_i$ | $d_j^2$ | — |
| nghiệm bình phương tối thiểu | $\hat\beta_{\text{OLS}}$ | — | $\hat\beta$ | $\hat\beta^{\text{ls}}$ | — |
| mức nhiễu | $\sigma$ | $\delta$ | $\sigma$ | $\sigma$ | — |

> **Chín chỗ trùng tên — những chỗ gây lỗi thật khi đọc hoặc chép mã:**
>
> 1. **$\lambda$ của ESL là mức phạt; $\lambda_i$ của Hoerl–Kennard là TRỊ RIÊNG.** Hai bài nguồn dùng
>    cùng chữ cho hai thứ khác nhau hoàn toàn. Cầu nối: $\lambda_i^{\text{HK}}=d_j^2$ của ESL.
> 2. **$\alpha$ ở ba nghĩa:** tham số chính quy hoá của Tikhonov (đúng vai trò $\lambda$) · vector toạ độ
>    chuẩn tắc $\alpha=P\beta$ của Hoerl–Kennard (4.12) · và `alpha` của scikit-learn, lại là mức phạt.
>    Thêm một nghĩa thứ tư ngay trong Tikhonov tr.504: **hàm** $\alpha(x,s)=A[x,\eta_s]$.
> 3. **$k$ của Hoerl–Kennard là mức phạt; $k(s)$ của Tikhonov là trọng số của số hạng đạo hàm** trong
>    phiếm hàm phạt. Đặt $k(s)=0$ ở Tikhonov mới ra ridge (mục 3.5).
> 4. **$K$:** ma trận đường chéo các mức phạt riêng $k_i$ của Hoerl–Kennard §5 · nhân $K(x,s)$ của
>    phương trình tích phân Tikhonov.
> 5. **$\gamma_1,\gamma_2$:** tổng phương sai và chệch² của Hoerl–Kennard (4.6) · hai hằng số kẹp tỉ số
>    $\delta^2/\alpha$ trong định lý 3 của Tikhonov.
> 6. **$Z$:** ma trận co $[I+k(X^\top X)^{-1}]^{-1}$ của Hoerl–Kennard (2.4) · lớp hàm compact $\bar Z$ của
>    Tikhonov.
> 7. **$P$ và $p$:** ma trận trực giao của Hoerl–Kennard · hàm trọng số $p(s)$ của Tikhonov · số cột $p$
>    của ESL và của bài này.
> 8. **$N$:** phiếm hàm độ lệch bình phương $N[z,\bar u]$ của Tikhonov · số mẫu $N$ của ESL (bài này
>    viết $n$).
> 9. **$L$:** khoảng cách $L_1$ của Hoerl–Kennard (1.4) — **không** phải chuẩn một · toán tử Euler
>    $L^\alpha$ của Tikhonov (3).

### 0.4 Luồng xử lý

Ridge không phải một kiến trúc, nhưng nó là **một quy trình có thứ tự bắt buộc**, và sai thứ tự là
nguồn lỗi phổ biến hơn cả sai công thức.

```
   X (n×p), y (n)
        │
        ▼
   ┌─────────────────────────┐
   │ 1. Tách hệ số chặn      │   trừ trung bình của X và y, khớp không có hệ số chặn
   │    ra khỏi khoản phạt   │   (nếu không: kết quả đổi theo gốc đo của y)
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐
   │ 2. Chuẩn hoá thang đo   │   mỗi cột về cùng độ lệch chuẩn
   │    của từng cột         │   (nếu không: cột đơn vị nhỏ bị phạt nặng hơn)
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐   ◄── chỉ chạy trên tập huấn luyện của từng mảnh
   │ 3. Chọn λ bằng          │       kiểm định chéo — không chạy trên toàn bộ dữ liệu
   │    kiểm định chéo       │       (nếu không: rò rỉ dữ liệu)
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐
   │ 4. Giải                 │   (XᵀX + λI) β̂ = Xᵀy
   │                         │   hoặc SVD, hoặc QR trên dữ liệu thêm p hàng giả
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐
   │ 5. Dự đoán              │   ŷ = β̂₀ + X_mới β̂, sau khi ÁP LẠI ĐÚNG phép trừ
   │                         │   trung bình và chia độ lệch chuẩn đã học ở bước 1–2
   └─────────────────────────┘
```

| Bước | Nhận vào | Trả ra | Bỏ đi thì hỏng gì |
|---|---|---|---|
| 1. Tách hệ số chặn | $X,y$ | $X,y$ đã trừ trung bình | hệ số chặn bị co về 0, dự đoán lệch một hằng số |
| 2. Chuẩn hoá cột | $X$ đã trừ trung bình | $X$ đã chia độ lệch chuẩn | phạt **không đều**: cột đo bằng đơn vị nhỏ có hệ số lớn nên bị phạt nặng hơn, dù không kém quan trọng hơn |
| 3. Chọn $\lambda$ | lưới $\lambda$, dữ liệu huấn luyện | một $\lambda$ | chọn bằng sai số huấn luyện luôn ra $\lambda=0$ |
| 4. Giải | $X,y,\lambda$ | $\hat\beta$ | — bước duy nhất có công thức đóng, và rẻ nhất |
| 5. Áp lại phép biến đổi | $X_\text{mới}$ | $\hat y$ | lỗi im lặng nguy hiểm nhất: mô hình vẫn trả số, và số đó sai |

**Bốn trong năm bước không dính gì tới công thức ridge.** Chúng là hệ quả của việc khoản phạt phụ
thuộc thang đo. Bình phương tối thiểu không cần bước 1–2 vì nghiệm của nó đổi đúng theo mọi phép đổi
đơn vị tuyến tính; ridge thì không.

### 0.5 Một vòng tính tay, bằng số thật

Dùng đúng ví dụ mục 0.1: hai cột trùng hệt nhau, $\lambda=1$, không có hệ số chặn.

**Bước 1 — lập $X^\top X$ và $X^\top y$.** Mỗi phần tử là tổng tích của hai cột:

$$X^\top X=\begin{pmatrix}1\cdot1+2\cdot2+3\cdot3 & 14\\ 14 & 14\end{pmatrix}=\begin{pmatrix}14&14\\14&14\end{pmatrix},\qquad
X^\top y=\begin{pmatrix}1\cdot2+2\cdot4+3\cdot6\\ 28\end{pmatrix}=\begin{pmatrix}28\\28\end{pmatrix}.$$

Định thức của $X^\top X$ bằng $14\cdot14-14\cdot14=0$: ma trận **suy biến**, hệ bình phương tối thiểu có
vô số nghiệm.

**Bước 2 — cộng $\lambda I$.** $X^\top X+I=\begin{pmatrix}15&14\\14&15\end{pmatrix}$, định thức
$15\cdot15-14\cdot14=29\ne0$. Một phép cộng vào đường chéo biến bài toán vô số nghiệm thành bài toán có
đúng một nghiệm — đó là toàn bộ câu *"adding small positive quantities to the diagonal of X'X"* của
Hoerl–Kennard tr.55.

**Bước 3 — giải hệ $2\times2$ bằng quy tắc Cramer.**

$$\hat\beta_1=\frac{15\cdot28-14\cdot28}{29}=\frac{28}{29},\qquad \hat\beta_2=\frac{15\cdot28-14\cdot28}{29}=\frac{28}{29}=0{,}965517.$$

**Bước 4 — dự đoán và hàm mục tiêu.** $\hat y=(1{,}931034;\ 3{,}862069;\ 5{,}793103)$. Tổng bình phương
phần dư $0{,}066587$, khoản phạt $\lambda\lVert\hat\beta\rVert^2=1{,}864447$, cộng lại
$1{,}931034$.

**Bước 5 — nhìn theo hướng.** Dữ liệu chỉ trải theo **một** hướng, $(1,1)/\sqrt2$, với
$d^2=28$. Ridge nhân thành phần của nghiệm theo hướng đó với $d^2/(d^2+\lambda)=28/29$ — đúng hệ số
co trong công thức (3.47) của ESL (mục 3.18). Hướng còn lại, $(1,-1)/\sqrt2$, có $d^2=0$: dữ liệu không
nói gì về nó, và ridge đặt thành phần theo hướng đó bằng **0**. Vì thế hai hệ số bằng nhau.

Mọi con số của mục này, tính bằng phân số chính xác: [`tools/v_ridge.py`](<../../../tools/v_ridge.py>)
mục 18.

### 0.6 Điểm dễ nhầm

- **"Ridge sửa đa cộng tuyến."** Không. Các cột vẫn tương quan y như cũ. Ridge chỉ làm **ước lượng**
  bớt nhạy với chuyện đó, và trả giá bằng việc hệ số không còn đọc được như hiệu ứng riêng của từng
  biến.
- **"$\lambda$ càng lớn càng an toàn."** Không. Hoerl–Kennard §4c tr.63 nói rõ cực tiểu sai số dời về
  $k=0$ khi $\lVert\beta\rVert^2$ lớn; mục 3.11 đo: $\beta^\top\beta$ tăng 1 000 lần thì mức phạt tối ưu
  giảm 1 191,2 lần.
- **"df$(\lambda)$ là số biến còn lại."** Không. Ridge không xoá biến nào; $\operatorname{df}$ đo độ
  linh hoạt và là số thực.
- **"ESL chọn df = 5 vì đó là cực tiểu sai số kiểm định chéo."** Đó là câu in ở tr.68, và chính Hình 3.7
  của sách bác nó: cực tiểu nằm ở df ≈ 7, còn df = 5 là lựa chọn của quy tắc một sai số chuẩn (mục
  3.15).
- **Nhầm $\lambda$ và $\alpha$ giữa các nguồn** — xem chín chỗ trùng tên ở mục 0.3.

### 0.7 Đọc bài theo thứ tự nào

| Muốn | Đọc |
|---|---|
| dùng được ridge ngay | mục 0 → mục 5 (thuật toán) → mục 7 (chọn $\lambda$) → mục 8 (giá phải trả) |
| hiểu vì sao cần nó | mục 1 (bế tắc của bình phương tối thiểu, bằng số) → mục 2 → mục 6 |
| đọc nguồn gốc | mục 3 theo thứ tự: Tikhonov 3.1–3.5 → Hoerl–Kennard 3.6–3.13 → ESL 3.14–3.19 |
| kiểm lại bài | mục 3.20 (hình dựng lại) → mục 10 (bảng đối chiếu) → mục 11 (điều kiện tái lập) |

### 0.8 Khi KHÔNG nên dùng

- **Cần một danh sách biến ngắn.** Ridge giữ mọi hệ số khác 0. Dùng [Lasso](<../Lasso Regression/Lasso Regression.md>) hoặc
  [Elastic Net](<../Elastic Net/Elastic Net.md>).
- **Các cột đã gần trực giao và $n\gg p$.** Mức lợi của ridge tăng theo độ trải của phổ trị riêng
  (Hoerl–Kennard tr.66); khi phổ phẳng thì cái lợi gần như bằng không, còn cái giá — một siêu tham số
  phải chọn và hệ số có chệch — thì vẫn còn.
- **Cần diễn giải hệ số như hiệu ứng nhân quả của từng biến.** Chệch có chủ đích làm chuyện đó sai.
- **Nhiễu có đuôi dày hoặc có điểm ngoại lai.** Ridge vẫn dùng tổng bình phương phần dư, nên nhạy với
  ngoại lai y như bình phương tối thiểu; cần [Huber Regression](<../Huber Regression/Huber Regression.md>).

---

## 1. Bài toán mà nó giải — và cách cũ tắc ở đâu

### 1.1 Bài toán

Mô hình quen thuộc $y=X\beta+\varepsilon$, nhiễu có trung bình 0 và phương sai $\sigma^2$. Cách cũ là
bình phương tối thiểu: $\hat\beta=(X^\top X)^{-1}X^\top y$. Nó là **ước lượng không chệch có phương sai
nhỏ nhất** trong các ước lượng tuyến tính (định lý Gauss–Markov) — tức tốt nhất nếu chỉ được phép chọn
trong lớp không chệch.

Hoerl và Kennard làm việc ở công ty hoá chất Du Pont. Họ mô tả ở tr.55 chuyện hay gặp trong dữ liệu
công nghiệp: các đại lượng đo cùng một quá trình nên tương quan mạnh với nhau, và khi đó hệ số bình
phương tối thiểu *"often do not make sense when put into the context of the physics, chemistry, and
engineering of the process which is generating the data"* (thường vô nghĩa khi đặt vào bối cảnh vật
lý, hoá học và kỹ thuật của quá trình sinh ra dữ liệu).

### 1.2 Cách cũ tắc ở đâu, đo bằng số

**Phép đo thứ nhất — công thức (1.8) của Hoerl–Kennard.** Khoảng cách bình phương trung bình từ
$\hat\beta$ tới $\beta$ bằng $\sigma^2\sum_i 1/\lambda_i$, với $\lambda_i$ là trị riêng của $X^\top X$
dạng tương quan. Một trị riêng nhỏ là đủ làm tổng nổ. Trên một thiết kế 8 cột sinh từ 2 hướng độc lập
(tương quan 0,99; [`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7):

| Đại lượng | Giá trị |
|---|---:|
| trị riêng lớn nhất / nhỏ nhất | 4,141922 / 0,008073 |
| tỉ số | $5{,}1308\cdot10^{2}$ |
| $\beta^\top\beta$ — độ dài bình phương của hệ số THẬT | 13 |
| $E[\hat\beta^\top\hat\beta]=\beta^\top\beta+\sigma^2\operatorname{tr}(X^\top X)^{-1}$ — công thức (1.5a) | 429,8554 |
| Monte Carlo 40 000 lần rút nhiễu | 429,9157 |
| $E[L_1^2]=\sigma^2\sum 1/\lambda_i$ — công thức (1.8) | 416,8554 |
| cận dưới $\sigma^2/\lambda_{\min}$ | 123,8741 |

Vector hệ số ước lượng **dài gấp khoảng 33 lần** vector thật tính theo bình phương độ dài
(429,8554 so với 13). Đó là con số đứng sau câu *"Estimated coefficients, β̂i, that are large in
absolute value have been observed by all who have tackled live nonorthogonal data problems"*
(tr.56).

**Phép đo thứ hai — hai cột gần trùng, 400 mẫu lặp lại.** Thiết kế: $n=50$, hai cột $x_1=z$ và
$x_2=z+0{,}01\cdot\text{nhiễu}$ — tương quan **0,999939**. Hệ số thật $\beta=(1,\,1)$, nên **tổng**
$\beta_1+\beta_2=2$ xác định rõ, còn **cách chia** thì hầu như không. Chạy lại trên 400 mẫu độc lập
cùng phân phối:

| | $\hat\beta_1$ trung bình | độ lệch chuẩn | khoảng $\hat\beta_1$ | $\lVert\hat\beta-\beta\rVert$ trung bình |
|---|---:|---:|:---:|---:|
| **Bình phương tối thiểu** | $+1{,}09$ | $14{,}27$ | $[-39{,}09,\ +42{,}57]$ | $16{,}096$ |
| **Ridge $\lambda=1$** | $+0{,}99$ | $0{,}08$ | $[+0{,}78,\ +1{,}18]$ | $0{,}097$ |

Bốn con số đáng nhớ:

1. **Bình phương tối thiểu cho $\hat\beta_1$ ngược dấu với sự thật ở 48% số lần chạy.** Gần như tung
   đồng xu. Ridge: 0%.
2. **Tổng thì bình phương tối thiểu ước lượng rất tốt.** Độ lệch chuẩn của $\hat\beta_1+\hat\beta_2$ chỉ
   $0{,}14$, của riêng $\hat\beta_1$ là $14{,}27$ — **gấp 99 lần**. Dữ liệu nói rõ tổng bằng bao
   nhiêu và gần như im lặng về cách chia; bình phương tối thiểu vẫn buộc phải chia, nên nó chia bừa.
3. **Sai số ước lượng tổng thể nhỏ hơn 165 lần** khi dùng ridge, trên cùng dữ liệu.
4. $\kappa(X)=1{,}83\cdot10^{2}$ còn $\kappa(X^\top X)=3{,}36\cdot10^{4}$ — bình phương lên. Đó là lý do
   không nên lập $X^\top X$ nếu tránh được (mục 5).

Mã sinh bảng này: [`tools/v_gd1ss.py`](<../../../tools/v_gd1ss.py>), mục `380  Ridge Regression`.

**Phép đo thứ ba — bài toán ngược của Tikhonov.** Phương trình tích phân loại một của Tikhonov
tr.501, rời rạc hoá 80 ẩn và 100 phương trình, nhiễu $\delta=10^{-3}$
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 4). Giải bằng bình phương tối thiểu trần:

| Cách giải | sai số đều $\max\lvert z-\bar z\rvert$ |
|---|---:|
| bình phương tối thiểu trần ($\alpha=0$) | $9{,}438\cdot10^{9}$ |
| chính quy hoá Tikhonov, $\alpha=10^{-6}$ | 0,022421 |

Nghiệm thật có biên độ 1,498940. Bình phương tối thiểu trả về một nghiệm mà sai số lớn hơn biên độ đó
$10^{9{,}7991}$ lần — gần **mười bậc độ lớn** — vì số điều kiện của ma trận là $6{,}088\cdot10^{18}$,
vượt quá nghịch đảo của độ chính xác số thực 64 bit (khoảng $4{,}5\cdot10^{15}$).

### 1.3 Nó đổi đúng cái gì

Cách cũ có hai lối thoát, và Hoerl–Kennard tr.55 bác cả hai: *"one is forced to treat the estimated
predicting function as a black box or to drop factors to destroy the correlation bonds among the Xi
used to form X'X. Both these alternatives are unsatisfactory if the original intent was to use the
estimated predictor for control and optimization."* (người ta buộc phải coi hàm dự đoán là hộp đen,
hoặc bỏ bớt biến để phá mối tương quan; cả hai đều không chấp nhận được nếu mục đích là dùng mô hình
để điều khiển và tối ưu hoá.)

Ridge đổi **đúng một chỗ**: thay $X^\top X$ bằng $X^\top X+\lambda I$. Không bỏ biến, không đổi mô hình,
không đổi dữ liệu. Cái giá là bỏ **tính không chệch** — thứ mà định lý Gauss–Markov lấy làm điều kiện.

---

## 2. Ý tưởng trung tâm — bằng lời, chưa công thức

Khi các cột gần trùng nhau, dữ liệu **không chứa thông tin** để phân biệt nhiều cách chia hệ số khác
nhau: vô số vector $\beta$ cho tổng bình phương phần dư gần như bằng nhau. Bình phương tối thiểu vẫn
chọn đúng một cái — cái nào thì do nhiễu quyết định, và cái đó thường **rất dài**.

Ridge thêm một ưu tiên: *trong các vector giải thích dữ liệu gần tốt như nhau, chọn vector ngắn
nhất*. Hoerl–Kennard tr.58 nói đúng lập luận đó: *"the worse the conditioning of X'X, the more β̂ can
be expected to be too long. On the other hand, the worse the conditioning, the further one can move
from β̂ without an appreciable increase in the residual sums of squares."* (ma trận càng xấu thì
$\hat\beta$ càng dễ quá dài; nhưng cũng chính khi đó, người ta đi được càng xa khỏi $\hat\beta$ mà tổng
bình phương phần dư gần như không tăng.)

Có ba cách phát biểu cùng ý đó, và ba nguồn của bài mỗi nguồn đi một đường:

| Cách nhìn | Phát biểu | Nguồn |
|---|---|---|
| **phạt** | cực tiểu hoá (độ lệch với dữ liệu) + $\lambda\cdot$(độ lớn nghiệm) | Tikhonov (2) · ESL (3.41) |
| **ràng buộc** | trên mặt cầu $\lVert\beta\rVert^2=R^2$, chọn điểm có tổng bình phương phần dư nhỏ nhất — hoặc ngược lại | Hoerl–Kennard (3.2), (3.7) · ESL (3.42) |
| **xác suất** | hệ số có tiên nghiệm Gauss quanh 0; nghiệm ridge là trung bình hậu nghiệm | Hoerl–Kennard §6 · ESL tr.64 |

Vì sao đổi lấy chệch lại có lời: phương sai của bình phương tối thiểu nằm dồn ở các hướng có trị
riêng nhỏ. Ridge co mạnh **đúng những hướng đó** và gần như để nguyên các hướng dữ liệu trải rộng. Nó
trả một ít chệch ở nơi dữ liệu yếu để lấy lại rất nhiều phương sai cũng ở chính nơi đó. Mục 6 đo cơ
chế này.

---

## 3. Phần ruột — theo đúng ba nguồn gốc

Phần này đi qua **toàn bộ** phần phương pháp của ba nguồn, theo đúng thứ tự in:

| Nguồn | Phần đưa vào | Vì sao là phần ruột |
|---|---|---|
| **Tikhonov 1963**, *Докл. АН СССР* 151(3):501–504 — **cả bài** | công thức (1), (2), (3), (4), (3′), (1′), định lý 1–3 và 1′–2′, dạng tích phân tr.504 | nguồn gốc của phép cộng khoản phạt vào bài toán không chỉnh, và của từ *chính quy hoá* |
| **Hoerl & Kennard 1970**, *Technometrics* 12(1):55–67 — **cả bài** | công thức (1.1)–(5.4), Hình 1, định lý 4.1–4.3, mục 6–8, bảng ký hiệu | bài báo đặt tên *ridge regression* và chứng minh định lý tồn tại |
| **ESL 2009**, §3.3.4 và 3.4.1, tr.61–68, cùng Bảng 3.4 và Hình 3.11 tr.71 | công thức (3.41)–(3.50), Hình 3.7–3.9, 3.11, Bảng 3.3–3.4 | nguồn hiện đại có dữ liệu thật để dựng lại từng con số |

**Cách trích.** Tikhonov 1963 là bản quét Math-Net.Ru; Hoerl–Kennard 1970 là bản quét JSTOR có sẵn
trong repo từ lần nhập Notion, **không có lớp chữ**. Mọi câu tiếng Nga và mọi câu của Hoerl–Kennard
dưới đây **chép từ ảnh trang**, và `kiemtrich.py` không kiểm được chúng. Hai nguồn này không có caption
cho khối công thức, nên "Hình" ở mục 3.1–3.13 phần lớn là **khung cắt từ trang in**; ô "caption gốc"
được thay bằng dòng nói khung chứa gì. Hoerl–Kennard có đúng **một** hình có caption (Hình 1 tr.61) và
nó có mục riêng. Câu tiếng Anh của ESL lấy từ lớp chữ của PDF.

**Mười một chỗ việc kiểm lại tìm ra.**

| # | Nguồn · chỗ | Nguồn in / nói | Kiểm lại cho thấy | Mục |
|---|---|---|---|---|
| 1 | Tikhonov tr.502, chứng minh định lý 2 | $\lVert u^\alpha-\bar u\rVert\le\alpha C$ | từ $M^\alpha\le\alpha C^2$ chỉ suy ra được $\le C\sqrt\alpha$; bất đẳng thức in **sai ở 9/10** giá trị $\alpha$ thử, và sai ở 10/10 sau khi đổi thang $\Omega$. Kết luận của định lý không đổi | 3.3 |
| 2 | Tikhonov tr.503, lưới sai phân | $h=\tfrac1n(a-b)$, $h_1=\tfrac1m(c-d)$ | bước lưới **âm** vì $a<b$; lưới $s_j=jh-0{,}5h$ còn thiếu số hạng $a$ | 3.4 |
| 3 | Tikhonov tr.504, dạng tích phân của (3) | $\alpha\{k z'+\int_0^s p\,z\,d\zeta\}$ | dấu phải là **trừ** và cận dưới là $a$: dấu trừ cho phần dư $9{,}758\cdot10^{-18}$, dấu cộng để lại đúng $2\alpha\int p\,z$ | 3.5 |
| 4 | Hoerl–Kennard tr.61–62 | định lý 4.1, hệ quả 4.1.1, định lý 4.2 | khối này **in hai lần** vắt qua trang; hệ quả 4.1.1 có hai cách diễn đạt khác nhau, và chứng minh của định lý 4.2 chen vào giữa | 3.11 |
| 5 | Hoerl–Kennard tr.59, công thức (3.3) | $(B-\hat\beta)'X'X(B-\hat\beta)'$ | dấu phẩy thứ hai làm phép nhân **không khớp chiều**; (3.4) ngay dưới dùng đúng dạng không có nó | 3.9 |
| 6 | Hoerl–Kennard tr.66, bảng ký hiệu | $Z^*=\{I+[(X^*)'(X^*)]^{-1}K]\}=I-KW$ | thiếu **nghịch đảo ngoài cùng** (lệch 25,74 so với $I-KW^*$; thêm vào thì lệch $1{,}1\cdot10^{-16}$), và vế phải phải là $W^*$ | 3.13 |
| 7 | Hoerl–Kennard tr.64 | *"in the example of Section 3, it has been demonstrated"* | §3 **không có ví dụ nào**; chính §3a tr.58 nói ví dụ nằm ở bài [12] | 3.13 |
| 8 | Hoerl–Kennard tr.64 | *"As Barnard [12] has recently pointed out"* | tài liệu [12] là bài của chính Hoerl–Kennard; Barnard là **[2]** | 3.13 |
| 9 | ESL tr.68 | *"In Figure 3.7 the minimum occurs at df(λ) = 5.0"* | toạ độ vectơ của Hình 3.7 đặt cực tiểu ở df ≈ 7; df = 5 là lựa chọn của **quy tắc một sai số chuẩn** — đúng như tr.61 và chú thích hình | 3.15 |
| 10 | ESL Bảng 3.3 tr.63 so với Hình 3.8 tr.65 | cùng mô hình, cùng dữ liệu | cột LS của bảng khớp chuẩn hoá trên **97** mẫu; đầu mút các đường của hình khớp chuẩn hoá trên **67** mẫu huấn luyện (lệch 0,0005 so với 0,0369) | 3.17 |
| 11 | ESL Bảng 3.3 tr.63, cột Ridge | hệ số, sai số kiểm tra 0,492, sai số chuẩn 0,165 | không quy ước chuẩn hoá nào dựng lại được cả 11 ô: hệ số khớp trong 0,00202 theo quy ước 97 mẫu, sai số kiểm tra 0,492 khớp theo quy ước 67 mẫu; sai số chuẩn 0,165 không khớp quy ước nào | 3.15 |

Thêm **một chỗ bỏ ngỏ** không phải lỗi in: vòng lặp chọn $k_i$ ở Hoerl–Kennard §7 tr.65 được chạy
*"until there is a stability achieved"*, nhưng nó **chỉ có điểm dừng khi** $\lambda_i\hat\alpha_i^2\ge4\hat\sigma^2$;
dưới ngưỡng đó $k_i\to\infty$ (mục 3.13).

---

### 3.1 Tikhonov 1963, tr.501 — bài toán không chỉnh và nguyên lý chính quy hoá

Hoerl–Kennard không dẫn Tikhonov, và Tikhonov không nói về thống kê. Nhưng phép toán ở trung tâm của hai
bài là **một**: cộng vào đại lượng cần cực tiểu hoá một khoản phạt nhân với một tham số dương. Mục 3.5
chứng minh bằng số rằng ridge là một trường hợp riêng của sơ đồ sai phân tr.503.

![Tikhonov 1963 tr.501 — phương trình Fredholm loại một (1)](<paper_ridge_tikhonov1963_tr501_bai_toan_1.png>)

**Hình 1 — Tikhonov 1963, tr.501: mở đầu §1 và phương trình (1).** А. Н. Тихонов, *О решении
некорректно поставленных задач и методе регуляризации*, Докл. АН СССР 151(3):501–504, 1963,
`_papers/1963 - Tikhonov - O reshenii nekorrektno postavlennykh zadach i metode regulyarizatsii.pdf`.

**Khung chứa gì.** Câu mở đầu, phương trình (1), và đoạn nói về tính duy nhất của nghiệm. Bài báo
không có hình hay bảng nào.

$$A[x,z(s)]=\int_a^b K(x,s)\,z(s)\,ds=u(x),\qquad c\le x\le d. \tag{1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $z(s)$ | hàm **cần tìm**, xác định trên $[a,b]$ | hàm một biến |
| $u(x)$ | hàm **đo được**, xác định trên $[c,d]$ | hàm một biến |
| $K(x,s)$ | nhân (*ядро*) — mô tả phép đo biến $z$ thành $u$ | hàm hai biến |
| $A$ | toán tử tích phân, $A[x,z]=\int K(x,s)z(s)ds$ | ánh xạ hàm → hàm |

**Tác giả nói gì.** Tr.501: *"Обратные задачи математической физики часто приводят к некорректно
поставленным задачам. Типичным примером является уравнение Фредгольма первого рода"* — rồi ngay sau
(1): *"Это уравнение имеет решение не для всякой функции u (x). Очевидно, что если K (x, s) имеет
определенный порядок гладкости по x, то не существует функции z (s) ∈ L₂, удовлетворяющей уравнению
(1), если u (x) имеет меньший порядок гладкости."*

(Bài toán ngược của vật lý toán thường dẫn tới bài toán đặt không chỉnh. Ví dụ điển hình là phương
trình Fredholm loại một. Phương trình này không có nghiệm với mọi hàm $u(x)$: nếu nhân $K(x,s)$ trơn tới
một bậc nào đó theo $x$ thì không tồn tại $z\in L_2$ thoả (1) khi $u$ kém trơn hơn bậc đó.)

**Ý nghĩa.** Công thức (1) là **hồi quy tuyến tính với vô số hệ số**. Rời rạc hoá $s$ thành $p$ điểm và
$x$ thành $n$ điểm thì $K$ thành một ma trận $n\times p$, $z$ thành vector hệ số, $u$ thành vector đáp án.
Câu về độ trơn nói điều cốt lõi: phép nhân với một nhân trơn **làm mịn**, nên nó xoá các thành phần tần
số cao của $z$. Đi ngược lại — từ $u$ về $z$ — là cố khôi phục thứ đã bị xoá, và nhiễu đo nhỏ tới đâu cũng
đủ thay thế nó.

![Tikhonov 1963 tr.501 — tính không chỉnh và nguyên lý chính quy hoá](<paper_ridge_tikhonov1963_tr501_nguyen_ly.png>)

**Hình 2 — Tikhonov 1963, tr.501: vì sao bài toán ngược không chỉnh, và định nghĩa họ nghiệm chính quy
hoá.**

**Khung chứa gì.** Hai đoạn cuối trang: ví dụ $z_2=z_1+p\cos\omega s$ và định nghĩa *регуляризованное
семейство*.

**Tác giả nói gì.** *"Следует иметь в виду, что обратная задача — задача нахождения z (s) по заданной
функции u (x) — некорректна. В самом деле, функциям z₁ (s) и z₂ (s) = z₁ (s) + p cos ω s, z₁ (s), z₂ (s)
∈ C̄₁, где p — любое (как угодно большое) фиксированное число, будут соответствовать функции u₁ (x) и u₂
(x), норма уклонения которых ‖u₁ (x) − u₂ (x)‖ как угодно мала, если ω достаточно велика."*

(Bài toán ngược — tìm $z$ khi biết $u$ — là không chỉnh. Thật vậy, hai hàm $z_1$ và
$z_2=z_1+p\cos\omega s$, với $p$ lớn tuỳ ý, ứng với hai hàm $u_1$, $u_2$ mà khoảng cách
$\lVert u_1-u_2\rVert$ nhỏ tuỳ ý nếu $\omega$ đủ lớn.)

Rồi nguyên lý: *"семейство функций z^α (s), зависящее от параметра α, мы будем называть регуляризованным
семейством приближенных решений, если: 1) u_α (x) = A [x, z^α (s)] → ū (x) при α → 0; 2) функции z^α (s)
при любом α принадлежат компактному классу функций Z̄, содержащему z̄ (s)."*

(Họ hàm $z^\alpha$ phụ thuộc tham số $\alpha$ được gọi là họ nghiệm gần đúng chính quy hoá nếu: một, ảnh
$A[z^\alpha]$ tiến về $\bar u$ khi $\alpha\to0$; hai, mọi $z^\alpha$ nằm trong một lớp compact $\bar Z$ chứa
nghiệm thật.)

**Dựng lại ví dụ $p\cos\omega s$.** Bài báo không cho số. Với nhân Gauss độ rộng 0,08 trên $[0,1]$ và
$p=1000$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 19):

| $\omega$ | $\lVert u_1-u_2\rVert$ (chuẩn $L_2$ tr.501) | $\lVert u_1-u_2\rVert\ /\ \max\lvert z_1-z_2\rvert$ |
|---:|---:|---:|
| 1 | $1{,}619842\cdot10^{2}$ | $1{,}619842\cdot10^{-1}$ |
| 10 | $9{,}706931\cdot10^{1}$ | $9{,}706931\cdot10^{-2}$ |
| 30 | $1{,}436592\cdot10^{1}$ | $1{,}436592\cdot10^{-2}$ |
| 100 | $1{,}549151$ | $1{,}549151\cdot10^{-3}$ |
| 300 | $8{,}880401\cdot10^{-1}$ | $8{,}880401\cdot10^{-4}$ |
| 1 000 | $2{,}191515\cdot10^{-1}$ | $2{,}191515\cdot10^{-4}$ |
| 3 000 | $1{,}966497\cdot10^{-2}$ | $1{,}966497\cdot10^{-5}$ |

Lưới 40 000 điểm, ít nhất 83 điểm mỗi chu kỳ ở $\omega=3000$.

**Cách đọc.** Hai hàm $z$ luôn cách nhau 1 000 ở đỉnh, vậy mà vế phải của chúng gần như trùng nhau khi
$\omega$ lớn: ở $\omega=3000$ tỉ số là $1{,}966497\cdot10^{-5}$. Tốc độ giảm **không đều**: từ $\omega=30$ tới
100 chuẩn đi từ $14{,}36592$ xuống $1{,}549151$, nhưng từ 100 tới 300 chỉ xuống $0{,}8880401$. Lý do nằm ở hai đầu đoạn: bên trong đoạn, nhân Gauss trung bình hoá
$\cos\omega s$ gần như triệt tiêu, nhưng sát biên $s=0$ và $s=1$ nhân bị cắt ngang nên phần triệt tiêu không
trọn. Chỉ tính $x\in(0{,}2;\ 0{,}8)$ thì ở $\omega=100$ chuẩn còn $4{,}254418\cdot10^{-2}$, và phần sát biên
chiếm 99,92 % bình phương chuẩn; ở $\omega=1000$ là 99,96 %. Người đo chỉ thấy $u$, và có nhiễu. Nếu nhiễu lớn hơn khoảng cách ở cột giữa, dữ liệu
**không phân biệt được** $z_1$ với $z_2$ — đúng khuôn của hai cột gần trùng ở mục 1.2, chỉ ở dạng hàm.

**Điều khung KHÔNG nói.** Tikhonov không nói phải chọn lớp compact $\bar Z$ nào; đó là việc của phiếm hàm
$\Omega$ ở trang sau. Và "compact" ở đây là trong không gian hàm vô hạn chiều — một chỗ quan trọng khi
chuyển sang ridge (mục 3.5).

### 3.2 Tikhonov 1963, tr.502 — phiếm hàm làm trơn (2), định lý 1, phương trình Euler (3) và (4)

![Tikhonov 1963 tr.502 — phiếm hàm (2), N, Ω và định lý 1](<paper_ridge_tikhonov1963_tr502_phiem_ham_2.png>)

**Hình 3 — Tikhonov 1963, tr.502: công thức (2), định nghĩa $N$ và $\Omega$, định lý 1.**

$$M^\alpha[z(s),\bar u(x)]=N[z(s),\bar u(x)]+\alpha\,\Omega[z(s)], \tag{2}$$

$$N[z,\bar u]=\int_c^d\big[A[x,z(s)]-\bar u(x)\big]^2dx,\qquad
\Omega[z]=\int_a^b\big[k(s)\,z'(s)^2+p(s)\,z^2(s)\big]ds\quad(k(s)>0,\ p(s)>0).$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $M^\alpha$ | **phiếm hàm làm trơn** (*сглаживающий функционал*) — thứ cần cực tiểu hoá | hàm của hàm → số |
| $N$ | độ lệch bình phương giữa ảnh $A[z]$ và dữ liệu $\bar u$ — ứng với tổng bình phương phần dư | số $\ge0$ |
| $\Omega$ | **phiếm hàm chính quy hoá** (*регуляризующий функционал*) — ứng với khoản phạt | số $\ge0$ |
| $\alpha$ | tham số chính quy hoá — ứng với $\lambda$ của bài này | số $>0$ |
| $k(s)$, $p(s)$ | trọng số của **độ dốc** $z'^2$ và của **độ lớn** $z^2$ trong khoản phạt | hàm dương |

**Tác giả nói gì.** *"Мы будем называть Ω [z] регуляризующим и M^α сглаживающим функционалами."* Và
định lý 1: *"Для любой функции ū (x) ∈ L₂ существует единственная непрерывная, дифференцируемая функция z^α
(s), реализующая минимум сглаживающего функционала M^α [z (s), ū (x)]."*

(Ta gọi $\Omega$ là phiếm hàm chính quy hoá và $M^\alpha$ là phiếm hàm làm trơn. Định lý 1: với mọi
$\bar u\in L_2$ tồn tại **duy nhất** một hàm liên tục, khả vi $z^\alpha$ làm $M^\alpha$ nhỏ nhất.)

**Ý nghĩa.** (2) là câu **"độ lệch với dữ liệu + $\alpha$ lần độ lớn của nghiệm"** — đúng hình dạng của
ESL (3.41) bốn mươi sáu năm sau. Chỗ khác duy nhất nằm ở $\Omega$: Tikhonov phạt **cả độ dốc** $z'^2$, nên
nghiệm của ông trơn; ridge chỉ phạt độ lớn. Định lý 1 là phiên bản vô hạn chiều của chuyện ở mục 0.5:
khoản phạt biến một bài toán có thể vô số nghiệm thành bài toán có đúng một nghiệm.

![Tikhonov 1963 tr.502 — phương trình Euler (3) và bài toán biên (4)](<paper_ridge_tikhonov1963_tr502_euler_3_4.png>)

**Hình 4 — Tikhonov 1963, tr.502: phương trình Euler (3), nhân $\bar K$, và bài toán biên (4).**

$$L^\alpha[z]=\alpha\Big\{\frac{d}{ds}\Big[k\frac{dz}{ds}\Big]-pz\Big\}-\Big\{\int_a^b\bar K(s,\zeta)z(\zeta)d\zeta-\bar b(s)\Big\}=0,\quad z'(a)=z'(b)=0, \tag{3}$$

$$\bar K(s,\zeta)=\int_c^d K(\xi,s)K(\xi,\zeta)\,d\xi,\qquad \bar b(s)=\int_c^d K(\xi,s)\,\bar u(\xi)\,d\xi,$$

$$L^{\omega}[z]=\frac{d}{ds}\Big[k(s)\frac{dz}{ds}\Big]-p(s)z(s)=f(s),\qquad z'(a)=z'(b)=0. \tag{4}$$

| Ký hiệu | Nghĩa | Ứng với ridge |
|---|---|---|
| $\bar K(s,\zeta)$ | "tích" của nhân với chính nó | $X^\top X$ |
| $\bar b(s)$ | nhân áp lên dữ liệu | $X^\top y$ |
| $\frac{d}{ds}[k\,z']-pz$ | toán tử Euler của $\Omega$ | $-I$ khi $k=0$, $p=1$ |
| $z'(a)=z'(b)=0$ | điều kiện biên **tự nhiên**: sinh ra từ phép lấy biến phân, không phải từ giả thiết thêm | — |

**Tác giả nói gì.** *"С помощью функции Грина для краевой задачи (4), определяемой оператором Эйлера для
регуляризующего функционала, уравнение (3) может быть преобразовано в уравнение Фредгольма 2-го рода,
для которого при α > 0 однородное уравнение имеет лишь тривиальное решение: отсюда и следует существование
z^α (s)."*

(Dùng hàm Green của bài toán biên (4), phương trình (3) chuyển được thành phương trình Fredholm **loại
hai**; với $\alpha>0$ phương trình thuần nhất chỉ có nghiệm tầm thường, nên nghiệm $z^\alpha$ tồn tại.)

**Ý nghĩa.** (3) là **phương trình chuẩn** của bài toán có phạt, viết cho hàm. Bỏ số hạng $\alpha$ đi thì
còn $\int\bar K z=\bar b$ — phương trình chuẩn của bình phương tối thiểu, vẫn là loại **một** và vẫn
không chỉnh. Số hạng $\alpha$ biến nó thành loại **hai**, và phương trình loại hai thì ổn định. Viết cho
ma trận, câu đó là $(X^\top X+\lambda I)\beta=X^\top y$: cộng một bội của ma trận đơn vị làm mọi trị riêng
xa khỏi 0 ít nhất $\lambda$.

**Điều khung KHÔNG nói.** Ký hiệu nhân $\bar K$ in một dấu phẩy thừa giữa hai thừa số
(*"K (ξ, s), K (ξ, ζ)"*); đó là phép nhân. Chỉ số trên của toán tử trong (4) in là $\omega$ — không liên
quan tới tần số $\omega$ ở tr.501.

### 3.3 Tikhonov 1963, tr.502–503 — định lý 2 (một bất đẳng thức in sai) và định lý 3 (chọn $\alpha$ theo $\delta^2$)

![Tikhonov 1963 tr.502 — định lý 2, chứng minh, và mở đầu định lý 3](<paper_ridge_tikhonov1963_tr502_dinh_ly_2_3.png>)

**Hình 5 — Tikhonov 1963, tr.502: định lý 2 kèm chứng minh, và nửa đầu định lý 3.**

**Tác giả nói gì.** Định lý 2: *"Если z̄ (s) ∈ C̄₁, ū (x) = A [x, z̄ (s)], то для любого ε > 0 существует
такое α (ε, z̄), что | z^α (s) − z̄ (s) | < ε для всех α < α₀ (ε, z̄)."* Chứng minh: *"В самом деле, M^α [z^α
(s); ū (x)] ≤ N [z̄, ū] + αΩ [z̄] = αC² (C² = Ω [z̄]), откуда следует, что z^α (s) удовлетворяет неравенству
1) Ω [z (s)] ≤ C², определяющему компактный класс функции Z̄, а также 2) ‖ u^α (x) − ū (x) ‖ ≤ αC → 0 при α →
0. Отсюда и следует теорема 2."*

(Định lý 2: nếu dữ liệu **không nhiễu**, $\bar u=A[\bar z]$, thì với mọi $\varepsilon>0$ có $\alpha_0$ sao
cho $\lvert z^\alpha-\bar z\rvert<\varepsilon$ với mọi $\alpha<\alpha_0$. Chứng minh: vì $z^\alpha$ là cực
tiểu, $M^\alpha[z^\alpha]\le M^\alpha[\bar z]=\alpha C^2$; suy ra $\Omega[z^\alpha]\le C^2$ và
$\lVert u^\alpha-\bar u\rVert\le\alpha C\to0$.)

**Chỗ in sai — bước 2) của chứng minh.** Từ $M^\alpha[z^\alpha]=N[z^\alpha]+\alpha\Omega[z^\alpha]\le\alpha
C^2$ và $\Omega\ge0$ ta chỉ có $N[z^\alpha]=\lVert u^\alpha-\bar u\rVert^2\le\alpha C^2$, tức

$$\lVert u^\alpha-\bar u\rVert\le C\sqrt\alpha,\quad\text{không phải}\quad\le\alpha C.$$

Kiểm trên bài toán rời rạc 80 ẩn, nhân Gauss, $\bar z=\cos\pi s+\tfrac12\cos3\pi s$, $k=p=1$, $C^2=\Omega[\bar
z]=16{,}649637$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 3):

| $\alpha$ | $\lVert u^\alpha-\bar u\rVert$ | $\alpha C$ (in) | $C\sqrt\alpha$ (đúng) | $\le\alpha C$? |
|---:|---:|---:|---:|:---:|
| $10^{-1}$ | $1{,}162817\cdot10^{-1}$ | $4{,}080397\cdot10^{-1}$ | $1{,}290335$ | có |
| $10^{-2}$ | $9{,}762613\cdot10^{-2}$ | $4{,}080397\cdot10^{-2}$ | $4{,}080397\cdot10^{-1}$ | **không** |
| $10^{-4}$ | $1{,}701015\cdot10^{-2}$ | $4{,}080397\cdot10^{-4}$ | $4{,}080397\cdot10^{-2}$ | **không** |
| $10^{-6}$ | $2{,}803856\cdot10^{-4}$ | $4{,}080397\cdot10^{-6}$ | $4{,}080397\cdot10^{-3}$ | **không** |
| $10^{-10}$ | $6{,}027514\cdot10^{-8}$ | $4{,}080397\cdot10^{-10}$ | $4{,}080397\cdot10^{-5}$ | **không** |

Trên 10 giá trị $\alpha$ từ $10^{-1}$ tới $10^{-10}$, bất đẳng thức in **sai ở 9**, cận $C\sqrt\alpha$ đúng
ở cả 10.

**Phép thử không cần bài toán cụ thể.** Nhân $k$ và $p$ với $c=10^4$, đồng thời chia $\alpha$ cho $c$:
phiếm hàm $M^\alpha$ không đổi, nên nghiệm và $\lVert u^\alpha-\bar u\rVert$ không đổi. Nhưng $C$ nhân
$\sqrt c$, nên $\alpha C$ **chia** $\sqrt c$, còn $C\sqrt\alpha$ đứng yên. Sau phép đổi thang đó, bất đẳng
thức in sai ở **10/10** giá trị $\alpha$. Một bất đẳng thức đổi đúng thành sai chỉ vì đổi đơn vị của
$\Omega$ thì không thể là hệ quả của một bất đẳng thức không phụ thuộc đơn vị.

**Ý nghĩa.** Kết luận của định lý 2 — ảnh tiến về dữ liệu khi $\alpha\to0$ — vẫn đúng: $C\sqrt\alpha$ cũng
về 0. Chỗ sai chỉ đổi **tốc độ**, từ bậc một thành bậc một nửa. Nhưng tốc độ đó quan trọng cho định lý
3: vì độ lệch tối thiểu cỡ $\sqrt\alpha$, muốn nó cân với nhiễu cỡ $\delta$ thì phải lấy $\alpha$ cỡ
$\delta^2$ — đúng điều kiện $\gamma_1\le\delta^2/\alpha\le\gamma_2$ mà định lý 3 dùng. Bản in có một bậc
sai ở bước chứng minh, và điều kiện đúng ở phát biểu.

![Tikhonov 1963 tr.503 — kết định lý 3, câu về ví dụ, lưới sai phân](<paper_ridge_tikhonov1963_tr503_dinh_ly_3_luoi.png>)

**Hình 6 — Tikhonov 1963, tr.503: nửa sau định lý 3, câu *"нетрудно убедиться на примерах"*, và lưới
sai phân.**

**Tác giả nói gì.** Định lý 3 (tr.502–503): *"Если z̄ ∈ C̄₁, то для любого ε > 0 и любых вспомогательных
чисел 0 < γ₁ ≤ γ₂ существует такое δ₀ (ε, γ₁, γ₂, z̄), что если: 1) норма уклонения функции ũ_δ (x) от
функции ū (x) меньше δ … 2) ᾱ = ᾱ (δ) удовлетворяет условиям γ₁ ≤ δ²/α ≤ γ₂ (или δ²/γ₂ < α < δ²/γ₁), то
z̃_δ^ᾱ (s), реализующая минимум сглаживающего функционала M^ᾱ [z, ũ_δ (x)], принадлежит ε-окрестности
функции z̄ (s)"*. Và ngay sau: *"Нетрудно убедиться на примерах, что функция z̃_δ^α (s), соответствующая
фиксированной функции ũ_δ (s) при малом δ, при α → 0 может выходить из ε-окрестности z̄ (s)."*

(Định lý 3: nếu dữ liệu nhiễu lệch khỏi dữ liệu thật dưới $\delta$, và $\alpha$ được chọn sao cho
$\delta^2/\alpha$ nằm giữa hai hằng số dương, thì nghiệm chính quy hoá nằm trong $\varepsilon$-lân cận của
nghiệm thật khi $\delta$ đủ nhỏ. Và: *không khó thấy qua ví dụ* rằng với **một** dữ liệu nhiễu cố định,
cho $\alpha\to0$ thì nghiệm có thể **rời khỏi** lân cận đó.)

$$\gamma_1\le\frac{\delta^2}{\alpha}\le\gamma_2.$$

| Ký hiệu | Nghĩa |
|---|---|
| $\delta$ | mức nhiễu của dữ liệu, $\lVert\tilde u_\delta-\bar u\rVert<\delta$ |
| $\gamma_1,\gamma_2$ | hai hằng số dương tuỳ chọn — **không** phải $\gamma_1,\gamma_2$ của Hoerl–Kennard |
| $\bar\alpha(\delta)$ | tham số chính quy hoá được chọn **theo** mức nhiễu |

**Ý nghĩa.** Đây là câu đầu tiên nói **chọn tham số chính quy hoá thế nào**: tỉ lệ với **bình phương**
mức nhiễu. Nhiễu nhỏ đi 10 lần thì $\alpha$ nhỏ đi 100 lần. Cho $\alpha\to0$ mà giữ nhiễu cố định là quay
về bài toán không chỉnh. Cách nhìn Bayes của Hoerl–Kennard §6 cho cùng một tỉ lệ bằng một lập luận
khác hẳn: mức phạt tối ưu là $\sigma^2/\tau^2$, cũng tỉ lệ với bình phương mức nhiễu (mục 3.13).

**Dựng lại "ví dụ" mà bài báo nói là dễ thấy.** Cùng bài toán rời rạc, nhiễu có chuẩn đúng bằng
$\delta$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 4).

(a) Chọn $\alpha=\delta^2$ ($\gamma_1=\gamma_2=1$) và cho $\delta\to0$:

| $\delta$ | $\alpha$ | $\max\lvert z-\bar z\rvert$ |
|---:|---:|---:|
| $10^{-1}$ | $10^{-2}$ | 1,326479 |
| $10^{-2}$ | $10^{-4}$ | 0,284306 |
| $10^{-3}$ | $10^{-6}$ | 0,022421 |
| $10^{-4}$ | $10^{-8}$ | 0,008231 |
| $10^{-5}$ | $10^{-10}$ | 0,002315 |

Sai số đều giảm đơn điệu — định lý 3 đúng trên ví dụ này.

(b) Giữ $\delta=10^{-3}$ và cho $\alpha\to0$:

| $\alpha$ | $10^{-2}$ | $10^{-4}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ | $10^{-14}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\max\lvert z-\bar z\rvert$ | 1,303721 | 0,286015 | **0,022421** | 0,067798 | 0,231674 | 1,549022 | 8,546764 |

Sai số chạm đáy ở $\alpha=10^{-6}=\delta^2$ rồi **tăng trở lại**: ở $\alpha=10^{-14}$ nó gấp 381,2 lần
mức đáy. Nếu chọn $\varepsilon=0{,}1$ thì nghiệm rời khỏi $\varepsilon$-lân cận từ $\alpha=10^{-9}$ trở
xuống. Đây đúng là câu *"при α → 0 может выходить из ε-окрестности"*, và nó cũng là **đường cong chữ U
của đánh đổi chệch–phương sai** mà Hoerl–Kennard vẽ ở Hình 1 bảy năm sau, chỉ đọc theo chiều ngược.

**Điều khung KHÔNG nói.** Tên hằng số trong phát biểu định lý 2 đổi giữa câu, từ $\alpha(\varepsilon,\bar
z)$ sang $\alpha_0(\varepsilon,\bar z)$. Và dạng tương đương trong ngoặc của định lý 3 dùng bất đẳng thức
**ngặt** $\delta^2/\gamma_2<\alpha<\delta^2/\gamma_1$ trong khi dạng chính dùng dấu $\le$; định lý 2′ ở tr.503
thì dùng $\le$ ở cả hai. Hai chỗ này không đổi nội dung.

### 3.4 Tikhonov 1963, tr.503 — sơ đồ sai phân: phiếm hàm $\hat M_h^\alpha$, hệ (3′), định lý 1′ và 2′

**Tác giả nói gì về lưới.** (Hình 6, nửa dưới) *"3. Перейдем к приближенным методам решения уравнения (1).
Рассмотрим метод конечных разностей. Возьмем сетку на (a, b): s_j = jh − 0,5 h (j = 1, . . ., n) и на (c,
d): x_i = ih₁ − 0,5 h₁ (i = 1, . . ., m), где h = 1/n (a − b) и h₁ = 1/m (c − d)."*

(Chuyển sang phương pháp gần đúng: sai phân hữu hạn. Lưới trên $(a,b)$ là $s_j=jh-0{,}5h$, trên $(c,d)$ là
$x_i=ih_1-0{,}5h_1$, với $h=\tfrac1n(a-b)$ và $h_1=\tfrac1m(c-d)$.)

**Chỗ in sai — dấu của bước lưới.** Vì $a<b$ nên $\tfrac1n(a-b)<0$. Lấy đúng công thức in
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 1):

| $(a,b)$, $n$ | $h$ in | lưới in $s_1..s_n$ | lưới đúng |
|---|---:|---|---|
| $(0,1)$, 10 | $-0{,}1$ | $-0{,}05,\ -0{,}15,\ \dots,\ -0{,}95$ | $0{,}05,\ 0{,}15,\ \dots,\ 0{,}95$ |
| $(1,3)$, 4 | $-0{,}5$ | $-0{,}25,\ -0{,}75,\ -1{,}25,\ -1{,}75$ | $1{,}25,\ 1{,}75,\ 2{,}25,\ 2{,}75$ |

Lưới in nằm **ngoài** đoạn $(a,b)$. Nó chỉ đúng khi $h=\tfrac1n(b-a)$ **và** $a=0$; với $a\ne0$ còn phải cộng
thêm $a$: $s_j=a+jh-0{,}5h$ — điểm giữa của ô thứ $j$.

![Tikhonov 1963 tr.503 — phiếm hàm sai phân, định lý 1′, hệ (3′)](<paper_ridge_tikhonov1963_tr503_he_3phay.png>)

**Hình 7 — Tikhonov 1963, tr.503: phiếm hàm làm trơn sai phân $\hat M_h^\alpha$, định lý 1′, hệ phương trình
(3′).**

$$\hat M_h^\alpha[\hat z,\hat u]=\sum_{i=1}^m\Big\{\sum_{j=1}^nK_{ij}\hat z_jh-\hat u_i\Big\}^2h_1+\alpha\sum_{j=1}^n\Big\{k_j(\hat z_{j+1}-\hat z_j)^2\frac1h+p_j\hat z_j^2h\Big\},$$

$$\hat L^\alpha[\hat z]=\alpha\Big\{\frac1{h^2}\big[k_j(\hat z_{j+1}-\hat z_j)-k_{j-1}(\hat z_j-\hat z_{j-1})\big]-p_j\hat z_j\Big\}-\Big\{\sum_{l=1}^n\bar K_{jl}\hat z_lh-\hat b_j\Big\}=0,\quad \hat z_0=\hat z_1,\ \hat z_{n+1}=\hat z_n, \tag{3′}$$

$$\bar K_{jl}=\sum_{i=1}^mK_{ij}K_{il}h_1,\qquad \hat b_j=\sum_{i=1}^mK_{ij}\hat u_ih_1.$$

| Ký hiệu | Nghĩa | Ứng với ridge |
|---|---|---|
| $K_{ij}$ | nhân tại $(x_i,s_j)$ | phần tử của $X$ (nhân thêm $h\sqrt{h_1}$) |
| $\hat z_j$, $\hat u_i$ | nghiệm và dữ liệu trên lưới | $\beta_j$, $y_i$ (nhân thêm $\sqrt{h_1}$) |
| $h$, $h_1$ | trọng số cầu phương (bước lưới) | — |
| $k_j$, $p_j$ | trọng số rời rạc của độ dốc và độ lớn | $k_j=0$, $p_j=1$ |
| $\hat z_0=\hat z_1$, $\hat z_{n+1}=\hat z_n$ | biên tự nhiên $z'=0$ viết cho lưới | — |

**Tác giả nói gì.** Định lý 1′: *"Для любой сетчатой функции û и α > 0 существует сетчатая функция ẑ^α,
реализующая минимум сглаживающего функционала M̂_h^α [ẑ, û]."* Và về cách lấy $k_j$, $p_j$: *"В частности,
например, k_j = k (s_j + 0,5 h), p_j = p (s_j)."*

(Định lý 1′: với mọi dữ liệu lưới và mọi $\alpha>0$ tồn tại nghiệm lưới cực tiểu hoá $\hat M_h^\alpha$. Có
thể lấy $k_j=k(s_j+0{,}5h)$, $p_j=p(s_j)$.)

**Kiểm (3′) có đúng là điều kiện cực tiểu không.** Bài báo in (3′) mà không in bước dẫn. Lấy đạo hàm
$\hat M_h^\alpha$ theo $\hat z_j$ rồi chia cho $-2h$ thì ra đúng (3′). Kiểm bằng số
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 2), lập (3′) **theo đúng từng số hạng in**:

| $\alpha$ | nghiệm (3′) so với cực tiểu tính trực tiếp | biên độ nghiệm |
|---:|---:|---:|
| $10^{-2}$ | $1{,}624\cdot10^{-14}$ | 0,1954 |
| $10^{-5}$ | $4{,}663\cdot10^{-15}$ | 1,4341 |
| $10^{-8}$ | $2{,}482\cdot10^{-13}$ | 1,4973 |

Gradient số của $\hat M_h^\alpha$ tại nghiệm của (3′), $\alpha=10^{-4}$: chuẩn vô cùng $2{,}168\cdot10^{-13}$.
Hệ (3′) đúng.

**Ý nghĩa.** (3′) là **phương trình chuẩn của một bài hồi quy có phạt**, viết ra bằng tay cho máy tính
năm 1963. Ma trận $\bar K$ chính là $X^\top X$, $\hat b$ là $X^\top y$, và khối $\alpha\{\dots\}$ là $-\alpha$
nhân một ma trận ba đường chéo (sai phân bậc hai cộng đường chéo $p$). Ridge thay khối đó bằng $-\lambda I$.

![Tikhonov 1963 tr.503 — định lý 2′](<paper_ridge_tikhonov1963_tr503_dinh_ly_2phay.png>)

**Hình 8 — Tikhonov 1963, tr.503: định lý 2′ (bản sai phân của định lý 3).** Phần kết của định lý nằm ở
đầu tr.504, trong Hình 9.

**Tác giả nói gì.** *"Если z (s) ∈ C̄₁, то для любого ε > 0 и любых вспомогательных чисел 0 < γ₁ ≤ γ₂
существуют такие δ₀ (ε, γ₁, γ₂, z̄) и h₀ (ε, γ₁, γ₂, z̄), что если: 1) норма уклонения функции ũ_δ (x) от ū
(x) меньше δ … 2) ᾱ = ᾱ (δ) удовлетворяет условиям γ₁ ≤ δ²/α ≤ γ₂ (или δ²/γ₂ ≤ α ≤ δ²/γ₁)"* — và tr.504:
*"то z_δ^ᾱ (s), реализующая минимум разностного сглаживающего функционала M̂_h^α [ẑ, ũ_δ], принадлежит
ε-окрестности функции z̄ (s) при δ ≤ δ₀ (ε, γ₁, γ₂, z̄), h < h₀ (ε, γ₁, γ₂, z̄)."*

(Như định lý 3, thêm một điều kiện: bước lưới $h$ đủ nhỏ.)

**Ý nghĩa.** Định lý 2′ nói **hai** giới hạn phải đi cùng nhau: nhiễu nhỏ **và** lưới mịn. Lưới mịn một
mình không cứu được — mục 1.2 cho thấy lưới 80 ẩn không có $\alpha$ thì sai gần mười bậc.

### 3.5 Tikhonov 1963, tr.504 — dạng tích phân (một dấu in sai), lời kết, và ridge là trường hợp $k=0$

![Tikhonov 1963 tr.504 — kết định lý 2′, phương trình (1′) và dạng tích phân của (3)](<paper_ridge_tikhonov1963_tr504_dang_tich_phan.png>)

**Hình 9 — Tikhonov 1963, tr.504: phần kết định lý 2′, nhận xét về máy tính, phương trình (1′) và dạng
tích phân của (3).**

$$A[x,z(s)]=u(x), \tag{1′}$$

$$\alpha(x,s)=A[x,\eta_s(\zeta)],\qquad \eta_s(\zeta)=\begin{cases}1,&\zeta\le s\\0,&\zeta>s,\end{cases}$$

**Ý nghĩa của hàm $\alpha(x,s)$.** Nó là ảnh qua toán tử của một **hàm bậc thang** tắt ở $s$ — với toán tử
tích phân thì $\alpha(x,s)=\int_a^sK(x,\zeta)d\zeta$. Đổi nhân $K$ thành nguyên hàm của nó theo biến thứ hai
là cách viết (3) mà **không cần đạo hàm của nhân**, nên cùng một công thức dùng được cho mọi toán tử bị chặn.

Với ký hiệu đó, dạng in:

$$\alpha\Big\{k(s)z'(s)+\int_0^sp(\zeta)z(\zeta)d\zeta\Big\}-\Big\{\int_c^dA[x,z(\zeta)]\,\alpha(x,s)dx-\int_c^d\alpha(x,s)\bar u(x)dx\Big\}=0,\quad z'(a)=z'(b)=0.$$

| Ký hiệu | Nghĩa |
|---|---|
| $A$ trong (1′) | một toán tử **bị chặn** bất kỳ — không nhất thiết là tích phân |
| $\eta_s$ | hàm bậc thang: 1 tới $s$, 0 sau $s$ |
| $\alpha(x,s)$ | ảnh của hàm bậc thang — **một hàm**, trùng tên với tham số $\alpha$ |

**Tác giả nói gì.** *"Уравнение (3′) представляет алгоритм для решения уравнения (1), дающий весьма
эффективные результаты с помощью электронных цифровых машин."* và *"Изложенный выше метод применим к
уравнениям типа A [x, z (s)] = u (x), (1′) где A [x, z (s)] — ограниченный оператор."*

(Hệ (3′) là thuật toán giải (1), cho kết quả rất hiệu quả trên máy tính điện tử số. Phương pháp áp dụng
được cho mọi phương trình dạng (1′) với toán tử bị chặn.)

**Chỗ in sai — dấu và cận của tích phân $p$.** Lấy tích phân (3) từ $a$ tới $s$: số hạng
$\frac{d}{ds}[kz']$ cho $k(s)z'(s)$ (vì $z'(a)=0$), số hạng $-pz$ cho $-\int_a^sp\,z$. Vậy dạng đúng là
$\alpha\{k z'-\int_a^s p\,z\,d\zeta\}$ — **dấu trừ**, **cận dưới $a$**. Kiểm trên bài toán rời rạc bằng
tổng dồn chính xác của (3′), $\alpha=10^{-4}$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 5):

| Dạng | phần dư lớn nhất |
|---|---:|
| dấu **cộng** như in | $5{,}225\cdot10^{-5}$ |
| dấu **trừ** | $9{,}758\cdot10^{-18}$ |
| biên độ số hạng $\alpha kz'$, để so | $4{,}736\cdot10^{-4}$ |

Phần dư của dạng in trùng với $2\alpha\int p\,z$ tới $9{,}717\cdot10^{-18}$ — tức nó sai **đúng** hai lần số
hạng $p$. Cận dưới $0$ chỉ trùng $a$ khi đoạn bắt đầu ở 0; với $a\ne0$, $z$ không xác định trên $(0,a)$.

![Tikhonov 1963 tr.504 — về cách chọn Ω, lời kết, tài liệu dẫn](<paper_ridge_tikhonov1963_tr504_ket.png>)

**Hình 10 — Tikhonov 1963, tr.504: đoạn về cách chọn $\Omega$, lời kết, ngày nhận bài và ba tài liệu
dẫn.**

**Tác giả nói gì.** *"Регуляризующий функционал Ω [z] может выбираться как квадратический функционал (не
обязательно дифференциальный) так, чтобы условие Ω (z) ≤ C определяло компактное множество и чтобы
оператор Эйлера для Ω (z) имел вполне непрерывный обратный оператор."* Và câu cuối: *"Сглаживающие
функционалы представляют удобный аппарат для решения уравнений второго рода на изолированной точке
спектра, а также при решении нелинейных задач."*

(Phiếm hàm chính quy hoá có thể chọn là một phiếm hàm bậc hai bất kỳ — **không nhất thiết có đạo hàm** —
miễn điều kiện $\Omega\le C$ xác định một tập compact và toán tử Euler của $\Omega$ có nghịch đảo hoàn toàn
liên tục. Phiếm hàm làm trơn cũng tiện cho phương trình loại hai tại điểm cô lập của phổ và cho bài toán
phi tuyến.)

**Ý nghĩa — và chỗ ridge nằm.** Câu *"не обязательно дифференциальный"* mở cửa cho $\Omega[z]=\int z^2$,
tức $k=0$, $p=1$ — **chính là ridge**. Nhưng câu đó kèm điều kiện: $\Omega\le C$ phải là tập compact. Trong
không gian hàm vô hạn chiều, quả cầu $\int z^2\le C$ **không** compact; trong không gian hữu hạn chiều thì
có. Nên ridge nằm gọn trong phạm vi bài báo **sau khi rời rạc hoá**, và nằm ngoài phạm vi các định lý 1–3
viết cho hàm.

**Kiểm bằng số.** Hệ (3′) với $k_j=0$, $p_j=1$, đặt $X=K\,h\sqrt{h_1}$, $y=\hat u\sqrt{h_1}$,
$\lambda=\alpha h$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 6):

| So sánh | lệch lớn nhất |
|---|---:|
| (3′) với $k=0$ so với $(X^\top X+\lambda I)^{-1}X^\top y$ | $2{,}619\cdot10^{-13}$ |
| so với `sklearn.linear_model.Ridge(alpha=λ, fit_intercept=False)` | $1{,}673\cdot10^{-13}$ |

Và cái giá của việc bỏ số hạng độ dốc, cùng nhiễu $\delta=10^{-3}$, cùng lưới $\alpha$:

| | $\alpha$ tốt nhất trên lưới | sai số đều nhỏ nhất |
|---|---:|---:|
| Tikhonov, $k=1$ | $10^{-6}$ | 0,022421 |
| ridge, $k=0$ | $10^{-5}$ | 0,180816 |

Ridge kém **8,0647 lần** trên bài toán này — vì nghiệm thật trơn, và chỉ Tikhonov nói điều đó cho mô hình.
Đây không phải lỗi của ridge: hồi quy thông thường không có "độ trơn" giữa hệ số cột 3 và cột 4, nên
không có gì để phạt độ dốc. Nhưng nó cho biết **khi các hệ số có thứ tự tự nhiên** (theo thời gian, theo
bước sóng, theo toạ độ), ridge thuần là lựa chọn yếu; biến thể $\lVert L\beta\rVert^2$ ở mục 9 là bản
Tikhonov đầy đủ.

### 3.6 Hoerl & Kennard 1970, tr.55 — tóm tắt và giới thiệu

![Hoerl–Kennard 1970 tr.55 — tóm tắt](<paper_ridge_hoerl1970_tr55_tom_tat.png>)

**Hình 11 — Hoerl & Kennard 1970, tr.55: tên bài, tác giả, tóm tắt.** A. E. Hoerl, R. W. Kennard,
*Ridge Regression: Biased Estimation for Nonorthogonal Problems*, Technometrics 12(1):55–67, 1970,
`_papers/1970 - Hoerl, Kennard - Ridge Regression Biased Estimation for Nonorthogonal Problems.pdf`
(bản quét JSTOR, 13 trang, không có lớp chữ).

**Khung chứa gì.** Tóm tắt bốn câu.

**Tác giả nói gì.** *"In multiple regression it is shown that parameter estimates based on minimum
residual sum of squares have a high probability of being unsatisfactory, if not incorrect, if the
prediction vectors are not orthogonal. Proposed is an estimation procedure based on adding small
positive quantities to the diagonal of X'X. Introduced is the ridge trace, a method for showing in two
dimensions the effects of nonorthogonality. It is then shown how to augment X'X to obtain biased
estimates with smaller mean square error."*

(Trong hồi quy bội, chứng minh rằng ước lượng dựa trên tổng bình phương phần dư nhỏ nhất có xác suất cao
là không đạt, thậm chí sai, nếu các vector dự báo không trực giao. Đề xuất một thủ tục ước lượng dựa trên
việc cộng những lượng dương nhỏ vào đường chéo của $X'X$. Giới thiệu vết ridge, cách thể hiện trên hai
chiều ảnh hưởng của sự không trực giao. Rồi chỉ ra cách tăng $X'X$ để được ước lượng có chệch nhưng sai
số bình phương trung bình nhỏ hơn.)

**Cách đọc.** Bốn câu, bốn đóng góp, theo đúng thứ tự các mục 1, 2, 3, 4 của bài. Chữ đáng để ý là
*"high probability"*: bài báo không nói bình phương tối thiểu **sai**, mà nói nó **có xác suất cao** nằm
xa giá trị thật. Đó là một phát biểu về phân phối của ước lượng, và mục 1 đo nó bằng kỳ vọng và phương
sai của khoảng cách.

![Hoerl–Kennard 1970 tr.55 — §0, giới thiệu](<paper_ridge_hoerl1970_tr55_gioi_thieu.png>)

**Hình 12 — Hoerl & Kennard 1970, tr.55: §0, *Introduction*.**

**Tác giả nói gì.** *"The usual estimation procedure for the unknown β is Gauss-Markov—linear functions of
Y = {yv} that are unbiased and have minimum variance. This estimation procedure is a good one if X'X, when
in the form of a correlation matrix, is nearly a unit matrix."* Và câu kết mục: *"Estimation based on the
matrix [X'X + kIp], k ≥ 0 rather than on X'X has been found to be a procedure that can be used to help
circumvent many of the difficulties associated with the usual least squares estimates."*

(Thủ tục thường dùng là Gauss–Markov — hàm tuyến tính của $Y$, không chệch và có phương sai nhỏ nhất. Nó
tốt nếu $X'X$ ở dạng ma trận tương quan gần là ma trận đơn vị. Ước lượng dựa trên $[X'X+kI_p]$, $k\ge0$,
thay cho $X'X$ đã được thấy là giúp tránh nhiều khó khăn của bình phương tối thiểu.)

**Cách đọc.** Hai điều kiện ẩn đứng ở đầu bài: $X'X$ **ở dạng tương quan** — tức mọi cột đã chuẩn hoá —
và tiêu chuẩn so sánh là **gần ma trận đơn vị**. Mọi con số của bài báo, kể cả Hình 1, giả định dạng tương
quan (tr.56: *"unless specified otherwise, the model is formulated to give an X'X in correlation form"*).
Đây là lý do bước 2 ở mục 0.4 là bắt buộc.

**Điều khung KHÔNG nói.** Ghi chú cuối trang cho biết bài nhận tháng 8 năm 1968 và sửa tháng 6 năm 1969.
Bài không có ví dụ số với dữ liệu thật: tr.58 dẫn hai bài toán ví dụ sang bài đi kèm [12], *Ridge
Regression. Applications to nonorthogonal problems*, cũng Technometrics 12, không có trong repo.

### 3.7 Hoerl & Kennard 1970, tr.56 — §1: tính chất của bình phương tối thiểu, (1.1)–(1.9)

![Hoerl–Kennard 1970 tr.56 — (1.1) tới (1.6)](<paper_ridge_hoerl1970_tr56_1_1_den_1_6.png>)

**Hình 13 — Hoerl & Kennard 1970, tr.56: §1, công thức (1.1)–(1.6).**

$$\hat\beta=(X'X)^{-1}X'Y \tag{1.1}$$
$$\phi(\hat\beta)=(Y-X\hat\beta)'(Y-X\hat\beta) \tag{1.2}$$
$$\operatorname{VAR}(\hat\beta)=\sigma^2(X'X)^{-1} \tag{1.3}$$
$$L_1^2=(\hat\beta-\beta)'(\hat\beta-\beta) \tag{1.4}$$
$$E[L_1^2]=\sigma^2\operatorname{Trace}(X'X)^{-1} \tag{1.5}$$
$$E[\hat\beta'\hat\beta]=\beta'\beta+\sigma^2\operatorname{Trace}(X'X)^{-1} \tag{1.5a}$$
$$\operatorname{VAR}[L_1^2]=2\sigma^4\operatorname{Trace}(X'X)^{-2} \tag{1.6}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $X'$ | chuyển vị của $X$ — bài báo viết dấu phẩy trên thay cho $\top$ | $p\times n$ |
| $\phi$ | tổng bình phương phần dư | số |
| $L_1$ | **khoảng cách** Euclid từ $\hat\beta$ tới $\beta$ — không phải chuẩn một | số $\ge0$ |
| $\operatorname{Trace}$ | vết | số |

**Tác giả nói gì.** *"To demonstrate the effects of this condition on the estimation of β, consider two
properties of β̂ — its variance-covariance matrix and its distance from its expected value."* (Để thấy
điều kiện này ảnh hưởng thế nào tới việc ước lượng $\beta$, xét hai tính chất của $\hat\beta$: ma trận hiệp
phương sai và khoảng cách tới kỳ vọng của nó.)

**Ý nghĩa.** (1.5) và (1.5a) là cùng một câu nói hai cách, và câu thứ hai đáng sợ hơn: **vector ước lượng
dài hơn vector thật một lượng cố định trung bình**, bằng tổng phương sai. Không có cách nào thoát bằng
thêm quan sát cùng thiết kế — chỉ có cách làm $X'X$ tốt hơn hoặc thay đổi ước lượng. (1.6) cho biết khoảng
cách đó không những lớn mà còn **dao động mạnh**, nên một lần chạy có thể rơi rất xa trung bình.

![Hoerl–Kennard 1970 tr.56 — (1.7) tới (1.9)](<paper_ridge_hoerl1970_tr56_1_7_den_1_9.png>)

**Hình 14 — Hoerl & Kennard 1970, tr.56: công thức (1.7)–(1.9) và đoạn kết mục 1.**

$$\lambda_{\max}=\lambda_1\ge\lambda_2\ge\dots\ge\lambda_p=\lambda_{\min}>0 \tag{1.7}$$
$$E[L_1^2]=\sigma^2\sum_{i=1}^p(1/\lambda_i) \tag{1.8}$$
$$\operatorname{VAR}[L_1^2]=2\sigma^4\sum(1/\lambda_i)^2 \tag{1.9}$$

| Ký hiệu | Nghĩa |
|---|---|
| $\lambda_i$ | **trị riêng** của $X'X$ — không phải mức phạt (mục 0.3, chỗ trùng tên số 1) |

**Tác giả nói gì.** *"Lower bounds for the average and variance are σ²/λmin and 2σ⁴/λ²min, respectively.
Hence, if the shape of the factor space is such that reasonable data collection results in an X'X with one
or more small eigenvalues, the distance from β̂ to β will tend to be large."* Và câu kết: *"The least
squares estimate (1.1) suffers from the deficiency of mathematical optimization techniques that give point
estimates; the estimation procedure does not have built into it a method for portraying the sensitivity of
the solution (1.1) to the optimization criterion (1.2)."*

(Cận dưới của trung bình và phương sai lần lượt là $\sigma^2/\lambda_{\min}$ và $2\sigma^4/\lambda_{\min}^2$. Nếu
hình dạng không gian nhân tố khiến $X'X$ có một hay vài trị riêng nhỏ, khoảng cách từ $\hat\beta$ tới
$\beta$ sẽ có xu hướng lớn. Bình phương tối thiểu mắc cái tật của các kỹ thuật tối ưu cho ra một điểm: nó
không có sẵn cách thể hiện độ nhạy của nghiệm với tiêu chuẩn tối ưu.)

**Kiểm bằng số.** Thiết kế mục 1.2, 8 cột dạng tương quan, $\sigma=1$, 40 000 lần rút nhiễu
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7):

| Công thức | Giá trị theo công thức | Monte Carlo |
|---|---:|---:|
| (1.8) $E[L_1^2]$ | 416,8554 | 416,3724 |
| (1.9) $\operatorname{VAR}[L_1^2]$ | 67 088,9817 | 66 502,4342 (tỉ số 0,9913) |
| (1.5a) $E[\hat\beta'\hat\beta]$ | 429,8554 | 429,9157 |
| cận dưới $\sigma^2/\lambda_{\min}$ | 123,8741 | — |
| cận dưới $2\sigma^4/\lambda^2_{\min}$ | 30 689,5632 | — |

**Ý nghĩa.** (1.8) chuyển câu hỏi "bình phương tối thiểu có tốt không" thành câu hỏi về **phổ trị riêng**:
riêng trị riêng nhỏ nhất $0{,}008073$ đóng góp 123,8741 trong tổng 416,8554, tức 29,72 %; sáu trị riêng nhỏ
cộng lại chiếm 99,88 %, còn hai trị riêng lớn — hai hướng dữ liệu thật sự trải ra — gần như không đóng góp
gì. Câu kết của mục nói điều mà ridge sẽ sửa: bình phương tối thiểu cho một **điểm**, không cho biết điểm
đó nhạy tới đâu. Vết ridge ở mục 3 là câu trả lời của bài báo.

### 3.8 Hoerl & Kennard 1970, tr.57 — §2: ước lượng ridge, (2.1)–(2.11)

![Hoerl–Kennard 1970 tr.57 — (2.1) tới (2.7)](<paper_ridge_hoerl1970_tr57_2_1_den_2_7.png>)

**Hình 15 — Hoerl & Kennard 1970, tr.57: §2, công thức (2.1)–(2.7).**

$$\hat\beta^*=[X'X+kI]^{-1}X'Y;\ k\ge0 \tag{2.1}$$
$$=WX'Y \tag{2.2}$$
$$\hat\beta^*=[I_p+k(X'X)^{-1}]^{-1}\hat\beta \tag{2.3}$$
$$=Z\hat\beta \tag{2.4}$$
$$\xi_i(W)=1/(\lambda_i+k) \tag{2.5}$$
$$\xi_i(Z)=\lambda_i/(\lambda_i+k) \tag{2.6}$$
$$Z=I-k(X'X+kI)^{-1}=I-kW \tag{2.7}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta^*$ | **ước lượng ridge** | $\mathbb{R}^p$ |
| $k$ | **mức phạt** — $\lambda$ của bài này | số $\ge0$ |
| $W$ | $(X'X+kI)^{-1}$ | $p\times p$ |
| $Z$ | ma trận biến $\hat\beta$ thành $\hat\beta^*$ | $p\times p$ |
| $\xi_i(\cdot)$ | trị riêng thứ $i$ của một ma trận | số |

**Tác giả nói gì.** *"A. E. Hoerl first suggested in 1962 [9] [11] that to control the inflation and
general instability associated with the least squares estimates, one can use"* (2.1). Và về cái tên:
*"The family of estimates given by k ≥ 0 has many mathematical similarities with the portrayal of
quadratic response functions [10]. For this reason, estimation and analysis built around (2.1) has been
labeled "ridge regression.""*

(A. E. Hoerl đề xuất lần đầu năm 1962 dùng (2.1) để kiểm soát sự phình to và mất ổn định của bình phương
tối thiểu. Họ ước lượng theo $k\ge0$ có nhiều nét tương đồng toán học với cách thể hiện mặt đáp ứng bậc
hai; vì thế phép ước lượng quanh (2.1) được gọi là *ridge regression*.)

**Ý nghĩa.** Chữ *ridge* không đến từ thống kê. Nó đến từ *ridge analysis* (Hoerl 1964, tài liệu [10]): đi
dọc **sống núi** của một mặt đáp ứng bậc hai. (2.3)–(2.6) nói ridge là **một phép co** bình phương tối
thiểu: cùng hướng riêng, mỗi hướng nhân với $\lambda_i/(\lambda_i+k)$. Hướng có $\lambda_i\gg k$ gần như giữ
nguyên; hướng có $\lambda_i\ll k$ gần như bị xoá. Đó chính là công thức (3.47) của ESL viết theo trị riêng
thay vì giá trị kỳ dị ($\lambda_i=d_j^2$).

**Kiểm bằng số.** Thiết kế mục 1.2, $k=0{,}05$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7):

| Công thức | lệch lớn nhất |
|---|---:|
| (2.3) $Z\hat\beta=\hat\beta^*$ | $1{,}033\cdot10^{-14}$ |
| (2.5) trị riêng của $W$ | $7{,}461\cdot10^{-14}$ |
| (2.6) trị riêng của $Z$ | $4{,}385\cdot10^{-15}$ |
| (2.7) $Z=I-kW$ | $1{,}332\cdot10^{-15}$ |

![Hoerl–Kennard 1970 tr.57 — (2.8) tới (2.11)](<paper_ridge_hoerl1970_tr57_2_8_den_2_11.png>)

**Hình 16 — Hoerl & Kennard 1970, tr.57: tính chất (iii), công thức (2.8)–(2.11).**

$$(\hat\beta^*)'(\hat\beta^*)<\hat\beta'\hat\beta \tag{2.8}$$
$$(\hat\beta^*)'(\hat\beta^*)\le\xi^2_{\max}(Z)\,\hat\beta'\hat\beta \tag{2.9}$$
$$\phi^*(k)=(Y-X\hat\beta^*)'(Y-X\hat\beta^*) \tag{2.10}$$
$$\phi^*(k)=Y'Y-(\hat\beta^*)'X'Y-k(\hat\beta^*)'(\hat\beta^*) \tag{2.11}$$

**Tác giả nói gì.** *"(iii) β̂* for k ≠ 0 is shorter than β̂"* và *"From (2.6) and (2.7) it is seen that
Z(0) = I and that Z approaches 0 as k → ∞."* Về (2.11): *"The expression shows that φ*(k) is the total sum
of squares less the "regression" sum of squares for β̂* with a modification depending upon the squared
length of β̂*."* (tr.58)

(Với $k\ne0$, $\hat\beta^*$ ngắn hơn $\hat\beta$. $Z(0)=I$ và $Z\to0$ khi $k\to\infty$. Biểu thức (2.11) cho
thấy $\phi^*(k)$ bằng tổng bình phương toàn phần trừ phần "hồi quy" của $\hat\beta^*$, cộng một hiệu chỉnh
theo độ dài bình phương của $\hat\beta^*$.)

**Kiểm bằng số.** Cùng thiết kế, $k=0{,}05$:

| Đại lượng | Giá trị |
|---|---:|
| $\lVert\hat\beta^*\rVert^2$ | 30,678181 |
| $\lVert\hat\beta\rVert^2$ | 723,465637 |
| cận (2.9) $\xi^2_{\max}(Z)\lVert\hat\beta\rVert^2$ | 706,309996 |
| (2.11) so với tính trực tiếp (2.10) | lệch $2{,}132\cdot10^{-14}$ |

**Ý nghĩa.** Chỉ với $k=0{,}05$ — nhỏ hơn trị riêng lớn nhất 82,8384 lần — độ dài bình phương đã co
**23,5824 lần**. Cận (2.9) đúng nhưng **rất lỏng** (706 so với 30,7), vì nó chỉ dùng trị riêng lớn nhất của $Z$, tức
hướng bị co ít nhất; phần co thật nằm ở các hướng trị riêng nhỏ.

### 3.9 Hoerl & Kennard 1970, tr.58–59 — §3: vết ridge, (3.1)–(3.9), và một dấu phẩy thừa

![Hoerl–Kennard 1970 tr.58 — §3a và 3b, (3.1)](<paper_ridge_hoerl1970_tr58_vet_ridge.png>)

**Hình 17 — Hoerl & Kennard 1970, tr.58: kết §2, §3a *Definition of the Ridge Trace*, §3b và
công thức (3.1).**

$$\phi=(Y-XB)'(Y-XB)=(Y-X\hat\beta)'(Y-X\hat\beta)+(B-\hat\beta)'X'X(B-\hat\beta)=\phi_{\min}+\phi(B) \tag{3.1}$$

| Ký hiệu | Nghĩa |
|---|---|
| $B$ | một ước lượng **bất kỳ** của $\beta$ |
| $\phi_{\min}$ | tổng bình phương phần dư nhỏ nhất, của bình phương tối thiểu |
| $\phi(B)$ | phần tăng thêm khi dời khỏi $\hat\beta$ — một dạng toàn phương |

**Tác giả nói gì.** Về vết ridge: *"But by computing β̂*(k) and φ*(k) for a set of values of k, such
insight can be obtained. A detailed study of two nonorthogonal problems and the conclusions that can be
drawn from their ridge traces is given in [12]."* Và lập luận trung tâm của cả bài: *"In particular, the
worse the conditioning of X'X, the more β̂ can be expected to be too long. On the other hand, the worse the
conditioning, the further one can move from β̂ without an appreciable increase in the residual sums of
squares. In view of (1.5a) it seems reasonable that if one moves away from the minimum sum of squares
point, the movement should be in a direction which will shorten the length of the regression vector."*

(Tính $\hat\beta^*(k)$ và $\phi^*(k)$ cho một dãy $k$ thì thấy được cấu trúc đó; nghiên cứu chi tiết hai bài
toán nằm ở [12]. $X'X$ càng xấu thì $\hat\beta$ càng dễ quá dài; nhưng cũng chính khi đó, người ta đi xa
được khỏi $\hat\beta$ mà tổng bình phương phần dư gần như không tăng. Theo (1.5a), nếu đã rời điểm tổng bình
phương nhỏ nhất thì nên đi theo hướng làm vector hồi quy **ngắn lại**.)

**Ý nghĩa.** (3.1) cho hình học: các mặt đồng mức của tổng bình phương phần dư là **ellipsoid tâm
$\hat\beta$**, và ellipsoid đó **dẹt dài** theo hướng trị riêng nhỏ. Hai câu trên ghép lại thành toàn bộ lập
luận: hướng mà $\hat\beta$ sai nhiều nhất cũng là hướng mà dời đi **tốn ít nhất**. Kiểm (3.1) với một $B$
ngẫu nhiên: lệch $1{,}563\cdot10^{-13}$.

**Điều khung KHÔNG nói.** Tr.58 **định nghĩa** vết ridge nhưng không vẽ vết nào. Mọi vết ridge thật của
Hoerl–Kennard nằm ở bài [12].

![Hoerl–Kennard 1970 tr.59 — (3.2) tới (3.7)](<paper_ridge_hoerl1970_tr59_3_2_den_3_7.png>)

**Hình 18 — Hoerl & Kennard 1970, tr.59: bài toán ràng buộc (3.2)–(3.7).**

$$\text{Minimize } B'B\ \text{ subject to }\ (B-\hat\beta)'X'X(B-\hat\beta)=\phi_0 \tag{3.2}$$
$$\text{Minimize } F=B'B+(1/k)[(B-\hat\beta)'X'X(B-\hat\beta)'-\phi_0] \tag{3.3}$$
$$\frac{\partial F}{\partial B}=2B+(1/k)[2(X'X)B-2(X'X)\hat\beta]=0 \tag{3.4}$$
$$B=\hat\beta^*=[X'X+kI]^{-1}X'Y \tag{3.5}$$
$$\phi^*(k)=\phi_{\min}+k^2\hat\beta^{*\prime}(X'X)^{-1}\hat\beta^* \tag{3.6}$$
$$F_1=(Y-XB)'(Y-XB)+(1/k)(B'B-R^2) \tag{3.7}$$

| Ký hiệu | Nghĩa |
|---|---|
| $\phi_0$ | lượng tăng tổng bình phương phần dư được **cho phép** |
| $1/k$ | nhân tử Lagrange — nên $k$ nhỏ ứng với ràng buộc lỏng |
| $R^2$ | bình phương độ dài vector hệ số được cố định — **không** phải hệ số xác định $R^2$ |

**Tác giả nói gì.** *"This is the ridge estimator. Of course, in practice it is easier to choose a k ≥ 0
and then compute φ0."* và *"A completely equivalent statement of the path is this: If the squared length
of the regression vector B is fixed at R², then β̂* is the value of B that gives a minimum sum of
squares."*

(Đó là ước lượng ridge. Tất nhiên thực tế dễ hơn là chọn $k\ge0$ rồi tính $\phi_0$. Một phát biểu hoàn toàn
tương đương của đường đi: nếu độ dài bình phương của vector hồi quy cố định bằng $R^2$, thì $\hat\beta^*$ là
giá trị của $B$ cho tổng bình phương nhỏ nhất.)

**Chỗ in sai — công thức (3.3).** Dấu phẩy trên thứ hai trong $(B-\hat\beta)'X'X(B-\hat\beta)'$ biến một
vector cột thành vector dòng, và tích $1\times p\cdot p\times p\cdot1\times p$ **không khớp chiều** — `numpy`
báo lỗi ngay ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7). Công thức (3.2) ngay trên và (3.4) ngay
dưới đều dùng dạng không có dấu đó, nên đây là lỗi sắp chữ.

**Kiểm (3.4)–(3.6).** Gradient của $F$ tại $B=\hat\beta^*$: chuẩn vô cùng $2{,}318\cdot10^{-13}$. (3.6) so
với tính trực tiếp: lệch $1{,}252\cdot10^{-13}$.

**Ý nghĩa.** Hai cách phát biểu (3.2) và (3.7) là **một cặp đối ngẫu**: cố định lượng tăng sai số rồi tìm
vector ngắn nhất, hoặc cố định độ dài rồi tìm sai số nhỏ nhất. Cả hai cho cùng một đường đi theo $k$. Đó là
dạng ràng buộc (3.42) của ESL. (3.6) cho biết cái giá về sai số huấn luyện của ridge **tăng như $k^2$** khi
$k$ nhỏ — bậc hai, nên lúc đầu gần như miễn phí.

![Hoerl–Kennard 1970 tr.59 — §3c, (3.8)–(3.9)](<paper_ridge_hoerl1970_tr59_3_8_3_9.png>)

**Hình 19 — Hoerl & Kennard 1970, tr.59: §3c *Likelihood Characterization of the Ridge Trace*, công
thức (3.8)–(3.9).**

$$(2\pi\sigma^2)^{-n/2}\exp\{-(1/2\sigma^2)(Y-X\beta)'(Y-X\beta)\} \tag{3.8}$$
$$(Y-X\beta)'(Y-X\beta)=(Y-X\hat\beta)'(Y-X\hat\beta)+(\beta-\hat\beta)'X'X(\beta-\hat\beta) \tag{3.9}$$

**Tác giả nói gì.** *"Although long vectors give the same likelihood values as shorter vectors, they will
not always have equal physical meaning. Implied is a restraint on the possible values of β̂ that is not made
explicit in the formulation of the general linear model given in the Introduction."*

(Vector dài cho cùng giá trị hợp lý như vector ngắn hơn, nhưng không phải lúc nào cũng có cùng ý nghĩa vật
lý. Ẩn ở đây là một ràng buộc lên các giá trị có thể của $\hat\beta$, thứ không được nói ra trong mô hình
tuyến tính tổng quát ở phần giới thiệu.)

**Ý nghĩa.** Câu này là **gốc của cách nhìn Bayes** ở mục 6 của bài báo: ridge đưa vào một thông tin mà dữ
liệu không chứa — rằng vector hệ số không dài vô lý. Bài báo gọi nó là *restraint*, ESL tr.64 gọi nó là
*prior*. Và nó giải thích vì sao ridge không bao giờ "miễn phí": nếu hệ số thật **đúng là** rất dài, ràng
buộc đó sai, và mục 3.11 đo điều gì xảy ra.

### 3.10 Hoerl & Kennard 1970, tr.60–61 — §4a: phân rã chệch–phương sai (4.2)–(4.10) và Hình 1

![Hoerl–Kennard 1970 tr.60 — (4.2) tới (4.10)](<paper_ridge_hoerl1970_tr60_4_2_den_4_10.png>)

**Hình 20 — Hoerl & Kennard 1970, tr.60: §4a *Variance and Bias of a Ridge Estimator*, công thức
(4.2)–(4.10).**

$$E[L_1^2(k)]=E[(\hat\beta^*-\beta)'(\hat\beta^*-\beta)]=E[(\hat\beta-\beta)'Z'Z(\hat\beta-\beta)]+(Z\beta-\beta)'(Z\beta-\beta) \tag{4.2}$$
$$=\sigma^2\operatorname{Trace}(X'X)^{-1}Z'Z+\beta'(Z-I)'(Z-I)\beta \tag{4.3}$$
$$=\sigma^2[\operatorname{Trace}(X'X+kI)^{-1}-k\operatorname{Trace}(X'X+kI)^{-2}]+k^2\beta'(X'X+kI)^{-2}\beta \tag{4.4}$$
$$=\sigma^2\sum_1^p\lambda_i/(\lambda_i+k)^2+k^2\beta'(X'X+kI)^{-2}\beta \tag{4.5}$$
$$=\gamma_1(k)+\gamma_2(k) \tag{4.6}$$
$$\hat\beta^*=Z\hat\beta=Z(X'X)^{-1}X'Y \tag{4.7}$$
$$\operatorname{VAR}[\hat\beta^*]=Z(X'X)^{-1}X'\operatorname{VAR}[Y]X(X'X)^{-1}Z'=\sigma^2Z(X'X)^{-1}Z' \tag{4.8}$$
$$\lim_{k\to0^+}(d\gamma_1/dk)=-2\sigma^2\Sigma(1/\lambda_i^2) \tag{4.9}$$
$$\lim_{k\to0^+}(d\gamma_2/dk)=0 \tag{4.10}$$

| Ký hiệu | Nghĩa |
|---|---|
| $\gamma_1(k)$ | **tổng phương sai** của các hệ số ridge |
| $\gamma_2(k)$ | **chệch bình phương** — bình phương khoảng cách từ $Z\beta$ tới $\beta$ |

**Ý nghĩa của (4.9)–(4.10) trong một câu.** Tại $k=0$ phương sai đang **rơi** còn chệch² **chưa kịp tăng**,
nên bước đầu tiên ra khỏi bình phương tối thiểu luôn có lời — phần "Ý nghĩa" dưới bảng kiểm nói kỹ.

Bài báo đánh số bắt đầu từ (4.2); không có công thức (4.1).

**Tác giả nói gì.** *"The total variance decreases as k increases, while the squared bias increases with
k. As is indicated by the dotted line, which is the sum of γ1(k) and γ2(k) and thus is E[L²1(k)], the
possibility exists that there are values of k (admissible values) for which the mean square error is less
for β̂* than it is for the usual solution β̂."* Và: *"However, the most significant feature is the value of
the derivative of each function in the neighborhood of the origin."*

(Tổng phương sai giảm khi $k$ tăng, chệch bình phương tăng theo $k$. Như đường chấm chỉ ra — nó là tổng
$\gamma_1+\gamma_2$, tức $E[L_1^2(k)]$ — có khả năng tồn tại những giá trị $k$ ("chấp nhận được") mà sai số
bình phương trung bình của $\hat\beta^*$ nhỏ hơn của nghiệm thường. Đặc điểm quan trọng nhất là **đạo hàm
của mỗi hàm gần gốc**.)

**Kiểm bằng số.** Thiết kế mục 1.2, $k=0{,}05$ ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7):

| Công thức | lệch |
|---|---:|
| (4.3) = (4.4), phần phương sai | $1{,}421\cdot10^{-14}$ |
| (4.4) = (4.5), phần phương sai | $9{,}237\cdot10^{-14}$ |
| (4.3) = (4.5), phần chệch² | $5{,}063\cdot10^{-14}$ |
| (4.8) vết $=\gamma_1$ | $7{,}816\cdot10^{-14}$ |
| (4.9) đạo hàm số tại $k=10^{-9}$ | $-6{,}708897\cdot10^{4}$ |
| (4.9) theo công thức $-2\sigma^2\Sigma1/\lambda_i^2$ | $-6{,}708898\cdot10^{4}$ |

**Ý nghĩa — hai đạo hàm là toàn bộ định lý.** Tại $k=0$, phương sai **giảm với tốc độ**
$2\sigma^2\Sigma1/\lambda_i^2$ — trên thiết kế này là 67 089 — trong khi chệch² **tăng với tốc độ 0**. Một
hàm dốc xuống cộng một hàm phẳng thì dốc xuống. Nên luôn có một đoạn $k>0$ nhỏ mà tổng nhỏ hơn giá trị ở 0.
Và tốc độ giảm tỉ lệ với $\Sigma1/\lambda_i^2$: ma trận càng xấu, lợi càng lớn. Với thiết kế trực giao,
$\lambda_i=1$, tốc độ chỉ còn $-2p\sigma^2=-16$.

![Hoerl–Kennard 1970 tr.61 — Hình 1](<paper_ridge_hoerl1970_tr61_hinh1.png>)

**Hình 21 — Hình 1 của Hoerl & Kennard 1970, tr.61: *"Mean Square Error Functions"*.**

**Caption gốc.** Bài báo chỉ in *"Figure 1"* dưới hình; tiêu đề trên hình là *"MEAN SQUARE ERROR
FUNCTIONS"*, trục đứng *"UNIT = (1/SIGMA SQ.)"*, trục ngang $k$ từ 0 tới 1,00. Bốn nhãn: *"Least
Squares"*, *"Ridge"* (đường đứt), *"Bias-Squared"*, *"Variance"*.

**Tác giả nói gì về hình.** Tr.60: *"Figure 1 shows in qualitative form the relationship between the
variances, the squared bias, and the parameter k."* Và tr.61: *"These properties lead to the conclusion
that it is possible to move to k > 0, take a little bias, and substantially reduce the variance, thereby
improving the mean square error of estimation and prediction."*

(Hình 1 thể hiện **ở dạng định tính** quan hệ giữa phương sai, chệch bình phương và tham số $k$. Các tính
chất đó dẫn tới kết luận: có thể đi tới $k>0$, nhận một ít chệch, và giảm phương sai đáng kể, nhờ đó cải
thiện sai số bình phương trung bình của ước lượng và dự đoán.)

**Cách đọc hình.** Đường ngang là sai số bình phương trung bình của bình phương tối thiểu — không phụ thuộc
$k$. Đường "Variance" bắt đầu ở mức đó tại $k=0$ rồi rơi rất nhanh. Đường "Bias-Squared" bắt đầu ở 0 rồi
tăng. Đường đứt "Ridge" là tổng của hai đường: nó chạm đáy ở một $k$ nhỏ, rồi đi lên và cắt lại đường
"Least Squares" — mọi $k$ bên trái chỗ cắt đó là "chấp nhận được" theo nghĩa của tác giả. Đáy và chỗ cắt
nằm ở vùng có chữ "Ridge" đè lên nét vẽ, nên bài này không đọc toạ độ của chúng.

**Số hoá bản quét.** Bài báo không cho số liệu, không cho $X$, không cho $\beta$, và tự gọi hình là **định
tính**. Nên phép kiểm duy nhất có nghĩa là kiểm **tính nhất quán bên trong**: đường đứt có bằng tổng hai
đường liền không. Số hoá ảnh trang 3680×5546 điểm ảnh: dò 11 hàng lưới (dư lớn nhất 6,91 điểm ảnh so với cấp
số cộng), 10 cột lưới (dư 4,40), rồi đọc ba đường ở bảy cột
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 17):

| $k$ | phương sai | chệch² | đường đứt | tổng hai đường | đứt − tổng |
|---:|---:|---:|---:|---:|---:|
| 0,25 | 8,24 | 51,83 | 59,74 | 60,08 | −0,34 |
| 0,35 | 6,11 | 61,85 | 68,03 | 67,96 | +0,07 |
| 0,45 | 4,92 | 70,85 | 75,45 | 75,77 | −0,32 |
| 0,55 | 4,12 | 78,61 | 82,62 | 82,73 | −0,11 |
| 0,65 | 3,64 | 85,16 | 88,78 | 88,80 | −0,02 |
| 0,75 | 3,04 | 92,84 | 96,09 | 95,88 | +0,21 |
| 0,95 | 2,47 | 105,87 | 108,10 | 108,34 | −0,25 |

Chênh lớn nhất 0,34 đơn vị, trong khi một ô lưới là 11 đơn vị. Hàng dài không thuộc lưới — đường *"Least
Squares"* — nằm ở 86,38. **Hình 1 nhất quán**: đường đứt đúng là tổng.

**Điều hình KHÔNG nói.** Không cho biết $p$, $\sigma$, trị riêng hay $\beta$ nào sinh ra các đường. Ở
$k=0{,}95$ chệch² đã là 105,87 và vẫn đang tăng, nên theo hệ quả 4.2.1 thì $\beta'\beta/\sigma^2$ của ví dụ
này lớn hơn 105,87. Và hình **không thể** hiện đạo hàm bằng 0 của chệch² tại gốc (4.10): ở độ phân giải của hình, đường
chệch² bắt đầu gần như dựng đứng — đoạn phẳng nằm trong một khoảng $k$ nhỏ hơn bề dày nét vẽ.

### 3.11 Hoerl & Kennard 1970, tr.61–63 — §4b và 4c: định lý 4.1–4.3, (4.11)–(4.16), và một khối in hai lần

![Hoerl–Kennard 1970 tr.61 — §4b, định lý 4.1, hệ quả 4.1.1, định lý 4.2](<paper_ridge_hoerl1970_tr61_dinh_ly_4_1.png>)

**Hình 22 — Hoerl & Kennard 1970, tr.61: đoạn kết §4a, §4b *Theorems on the Mean Square Function*,
định lý 4.1, hệ quả 4.1.1, định lý 4.2.**

![Hoerl–Kennard 1970 tr.62 — khối lặp, định lý 4.2 với chứng minh, định lý 4.3](<paper_ridge_hoerl1970_tr62_dinh_ly_4_2_4_3.png>)

**Hình 23 — Hoerl & Kennard 1970, tr.62: dòng chứng minh lạc chỗ, hệ quả 4.1.1 và định lý 4.2 in lại, chứng
minh định lý 4.2, hệ quả 4.2.1–4.2.2, định lý 4.3, công thức (4.11)–(4.15).**

$$\gamma_2(k)=k^2\sum_1^p\alpha_i^2/(\lambda_i+k)^2 \tag{4.11}$$
$$\text{where }\alpha=P\beta \tag{4.12}$$
$$\gamma_2(k)=\sum_1^p\alpha_i^2/[1+(\lambda_i/k)]^2 \tag{4.13}$$
$$d\gamma_2(k)/dk=2k\sum_1^p\lambda_i\alpha_i^2/(\lambda_i+k)^3 \tag{4.14}$$
$$dE[L_1^2(k)]/dk=-2\sigma^2\sum_1^p\lambda_i/(\lambda_i+k)^3+2k\sum_1^p\lambda_i\alpha_i^2/(\lambda_i+k)^3 \tag{4.15}$$

| Ký hiệu | Nghĩa |
|---|---|
| $P$ | ma trận trực giao sao cho $X'X=P'\Lambda P$ — cột của $P'$ là vector riêng |
| $\alpha_i$ | **toạ độ của hệ số thật** theo hướng riêng thứ $i$ — **không** phải tham số chính quy hoá |
| $\Lambda$ | ma trận đường chéo các trị riêng |

**Tác giả nói gì.** Định lý 4.1: *"The total variance γ1(k) is a continuous, monotonically decreasing
function of k."* Định lý 4.2: *"The squared bias γ2(k) is a continuous, monotonically increasing function
of k."* Hệ quả 4.2.1: *"The squared bias γ2(k) approaches β'β as an upper limit."* Định lý 4.3: *"(Existence
Theorem) There always exists a k > 0 such that E[L²1(k)] < E[L²1(0)] = σ² Σ(1/λi)."*

(Tổng phương sai là hàm liên tục, giảm đơn điệu theo $k$. Chệch bình phương là hàm liên tục, tăng đơn điệu,
và có giới hạn trên $\beta'\beta$. Định lý tồn tại: **luôn** có $k>0$ làm sai số bình phương trung bình nhỏ
hơn của bình phương tối thiểu.)

**Chỗ in sai — một khối in hai lần.** Cuối tr.61 in định lý 4.1, hệ quả 4.1.1, câu *"Both the theorem and
the corollary are readily proved…"* và định lý 4.2. Đầu tr.62 mở bằng dòng *"Proof: From (4.5) γ2(k) =
k²β'(X'X + kI)⁻²β."* — chứng minh của định lý 4.2 — rồi in **lại** hệ quả 4.1.1, câu *"Both the theorem…"* và
định lý 4.2, rồi mới tới chứng minh đầy đủ. Hai bản hệ quả 4.1.1 **khác chữ**:

| Chỗ | Nguyên văn |
|---|---|
| tr.61 | *"The first derivative with respect to k of the total variance γ'1(k), approaches − ∞ as k → 0+ and λp → 0."* |
| tr.62 | *"The first derivative of the total variance, γ'1(k), approaches − ∞ as k → 0+ and the matrix X'X becomes singular."* |

Hai phát biểu nói cùng một điều ($\lambda_p\to0$ nghĩa là $X'X$ suy biến), nên lỗi không đổi nội dung. Nhưng
nó cho thấy bản in giữ lại **hai phiên bản** của một đoạn — dấu vết một lần sửa bài chưa xoá hết. Định lý 4.1
không có chứng minh riêng nào, chỉ có câu *"readily proved"*.

![Hoerl–Kennard 1970 tr.63 — (4.16), §4c, §5](<paper_ridge_hoerl1970_tr63_4_16_muc5.png>)

**Hình 24 — Hoerl & Kennard 1970, tr.63: kết chứng minh định lý 4.3 với (4.16), §4c *Some Comments On
The Mean Square Error Function*, §5 *A General Form of Ridge Regression* với (5.1)–(5.4).**

$$k<\sigma^2/\alpha^2_{\max} \tag{4.16}$$

**Tác giả nói gì.** Kết chứng minh: *"Thus, to prove the theorem, it is only necessary to show that there
always exists a k > 0 such that dE[L²1(k)]/dk < 0. The condition for this is shown by (4.15) to be: k <
σ²/α²max q.e.d."* Mục 4c: *"And since γ2(k) approaches β'β as a limit as k → ∞, this minimum will move
toward k = 0 as the magnitude of β'β increases. Since β'β is the squared length of the unknown regression
vector, it would appear to be impossible to choose a value of k ≠ 0 and thus achieve a smaller mean square
error without being able to assign an upper bound to β'β."*

(Chỉ cần chỉ ra luôn có $k>0$ làm đạo hàm âm; theo (4.15) điều kiện là $k<\sigma^2/\alpha^2_{\max}$. Mục 4c: vì
chệch² tiến tới $\beta'\beta$, cực tiểu sẽ dời về $k=0$ khi $\beta'\beta$ tăng; và vì $\beta'\beta$ là độ dài
bình phương của vector **chưa biết**, có vẻ không thể chọn $k\ne0$ để chắc chắn giảm sai số mà không đặt được
một cận trên cho $\beta'\beta$.)

**Ý nghĩa.** (4.16) là một điều kiện **đủ**, không phải cần: nếu $k\alpha_i^2<\sigma^2$ với mọi $i$ thì từng
số hạng dương của (4.15) nhỏ hơn số hạng âm cùng chỉ số. Nó cho một $k$ an toàn **nếu biết $\beta$** — mà
không ai biết. Mục 4c tự nói ra đúng điều đó, và đây là câu trung thực nhất của bài báo: định lý tồn tại
không cho cách **tìm** $k$.

**Kiểm bằng số — cận (4.16) chặt tới đâu.** Thiết kế mục 1.2 ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>)
mục 8):

| Đại lượng | Giá trị |
|---|---:|
| $E[L_1^2(0)]$ | 416,855380 |
| $k$ tối ưu trên lưới | 0,695024 |
| $E[L_1^2]$ tại đó | 9,986065 — giảm 41,7437 lần |
| cận (4.16) $\sigma^2/\alpha^2_{\max}$ | 0,293847 |
| $k$ tối ưu / cận | 2,3653 |
| $dE/dk<0$ tại 200 điểm trong $(0,\ \sigma^2/\alpha^2_{\max}]$ | đúng |

Trên thiết kế này **mọi** $k$ trên lưới tới 100 đều thắng bình phương tối thiểu, vì giới hạn khi $k\to\infty$
là $\beta'\beta=13$, đã nhỏ hơn 416,86. Cận (4.16) nằm dưới $k$ tối ưu 2,3653 lần: đủ an toàn, không tối ưu.

**Kiểm câu của mục 4c.** Nhân $\beta$ với một hằng số, giữ nguyên thiết kế
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 9):

| $\beta'\beta$ | 1,3 | 3,9 | 13 | 39 | 130 | 390 | 1 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $k$ tối ưu | 6,223 | 2,18776 | 0,695024 | 0,231206 | 0,0654636 | 0,0198609 | 0,00522396 |
| giảm MSE (lần) | 357,3682 | 128,9497 | 41,7437 | 14,8683 | 5,1351 | 2,3411 | 1,3850 |

$k$ tối ưu giảm đơn điệu; $\beta'\beta$ tăng 1 000 lần thì $k$ tối ưu giảm 1 191,2 lần, và cái lợi co từ 357
lần xuống 1,385 lần. Câu của mục 4c đúng, và nó định lượng được: **ridge lợi nhiều nhất khi hệ số thật
ngắn so với nhiễu**.

### 3.12 Hoerl & Kennard 1970, tr.63 — §5: dạng tổng quát (5.1)–(5.4)

(Khung ở Hình 24, nửa dưới.)

$$X=X^*P \tag{5.1}$$
$$Y=X^*\alpha+\varepsilon \tag{5.2}$$
$$\alpha=P\beta,\quad(X^*)'(X^*)=\Lambda,\quad\alpha'\alpha=\beta'\beta \tag{5.3}$$
$$\hat\alpha^*=[(X^*)'(X^*)+K]^{-1}(X^*)'Y,\qquad K=(\delta_{ij}k_i),\ k_i\ge0 \tag{5.4}$$

| Ký hiệu | Nghĩa |
|---|---|
| $X^*$ | dữ liệu viết trong hệ toạ độ riêng — các cột trực giao |
| $K$ | ma trận đường chéo, **mỗi hướng một mức phạt** $k_i$ — không phải nhân $K(x,s)$ của Tikhonov |
| $\delta_{ij}$ | ký hiệu Kronecker — không phải mức nhiễu $\delta$ của Tikhonov |

**Tác giả nói gì.** *"In the general form; one seeks a ki for each canonical variate defined by X*. By
defining (L*1)² = (α̂* − α)'(α̂* − α) it can be shown that the optimal values for the ki will be ki =
σ²/α²i. There is no graphical equivalent to the RIDGE TRACE but an iterative procedure initiated at k̂i =
σ̂²/α̂²i can be used."*

(Ở dạng tổng quát, tìm một $k_i$ cho mỗi biến chuẩn tắc. Có thể chứng minh giá trị tối ưu là
$k_i=\sigma^2/\alpha_i^2$. Không có dạng đồ thị tương đương vết ridge, nhưng có thể dùng một thủ tục lặp bắt
đầu từ $\hat k_i=\hat\sigma^2/\hat\alpha_i^2$.)

**Kiểm "$k_i=\sigma^2/\alpha_i^2$" — và điều bài báo không nhấn mạnh.** Sai số của một hướng là
$\sigma^2\lambda_i/(\lambda_i+k)^2+k^2\alpha_i^2/(\lambda_i+k)^2$; đạo hàm bằng 0 tại $k=\sigma^2/\alpha_i^2$
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 10):

| $\lambda_i$ | $\alpha_i$ | argmin trên lưới | $\sigma^2/\alpha_i^2$ |
|---:|---:|---:|---:|
| 0,01 | 1 | 1,00000 | 1,00000 |
| 1 | 1 | 1,00000 | 1,00000 |
| 100 | 1 | 1,00000 | 1,00000 |
| 1 | 0,3 | 11,11118 | 11,11111 |
| 1 | 3 | 0,11111 | 0,11111 |

Mức phạt tối ưu của một hướng **không phụ thuộc trị riêng của hướng đó** — chỉ phụ thuộc tỉ số nhiễu trên
tín hiệu. Trên thiết kế mục 1.2, phạt riêng từng hướng cho $E[L_1^2]=9{,}733636$, so với 9,986065 của một $k$
chung tốt nhất và 416,855380 của bình phương tối thiểu: dạng tổng quát lợi thêm rất ít so với dạng một
tham số, trong khi phải ước lượng $p$ tham số thay vì một.

### 3.13 Hoerl & Kennard 1970, tr.64–66 — §6, 7, 8 và bảng ký hiệu

![Hoerl–Kennard 1970 tr.64 — §6 và đầu §7](<paper_ridge_hoerl1970_tr64_muc6_muc7.png>)

**Hình 25 — Hoerl & Kennard 1970, tr.64: §6 *Relation to Other Work in Regression* và đầu §7
*Selecting a Better Estimate of β*.**

**Tác giả nói gì.** Về Bayes: *"Viewed in this context, each ridge estimate can be considered as the
posterior mean based on giving the regression coefficients, β, a prior normal distribution with mean zero
and variance-covariance matrix Σ = (δijσ²/k). For those that do not like the philosophical implications of
assuming β to be a random variable, all this is equivalent to constrained estimation by a nonuniform
weighting on the values of β."* Về ràng buộc: *"Under the constraint, if β̂'β̂ ≤ R², than B is chosen to be
β̂; otherwise B is chosen to be β̂* where k is chosen so that (β̂*)'(β̂*) = R²."* Mở đầu mục 7: *"In Section 2
and in the example of Section 3, it has been demonstrated that the ordinary least squares estimate of the
regression vector β suffers from a number of deficiencies"* và *"As Barnard [12] has recently pointed out,
an alternative to unbiasedness in the logic of the least squares estimator β̂ is the prior assurance of
bounded mean square error with no boundedness assumption on β."*

(Mỗi ước lượng ridge có thể xem là trung bình hậu nghiệm khi cho hệ số tiên nghiệm chuẩn, trung bình 0, ma
trận hiệp phương sai $\sigma^2/k$ nhân ma trận đơn vị; ai không thích coi $\beta$ là biến ngẫu nhiên thì điều
đó tương đương ước lượng có ràng buộc. Dưới ràng buộc mặt cầu bán kính $R$: nếu $\hat\beta'\hat\beta\le R^2$
thì giữ $\hat\beta$, ngược lại chọn $k$ sao cho $\hat\beta^*$ nằm trên mặt cầu.)

**Kiểm bằng số** ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 12), $k=0{,}3$:

| So sánh | Kết quả |
|---|---:|
| trung bình hậu nghiệm với tiên nghiệm $N(0,\sigma^2/k\cdot I)$ so với $[X'X+kI]^{-1}X'Y$ | lệch $3{,}109\cdot10^{-15}$ |
| tổng bình phương phần dư của $\hat\beta^*$, trên mặt cầu $R^2=\lVert\hat\beta^*\rVert^2=4{,}020818$ | 49,408973 |
| nhỏ nhất trên 20 000 điểm ngẫu nhiên cùng mặt cầu | 49,548261 |

**Hai chỗ in sai ở đầu mục 7.**

1. *"in the example of Section 3"* — mục 3 (tr.58–59) **không có ví dụ nào**. Chính mục 3a tr.58 nói: *"A
   detailed study of two nonorthogonal problems and the conclusions that can be drawn from their ridge
   traces is given in [12]."* Ví dụ nằm ở bài đi kèm, không nằm trong bài này.
2. *"As Barnard [12] has recently pointed out"* — danh mục tài liệu tr.66–67 ghi [12] là *Hoerl, A. E. and
   Kennard, R. W. (1970). Ridge Regression. Applications to nonorthogonal problems.* Bài của Barnard —
   *"The logic of least squares"*, JRSS B 25, 1963 — là **[2]**.

![Hoerl–Kennard 1970 tr.65 — §7 tiếp theo và §8](<paper_ridge_hoerl1970_tr65_muc7_muc8.png>)

**Hình 26 — Hoerl & Kennard 1970, tr.65: cách chọn $k$ bằng vết ridge, thủ tục lặp cho $k_i$, và §8
*Conclusions*.**

**Tác giả nói gì.** Bốn tiêu chí đọc vết ridge: *"At a certain value of k the system will stabilize and
have the general characteristics of an orthogonal system. • Coefficients will not have unreasonable
absolute values with respect to the factors for which they represent rates of change. • Coefficients with
apparently incorrect signs at k = 0 will have changed to have the proper sign. • The residual sum of
squares will not have been inflated to an unreasonable value."* Rồi thủ tục lặp: *"The k̂i0 will tend to be
too small because of the tendency to overestimate α'α. Since use of the k̂i0 will shorten the length of the
estimated regression vector, ki0 can be re-estimated using the α̂*i. This re-estimation can be continued
until there is a stability achieved in (α̂*)'(α̂*) and k̂i0 = σ̂²/(α̂*i)²."*

(Ở một $k$ nào đó hệ sẽ ổn định và mang dáng một hệ trực giao; hệ số không có trị tuyệt đối vô lý; hệ số
có dấu có vẻ sai ở $k=0$ đã đổi sang dấu đúng; tổng bình phương phần dư chưa bị thổi phồng tới mức vô lý.
Với thủ tục lặp: $\hat k_{i0}$ ban đầu có xu hướng quá nhỏ vì $\alpha'\alpha$ bị ước lượng quá; vì dùng
$\hat k_{i0}$ làm vector ngắn lại, có thể ước lượng lại $k_{i0}$ bằng $\hat\alpha^*_i$, và lặp tới khi ổn định.)

**Kiểm "$\alpha'\alpha$ bị ước lượng quá".** $E[\hat\alpha'\hat\alpha]=\alpha'\alpha+\sigma^2\Sigma1/\lambda_i$;
Monte Carlo 40 000 lần cho 429,9157, so với $\alpha'\alpha=13$ cộng 416,8554 bằng 429,8554. Câu đúng.

**Chỗ bỏ ngỏ — vòng lặp không phải lúc nào cũng dừng.** Điểm dừng thoả
$k=\sigma^2(\lambda+k)^2/(\lambda^2\hat\alpha^2)$, tức $\sigma^2k^2+(2\sigma^2\lambda-\lambda^2\hat\alpha^2)k+\sigma^2\lambda^2=0$.
Phương trình bậc hai này có nghiệm thực **khi và chỉ khi** $\lambda\hat\alpha^2\ge4\sigma^2$ — tức thống kê
$t=\hat\alpha/(\sigma/\sqrt\lambda)$ của hướng đó có $\lvert t\rvert\ge2$. Dưới ngưỡng, mỗi vòng làm
$\hat\alpha^*$ nhỏ đi và $k$ lớn lên, không bao giờ dừng ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục
11):

| $\lambda$ | $\hat\alpha$ | $t^2$ | số vòng | $k$ cuối | kết cục |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 9,000 | 19 | 0,145898 | dừng |
| 1 | 2,2 | 4,840 | 48 | 0,411833 | dừng |
| 1 | 2 | 4,000 | 100 000 | 0,99996 | dừng rất chậm, đúng ở biên |
| 1 | 1,9 | 3,610 | 21 | $1{,}84\cdot10^{23}$ | **phân kỳ** |
| 1 | 1 | 1,000 | 6 | $4{,}41\cdot10^{22}$ | **phân kỳ** |
| 0,05 | 5 | 1,250 | 6 | $1{,}39\cdot10^{14}$ | **phân kỳ** |
| 0,05 | 12 | 7,200 | 21 | 0,01 | dừng |

Trên thiết kế mục 1.2 với một lần rút dữ liệu, **6 trong 8** hướng có $t^2<4$. Với những hướng đó thủ tục
lặp đẩy hệ số về 0 hoàn toàn — một kết quả có thể chấp nhận được, nhưng là **một phép chọn biến**, và bài
báo không nói ra rằng thủ tục của mình làm điều đó.

**Tác giả nói gì trong kết luận.** *"A study of the properties of the estimator β̂* shows that it can be used
to improve the mean square error of estimation, and the magnitude of this improvement increases with an
increase in spread of the eigenvalue spectrum. An estimate based on β̂* is biased and the use of a biased
estimator implies some prior bound on the regression vector β."* (tr.66)

(Ước lượng $\hat\beta^*$ có thể cải thiện sai số bình phương trung bình, và mức cải thiện tăng theo **độ
trải của phổ trị riêng**. Ước lượng này có chệch, và dùng ước lượng có chệch là ngầm đặt một cận trước lên
vector hồi quy.)

![Hoerl–Kennard 1970 tr.66 — kết luận và bảng ký hiệu](<paper_ridge_hoerl1970_tr66_ket_ky_hieu.png>)

**Hình 27 — Hoerl & Kennard 1970, tr.66: đoạn cuối kết luận và bảng *Nomenclature*.**

**Khung chứa gì.** 16 dòng định nghĩa, từ $\hat\beta$ tới $Z^*$.

**Chỗ in sai — dòng cuối bảng ký hiệu.** In: $Z^*=\{I+[(X^*)'(X^*)]^{-1}K]\}=I-KW$. Ba vấn đề: dấu ngoặc
không cân (một `]` thừa); **thiếu nghịch đảo ngoài cùng**; vế phải phải là $W^*$, không phải $W$. Dạng đúng,
suy từ $Z=[I+k(X'X)^{-1}]^{-1}=I-kW$ ở dòng thứ tư của cùng bảng:
$Z^*=\{I+[(X^*)'(X^*)]^{-1}K\}^{-1}=I-KW^*$. Kiểm với $K$ đường chéo ngẫu nhiên
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 7): dạng in lệch **25,74** so với $I-KW^*$; thêm nghịch đảo
thì lệch $1{,}110\cdot10^{-16}$.

**Điều khung KHÔNG nói.** Bảng ký hiệu ghi $\hat\beta^*(k)$ với *"k > 0"* trong khi (2.1) ghi $k\ge0$ — hai
dòng khác nhau ở trường hợp $k=0$, lúc ridge trùng bình phương tối thiểu. Không đổi nội dung.

### 3.14 ESL 2009, tr.61 — §3.3.4 (dữ liệu tuyến tiền liệt) và mở đầu §3.4

Hoerl–Kennard không có dữ liệu thật, Tikhonov không có số. Nguồn thứ ba lấp chỗ trống đó: sách của Hastie,
Tibshirani và Friedman dùng một bộ dữ liệu y khoa có sẵn công khai — 97 bệnh nhân sắp mổ tuyến tiền liệt,
8 chỉ số lâm sàng, đáp án là log của nồng độ kháng nguyên PSA — và in hệ số, sai số kiểm tra, sai số chuẩn
cho sáu phương pháp. Dữ liệu lấy từ gói R `ElemStatLearn` 2015.6.26.2 trên kho lưu trữ CRAN, lưu ở
`tools/prostate.csv`: 97 dòng, cột `train` đánh dấu 67 mẫu huấn luyện và 30 mẫu kiểm tra đúng như sách
chia.

![ESL 2009 tr.61 — §3.3.4 và đầu §3.4.1](<paper_ridge_esl2009_tr61_muc_3_3_4_va_3_4_1.png>)

**Hình 28 — ESL 2009, tr.61: §3.3.4 *Prostate Cancer Data Example (Continued)*, mở đầu §3.4
*Shrinkage Methods* và dòng đầu §3.4.1.** T. Hastie, R. Tibshirani, J. Friedman, *The Elements of
Statistical Learning*, bản 2 (in lần 12, 2017), Springer, tr.61 (PDF trang 80), `_papers/2009 - Hastie,
Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf`.

**Khung chứa gì.** Văn xuôi; không có công thức hay hình.

**Tác giả nói gì.** *"Each method has a complexity parameter, and this was chosen to minimize an estimate
of prediction error based on tenfold cross-validation"* — rồi, chỉ vài dòng sau: *"We have used the
"one-standard-error" rule—we pick the most parsimonious model within one standard error of the minimum
(Section 7.10, page 244)."* Về §3.4: *"Shrinkage methods are more continuous, and don't suffer as much
from high variability."*

(Mỗi phương pháp có một tham số độ phức tạp, chọn để cực tiểu hoá một ước lượng sai số dự đoán bằng kiểm
định chéo 10 mảnh. Sách dùng quy tắc một sai số chuẩn: chọn mô hình tiết kiệm nhất trong phạm vi một
sai số chuẩn của cực tiểu. Phương pháp co liên tục hơn chọn tập con, và không bị phương sai cao nhiều bằng.)

**Cách đọc.** Hai câu đầu **mâu thuẫn nhẹ ngay trong một đoạn**: câu thứ nhất nói tham số được chọn "để cực
tiểu hoá" sai số, câu thứ hai nói thực ra chọn theo quy tắc một sai số chuẩn — tức **không** ở cực tiểu. Câu
thứ hai là câu đúng với việc sách làm; mục 3.15 đo điều đó bằng chính Hình 3.7.

**Điều khung KHÔNG nói.** Cách chia 10 mảnh không in, hạt giống không in. Nên đường cong kiểm định chéo
của sách **không dựng lại được từng điểm**; chỉ dựng lại được phân phối của nó qua nhiều cách chia.

### 3.15 ESL 2009, tr.62–63 — Hình 3.7, Bảng 3.3, công thức (3.41)–(3.42)

![ESL 2009 tr.62 — Hình 3.7](<paper_ridge_esl2009_tr62_hinh3_7.png>)

**Hình 29 — Hình 3.7 của ESL 2009, tr.62 (PDF trang 81).**

**Caption gốc.** *"Estimated prediction error curves and their standard errors for the various selection
and shrinkage methods. Each curve is plotted as a function of the corresponding complexity parameter for
that method. The horizontal axis has been chosen so that the model complexity increases as we move from
left to right. The estimates of prediction error and their standard errors were obtained by tenfold
cross-validation; full details are given in Section 7.10. The least complex model within one standard error
of the best is chosen, indicated by the purple vertical broken lines."*

(Đường sai số dự đoán ước lượng kèm sai số chuẩn cho các phương pháp chọn và co. Mỗi đường vẽ theo tham số
độ phức tạp của phương pháp đó, trục ngang chọn sao cho độ phức tạp tăng từ trái sang phải. Ước lượng và
sai số chuẩn lấy từ kiểm định chéo 10 mảnh. Mô hình ít phức tạp nhất trong phạm vi một sai số chuẩn của mô
hình tốt nhất được chọn, đánh dấu bằng vạch đứt đứng màu tím.)

**Tác giả nói gì về hình.** Tr.61: *"The estimated prediction error curves are shown in Figure 3.7. Many of
the curves are very flat over large ranges near their minimum."* Và tr.68: *"In Figure 3.7 the minimum
occurs at df(λ) = 5.0."*

(Nhiều đường rất phẳng trên một khoảng rộng quanh cực tiểu. — Ở Hình 3.7 cực tiểu nằm tại df = 5,0.)

**Cách đọc hình.** Chỉ bảng *"Ridge Regression"* thuộc bài này. Trục ngang là $\operatorname{df}(\lambda)$
từ 0 (phạt vô cùng, mô hình chỉ còn trung bình) tới 8 (không phạt, bình phương tối thiểu). Chấm cam là sai
số kiểm định chéo, thanh xanh là một sai số chuẩn, vạch ngang tím là "cực tiểu + một sai số chuẩn", vạch
đứng tím là mô hình được chọn.

**Chỗ tự mâu thuẫn — "the minimum occurs at df(λ) = 5.0".** Hình 3.7 là đồ thị **vectơ** trong PDF, nên toạ
độ đọc thẳng được từ lệnh vẽ, không qua điểm ảnh. Hiệu chuẩn bằng vị trí 5 nhãn trục ngang và 7 nhãn trục
dọc ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 16):

| df | −0,005 | 1,001 | 1,988 | 3,022 | 3,991 | **4,996** | 6,021 | **7,146** | 7,995 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CV đọc được | 1,4765 | 1,0635 | 0,8401 | 0,7156 | 0,6465 | **0,6030** | 0,5734 | **0,5536** | 0,5575 |

| Đọc từ hình | Giá trị |
|---|---:|
| vạch tím **đứng** | df = 5,001 |
| vạch tím **ngang** | CV = 0,6332 |
| cực tiểu của đường | df = 7,146, CV = 0,5536 |
| vạch ngang − cực tiểu = một sai số chuẩn | 0,0796 |
| điểm nhỏ nhất có CV ≤ vạch ngang | df = 4,996 |

Cực tiểu nằm ở **df ≈ 7**. Điểm df = 5 là điểm **đầu tiên** chui xuống dưới vạch "cực tiểu + một sai số
chuẩn" — đúng như chú thích hình và như tr.61 nói, và **khác** câu tr.68.

**Và phân phối qua các cách chia.** 500 cách chia 10 mảnh ngẫu nhiên, lưới df = 0…8, quy ước chuẩn hoá
khớp cột LS của Bảng 3.3 ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 15):

| df | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| argmin CV (số lần trên 500) | 0 | 0 | **0** | 14 | **420** | 66 |
| quy tắc một sai số chuẩn (số lần) | 78 | **377** | 44 | 0 | 0 | 0 |

df = 5 **không một lần nào** là cực tiểu. Quy tắc một sai số chuẩn thì chọn df = 4 thường nhất (377/500) và
df = 5 ở 44/500; cách chia của sách rơi vào nhóm thiểu số đó. Chỗ lệch giữa phép dựng lại và hình (4 so với
5) chỉ ra rằng **lựa chọn "df = 5" là một mẫu của một đại lượng ngẫu nhiên**, không phải một hằng số của dữ
liệu.

![ESL 2009 tr.63 — Bảng 3.3](<paper_ridge_esl2009_tr63_bang3_3.png>)

**Hình 30 — Bảng 3.3 của ESL 2009, tr.63 (PDF trang 82).**

**Caption gốc.** *"Estimated coefficients and test error results, for different subset and shrinkage
methods applied to the prostate data. The blank entries correspond to variables omitted."*

(Hệ số ước lượng và sai số kiểm tra cho các phương pháp chọn tập con và co, trên dữ liệu tuyến tiền liệt. Ô
trống là biến bị bỏ.)

**Tác giả nói gì về bảng.** Tr.68: *"Table 3.3 shows that ridge regression reduces the test error of the
full least squares estimates by a small amount."* (Bảng 3.3 cho thấy ridge giảm sai số kiểm tra của bình
phương tối thiểu đầy đủ **một lượng nhỏ**.)

**Cách đọc bảng.** Chỉ hai cột thuộc bài này: **LS** và **Ridge**. Hai dòng cuối là sai số kiểm tra —
trung bình bình phương sai số trên 30 mẫu kiểm tra — và sai số chuẩn của trung bình đó.

**Dựng lại cột LS — và phát hiện quy ước chuẩn hoá.** Sách viết (tr.50) *"after first standardizing the
predictors to have unit variance"* mà không nói chuẩn hoá trên tập nào. Thử bốn quy ước
([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 13):

| Quy ước chuẩn hoá | lệch làm tròn lớn nhất so với cột LS |
|---|---:|
| trên **97** mẫu, độ lệch chuẩn chia $n-1$ | **0,000** |
| trên 97 mẫu, chia $n$ | 0,004 |
| trên 67 mẫu huấn luyện, chia $n$ | 0,031 |
| trên 67 mẫu huấn luyện, chia $n-1$ | 0,036 |

Theo quy ước khớp, sai số kiểm tra LS = 0,521274 (in 0,521), sai số chuẩn 0,178724 (in 0,179), và mốc
*"base error rate"* tr.51 — dự đoán mọi bệnh nhân bằng trung bình huấn luyện — là 1,056733 (in 1,057). **Cột
LS khớp cả 11 ô.**

**Dựng lại cột Ridge — không quy ước nào khớp cả 11 ô.** Tìm $\lambda$ sao cho $\operatorname{df}(\lambda)=5{,}0$
theo (3.50), rồi so từng ô (`v_ridge.py` mục 14):

| Quy ước | ô khớp sau làm tròn | lệch lớn nhất 8 hệ số | sai số kiểm tra | sai số chuẩn |
|---|---:|---:|---:|---:|
| trung bình 97 · độ lệch chuẩn 97 | 3/9 | 0,00202 | 0,490361 | 0,162269 |
| trung bình 67 · độ lệch chuẩn 97 | 4/9 | 0,00202 | 0,490361 | 0,162269 |
| trung bình 67 · độ lệch chuẩn 67, chia $n-1$ | 1/9 | 0,01593 | **0,492499** | 0,161677 |
| trung bình 67 · độ lệch chuẩn 67, chia $n$ | 2/9 | 0,01397 | **0,492499** | 0,161677 |
| **in trong sách** | — | — | 0,492 | 0,165 |

Ba điều đọc ra được:

1. **Hệ số chặn in 2,452 bằng đúng trung bình $y$ của 67 mẫu huấn luyện** (2,452345), **không** bằng hệ số
   chặn LS 2,465 của cùng bảng. Tức hai cột dùng hai cách trừ trung bình khác nhau.
2. **Tám hệ số** khớp quy ước 97 mẫu trong 0,00202; **sai số kiểm tra** 0,492 chỉ khớp quy ước 67 mẫu.
3. **Sai số chuẩn 0,165** không khớp quy ước nào (0,1623 hoặc 0,1617). Quét df từ 4,8 tới 5,2 cũng không cho
   quá 3/9 ô hệ số khớp.

Kết luận trung thực: cột Ridge của Bảng 3.3 **không dựng lại được tới chữ số in** từ dữ liệu công khai; nó
dựng lại được **trong 0,002** ở hệ số và **trong 0,002** ở sai số kiểm tra, nhưng không với cùng một quy ước.

**Kiểm câu "reduces … by a small amount".** Trên 30 mẫu kiểm tra, sai số LS trừ sai số Ridge có trung bình
0,030913 và sai số chuẩn của hiệu (ghép cặp) 0,033824 — tỉ số 0,9139. Ridge tốt hơn LS ở 16/30 bệnh nhân.
Câu của sách đúng về chiều và đúng khi nói *"small"*: **mức giảm nhỏ hơn một sai số chuẩn**, nên trên 30 mẫu
không phân biệt được với không giảm.

![ESL 2009 tr.63 — công thức (3.41) và (3.42)](<paper_ridge_esl2009_tr63_cong_thuc_3_41_3_42.png>)

**Hình 31 — ESL 2009, tr.63: công thức (3.41) và (3.42), và đoạn về tương quan và chuẩn hoá.**

$$\hat\beta^{\text{ridge}}=\underset{\beta}{\operatorname{argmin}}\Big\{\sum_{i=1}^N\Big(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\Big)^2+\lambda\sum_{j=1}^p\beta_j^2\Big\} \tag{3.41}$$

$$\hat\beta^{\text{ridge}}=\underset{\beta}{\operatorname{argmin}}\sum_{i=1}^N\Big(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\Big)^2\quad\text{subject to}\quad\sum_{j=1}^p\beta_j^2\le t \tag{3.42}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $N$ | số mẫu — bài này viết $n$ | số nguyên |
| $x_{ij}$ | giá trị cột $j$ của mẫu $i$ | số |
| $\beta_0$ | hệ số chặn — **không** nằm trong khoản phạt | số |
| $\lambda$ | mức phạt — **không** phải trị riêng $\lambda_i$ của Hoerl–Kennard | số $\ge0$ |
| $t$ | bán kính bình phương của ràng buộc | số $>0$ |

**Tác giả nói gì.** *"Here λ ≥ 0 is a complexity parameter that controls the amount of shrinkage: the
larger the value of λ, the greater the amount of shrinkage. The coefficients are shrunk toward zero (and
each other). The idea of penalizing by the sum-of-squares of the parameters is also used in neural
networks, where it is known as weight decay (Chapter 11)."* Và: *"A wildly large positive coefficient on
one variable can be canceled by a similarly large negative coefficient on its correlated cousin."* Và:
*"The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the
inputs before solving (3.41)."*

(λ là tham số độ phức tạp điều khiển lượng co: càng lớn thì co càng mạnh. Hệ số bị co về 0 — và về nhau. Ý
tưởng phạt tổng bình phương tham số cũng dùng trong mạng nơ-ron dưới tên *weight decay*. Một hệ số dương lớn
vô lý trên một biến có thể bị triệt tiêu bằng một hệ số âm lớn tương tự trên biến "anh em" tương quan với
nó. Nghiệm ridge không đổi đúng theo phép đổi thang đầu vào, nên người ta thường chuẩn hoá đầu vào trước.)

**Ý nghĩa.** (3.41) là (2) của Tikhonov với $A$ là ma trận và $\Omega=\lVert\beta\rVert^2$; (3.42) là (3.7)
của Hoerl–Kennard. Câu *"canceled by … its correlated cousin"* là cơ chế của bảng 400 mẫu ở mục 1.2: $+42$
trên cột này bù $-40$ trên cột kia. Câu cuối là lời cảnh báo mà chính hai trang sau của sách không giữ được
nhất quán — mục 3.17.

### 3.16 ESL 2009, tr.64 — công thức (3.43)–(3.45), hệ số chặn và cách nhìn Bayes

![ESL 2009 tr.64 — (3.43) tới (3.45)](<paper_ridge_esl2009_tr64_cong_thuc_3_43_3_45.png>)

**Hình 32 — ESL 2009, tr.64: hệ số chặn, công thức (3.43)–(3.44), đầu vào trực chuẩn, cách nhìn Bayes, và
(3.45).**

$$\operatorname{RSS}(\lambda)=(\mathbf y-\mathbf X\beta)^T(\mathbf y-\mathbf X\beta)+\lambda\beta^T\beta \tag{3.43}$$
$$\hat\beta^{\text{ridge}}=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y \tag{3.44}$$
$$\mathbf X=\mathbf U\mathbf D\mathbf V^T \tag{3.45}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathbf X$ | ma trận đầu vào **đã trừ trung bình từng cột**, không có cột hệ số chặn | $N\times p$ |
| $\mathbf U$ | cột trực chuẩn, sinh không gian cột của $\mathbf X$ | $N\times p$ |
| $\mathbf D$ | đường chéo các giá trị kỳ dị $d_1\ge\dots\ge d_p\ge0$ | $p\times p$ |
| $\mathbf V$ | ma trận trực giao, sinh không gian dòng | $p\times p$ |

**Tác giả nói gì.** Về hệ số chặn: *"Penalization of the intercept would make the procedure depend on the
origin chosen for Y; that is, adding a constant c to each of the targets yi would not simply result in a
shift of the predictions by the same amount c."* Về (3.44): *"This makes the problem nonsingular, even if
XTX is not of full rank, and was the main motivation for ridge regression when it was first introduced in
statistics (Hoerl and Kennard, 1970)."* Về đầu vào trực chuẩn: *"In the case of orthonormal inputs, the
ridge estimates are just a scaled version of the least squares estimates, that is, β̂ridge = β̂/(1 + λ)."* Về
Bayes: *"Thus the ridge estimate is the mode of the posterior distribution; since the distribution is
Gaussian, it is also the posterior mean."*

(Phạt hệ số chặn làm thủ tục phụ thuộc gốc của $Y$. Phép cộng $\lambda$ vào đường chéo làm bài toán không suy
biến kể cả khi $X^TX$ không đủ hạng — và đó là động cơ chính khi ridge ra đời trong thống kê. Với đầu vào
trực chuẩn, ridge chỉ là bình phương tối thiểu chia $1+\lambda$. Ước lượng ridge là mode của hậu nghiệm; vì
hậu nghiệm Gauss, nó cũng là trung bình.)

**Kiểm bằng số.** Đầu vào trực chuẩn $40\times5$, $\lambda=0{,}7$: (3.44) so với $\hat\beta/(1+\lambda)$ lệch
$2{,}220\cdot10^{-16}$ (`v_ridge.py` mục 14). Trung bình hậu nghiệm so với ridge: lệch $3{,}109\cdot10^{-15}$ (mục 12).

**Ý nghĩa — và một chỗ ESL đọc Hoerl–Kennard khác chính họ.** ESL gọi động cơ chính của ridge là **làm
bài toán không suy biến**. Tóm tắt của Hoerl–Kennard (Hình 11) nói động cơ là **sai số bình phương trung
bình nhỏ hơn**, và định lý chính của họ (4.3) **giả định** $X'X$ đủ hạng ($\lambda_p>0$ ở (1.7)). Hai động
cơ khác nhau: thứ nhất là chuyện số học, thứ hai là chuyện thống kê. Bài báo gốc lo chuyện thứ hai; chuyện
thứ nhất là của Tikhonov.

### 3.17 ESL 2009, tr.65 — Hình 3.8: vết ridge trên dữ liệu thật, và quy ước chuẩn hoá khác Bảng 3.3

![ESL 2009 tr.65 — Hình 3.8](<paper_ridge_esl2009_tr65_hinh3_8.png>)

**Hình 33 — Hình 3.8 của ESL 2009, tr.65 (PDF trang 84).**

**Caption gốc.** *"Profiles of ridge coefficients for the prostate cancer example, as the tuning parameter
λ is varied. Coefficients are plotted versus df(λ), the effective degrees of freedom. A vertical line is
drawn at df = 5.0, the value chosen by cross-validation."*

(Đường đi của hệ số ridge cho ví dụ tuyến tiền liệt khi thay đổi λ. Hệ số vẽ theo df(λ), bậc tự do hiệu
dụng. Vạch đứng vẽ ở df = 5,0, giá trị chọn bằng kiểm định chéo.)

**Tác giả nói gì về hình.** Tr.64: *"Figure 3.8 shows the ridge coefficient estimates for the prostate
cancer example, plotted as functions of df(λ), the effective degrees of freedom implied by the penalty λ
(defined in (3.50) on page 68)."*

**Cách đọc hình.** Đây là **vết ridge** của Hoerl–Kennard, 39 năm sau, trên dữ liệu thật — với một thay đổi:
trục ngang là $\operatorname{df}(\lambda)$ thay vì $k$, nên phạt mạnh ở **trái** và không phạt ở **phải**. Mỗi
đường là một biến; đầu mút phải là hệ số bình phương tối thiểu; mọi đường hội tụ về 0 ở trái. Đường
`lcp` đổi dấu gần vạch đứng: ở df = 5 hệ số dựng lại là −0,00009 (`v_ridge.py` mục 14), nên Bảng 3.3 in 0,000.

**Đọc toạ độ vectơ và phát hiện hình dùng quy ước khác bảng.** Hình có 200 chấm xanh (8 biến × 25 điểm). Lấy
8 chấm ở df lớn nhất (df = 7,9981), gán cho biến theo thứ tự nhãn chữ in cạnh đầu mút, rồi so với hệ số
bình phương tối thiểu theo ba quy ước chuẩn hoá (`v_ridge.py` mục 16):

| Biến | chấm cuối trong hình | LS chuẩn hoá trên 97 mẫu (khớp Bảng 3.3) | LS chuẩn hoá trên 67 mẫu, chia $n-1$ |
|---|---:|---:|---:|
| lcavol | 0,7165 | 0,6795 | 0,7164 |
| lweight | 0,2931 | 0,2631 | 0,2926 |
| age | −0,1422 | −0,1415 | −0,1425 |
| lbph | 0,2124 | 0,2101 | 0,2120 |
| svi | 0,3098 | 0,3052 | 0,3096 |
| lcp | −0,2886 | −0,2885 | −0,2890 |
| gleason | −0,0204 | −0,0213 | −0,0209 |
| pgg45 | 0,2775 | 0,2670 | 0,2773 |
| **lệch lớn nhất** | — | **0,0369** | **0,0005** |

Và dọc toàn bộ đường (184 chấm ở giữa, so với đường ridge dựng lại tại đúng df của từng chấm):

| Quy ước | lệch trung vị | lệch lớn nhất |
|---|---:|---:|
| 97 mẫu, chia $n-1$ | 0,0029 | 0,0311 |
| 67 mẫu, chia $n$ | 0,0011 | 0,0053 |
| **67 mẫu, chia $n-1$** | **0,0004** | **0,0008** |

**Hình 3.8 chuẩn hoá trên 67 mẫu huấn luyện; cột LS của Bảng 3.3, cách hai trang, chuẩn hoá trên cả 97
mẫu.** Chênh lệch lớn nhất ở `lcavol`: 0,7165 trong hình so với 0,680 trong bảng, cho **cùng một** mô hình
bình phương tối thiểu. Không quy ước nào sai — chuẩn hoá chỉ trên tập huấn luyện thậm chí là cách đúng hơn,
vì không để tập kiểm tra rò vào — nhưng người đọc đặt bảng cạnh hình sẽ thấy hai con số khác nhau cho cùng
một hệ số mà sách không giải thích. Đó chính là tính *"not equivariant under scaling"* mà tr.63 cảnh báo, ở
dạng cụ thể nhất.

**Điều hình KHÔNG nói.** Vạch đứng ở df = 5,0 được gọi là *"the value chosen by cross-validation"*, không nói
theo quy tắc nào; mục 3.15 đã cho thấy đó là quy tắc một sai số chuẩn, không phải cực tiểu.

### 3.18 ESL 2009, tr.66–68 — (3.46)–(3.50), Hình 3.9: ridge co theo từng hướng chính

![ESL 2009 tr.66 — (3.46) tới (3.49)](<paper_ridge_esl2009_tr66_cong_thuc_3_46_3_49.png>)

**Hình 34 — ESL 2009, tr.66: công thức (3.46)–(3.49).**

$$\mathbf X\hat\beta^{\text{ls}}=\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y=\mathbf U\mathbf U^T\mathbf y \tag{3.46}$$

$$\mathbf X\hat\beta^{\text{ridge}}=\mathbf X(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y=\mathbf U\mathbf D(\mathbf D^2+\lambda\mathbf I)^{-1}\mathbf D\mathbf U^T\mathbf y=\sum_{j=1}^p\mathbf u_j\frac{d_j^2}{d_j^2+\lambda}\mathbf u_j^T\mathbf y \tag{3.47}$$

$$\mathbf X^T\mathbf X=\mathbf V\mathbf D^2\mathbf V^T \tag{3.48}$$

$$\operatorname{Var}(\mathbf z_1)=\operatorname{Var}(\mathbf X v_1)=\frac{d_1^2}{N} \tag{3.49}$$

| Ký hiệu | Nghĩa |
|---|---|
| $\mathbf u_j$ | cột $j$ của $\mathbf U$ — một hướng trực chuẩn trong không gian mẫu |
| $\mathbf u_j^T\mathbf y$ | toạ độ của $\mathbf y$ theo hướng đó |
| $d_j^2/(d_j^2+\lambda)$ | **hệ số co** của hướng $j$ — bằng $\xi_i(Z)$ (2.6) của Hoerl–Kennard với $\lambda_i=d_j^2$, $k=\lambda$ |
| $v_j$, $\mathbf z_j$ | hướng thành phần chính và biến thành phần chính thứ $j$ |

**Tác giả nói gì.** *"Like linear regression, ridge regression computes the coordinates of y with respect to
the orthonormal basis U. It then shrinks these coordinates by the factors d²j/(d²j + λ). This means that a
greater amount of shrinkage is applied to the coordinates of basis vectors with smaller d²j."*

(Như hồi quy tuyến tính, ridge tính toạ độ của $y$ theo cơ sở trực chuẩn $U$; rồi co các toạ độ đó theo hệ
số $d_j^2/(d_j^2+\lambda)$. Tức hướng có $d_j^2$ nhỏ bị co mạnh hơn.)

**Kiểm bằng số.** Trên dữ liệu tuyến tiền liệt, 67 mẫu, df = 5,0, $\lambda=23{,}998908$ (`v_ridge.py` mục 14):

| Hướng $j$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $d_j$ | 15,374 | 10,8574 | 8,4546 | 6,45 | 5,8238 | 5,1405 | 4,3205 | 3,4483 |
| $d_j^2/(d_j^2+\lambda)$ | 0,9078 | 0,8309 | 0,7486 | 0,6342 | 0,5856 | 0,5241 | 0,4375 | 0,3313 |

Tổng tám hệ số co đúng bằng 5,0 — đó là định nghĩa (3.50). Nghiệm (3.44) so với dạng SVD (3.47): lệch
$2{,}776\cdot10^{-16}$.

**Ý nghĩa.** (3.47) là (2.6) của Hoerl–Kennard viết cho **dự đoán** thay vì cho **hệ số**. Hai nguồn cách
nhau 39 năm nói cùng một câu: ridge không co đều, nó co theo hướng, và mức co của một hướng chỉ phụ thuộc
dữ liệu trải bao nhiêu theo hướng đó so với $\lambda$. Trên dữ liệu thật, hướng mạnh nhất giữ 90,78 %, hướng
yếu nhất chỉ giữ 33,13 %.

![ESL 2009 tr.67 — Hình 3.9](<paper_ridge_esl2009_tr67_hinh3_9.png>)

**Hình 35 — Hình 3.9 của ESL 2009, tr.67 (PDF trang 86).**

**Caption gốc.** *"Principal components of some input data points. The largest principal component is the
direction that maximizes the variance of the projected data, and the smallest principal component minimizes
that variance. Ridge regression projects y onto these components, and then shrinks the coefficients of the
low–variance components more than the high-variance components."*

(Thành phần chính của một số điểm dữ liệu. Thành phần chính lớn nhất là hướng làm phương sai của dữ liệu
chiếu lớn nhất, thành phần nhỏ nhất làm nó nhỏ nhất. Ridge chiếu $y$ lên các thành phần này rồi co hệ số của
thành phần phương sai thấp mạnh hơn thành phần phương sai cao.)

**Tác giả nói gì về hình.** Tr.67: *"Ridge regression protects against the potentially high variance of
gradients estimated in the short directions. The implicit assumption is that the response will tend to vary
most in the directions of high variance of the inputs. This is often a reasonable assumption, since
predictors are often chosen for study because they vary with the response variable, but need not hold in
general."*

(Ridge bảo vệ khỏi phương sai cao của độ dốc ước lượng theo các hướng ngắn. Giả định ngầm là đáp án thay đổi
nhiều nhất theo các hướng mà đầu vào trải rộng nhất. Đó thường là giả định hợp lý, nhưng **không nhất thiết
đúng**.)

**Cách đọc hình.** Đám mây điểm dẹt theo đường chéo. Đường tím dài là hướng dữ liệu trải rộng — độ dốc của
một mặt phẳng khớp theo hướng này ước lượng được chính xác. Đường tím ngắn là hướng dữ liệu gần như không
trải — độ dốc theo hướng này ước lượng rất kém, và ridge co nó mạnh nhất.

**Điều hình KHÔNG nói.** Câu cuối của đoạn bình luận là **giới hạn thật** của ridge: nếu đáp án phụ thuộc
mạnh vào đúng hướng yếu nhất, ridge co mất tín hiệu thật. Hình không cho dữ liệu đáp án, nên không thấy được
trường hợp đó. Hoerl–Kennard §4c nói cùng ý bằng ngôn ngữ khác: cực tiểu dời về $k=0$ khi $\alpha_i$ lớn.

![ESL 2009 tr.68 — (3.50)](<paper_ridge_esl2009_tr68_cong_thuc_3_50.png>)

**Hình 36 — ESL 2009, tr.68: công thức (3.50) và đoạn kết §3.4.1.**

$$\operatorname{df}(\lambda)=\operatorname{tr}[\mathbf X(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T]=\operatorname{tr}(\mathbf H_\lambda)=\sum_{j=1}^p\frac{d_j^2}{d_j^2+\lambda} \tag{3.50}$$

| Ký hiệu | Nghĩa |
|---|---|
| $\mathbf H_\lambda$ | **ma trận mũ** của ridge: biến $\mathbf y$ thành $\hat{\mathbf y}$ |
| $\operatorname{tr}$ | vết |

**Tác giả nói gì.** *"The idea is that although all p coefficients in a ridge fit will be non-zero, they
are fit in a restricted fashion controlled by λ. Note that df(λ) = p when λ = 0 (no regularization) and
df(λ) → 0 as λ → ∞. Of course there is always an additional one degree of freedom for the intercept, which
was removed apriori."*

(Dù cả $p$ hệ số của ridge đều khác 0, chúng bị khớp theo một cách hạn chế do λ điều khiển. df = $p$ khi
λ = 0 và df → 0 khi λ → ∞. Luôn có thêm một bậc tự do cho hệ số chặn, đã tách ra từ trước.)

**Ý nghĩa.** (3.50) đổi núm điều khiển $\lambda$ — một con số không có đơn vị trực quan, phụ thuộc thang đo —
thành một con số **đọc được**: mô hình linh hoạt như bao nhiêu tham số tự do. Đó là lý do trục ngang Hình 3.7
và 3.8 là df chứ không phải λ: nó so được giữa các phương pháp. Với bình phương tối thiểu $\mathbf H$ là phép
chiếu và vết của nó bằng hạng; với ridge $\mathbf H_\lambda$ không còn là phép chiếu, và vết của nó là số thực.

### 3.19 ESL 2009, tr.71 — Bảng 3.4 và Hình 3.11: ridge đặt cạnh lasso và chọn tập con

![ESL 2009 tr.71 — Bảng 3.4 và Hình 3.11](<paper_ridge_esl2009_tr71_bang3_4_hinh3_11.png>)

**Hình 37 — Bảng 3.4 và Hình 3.11 của ESL 2009, tr.71 (PDF trang 90).**

**Caption gốc của Bảng 3.4.** *"Estimators of βj in the case of orthonormal columns of X. M and λ are
constants chosen by the corresponding techniques; sign denotes the sign of its argument (±1), and x+ denotes
"positive part" of x. Below the table, estimators are shown by broken red lines. The 45° line in gray shows
the unrestricted estimate for reference."*

**Caption gốc của Hình 3.11.** *"Estimation picture for the lasso (left) and ridge regression (right). Shown
are contours of the error and constraint functions. The solid blue areas are the constraint regions |β1| +
|β2| ≤ t and β²1 + β²2 ≤ t², respectively, while the red ellipses are the contours of the least squares error
function."*

(Bảng 3.4: ước lượng của $\beta_j$ khi các cột của $X$ trực chuẩn; dưới bảng, các ước lượng vẽ bằng đường đứt
đỏ, đường xám 45° là ước lượng không ràng buộc. Hình 3.11: bức tranh ước lượng của lasso (trái) và ridge
(phải); vùng xanh là miền ràng buộc, ellipse đỏ là đường đồng mức của sai số bình phương tối thiểu.)

**Tác giả nói gì.** Tr.69: *"Ridge regression does a proportional shrinkage."* Tr.71–72: *"Both methods find
the first point where the elliptical contours hit the constraint region. Unlike the disk, the diamond has
corners; if the solution occurs at a corner, then it has one parameter βj equal to zero."*

(Ridge co **theo tỉ lệ**. Cả hai phương pháp tìm điểm đầu tiên mà ellipse chạm miền ràng buộc. Khác hình
tròn, hình thoi có góc; nếu nghiệm rơi vào góc thì một tham số bằng 0.)

**Cách đọc.** Bảng nhỏ giữa trang: đường đứt của ridge là **một đường thẳng qua gốc** có độ dốc $1/(1+\lambda)$
— không có đoạn nào nằm ngang ở 0. Hình dưới: ellipse đỏ là (3.1) của Hoerl–Kennard vẽ cho hai hệ số; hình
tròn xanh là ràng buộc (3.42). Hình tròn không có góc, nên điểm chạm hầu như không bao giờ nằm trên trục —
đó là lý do ridge không xoá biến.

**Điều khung KHÔNG nói.** Caption Hình 3.11 viết miền ridge là $\beta_1^2+\beta_2^2\le t^2$, trong khi (3.42) và
thân bài tr.72 viết $\le t$. Hai cách viết khác nhau ở ý nghĩa của $t$ (bán kính hay bán kính bình phương),
không đổi hình.

### 3.20 Bốn phép kiểm đặt cạnh nhau — hình tự vẽ

Bốn bảng số của mục 3.3, 3.10, 3.15 và 3.17 vẽ lên một hình, để thấy bằng mắt cái mà bảng nói bằng số.

![Bốn phép kiểm nguồn gốc bài Ridge](<ridge_dung_lai.png>)

**Hình 38 — Bốn phép kiểm lại ba nguồn gốc (hình tự vẽ).**

**Cách đọc hình.**

- **Trái trên — Tikhonov tr.503, câu "при α → 0 может выходить из ε-окрестности".** Sai số đều theo $\alpha$
  ở nhiễu cố định $\delta=10^{-3}$. Đường liền (có số hạng độ dốc, $k=1$) chạm đáy ở $\alpha=\delta^2$ rồi
  **đi lên**. Đường đứt là ridge thuần ($k=0$): cũng hình chữ U, đáy cao hơn 8,0647 lần. Vạch đứng là
  $\alpha=\delta^2$.
- **Phải trên — Hoerl–Kennard Hình 1, số hoá.** Chấm: phương sai, chệch², đường đứt đọc từ bản quét. Dấu
  cộng: tổng phương sai + chệch². Dấu cộng rơi đúng lên chấm của đường đứt — hình nhất quán.
- **Trái dưới — ESL Hình 3.7, bảng Ridge.** Toạ độ vectơ của chín chấm; vạch ngang là "cực tiểu + một sai số
  chuẩn" đọc từ hình. Chấm đỏ là cực tiểu (df ≈ 7), chấm xanh là điểm đầu tiên dưới vạch ngang (df ≈ 5).
- **Phải dưới — ESL Bảng 3.3 so với Hình 3.8.** Với mỗi biến: hệ số LS in trong bảng (vuông rỗng), đầu mút
  đường trong hình (tròn đặc). Ở `lcavol`, `lweight` và `pgg45` hai dấu tách hẳn nhau; trục dọc là hiệu
  "hình − bảng".

<details>
<summary><b>Code sinh Hình 38</b> — khối tự chứa, số liệu dán từ đầu ra <code>tools/v_ridge.py</code></summary>

```python
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

# v_ridge.py mục 4(b) và mục 6: sai số đều max|z − z̄| theo α, nhiễu δ = 1e-3
alpha_k1 = 10.0 ** -np.arange(2, 15)
sai_k1 = np.array([1.303721, 0.777038, 0.286015, 0.066032, 0.022421, 0.030404, 0.067798,
                   0.119298, 0.231674, 0.712750, 1.549022, 3.399091, 8.546764])
alpha_k0 = 10.0 ** -np.arange(2, 11)
sai_k0 = np.array([0.858877, 0.408916, 0.219624, 0.180816, 0.208411, 0.460691, 1.386116,
                   2.124955, 5.386052])

# v_ridge.py mục 17: số hoá Hình 1 của Hoerl & Kennard 1970
k_hk = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95])
phuong_sai = np.array([8.24, 6.11, 4.92, 4.12, 3.64, 3.04, 2.47])
chech2 = np.array([51.83, 61.85, 70.85, 78.61, 85.16, 92.84, 105.87])
net_dut = np.array([59.74, 68.03, 75.45, 82.62, 88.78, 96.09, 108.10])
muc_ls = 86.38

# v_ridge.py mục 16: toạ độ vectơ bảng Ridge của Hình 3.7 ESL
df_37 = np.array([-0.004968, 1.001, 1.988, 3.022, 3.991, 4.996, 6.021, 7.146, 7.995])
cv_37 = np.array([1.4765, 1.0635, 0.8401, 0.7156, 0.6465, 0.6030, 0.5734, 0.5536, 0.5575])
vach_ngang = 0.6332

# v_ridge.py mục 16 và Bảng 3.3 tr.63: hệ số LS theo biến
bien = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
bang_33 = np.array([0.680, 0.263, -0.141, 0.210, 0.305, -0.288, -0.021, 0.267])
hinh_38 = np.array([0.7165, 0.2931, -0.1422, 0.2124, 0.3098, -0.2886, -0.0204, 0.2775])

fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))

a = ax[0, 0]
a.loglog(alpha_k1, sai_k1, "o-", color="#2563eb", ms=4, lw=1.3, label="Tikhonov, k = 1")
a.loglog(alpha_k0, sai_k0, "s--", color="#dc2626", ms=4, lw=1.1, label="ridge, k = 0")
a.axvline(1e-6, color="#111827", lw=0.8)
a.invert_xaxis()
a.set_xlabel("α (giảm dần sang phải)")
a.set_ylabel("max |z − z̄|")
a.set_title("Tikhonov 1963 tr.503: δ = 1e-3 cố định, α → 0\nsai số chạm đáy ở α = δ² rồi tăng lại", fontsize=9)
a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(muc_ls, color="#6b7280", lw=1.0, label="Least Squares (86,38)")
a.plot(k_hk, phuong_sai, "o", color="#059669", ms=5, label="Variance")
a.plot(k_hk, chech2, "o", color="#d97706", ms=5, label="Bias-Squared")
a.plot(k_hk, net_dut, "o", color="#2563eb", ms=6, mfc="none", label="Ridge (nét đứt)")
a.plot(k_hk, phuong_sai + chech2, "+", color="#111827", ms=10, label="Variance + Bias²")
a.set_xlabel("k")
a.set_ylabel("đơn vị σ²")
a.set_title("Hoerl & Kennard 1970 Hình 1, số hoá bản quét\nđường đứt = tổng hai đường liền", fontsize=9)
a.legend(fontsize=7.5, loc="center right")

a = ax[1, 0]
a.plot(df_37, cv_37, "o-", color="#d97706", ms=5, lw=1.2)
a.axhline(vach_ngang, color="#7c3aed", lw=1.0, ls="--", label="cực tiểu + 1 SE (0,6332)")
jm = int(np.argmin(cv_37))
j1 = int(np.nonzero(cv_37 <= vach_ngang)[0][0])
a.plot(df_37[jm], cv_37[jm], "o", color="#dc2626", ms=9, label="cực tiểu: df = %.3f" % df_37[jm])
a.plot(df_37[j1], cv_37[j1], "o", color="#2563eb", ms=9, label="1-SE: df = %.3f" % df_37[j1])
a.set_xlabel("df(λ)")
a.set_ylabel("CV error")
a.set_title("ESL Hình 3.7 (Ridge): tr.68 viết 'minimum at df = 5.0'\ncực tiểu thật ở df ≈ 7, df = 5 là quy tắc 1-SE", fontsize=9)
a.legend(fontsize=7.5)

a = ax[1, 1]
xs = np.arange(len(bien))
a.axhline(0, color="#111827", lw=0.7)
a.bar(xs, hinh_38 - bang_33, color="#93c5fd", edgecolor="#1d4ed8")
a.set_xticks(xs)
a.set_xticklabels(bien, rotation=35, fontsize=8)
a.set_ylabel("đầu mút Hình 3.8 − cột LS Bảng 3.3")
a.set_title("ESL: cùng mô hình LS, hai quy ước chuẩn hoá\nHình 3.8 (67 mẫu) so với Bảng 3.3 (97 mẫu)", fontsize=9)

fig.tight_layout()
fig.savefig("ridge_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")

print("Tikhonov: α tốt nhất k=1 = %.0e (%.6f) ; k=0 = %.0e (%.6f) ; tỉ số %.4f"
      % (alpha_k1[np.argmin(sai_k1)], sai_k1.min(), alpha_k0[np.argmin(sai_k0)], sai_k0.min(),
         sai_k0.min() / sai_k1.min()))
print("Hoerl-Kennard: |nét đứt − tổng| lớn nhất = %.2f" % np.abs(net_dut - phuong_sai - chech2).max())
print("ESL Hình 3.7: cực tiểu df = %.3f ; 1-SE df = %.3f" % (df_37[jm], df_37[j1]))
print("ESL Hình 3.8 − Bảng 3.3: lệch lớn nhất %.4f ở %s"
      % (np.abs(hinh_38 - bang_33).max(), bien[int(np.argmax(np.abs(hinh_38 - bang_33)))]))
```

Khối này không tính lại gì: nó dán đúng các con số `tools/v_ridge.py` in ra ở mục 4, 6, 16 và 17, rồi vẽ.
Chạy dưới một giây; `tools/fig_ridge.py` là bản trích nguyên văn khối này.

</details>

---

## 4. Công thức và ký hiệu hiện đại

### 4.1 Bài toán

$$\hat\beta_{\text{ridge}}=\arg\min_\beta\ \lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_2^2$$

với $X$ **đã trừ trung bình từng cột** (và thường đã chia độ lệch chuẩn), $y$ đã trừ trung bình, hệ số chặn
ước lượng riêng bằng $\bar y-\bar x^\top\hat\beta$. Đây là (3.41)–(3.43) của ESL, (2) của Tikhonov với
$\Omega=\lVert\beta\rVert^2$.

### 4.2 Nghiệm, chệch và phương sai

$$\hat\beta_{\text{ridge}}=(X^\top X+\lambda I)^{-1}X^\top y$$

$$\operatorname{E}[\hat\beta]-\beta=-\lambda(X^\top X+\lambda I)^{-1}\beta,\qquad
\operatorname{Var}(\hat\beta)=\sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}$$

Công thức phương sai là (4.8) của Hoerl–Kennard viết lại: $Z(X^\top X)^{-1}Z^\top$ với
$Z=(X^\top X+\lambda I)^{-1}X^\top X$.

### 4.3 Dạng theo hướng riêng

Viết $X=UDV^\top$. Khi đó

$$\hat\beta_{\text{ridge}}=\sum_{j=1}^p v_j\,\frac{d_j}{d_j^2+\lambda}\,u_j^\top y,\qquad
\operatorname{df}(\lambda)=\sum_{j=1}^p\frac{d_j^2}{d_j^2+\lambda},$$

$$E\lVert\hat\beta-\beta\rVert^2=\underbrace{\sigma^2\sum_j\frac{d_j^2}{(d_j^2+\lambda)^2}}_{\gamma_1:\ \text{phương sai}}+\underbrace{\lambda^2\sum_j\frac{\alpha_j^2}{(d_j^2+\lambda)^2}}_{\gamma_2:\ \text{chệch}^2},\qquad\alpha_j=v_j^\top\beta.$$

Dòng cuối là (4.5) và (4.11) của Hoerl–Kennard với $\lambda_i=d_j^2$, $k=\lambda$.

### 4.4 Ba cách viết cùng một nghiệm

| Cách | Công thức | Dùng khi |
|---|---|---|
| phương trình chuẩn có phạt | $(X^\top X+\lambda I)\beta=X^\top y$, giải bằng Cholesky | $p$ nhỏ, $X^\top X$ lập được |
| thêm $p$ dòng giả | bình phương tối thiểu trên $\tilde X=\binom{X}{\sqrt\lambda I}$, $\tilde y=\binom{y}{0}$ | muốn độ ổn định số của QR, không lập $X^\top X$ |
| SVD | công thức mục 4.3 | cần nhiều giá trị $\lambda$: một lần phân rã dùng cho mọi $\lambda$ |

Cách thứ hai đọc ra một điều đáng nhớ: **ridge là bình phương tối thiểu trên dữ liệu có thêm $p$ mẫu giả**,
mỗi mẫu nói "hệ số này bằng 0" với trọng số $\sqrt\lambda$. Đó là (3.7) của Hoerl–Kennard — ràng buộc độ
dài — viết bằng dữ liệu.

---

## 5. Thuật toán từng bước

| Bước | Làm gì | Vào → ra | Bỏ đi thì hỏng chỗ nào |
|---|---|---|---|
| 1 | tách tập kiểm tra ra trước mọi thứ | $n$ mẫu → huấn luyện, kiểm tra | con số cuối cùng đo trên dữ liệu đã dùng để chọn, nên lạc quan giả |
| 2 | trên tập huấn luyện: trừ trung bình, chia độ lệch chuẩn từng cột; lưu hai vector đó | $X_{\text{tr}}$ → $X_{\text{tr}}$ chuẩn hoá | phạt không đều giữa các cột (mục 3.17 cho thấy kể cả sách giáo khoa cũng lẫn hai quy ước) |
| 3 | chọn lưới $\lambda$ theo thang log, rộng tới mức cực tiểu không nằm ở biên | → 50–100 giá trị | cực tiểu ở biên lưới nghĩa là chưa tìm thấy cực tiểu |
| 4 | kiểm định chéo $K$ mảnh trên tập huấn luyện; **chuẩn hoá lại trong từng mảnh** | → đường CV theo $\lambda$ | rò rỉ: thông tin của mảnh kiểm định lọt vào phép chuẩn hoá |
| 5 | chọn $\lambda$: cực tiểu, hoặc quy tắc một sai số chuẩn | → một $\lambda$ | — |
| 6 | khớp lại trên toàn tập huấn luyện với $\lambda$ đó: SVD hoặc Cholesky | → $\hat\beta$, $\hat\beta_0$ | — |
| 7 | trên tập kiểm tra: áp **đúng** trung bình và độ lệch chuẩn đã lưu ở bước 2, dự đoán, đo sai số | → một con số | lỗi im lặng: mô hình vẫn trả số, và số đó sai |

**Cài đặt từ đầu, hai cách, và phép so.**

```python
import numpy as np


def ridge_closed(X, y, lam):
    """Ridge có nghiệm đóng: giải (XᵀX + λI) β = Xᵀy."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def ridge_augmented(X, y, lam):
    """Bình phương tối thiểu trên dữ liệu có thêm p dòng giả.
    Giữ được độ ổn định của QR vì không bao giờ lập XᵀX."""
    n, p = X.shape
    Xa = np.vstack([X, np.sqrt(lam) * np.eye(p)])
    ya = np.r_[y, np.zeros(p)]
    return np.linalg.lstsq(Xa, ya, rcond=None)[0]


X = np.array([[1., 1.], [2., 2.], [3., 3.]])
y = np.array([2., 4., 6.])
a = ridge_closed(X, y, lam=1.0)
b = ridge_augmented(X, y, lam=1.0)
print("closed   =", np.round(a, 6))
print("augmented=", np.round(b, 6))
print("lệch lớn nhất =", np.max(np.abs(a - b)))
```

```text
closed   = [0.965517 0.965517]
augmented= [0.965517 0.965517]
lệch lớn nhất = 4.440892098500626e-16
```

Cả hai cho đúng $28/29=0{,}965517$ của mục 0.5. Cách thứ ba — vòng lặp toạ độ, dùng chung với bài
[Elastic Net](<../Elastic Net/Elastic Net.md>) — cũng khớp nghiệm đóng
([`tools/demo_shrink.py`](<../../../tools/demo_shrink.py>), 80 mẫu, 12 cột):

```
   ridge λ=0.1     vòng lặp toạ độ vs nghiệm đóng: lệch 2.073e-13
   ridge λ=1       vòng lặp toạ độ vs nghiệm đóng: lệch 3.858e-13
   ridge λ=10      vòng lặp toạ độ vs nghiệm đóng: lệch 1.381e-13
   ridge λ=100     vòng lặp toạ độ vs nghiệm đóng: lệch 4.105e-14
```

Chữ số cuối của hai dòng đầu và dòng ba **đổi theo máy**: bản chạy trên Windows năm 2026-08-31 in
2.074e-13 và 1.382e-13. Ba cài đặt khác nhau trùng nhau tới $10^{-13}$ là đủ để tin cả ba đúng; chữ số thứ
tư của một sai số cỡ $10^{-13}$ thì không mang thông tin.

**Dùng thư viện, trên dữ liệu thật.** Quy trình bảy bước ở trên, với `scikit-learn` và dữ liệu tuyến tiền
liệt của mục 3.14:

```python
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

# tools/prostate.csv: 97 bệnh nhân, cột 'train' chia 67/30 đúng như ESL tr.50
df = pd.read_csv("prostate.csv")
cot = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
tr = df["train"].astype(str) == "True"
X_tr, y_tr = df.loc[tr, cot].to_numpy(), df.loc[tr, "lpsa"].to_numpy()
X_te, y_te = df.loc[~tr, cot].to_numpy(), df.loc[~tr, "lpsa"].to_numpy()

luoi = np.logspace(-3, 3, 61)
mo_hinh = make_pipeline(
    StandardScaler(),                                  # học trung bình/độ lệch chuẩn trên 67 mẫu
    RidgeCV(alphas=luoi, cv=KFold(10, shuffle=True, random_state=0),
            scoring="neg_mean_squared_error"),
)
mo_hinh.fit(X_tr, y_tr)
ridge = mo_hinh[-1]
ls = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)

sai_r = (y_te - mo_hinh.predict(X_te)) ** 2
sai_l = (y_te - ls.predict(X_te)) ** 2
d = np.linalg.svd(mo_hinh[0].transform(X_tr) - mo_hinh[0].transform(X_tr).mean(0), compute_uv=False)

print("alpha chọn bằng CV 10 mảnh  = %.6f" % ridge.alpha_)
print("alpha nằm trong lưới [%.0e, %.0e], không chạm biên: %s"
      % (luoi[0], luoi[-1], luoi[0] < ridge.alpha_ < luoi[-1]))
print("df(alpha) = %.4f" % (d ** 2 / (d ** 2 + ridge.alpha_)).sum())
print("hệ số chặn = %.4f" % ridge.intercept_)
for t, b in zip(cot, ridge.coef_):
    print("  %-8s %+.4f" % (t, b))
print("sai số kiểm tra: Ridge %.6f (SE %.6f) ; LS %.6f (SE %.6f)"
      % (sai_r.mean(), sai_r.std(ddof=1) / np.sqrt(len(sai_r)),
         sai_l.mean(), sai_l.std(ddof=1) / np.sqrt(len(sai_l))))
```

Chạy trong `tools/` (cần `tools/prostate.csv`):

```text
alpha chọn bằng CV 10 mảnh  = 5.011872
alpha nằm trong lưới [1e-03, 1e+03], không chạm biên: True
df(alpha) = 6.9415
hệ số chặn = 2.4523
  lcavol   +0.6059
  lweight  +0.2842
  age      -0.1098
  lbph     +0.2003
  svi      +0.2828
  lcp      -0.1603
  gleason  +0.0125
  pgg45    +0.2049
sai số kiểm tra: Ridge 0.494124 (SE 0.163635) ; LS 0.521274 (SE 0.178724)
```

**Cách đọc.** `StandardScaler` nằm **trong** pipeline, nên nó học trung bình và độ lệch chuẩn trên 67 mẫu
huấn luyện và, trong kiểm định chéo, lại học lại trên từng 9/10 — đúng bước 2 và 4. `alpha` của
scikit-learn là $\lambda$ của bài này. Con số khác Bảng 3.3 vì cách chia 10 mảnh khác và lưới khác; mục 3.15
đã cho thấy lựa chọn này là một đại lượng ngẫu nhiên.

---

## 6. Vì sao chệch lại ăn thua — cơ chế và phép đo

**Cơ chế** (Hoerl–Kennard §4a, mục 3.10 của bài này). Tại $\lambda=0$, phương sai giảm với tốc độ
$2\sigma^2\sum1/d_j^4$ còn chệch² tăng với tốc độ **0**. Một hàm dốc xuống cộng một hàm phẳng thì dốc xuống,
nên luôn có một đoạn $\lambda>0$ nhỏ thắng bình phương tối thiểu. Tốc độ giảm lớn khi có $d_j$ nhỏ — tức
đúng khi các cột gần trùng nhau.

**Phép đo một-thay-đổi.** Giữ nguyên thiết kế, nhiễu, hệ số thật; chỉ đổi $\lambda$. Thiết kế 8 cột sinh từ 2
hướng, tương quan 0,99, $n=50$, 4 000 lần rút nhiễu:

![Vết ridge và phân rã chệch–phương sai](<ridge_trace.png>)

**Hình 39 — Vết ridge, phân rã chệch–phương sai và hệ số co theo từng hướng (hình tự vẽ).**

**Cách đọc hình.** Ba bảng từ trái sang:

- **Trái** — vết ridge theo định nghĩa của Hoerl–Kennard §3a. Ở $\lambda$ nhỏ các hệ số dao động mạnh và
  có hệ số **sai dấu**; khi $\lambda$ tăng chúng ổn định lại rồi cùng bò về 0. Đây là tiêu chí thứ nhất và
  thứ ba của Hoerl–Kennard tr.65 (*"the system will stabilize"*, *"coefficients with apparently incorrect
  signs … will have changed"*).
- **Giữa** — phân rã $E\lVert\hat\beta-\beta\rVert^2=\gamma_1+\gamma_2$ đo bằng Monte Carlo. Đường xanh lá
  (phương sai) giảm nhanh, đường đỏ (chệch²) tăng chậm, tổng có cực tiểu. Vạch đen là $k^*$. Đây là Hình 1
  của Hoerl–Kennard dựng trên một thiết kế cụ thể thay vì vẽ định tính.
- **Phải** — hệ số co $d_j^2/(d_j^2+\lambda)$ ở ba mức $\lambda$, đúng (2.6) và (3.47). Hướng có $d_j^2$ lớn gần
  như không bị đụng tới.

Đầu ra của khối mã:

```
κ(XᵀX) = 5.013e+02
OLS: MSE 7.2612 = chệch² 0.0007 + phương sai 7.2605
k* 0.6261: MSE 4.3271 = chệch² 2.0812 + phương sai 2.2459
k lớn nhất trên lưới còn thắng OLS: 4.238
```

<details>
<summary><b>Code sinh Hình 39</b> — khối tự chứa, copy và chạy trực tiếp</summary>

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


def collinear(n=100, p=8, rho=0.99, noise=1.0, seed=0):
    """Thiet ke gan suy bien - dung boi canh Hoerl & Kennard 1970 mo ta:
    8 bien sinh ra tu chi 2 huong doc lap, nen XtX gan suy bien."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    X = np.zeros((n, p))
    for j in range(p):
        X[:, j] = rho * Z[:, j % 2] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    beta = np.zeros(p); beta[:2] = [3.0, -2.0]
    return X, X @ beta + noise * rng.normal(size=n), beta


X, y, beta = collinear(n=50, p=8, rho=0.99, noise=1.0, seed=3)
n, p = X.shape
XtX, Xty = X.T @ X, X.T @ y

# --- mo phong: chech^2 va phuong sai cua ridge theo lambda, 4000 lan rut nhieu ---
lam = np.array(sorted(set(np.r_[0, np.logspace(-4, 3, 60)])))
rng = np.random.default_rng(11)
REP = 4000
BB = {i: [] for i in range(len(lam))}
for _ in range(REP):
    Xty_ = X.T @ (X @ beta + rng.normal(size=n))
    for i, l in enumerate(lam):
        BB[i].append(np.linalg.solve(XtX + l * np.eye(p), Xty_))
bias2 = np.array([((np.array(BB[i]).mean(0) - beta) ** 2).sum() for i in range(len(lam))])
var = np.array([np.array(BB[i]).var(0).sum() for i in range(len(lam))])
mse = np.array([((np.array(BB[i]) - beta) ** 2).sum(1).mean() for i in range(len(lam))])
ibest = int(np.argmin(mse))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

# bang trai: VET RIDGE - he so theo lambda, khop tren du lieu that (khong mo phong)
lam2 = np.logspace(-4, 4, 120)
Bp = np.array([np.linalg.solve(XtX + l * np.eye(p), Xty) for l in lam2])
for j in range(p):
    axes[0].semilogx(lam2, Bp[:, j], lw=1.2)
axes[0].axhline(0, color="#111827", lw=.7)
axes[0].set_xlabel("k (λ)"); axes[0].set_ylabel(r"$\hat\beta_j(k)$")
axes[0].set_title("Vết ridge — 8 biến, tương quan ρ = 0.99\n"
                  "(Hoerl & Kennard gọi là ridge trace)", fontsize=9)

# bang giua: phan ra MSE = chech^2 + phuong sai
axes[1].semilogx(lam[1:], mse[1:], color="#2563eb", lw=1.6, label="MSE")
axes[1].semilogx(lam[1:], bias2[1:], color="#dc2626", lw=1.1, ls="--", label="chệch²")
axes[1].semilogx(lam[1:], var[1:], color="#059669", lw=1.1, ls=":", label="phương sai")
axes[1].axhline(mse[0], color="#6b7280", lw=1.0, label="MSE của OLS")
axes[1].axvline(lam[ibest], color="#111827", lw=.8)
axes[1].set_yscale("log"); axes[1].set_xlabel("k (λ)"); axes[1].legend(fontsize=7.5)
axes[1].set_title("MSE = chệch² + phương sai, %d lần lặp\nk* = %.3g giảm MSE %.1f lần"
                  % (REP, lam[ibest], mse[0] / mse[ibest]), fontsize=9)

# bang phai: he so co theo tung huong rieng (muc 3.1)
s_ = np.linalg.svd(X, full_matrices=False)[1]
for l, col in [(0.1, "#2563eb"), (1.0, "#d97706"), (10.0, "#dc2626")]:
    axes[2].plot(s_ ** 2, s_ ** 2 / (s_ ** 2 + l), "o-", color=col, ms=4, lw=1.1,
                 label="λ = %g" % l)
axes[2].set_xscale("log"); axes[2].legend(fontsize=7.5)
axes[2].set_xlabel(r"$d_j^2$ (giá trị riêng của $X^\top X$)")
axes[2].set_ylabel(r"$d_j^2/(d_j^2+\lambda)$")
axes[2].set_title("Ridge co hướng nào? Hướng phương sai nhỏ\n"
                  "bị co mạnh, hướng lớn gần như giữ nguyên", fontsize=9)

fig.savefig("ridge_trace.png", dpi=130, bbox_inches="tight", facecolor="white")
better = lam[mse < mse[0]]
print("κ(XᵀX) = %.3e" % np.linalg.cond(XtX))
print("OLS: MSE %.4f = chệch² %.4f + phương sai %.4f"
      % (mse[0], bias2[0], var[0]))
print("k* %.4g: MSE %.4f = chệch² %.4f + phương sai %.4f"
      % (lam[ibest], mse[ibest], bias2[ibest], var[ibest]))
print("k lớn nhất trên lưới còn thắng OLS: %.4g" % better.max())
```

Kết quả Monte Carlo có thể mất vài giây đến vài phút tùy máy. Với seed và lưới trong code,
$k^*$ trên lưới phải trùng con số $0{,}6261$ ở khối đầu ra phía trên.

</details>

**Đối chiếu Monte Carlo với công thức (4.5).** Cùng thiết kế, tính thẳng $\gamma_1+\gamma_2$ bằng trị riêng —
không rút nhiễu, lưới 200 001 điểm ([`tools/v_ridge.py`](<../../../tools/v_ridge.py>) mục 8):

| Đại lượng | Monte Carlo, lưới 61 điểm | Công thức (4.5), lưới mịn |
|---|---:|---:|
| $E\lVert\hat\beta-\beta\rVert^2$ tại $\lambda=0$ | 7,2612 | 7,2896 |
| $\lambda$ tối ưu | 0,6261 | 0,5544 |
| $E\lVert\hat\beta-\beta\rVert^2$ tại tối ưu | 4,3271 | 4,3114 |
| phương sai / chệch² tại tối ưu | 2,2459 / 2,0812 | 2,4966 / 1,8148 |
| mức giảm | 1,7 lần | 1,6908 lần |
| đầu mút khoảng thắng bình phương tối thiểu | 4,238 | 4,6009 |
| cận đủ (4.16) $\sigma^2/\alpha^2_{\max}$ | — | 0,3080 |

Hai cột lệch nhau ở chữ số thứ hai, vì hai lý do đều thấy được: Monte Carlo 4 000 lần có sai số lấy mẫu,
và lưới 61 điểm theo log **không có** điểm 0,5544. Kết luận thì trùng: theo công thức, chệch² đi từ 0 lên
1,8148 trong khi phương sai đi từ 7,2896 xuống 2,4966 — **trả một ít chệch để lấy lại phần lớn phương sai** —
và khoảng $\lambda$ an toàn hữu hạn.

> **Điều hình KHÔNG nói.** Đầu mút 4,6 của khoảng thắng **không đọc được từ vết ridge**: vết ridge chỉ vẽ hệ
> số, không vẽ sai số, vì sai số cần biết $\beta$ thật. Chọn $\lambda$ quá lớn thì ridge tệ hơn bình phương
> tối thiểu, và tệ theo cách khó thấy vì hệ số vẫn "trông ổn định". Đó là lý do cách chọn bằng mắt của
> Hoerl–Kennard đã nhường chỗ cho kiểm định chéo (mục 7).

---

## 7. Chọn $\lambda$ như thế nào — ba nguồn, bốn cách

| Cách | Nguồn | Cần biết gì | Được | Mất |
|---|---|---|---|---|
| **vết ridge bằng mắt** — chọn $k$ nhỏ nhất mà hệ số đã ổn định, dấu đã hợp lý, tổng bình phương phần dư chưa phình | Hoerl–Kennard tr.65 | hiểu biết về quá trình sinh dữ liệu (*"proper sign"*) | thấy cấu trúc; không cần tách dữ liệu | chủ quan; không thấy được đầu mút của khoảng an toàn (§6) |
| **$k_i=\hat\sigma^2/\hat\alpha_i^2$ lặp** | Hoerl–Kennard tr.63, tr.65 | $\hat\sigma^2$ | tự động, mỗi hướng một mức | **không dừng** ở các hướng có $\lvert t\rvert<2$ — mục 3.13 |
| **$\alpha$ tỉ lệ $\delta^2$** | Tikhonov tr.502–503 | mức nhiễu $\delta$ | có bảo đảm hội tụ khi $\delta\to0$ | cần biết $\delta$; hằng số $\gamma_1,\gamma_2$ không được cho |
| **kiểm định chéo $K$ mảnh + quy tắc một sai số chuẩn** | ESL tr.61 | không | dựa trên sai số dự đoán, không cần giả định | là một đại lượng ngẫu nhiên: trên 500 cách chia, quy tắc 1-SE chọn df = 4 ở 377 lần, df = 5 ở 44 lần (mục 3.15) |

Cách thứ năm, đáng biết vì ít người dùng: với mô hình tuyến tính, sai số bỏ-một-ra tính được **không cần khớp
lại lần nào**:

$$\text{LOOCV}(\lambda)=\frac1n\sum_{i=1}^n\left(\frac{y_i-\hat y_i}{1-h_{ii}(\lambda)}\right)^2,$$

với $h_{ii}$ là phần tử đường chéo của $H_\lambda=X(X^\top X+\lambda I)^{-1}X^\top$ — ma trận mà vết của nó là
$\operatorname{df}(\lambda)$ ở (3.50). `RidgeCV` của scikit-learn dùng công thức này khi không truyền `cv`.

**Quy trình dùng thật.**

1. Tách tập kiểm tra trước. Pipeline gồm: điền thiếu, mã hoá → chuẩn hoá → ridge.
2. Dò $\lambda$ trên thang log; kiểm rằng giá trị chọn không nằm ở biên lưới.
3. Chọn bằng kiểm định chéo với thước đo đúng mục tiêu; nếu đường CV phẳng quanh cực tiểu (ESL tr.61:
   *"Many of the curves are very flat over large ranges near their minimum"*), quy tắc một sai số chuẩn cho
   mô hình co mạnh hơn mà không đo được là tệ hơn.
4. Khớp lại trên toàn tập huấn luyện, đánh giá tập kiểm tra **một lần**.
5. Báo cáo cả $\lambda$, quy ước chuẩn hoá (trên tập nào, chia $n$ hay $n-1$), và việc hệ số chặn có bị phạt
   hay không. Mục 3.15–3.17 cho thấy thiếu dòng này thì ngay cả một bảng in trong sách giáo khoa cũng không
   dựng lại được tới chữ số.

---

## 8. Giá phải trả — nó hỏng khi nào, và khi nào đừng dùng

### 8.1 Ba giới hạn thật

1. **Không tạo mô hình thưa.** Mọi hệ số khác 0. Với 10 000 gen và 200 bệnh nhân, ridge trả về gần đủ 10 000
   hệ số. Không có $\lambda$ nào sửa được điều đó.
2. **Nhạy với thang đo.** Đổi đơn vị một cột là đổi mức phạt lên cột đó. ESL tr.63: *"The ridge solutions are
   not equivariant under scaling of the inputs"*.
3. **Lợi ích phụ thuộc cái chưa biết.** Hoerl–Kennard §4c: cực tiểu dời về $\lambda=0$ khi $\beta^\top\beta$
   lớn. Mục 3.11 đo: $\beta^\top\beta$ tăng 1 000 lần thì mức giảm sai số co từ 357,3682 lần xuống 1,3850 lần.
   Và ESL tr.67: giả định ngầm là đáp án thay đổi theo hướng đầu vào trải rộng — *"need not hold in general"*.

### 8.2 Bảng giá

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Nguồn số |
|---|---|---|---|
| hệ số hết nhảy giữa các mẫu | độ lệch chuẩn $\hat\beta_1$: $14{,}27\to0{,}08$ | ước lượng có chệch; hệ số không còn là hiệu ứng riêng của từng biến | mục 1.2 · `v_gd1ss.py` |
| hết ngược dấu | 48% → 0% số lần chạy | — | mục 1.2 |
| sai số ước lượng nhỏ hơn | 41,7437 lần trên thiết kế 8 cột; 165 lần trên thiết kế 2 cột | phải chọn $\lambda$; chọn quá lớn thì tệ hơn bình phương tối thiểu | mục 3.11 · mục 1.2 |
| bài toán ngược giải được | sai số đều từ $9{,}438\cdot10^{9}$ xuống 0,022421 (Tikhonov) hoặc 0,180816 (ridge thuần) | thuần ridge kém dạng có số hạng độ dốc 8,0647 lần khi nghiệm trơn | mục 1.2 · mục 3.5 |
| sai số dự đoán trên dữ liệu thật | ESL tr.63: 0,521 → 0,492 | **nhỏ hơn một sai số chuẩn** (tỉ số 0,9139); không phân biệt được với không cải thiện trên 30 mẫu | mục 3.15 |
| chạy được khi $p>n$ hoặc cột trùng nhau | nghiệm duy nhất $28/29$ ở ví dụ mục 0.5 | kết quả phụ thuộc đơn vị đo từng cột | mục 0.5 |
| **tạo mô hình thưa** | **không được gì** — 0 hệ số bằng 0 | phải đổi sang Lasso | mục 8.3 |

Dòng cuối là dòng thua thật.

### 8.3 So với các cách khác, trên cùng dữ liệu

Ba cột gần trùng nhau (tương quan 0,9999), đóng góp thật chia đều
([`tools/demo_shrink.py`](<../../../tools/demo_shrink.py>)):

```
      Ridge   hệ số ba cột nhóm = [ 0.9505  0.9517  0.9531]   hội tụ sau    241 vòng
      ElNet   hệ số ba cột nhóm = [ 0.9497  0.9537  0.9520]   hội tụ sau    388 vòng
      Lasso   hệ số ba cột nhóm = [ 0.0000  2.2915  0.5577]   hội tụ sau 120718 vòng
```

Ridge chia gần đều — ba hệ số lệch nhau chưa tới 0,3 %. Lasso xoá hẳn một cột và chia phần còn lại 2,29 /
0,56 giữa hai cột gần như không phân biệt được. Đó là lý do thực tế nhất để chọn ridge khi các biến đi theo
nhóm. Số vòng lặp chỉ là của cài đặt vòng lặp toạ độ trong script, không phải tỉ lệ tốc độ phổ quát.

| Phương pháp | Phạt | Có hệ số bằng 0? | Với nhóm biến tương quan |
|---|---|---|---|
| [Bình phương tối thiểu](<../Linear Regression/Linear Regression.md>) | không | không | hệ số bất ổn |
| **Ridge** | $\lambda\lVert\beta\rVert_2^2$ | không | chia đều trong nhóm |
| [Lasso](<../Lasso Regression/Lasso Regression.md>) | $\lambda\lVert\beta\rVert_1$ | có | chọn một biến, không ổn định |
| [Elastic Net](<../Elastic Net/Elastic Net.md>) | cả hai | có | vừa thưa vừa chia nhóm |
| Tikhonov tổng quát | $\lambda\lVert L\beta\rVert_2^2$, $L$ là toán tử sai phân | không | ép hệ số liền kề giống nhau — mục 3.5 |
| Hoerl–Kennard tổng quát | $\beta^\top K\beta$, $K$ đường chéo | không | mỗi hướng riêng một mức — mục 3.12 |

### 8.4 Khi nào đừng dùng

Xem mục 0.8. Tóm lại: cần danh sách biến ngắn · cột đã gần trực giao và $n\gg p$ · cần diễn giải hệ số như
hiệu ứng riêng · dữ liệu có ngoại lai.

---

## 9. Ứng dụng thật

- **Bài toán ngược trong vật lý, địa vật lý, ảnh.** Khử mờ, chụp cắt lớp, tìm cấu trúc lòng đất từ số đo trên
  mặt — địa hạt gốc của Tikhonov (tr.501: *"Обратные задачи математической физики"*). Thường dùng
  $\lVert L\beta\rVert^2$ với $L$ là toán tử đạo hàm, đúng như $\Omega$ có số hạng $k(s)z'^2$ ở (2).
- **Dữ liệu công nghiệp, hoá chất.** Địa hạt gốc của Hoerl–Kennard: các đại lượng đo cùng một quá trình nên
  tương quan mạnh.
- **Tin sinh học, $p\gg n$.** Ridge cho mô hình dự đoán ổn định; muốn biết gen nào thì cần Lasso hoặc Elastic
  Net.
- **Học sâu.** *Weight decay* chính là phạt $\lambda\lVert w\rVert^2$ lên trọng số (ESL tr.63: *"also used in
  neural networks, where it is known as weight decay"*). Cẩn thận: với bộ tối ưu thích nghi như
  [Adam](<../../Optimization - Training/Adam/Adam.md>), cộng khoản phạt vào hàm mất mát **không** tương đương với trừ $\lambda w$ khỏi trọng số, vì
  gradient bị chia cho căn của mô-men bậc hai — nội dung của AdamW.
- **Kernel ridge regression.** Thay tích vô hướng bằng hạt nhân, ridge thành phương pháp phi tuyến, họ hàng
  gần của [SVR](<../Support Vector Regression (SVR)/Support Vector Regression (SVR).md>) — khác ở hàm mất mát.

---

## 10. Đối chiếu với nguồn gốc — đã lấy gì, bỏ gì, vì sao

### 10.1 Tikhonov 1963 — theo khối công thức và định lý

Bài báo **không có hình và không có bảng nào**. Bảng này thay bằng các khối công thức và định lý — phần ruột
của một bài toán học không có hình.

| Khối | Trang | Lấy? | Ở đâu |
|---|---|---|---|
| phương trình (1) và câu về độ trơn | tr.501 | **có** | Hình 1, mục 3.1 |
| chuẩn trong $\bar C_1$ và trong $U$ | tr.501 | có | mục 3.1 (bài dùng chuẩn $L_2$ tr.501 ở mọi phép đo) |
| ví dụ $z_2=z_1+p\cos\omega s$ | tr.501 | **có, dựng lại** | Hình 2, mục 3.1 |
| định nghĩa họ nghiệm chính quy hoá | tr.501 | **có** | Hình 2, mục 3.1 |
| công thức (2), $N$, $\Omega$ | tr.502 | **có** | Hình 3, mục 3.2 |
| định lý 1 | tr.502 | có | Hình 3, mục 3.2 |
| phương trình Euler (3), $\bar K$, $\bar b$ | tr.502 | **có** | Hình 4, mục 3.2 |
| bài toán biên (4) | tr.502 | có | Hình 4, mục 3.2 |
| định lý 2 và chứng minh | tr.502 | **có — một bất đẳng thức in sai, đã đo** | Hình 5, mục 3.3 |
| định lý 3 | tr.502–503 | **có, dựng lại** | Hình 5, 6, mục 3.3 |
| câu *"нетрудно убедиться на примерах"* | tr.503 | **có, dựng lại ví dụ bài báo không in** | Hình 6, mục 3.3 |
| lưới sai phân | tr.503 | **có — dấu của $h$ in sai** | Hình 6, mục 3.4 |
| phiếm hàm $\hat M_h^\alpha$, định lý 1′ | tr.503 | **có** | Hình 7, mục 3.4 |
| hệ (3′), $\bar K_{jl}$, $\hat b_j$ | tr.503 | **có, kiểm là điều kiện cực tiểu** | Hình 7, mục 3.4 |
| định lý 2′ | tr.503–504 | có | Hình 8, 9, mục 3.4 |
| nhận xét về máy tính điện tử, khai triển trực giao | tr.504 | có | Hình 9, mục 3.5 |
| (1′) và dạng tích phân của (3) | tr.504 | **có — dấu và cận in sai, đã đo** | Hình 9, mục 3.5 |
| cách chọn $\Omega$, *"не обязательно дифференциальный"* | tr.504 | **có** — chỗ ridge nằm | Hình 10, mục 3.5 |
| câu kết về phương trình loại hai và bài toán phi tuyến | tr.504 | có | Hình 10, mục 3.5 |
| danh mục 3 tài liệu dẫn | tr.504 | không | [¹] Tikhonov 1943 không có trên Math-Net.Ru; [²] Tikhonov–Samarskii 1961 và [³] Sobolev 1950 là nền giải tích số, ngoài phạm vi |

### 10.2 Tikhonov 1963 — theo mục

| Mục | Trang | Nội dung | Lấy? | Ở đâu |
|---|---|---|---|---|
| 1 | tr.501 | bài toán không chỉnh, nguyên lý chính quy hoá | **có** | mục 3.1 |
| 2 | tr.502–503 | phiếm hàm làm trơn, định lý 1–3 | **có** | mục 3.2, 3.3 |
| 3 | tr.503–504 | sơ đồ sai phân, định lý 1′–2′, mở rộng (1′), cách chọn $\Omega$ | **có** | mục 3.4, 3.5 |
| tài liệu dẫn | tr.504 | 3 mục | không | xem 10.1 |

### 10.3 Hoerl & Kennard 1970 — theo hình và bảng

Bài báo có **1 hình và 0 bảng**.

| Hình / bảng | Trang | Nhóm | Lấy? | Nhận xét của tác giả đã dẫn | Ở đâu |
|---|---|---|---|---|---|
| Figure 1 — *Mean Square Error Functions* | tr.61 | RUỘT | **có, số hoá bản quét** | tr.60, tr.61 | Hình 21, mục 3.10 |

### 10.4 Hoerl & Kennard 1970 — theo mục

| Mục | Trang | Nội dung | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|---|
| tóm tắt | tr.55 | bốn đóng góp | **có** | Hình 11, mục 3.6 |
| 0. Introduction | tr.55 | Gauss–Markov, dạng tương quan, đề xuất $X'X+kI$ | **có** | Hình 12, mục 3.6, 1.3 |
| 1. Properties of Best Linear Unbiased Estimation | tr.56 | (1.1)–(1.9) | **có, kiểm bằng Monte Carlo** | Hình 13, 14, mục 3.7, 1.2 |
| 2. Ridge Regression | tr.57–58 | (2.1)–(2.11), nguồn gốc cái tên | **có, kiểm từng công thức** | Hình 15, 16, mục 3.8 |
| 3a. Definition of the Ridge Trace | tr.58 | định nghĩa, dẫn ví dụ sang [12] | **có** | Hình 17, mục 3.9 |
| 3b. Characterization of the Ridge Trace | tr.58–59 | (3.1)–(3.7) | **có — (3.3) in sai** | Hình 17, 18, mục 3.9 |
| 3c. Likelihood Characterization | tr.59 | (3.8)–(3.9) | **có** | Hình 19, mục 3.9 |
| 4a. Variance and Bias of a Ridge Estimator | tr.60–61 | (4.2)–(4.10), Figure 1 | **có, kiểm và số hoá** | Hình 20, 21, mục 3.10 |
| 4b. Theorems on the Mean Square Function | tr.61–63 | định lý 4.1–4.3, (4.11)–(4.16) | **có — khối in hai lần; cận (4.16) đo độ chặt** | Hình 22, 23, 24, mục 3.11 |
| 4c. Some Comments On The Mean Square Error Function | tr.63 | cực tiểu dời theo $\beta'\beta$ | **có, đo** | Hình 24, mục 3.11 |
| 5. A General Form of Ridge Regression | tr.63 | (5.1)–(5.4), $k_i=\sigma^2/\alpha_i^2$ | **có, kiểm** | Hình 24, mục 3.12 |
| 6. Relation to Other Work in Regression | tr.64 | Stein, Bayes, ràng buộc | **có, kiểm Bayes và ràng buộc** | Hình 25, mục 3.13; phần James–Stein chỉ nêu tên, không đo — thuộc bài ước lượng co khác |
| 7. Selecting a Better Estimate of β | tr.64–65 | tiêu chí vết ridge, thủ tục lặp | **có — hai dẫn chiếu in sai; vòng lặp đo điều kiện dừng** | Hình 25, 26, mục 3.13, 7 |
| 8. Conclusions | tr.65–66 | tổng kết | **có** | Hình 26, 27, mục 3.13 |
| Nomenclature | tr.66 | 16 dòng ký hiệu | **có — dòng $Z^*$ in sai** | Hình 27, mục 3.13, 0.3 |
| References | tr.66–67 | 21 tài liệu | không | danh mục; chỉ dẫn [2], [9], [10], [12] khi cần ở mục 3.8, 3.13 |

### 10.5 ESL 2009 — theo hình và bảng

Phạm vi: §3.3.4 tới hết §3.4 (tr.61–79), cộng hình duy nhất ngoài phạm vi đó có nhắc tới ridge.

| Hình / bảng | Trang | Nhóm | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|---|
| Figure 3.7 — đường CV của 5 phương pháp | tr.62 | RUỘT (bảng Ridge) | **có, đọc toạ độ vectơ** | Hình 29, mục 3.15; bốn bảng còn lại thuộc bài chọn tập con, Lasso, PCR, PLS |
| Table 3.3 — hệ số và sai số kiểm tra | tr.63 | RUỘT (cột LS, Ridge) | **có, dựng lại** | Hình 30, mục 3.15; bốn cột còn lại thuộc bài khác |
| Figure 3.8 — vết ridge trên dữ liệu tuyến tiền liệt | tr.65 | RUỘT | **có, đọc toạ độ vectơ** | Hình 33, mục 3.17 |
| Figure 3.9 — thành phần chính | tr.67 | RUỘT | **có** | Hình 35, mục 3.18 |
| Figure 3.10 — vết lasso | tr.70 | NGOÀI | không | thuộc bài [Lasso](<../Lasso Regression/Lasso Regression.md>) |
| Table 3.4 — ba ước lượng khi đầu vào trực chuẩn | tr.71 | RUỘT (dòng Ridge) | **có** | Hình 37, mục 3.19 |
| Figure 3.11 — miền ràng buộc lasso và ridge | tr.71 | RUỘT | **có** | Hình 37, mục 3.19 |
| Figure 3.12 — đường đồng mức $\sum\lvert\beta_j\rvert^q$ | tr.72 | NGOÀI | không | họ phạt $q$ tổng quát; ridge chỉ là $q=2$, không thêm gì cho bài này |
| Figure 3.13 — $q=1{,}2$ so với elastic net | tr.73 | NGOÀI | không | thuộc bài [Elastic Net](<../Elastic Net/Elastic Net.md>) |
| Figure 3.14–3.16 — LAR | tr.75, 78 | NGOÀI | không | thuộc bài [LARS](<../Least Angle Regression (LARS)/Least Angle Regression (LARS).md>) |
| Figure 3.17 — ridge co hệ số của các thành phần chính | tr.80 | NGOÀI | không | nằm trong §3.5.1 (PCR); nội dung hệ số co theo hướng đã có đủ ở (3.47), Hình 35 |

### 10.6 ESL 2009 — theo mục và công thức

| Mục / công thức | Trang | Lấy? | Ở đâu / vì sao không |
|---|---|---|---|
| 3.2.1 *Example: Prostate Cancer*, chuẩn hoá và chia 67/30 | tr.49–50 | có | mục 3.14, 3.15 (quy ước chuẩn hoá) |
| 3.2 phần *"base error rate"* 1,057 | tr.51 | **có, dựng lại** | mục 3.15 |
| 3.3.4 *Prostate Cancer Data Example (Continued)* | tr.61 | **có** | Hình 28, mục 3.14 |
| 3.4 *Shrinkage Methods*, đoạn mở | tr.61 | có | Hình 28, mục 3.14 |
| 3.4.1, công thức (3.41), (3.42) | tr.63 | **có** | Hình 31, mục 3.15 |
| 3.4.1, đoạn hệ số chặn và (3.43), (3.44) | tr.64 | **có** | Hình 32, mục 3.16 |
| 3.4.1, đầu vào trực chuẩn $\hat\beta/(1+\lambda)$ | tr.64 | **có, kiểm** | mục 3.16 |
| 3.4.1, cách nhìn Bayes, $\lambda=\sigma^2/\tau^2$ | tr.64 | **có, kiểm** | mục 3.16 |
| 3.4.1, (3.45)–(3.49), SVD và thành phần chính | tr.64–67 | **có, kiểm** | Hình 32, 34, mục 3.18 |
| 3.4.1, (3.50), df, câu *"minimum occurs at df(λ) = 5.0"* | tr.68 | **có — câu đó mâu thuẫn Hình 3.7** | Hình 36, mục 3.15, 3.18 |
| 3.4.2 *The Lasso*, (3.51)–(3.52) | tr.68–69 | không | thuộc bài Lasso |
| 3.4.3 *Discussion*, Bảng 3.4, Hình 3.11 | tr.69–72 | **có phần ridge** | Hình 37, mục 3.19 |
| 3.4.3, (3.53)–(3.54), phạt $q$ và elastic net | tr.72–73 | không | họ phạt tổng quát; thuộc bài Lasso và Elastic Net |
| 3.4.4 *Least Angle Regression* | tr.73–79 | không | thuộc bài LARS |
| Bài tập 3.5, 3.6 (tách hệ số chặn; hậu nghiệm) | cuối chương 3 | không | bài tập của sách; kết quả của chúng đã được kiểm bằng số ở mục 3.16 |
| §7.10 (kiểm định chéo, quy tắc một sai số chuẩn) | tr.241–249 | không | chương khác của sách; quy tắc đã phát biểu đủ ở tr.61 |

---

## 11. Tài liệu tham khảo và điều kiện tái lập

```
[1] А. Н. Тихонов, "О решении некорректно поставленных задач и методе регуляризации",
    Доклады Академии наук СССР 151(3):501–504, 1963.
    → Nguồn ruột. Bản quét Math-Net.Ru, truy cập mở.
    AI algorithms/_papers/1963 - Tikhonov - O reshenii nekorrektno postavlennykh zadach i metode regulyarizatsii.pdf

[2] A. E. Hoerl, R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    Technometrics 12(1):55–67, 1970. https://doi.org/10.1080/00401706.1970.10488634
    → Nguồn ruột. Sau tường phí (Unpaywall: is_oa = false). Bản quét JSTOR 13 trang đã có trong
      repo từ lần nhập Notion, ở "Ridge Regression (Tikhonov Regularization)/Untitled/image3–15.png";
      ghép thành một PDF không có lớp chữ:
    AI algorithms/_papers/1970 - Hoerl, Kennard - Ridge Regression Biased Estimation for Nonorthogonal Problems.pdf

[3] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, bản 2, in lần 12
    (2017), Springer. Mục 3.2.1, 3.3.4, 3.4.1, 3.4.3.
    → Nguồn hiện đại. Tác giả phát hành miễn phí; trang của tác giả (hastie.su.domains) hết thời gian
      chờ lúc tải, bản trong repo lấy từ bản sao cùng tên trên course.ccs.neu.edu, 764 trang.
    AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf

[4] Dữ liệu prostate: T. Stamey et al., Journal of Urology 16:1076–1083, 1989; lấy từ gói R
    ElemStatLearn 2015.6.26.2, CRAN archive, tệp data/prostate.RData (bản đã sửa lỗi dữ liệu mà
    chú thích tr.3 của [3] nhắc tới).
    tools/prostate.csv

[5] А. Н. Тихонов, "Об устойчивости обратных задач", ДАН СССР 39(5):195–198, 1943.
    → KHÔNG có trong repo: Math-Net.Ru không lưu số tạp chí năm 1943. Chỉ nêu tên.

[6] A. E. Hoerl, R. W. Kennard, "Ridge Regression. Applications to nonorthogonal problems",
    Technometrics 12:69–82, 1970 — tài liệu [12] của [2], nơi có hai ví dụ số và vết ridge thật.
    → KHÔNG có trong repo (sau tường phí). Bài này không trích số nào từ nó.

[7] H. Zou, T. Hastie, "Regularization and variable selection via the elastic net", JRSS B
    67(2):301–320, 2005. → Nguồn cho so sánh ridge–lasso ở bài Elastic Net.
    AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf

[8] I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019. → mục 9.
    AI algorithms/_papers/2019 - Loshchilov, Hutter - Decoupled Weight Decay Regularization (AdamW).pdf
```

**Script.**

| Script | Sinh ra |
|---|---|
| [`tools/v_ridge.py`](<../../../tools/v_ridge.py>) | mọi phép kiểm Tikhonov (mục 1–6, 19), Hoerl–Kennard (mục 7–12, 17), ESL (mục 13–16), ví dụ tay (mục 18). Đầu ra lưu ở `tools/_out_ridge.txt` |
| [`tools/v_gd1ss.py`](<../../../tools/v_gd1ss.py>) mục 380 | bảng 400 mẫu, mục 1.2 |
| [`tools/demo_shrink.py`](<../../../tools/demo_shrink.py>) | ba cài đặt khớp nhau, và bảng nhóm ba cột, mục 5 và 8.3 |
| [`tools/v_reg.py`](<../../../tools/v_reg.py>) mục 3 | Hình 39 (khối mã in trong mục 6 là bản trích của nó) |
| [`tools/fig_ridge.py`](<../../../tools/fig_ridge.py>) | Hình 38 — trích từ chính bài bằng `extract_fig.py` |

**Điều kiện tái lập.** numpy 2.4.4 · scikit-learn 1.9.1 · matplotlib 3.10.9 · pypdfium2 · Pillow · Linux x86-64.

- **Tái lập tuyệt đối** (phân số chính xác): ví dụ tay mục 0.5.
- **Tái lập tới chữ số in** trên mọi máy đã thử: mọi phép kiểm công thức của Tikhonov và Hoerl–Kennard, dựng
  lại Bảng 3.3 và toạ độ vectơ Hình 3.7, 3.8 (đọc thẳng lệnh vẽ trong PDF, không qua điểm ảnh).
- **Phụ thuộc cách render**: bảy cột số hoá Hình 1 của Hoerl–Kennard (mục 3.10) — đổi độ phân giải ảnh quét
  thì chữ số thứ hai sau dấu phẩy có thể đổi.
- **Phụ thuộc máy ở chữ số cuối**: sai số $10^{-13}$ của vòng lặp toạ độ (mục 5).
- **Phụ thuộc cách chia ngẫu nhiên**: $\lambda$ chọn bằng kiểm định chéo (mục 5, 3.15) — cố định bằng
  `random_state` và hạt giống 0–499.

---

## 12. Kiểm tra trước khi đóng bài

- [x] **IO** — mục 0.1: bảng đầu vào/đầu ra kèm kích thước và khối mã chạy được
- [x] **KN** — mục 0.2: bảng thuật ngữ định nghĩa tại chỗ, gồm cả thuật ngữ tiếng Nga của Tikhonov
- [x] **Bảng ký hiệu** — mục 0.3: bốn hệ ký hiệu đặt cạnh nhau và chín chỗ trùng tên
- [x] **SS** — mục 1.2: bế tắc của bình phương tối thiểu nêu bằng ba phép đo — $E[\hat\beta'\hat\beta]$ gấp
      33,0658 lần $\beta'\beta$, ngược dấu 48 % số lần, sai số bài toán ngược $10^{9{,}7991}$ lần biên độ
- [x] **CT** — mục 3–5: toàn bộ ba nguồn theo trang, công thức hiện đại, thuật toán bảy bước kèm "bỏ đi thì
      hỏng chỗ nào"
- [x] **VS** — mục 6: cơ chế hai đạo hàm tại 0, kèm phép đo một-thay-đổi và đối chiếu với công thức (4.5)
- [x] **KQ** — mục 3.15–3.17: kết quả của ESL kèm số hiệu Bảng/Hình và trang, đã dựng lại trên dữ liệu thật
- [x] **GIÁ** — mục 8.2: bảng bốn cột có dòng thua (không thưa) và dòng lợi không phân biệt được với nhiễu
- [x] **Luật 6** — 37 khung cắt từ nguồn gốc, mỗi khung có dòng caption hoặc dòng "khung chứa gì", lời tác giả
      nguyên văn kèm trang, và cách đọc; mọi công thức đánh số có ký hiệu và đoạn ý nghĩa; sáu bảng đối chiếu
      ở mục 10
- [x] **E.1d** — Tikhonov không có ví dụ số: dựng lại hai "ví dụ" bài báo nói mà không in (tr.501 và tr.503);
      Hoerl–Kennard không có ví dụ số: số hoá Hình 1 và kiểm tính cộng; ESL: dựng lại Bảng 3.3, Hình 3.7, 3.8
- [x] Hình tự vẽ kèm khối mã trong bài — Hình 38 và Hình 39
- [x] Kết quả âm và chỗ nguồn sai giữ nguyên, kèm chẩn đoán — bảng mười một chỗ đầu mục 3

**Còn mở, đã biết:**

1. **Cột Ridge của Bảng 3.3** không dựng lại được tới chữ số in bằng một quy ước chuẩn hoá duy nhất; sai số
   chuẩn 0,165 không khớp quy ước nào đã thử (mục 3.15).
2. **Câu tiếng Nga và câu của Hoerl–Kennard chép từ ảnh trang**; `kiemtrich.py` không kiểm được vì hai PDF
   không có lớp chữ. Lớp OCR của lần nhập Notion (`Untitled.md`) có nhưng thiếu và vỡ công thức, nên không
   dùng làm nguồn đối chiếu.
3. **Bài đi kèm của Hoerl–Kennard** (tài liệu [6]) chưa đọc; mọi vết ridge thật của họ nằm ở đó.
4. **Tikhonov 1943** chưa đọc — Math-Net.Ru không có số đó.
