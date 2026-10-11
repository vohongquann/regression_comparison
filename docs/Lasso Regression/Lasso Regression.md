# Lasso Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 390
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Robert Tibshirani
Date: January 1, 1996
url: https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

> **Lasso — *least absolute shrinkage and selection operator*.** Lasso khớp một mô hình tuyến tính như
> bình phương tối thiểu, nhưng **giới hạn tổng trị tuyệt đối của các hệ số**. Giới hạn đó làm một số hệ
> số bằng **đúng 0**, tức mô hình tự bỏ bớt biến. Với 200 cột dữ liệu mà chỉ vài cột thật sự có ích,
> lasso trả về một mô hình chỉ dùng vài cột, và làm điều đó bằng **một** bài toán tối ưu lồi thay vì thử
> hết $2^{200}$ tổ hợp.
>
> Bài này đi theo **năm nguồn gốc**:
>
> 1. **R. Tibshirani 1996**, *J. R. Statist. Soc. B* 58(1):267–288, 22 trang — bài báo đặt tên và định
>    nghĩa lasso. **Bản PDF là bản tải JSTOR, không phải bản truy cập mở** (Unpaywall báo `is_oa = false`),
>    nên nó chỉ nằm trên máy dựng bài, **không có trong repo công khai** (xem mục 11). Mọi hình của bài báo
>    dưới đây là khung cắt từ bản đó.
> 2. **R. Tibshirani 2011**, *J. R. Statist. Soc. B* 73(3):273–282 — bài hồi tưởng của chính tác giả,
>    kèm thảo luận của Bühlmann, Holmes và phần trả lời.
> 3. **T. Hastie, R. Tibshirani & J. Friedman, *The Elements of Statistical Learning*, bản 2, §3.4.2–3.4.3**
>    — nguồn hiện đại có dữ liệu thật (ung thư tuyến tiền liệt, 97 bệnh nhân).
> 4. **B. Efron, T. Hastie, I. Johnstone & R. Tibshirani 2004**, *Annals of Statistics* 32(2) — phần nói
>    về lasso: (1.1)–(1.5), Bảng 1, Hình 1 và §3.1. Phần còn lại là ruột của bài
>    [Least Angle Regression](<../Least Angle Regression (LARS)/Least Angle Regression (LARS).md>).
> 5. **J. Friedman, T. Hastie & R. Tibshirani 2010**, *J. Statistical Software* 33(1) — thuật toán hạ theo
>    toạ độ mà scikit-learn và glmnet dùng: (1)–(10), §2.5–2.6.
>
> Mọi công thức đánh số và mọi bảng số của năm nguồn được kiểm lại bằng
> [`tools/v_lasso.py`](<../../../tools/v_lasso.py>). Việc kiểm tìm ra **hai mươi ba chỗ** nguồn in sai, tự
> mâu thuẫn hoặc không dựng lại được (bảng đầu mục 3). Hai chỗ đáng nhớ nhất: **Bảng 1 của bài báo 1996
> được tính trên bản dữ liệu prostate có lỗi nhập liệu**, và **ví dụ mô phỏng số 1 khớp nhiễu $\sigma=2$,
> không phải $\sigma=3$ như văn bản viết**. Không chỗ nào lật định nghĩa hay kết luận chính của lasso.

---

## 0. Bắt đầu từ đây

### 0.1 Nhận gì, trả gì

Lasso nhận **đúng những gì hồi quy tuyến tính nhận**, cộng một con số do người dùng chọn. Chỗ khác biệt
nằm ở **hình dạng của đầu ra**: nhiều ô trong vector hệ số bằng đúng 0.

| | Cách hiểu thông thường | Ký hiệu | Ví dụ ngay dưới |
|---|---|---|---|
| **Đầu vào** | một bảng gồm $n$ mẫu và $p$ cột số, mỗi cột đã chuẩn hoá | $X\in\mathbb{R}^{n\times p}$ | $442\times10$ |
| **Đáp án khi huấn luyện** | một giá trị số cho mỗi mẫu | $y\in\mathbb{R}^{n}$ | $442$ số |
| **Siêu tham số** | "cho phép hệ số lớn tới đâu" — hai cách viết tương đương | cận $t\ge0$, hoặc mức phạt $\lambda\ge0$ | `alpha` của scikit-learn |
| **Đầu ra của việc học** | một hệ số cho mỗi cột, **nhiều ô bằng đúng 0**, cộng một hệ số chặn | $\hat\beta\in\mathbb{R}^{p}$, $\hat\beta_0$ | $10$ số |
| **Giá đỡ** | danh sách tên các biến còn lại | $\{j:\hat\beta_j\ne0\}$ | `['bmi', 'bp', 's5']` |
| **Dự đoán** | nhân từng dòng với các hệ số rồi cộng hệ số chặn | $\hat y=\hat\beta_0+X\hat\beta$ | $442$ số |

Chạy thật trên bộ `diabetes` của scikit-learn — 442 bệnh nhân, 10 biến, mỗi cột đã được thư viện chuẩn
hoá về độ dài 1 ([`tools/v_lasso.py`](<../../../tools/v_lasso.py>) mục 21):

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

d = load_diabetes()
X, y = d.data, d.target
for a in (0.05, 1.0, 2.0):
    m = Lasso(alpha=a, max_iter=100000).fit(X, y)
    print("alpha = %.2f:" % a, np.round(m.coef_).astype(int), (m.coef_ != 0).sum(), "biến")
m = Lasso(alpha=1.0, max_iter=100000).fit(X, y)
print([d.feature_names[j] for j in np.nonzero(m.coef_)[0]], m.predict(X).shape)
```

```text
alpha = 0.05: [   0 -194  522  295  -99    0 -223    0  512   53] 7 biến
alpha = 1.00: [  0   0 368   6   0   0   0   0 308   0] 3 biến
alpha = 2.00: [ 0  0 64  0  0  0  0  0  4  0] 2 biến
['bmi', 'bp', 's5'] (442,)
```

Hệ số làm tròn về số nguyên. Đọc theo cột: `age`, `s2` và `s4` bị xoá ngay từ `alpha = 0.05`; hai biến
sống lâu nhất là **`bmi`** (chỉ số khối cơ thể) và **`s5`**.

Một chi tiết đáng nhìn kỹ: từ `alpha = 1` sang `alpha = 2`, `bmi` tụt từ 368 xuống 64 và `s5` từ 308
xuống 4. **Giá đỡ vẫn còn 2 biến, nhưng mô hình đã gần như bằng 0.** Đếm số biến còn lại không cho biết mô
hình còn dự đoán được gì; phải nhìn cả độ lớn hệ số.

Bốn điều dễ hiểu sai, nói rõ ngay:

- **Số 0 là số 0 thật, không phải số rất nhỏ.** Đây là điểm khác ridge. Nhưng nếu bộ giải dừng sớm vì
  ngưỡng hội tụ chưa siết, có thể thấy những số cỡ $10^{-3}$ giả dạng "khác 0" — mục 6.4 có một ví dụ
  đo được.
- **Mức phạt không học được từ sai số huấn luyện.** Chọn bằng sai số huấn luyện thì luôn ra "không phạt",
  tức quay về bình phương tối thiểu. Mục 7 có năm cách chọn, và năm nguồn dùng năm cách khác nhau.
- **Kết quả phụ thuộc ĐƠN VỊ của từng cột.** Phải chuẩn hoá trước — và **chuẩn hoá trên tập nào** cũng
  đổi kết quả: mục 3.21 cho thấy một bảng của ESL chỉ dựng lại được khi biết đúng quy ước đó.
- **"Biến bị loại" ≠ "biến vô dụng".** Nếu hai cột gần trùng nhau, lasso giữ một và bỏ cái kia gần như
  tuỳ tiện (mục 6.4). Đây là giới hạn thật, không phải chi tiết kỹ thuật.

### 0.2 Từ mới trong bài này

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Trong bài này | Nghĩa |
|---|---|---|
| **Bình phương tối thiểu** (OLS, *least squares*) | $\hat\beta^\circ$ của Tibshirani 1996 | chọn hệ số làm **tổng bình phương phần dư** nhỏ nhất |
| **Phần dư** (*residual*) | $y_i-\hat y_i$ | giá trị thật trừ dự đoán |
| **Chọn tập con** (*subset selection*) | đối thủ số một của lasso | thử các tập biến, giữ tập cho sai số thấp nhất; *best subset* là thử **hết** các tập mỗi cỡ |
| **Ridge** (*ridge regression*) | đối thủ số hai | cộng khoản phạt $\lambda\sum\beta_j^2$; co mọi hệ số về gần 0 mà **không** đưa hệ số nào về đúng 0 |
| **Garotte không âm** (*non-negative garotte*) | công thức (2) của Tibshirani 1996 | phương pháp của Breiman: nhân từng hệ số OLS với một hệ số co $c_j\ge0$, giới hạn $\sum c_j$. Lasso sinh ra từ nó |
| **Co hệ số** (*shrinkage*) | chữ "S" thứ nhất trong tên | kéo độ lớn hệ số về gần 0 |
| **Thưa** (*sparse*) | đầu ra của lasso | nhiều hệ số bằng đúng 0 |
| **Giá đỡ** (*support*) | `['bmi', 'bp', 's5']` ở trên | tập chỉ số có hệ số khác 0 |
| **Ngưỡng mềm** (*soft threshold*) | công thức (3) và (13) | co độ lớn một lượng cố định rồi chặn tại 0: $\operatorname{sign}(z)(\lvert z\rvert-\gamma)^+$ |
| **Ngưỡng cứng** (*hard threshold*) | chọn tập con khi cột trực giao | giữ nguyên nếu vượt ngưỡng, xoá nếu không |
| **Thiết kế trực chuẩn** (*orthonormal design*) | $X^\top X=I$ | các cột vuông góc với nhau và có độ dài 1 — trường hợp mọi phương pháp có nghiệm viết tay được |
| **Quy hoạch toàn phương** (*quadratic programming*) | cách giải (1) năm 1996 | cực tiểu một hàm bậc hai dưới các ràng buộc tuyến tính |
| **Điều kiện Kuhn–Tucker** | Tibshirani 1996 §6 | điều kiện cần và đủ để một điểm là nghiệm của bài toán tối ưu lồi có ràng buộc |
| **Dưới vi phân** (*subgradient*) | mục 4.2 | "đạo hàm" của hàm có góc: tại $0$, dưới vi phân của $\lvert\beta\rvert$ là cả đoạn $[-1,1]$ |
| **Đường nghiệm** (*solution path*) | Hình 5, Hình 3.10, Hình 1 của Efron | hệ số vẽ theo mức phạt; của lasso là **tuyến tính từng khúc** |
| **Kiểm định chéo** (*cross-validation*, CV) | 5 mảnh ở Tibshirani, 10 mảnh ở ESL | chia dữ liệu thành $K$ phần; lần lượt khớp trên $K-1$ phần, đo sai số trên phần còn lại |
| **Kiểm định chéo tổng quát** (GCV) | công thức (10) | xấp xỉ kiểm định chéo bằng một công thức, không cần khớp lại mô hình |
| **Ước lượng rủi ro không chệch của Stein** (SURE) | công thức (11) | công thức tính **trung bình** sai số của một ước lượng khi chỉ có **một** mẫu dữ liệu, với nhiễu chuẩn |
| **Bootstrap** | Bảng 2, Hình 6 | lấy mẫu lại có hoàn lại từ chính dữ liệu để đo độ dao động của ước lượng |
| **Phân phối Laplace** (*double-exponential*) | §5, Hình 7 | mật độ $\tfrac1{2\tau}e^{-\lvert\beta\rvert/\tau}$: nhọn ở 0, đuôi dày hơn phân phối chuẩn |
| **Mode hậu nghiệm** (MAP) | §5 | giá trị tham số có mật độ hậu nghiệm lớn nhất |
| **Hạ theo toạ độ** (*coordinate descent*) | Friedman 2010 | tối ưu lần lượt từng hệ số, giữ nguyên các hệ số khác |
| **Khởi động ấm** (*warm start*) | Friedman 2010 §2.5 | dùng nghiệm ở mức phạt trước làm điểm xuất phát cho mức phạt kế tiếp |

### 0.3 Bảng ký hiệu — và chỗ trùng tên giữa năm nguồn

Bảng đầu là ký hiệu của chính bài này.

| Ký hiệu | Đọc là | Nghĩa | Chiều / kiểu |
|---|---|---|---|
| $N$ hoặc $n$ | "en" | số mẫu | số nguyên |
| $p$ | "pê" | số biến (cột) | số nguyên |
| $x_{ij}$ | "ích i j" | giá trị biến $j$ của mẫu $i$ | vô hướng |
| $x_j$ | "ích j" | **một cột** của $X$ | $\mathbb{R}^n$ |
| $\beta$, $\hat\beta$ | "bê-ta", "bê-ta mũ" | hệ số thật; hệ số ước lượng | $\mathbb{R}^p$ |
| $\hat\beta^\circ$ | "bê-ta mũ tròn" | nghiệm bình phương tối thiểu đầy đủ (Tibshirani 1996) | $\mathbb{R}^p$ |
| $\alpha$ (trong (1)) | "an-pha" | hệ số chặn; nghiệm luôn là $\bar y$ | vô hướng |
| $t$ | "tê" | **cận** của $\sum_j\lvert\beta_j\rvert$ | $t\ge0$ |
| $t_0$ | "tê không" | $\sum_j\lvert\hat\beta^\circ_j\rvert$ — cận mà từ đó trở lên lasso trùng OLS | vô hướng |
| $s$ | "ét" | cận chuẩn hoá $t/t_0\in[0,1]$ | vô hướng |
| $\lambda$ | "lam-đa" | **mức phạt** trong dạng $\text{RSS}+\lambda\sum\lvert\beta_j\rvert$ | $\lambda\ge0$ |
| $\gamma$ | "gam-ma" | ngưỡng của phép co ở (3), (5), (13) | $\gamma\ge0$ |
| $c_j$ | "xê j" | hệ số co của garotte | $c_j\ge0$ |
| $\rho$ | "rô" | tương quan giữa hai biến | $[-1,1]$ |
| $\sigma^2$, $\hat\sigma^2$ | "xích-ma bình phương" | phương sai nhiễu; ước lượng của nó | vô hướng |
| $\tau$ | "tô" | tham số thang của tiên nghiệm Laplace (§5); độ lệch chuẩn của $\hat\beta^\circ_j$ (§4) — **hai nghĩa** | vô hướng |
| $W$, $W^-$ | "đáp-bơ-liu", "nghịch đảo suy rộng" | $\operatorname{diag}(\lvert\tilde\beta_j\rvert)$ và nghịch đảo suy rộng của nó (§2.5, §4) | $p\times p$ |
| $V$ | "vê" | ma trận hiệp phương sai tổng thể của $x$ | $p\times p$ |
| ME, PE | | sai số trung bình của **hàm hồi quy** và của **dự đoán** (8) | vô hướng |
| $\delta_i$ | "đen-ta i" | một vector dấu $(\pm1,\dots,\pm1)$ (§6) | $\{-1,1\}^p$ |
| $E$, $S$ | | tập ràng buộc **đang chặt** và **đang lỏng** (§6) | tập chỉ số |
| $\operatorname{sign}(z)$ | "dấu của dét" | $+1$, $-1$ hoặc $0$ | |
| $(z)^+$ | "phần dương của dét" | $\max(z,0)$ — **chính chỗ sinh ra số 0** | |
| $\lVert\beta\rVert_1$ | "chuẩn một" | $\sum_j\lvert\beta_j\rvert$ | |

**Chỗ trùng tên — bảy chỗ, và cả bảy gây lỗi thật khi đọc chéo các nguồn hoặc viết mã:**

1. **$s$ ở Tibshirani 1996 là cận CHUẨN HOÁ $t/t_0$; ở bài hồi tưởng 2011 tr.273, $s$ là chính CẬN** —
   *"a constraint of the form Σ|βj| ⩽ s"*. Cùng tác giả, hai nghĩa.
2. **$\gamma$ ở Tibshirani 1996 là ngưỡng co; ở bài hồi tưởng 2011 tr.274, $\gamma$ là SỐ MŨ** của bridge,
   $\lambda\sum\lvert\beta_j\rvert^\gamma$. Tibshirani 1996 gọi số mũ đó là $q$.
3. **Bốn cách đặt mức phạt, bốn thang khác nhau.** Tibshirani 1996 (§5) và hồi tưởng 2011:
   $\text{RSS}+\lambda\sum\lvert\beta_j\rvert$. ESL (3.52): $\tfrac12\text{RSS}+\lambda\sum\lvert\beta_j\rvert$.
   Friedman 2010 (1): $\tfrac1{2N}\text{RSS}+\lambda P_\alpha(\beta)$. scikit-learn:
   $\tfrac1{2n}\text{RSS}+\texttt{alpha}\cdot\lVert\beta\rVert_1$. Cùng dữ liệu thì $\lambda_{\text{ESL}}=n\cdot\texttt{alpha}$
   và $\lambda_{1996}=2n\cdot\texttt{alpha}$ — quên hệ số là lệch hàng trăm lần.
4. **$\alpha$ mang bốn nghĩa.** Tibshirani 1996 (1): hệ số chặn. Friedman 2010 (2): tỉ lệ trộn, $\alpha=1$ là
   lasso. ESL (3.54): tỉ lệ trộn **ngược chiều** — $\alpha$ nhân với $\beta_j^2$, nên $\alpha=0$ mới là lasso.
   scikit-learn: `alpha` là **mức phạt**, còn tỉ lệ trộn tên là `l1_ratio`.
5. **$\tau$ hai nghĩa trong cùng bài 1996:** thang Laplace ở §5, độ lệch chuẩn $\hat\sigma/\sqrt N$ ở §4.
6. **$\lambda$ trong (7) và (9) không phải $\lambda$ của §5.** Ở (7), (9) nó là mức phạt của một bài toán
   *ridge* xấp xỉ, $\sum\beta_j^2/\lvert\tilde\beta_j\rvert$; ở §5 là mức phạt lasso. Mục 3.7 đo: tại cùng
   một nghiệm, hai số chênh đúng 2 lần.
7. **$x_j$ trong bài này là một CỘT**, vì mọi phép toán làm theo từng biến. Nhiều tài liệu khác dùng $x_i$
   cho một **dòng**. Nhìn chỉ số: $i$ chạy theo mẫu, $j$ chạy theo cột.

### 0.4 Luồng xử lý

```
 X (n×p), y (n)
      │
      ▼
 ① chuẩn hoá từng cột (trừ trung bình, chia độ lệch chuẩn) ; trừ trung bình của y
      │            ← chuẩn hoá trên TẬP NÀO là một quyết định (mục 3.21)
      ▼
 ② chọn cách đặt: cận t  hoặc  mức phạt λ        ← cùng một đường nghiệm (mục 4.1)
      │
      ▼
 ③ giải:  min ‖y − Xβ‖²  với  Σ|βⱼ| ≤ t
      │    · 1996: quy hoạch toàn phương, thêm dần ràng buộc dấu (mục 3.12)
      │    · 2004: LARS có sửa đổi lasso — cả đường nghiệm (mục 3.24–3.25)
      │    · 2010: hạ theo toạ độ, mỗi bước là một NGƯỠNG MỀM (mục 3.27)
      ▼
 ④ chọn t (hoặc λ): CV, GCV, Stein, quy tắc 1 sai số chuẩn (mục 7)
      │
      ▼
 ⑤ β̂ (nhiều số 0) ; β̂₀ = ȳ ; trả về thang gốc: chia cho độ lệch chuẩn đã dùng ở ①
      │
      ▼
 ŷ = β̂₀ + X β̂
```

### 0.5 Một vòng tính tay, bằng số thật

**Ví dụ 1 — bốn hệ số trực chuẩn.** Khi $X^\top X=I$, lasso là ngưỡng mềm của nghiệm OLS (công thức (3)).
Lấy $\hat\beta^\circ=(0{,}5;\ 1{,}0;\ 3{,}0;\ -2{,}0)$ và ngưỡng $\gamma=1$:

| $\hat\beta^\circ_j$ | $\lvert\hat\beta^\circ_j\rvert-1$ | $(\cdot)^+$ | lasso $=\operatorname{sign}\cdot(\cdot)^+$ | ridge $\hat\beta^\circ_j/(1+1)$ |
|---:|---:|---:|---:|---:|
| 0,5 | −0,5 | 0 | **0** | 0,25 |
| 1,0 | 0 | 0 | **0** | 0,50 |
| 3,0 | 2,0 | 2,0 | 2,0 | 1,50 |
| −2,0 | 1,0 | 1,0 | −1,0 | −1,00 |

Hai hệ số đầu bị **xoá hẳn**; hai hệ số sau bị kéo về 0 đúng **một** đơn vị. Ridge không xoá gì — nó chia
đôi mọi thứ. Đầu ra `v_lasso.py` mục 21: ngưỡng mềm `[ 0.  0.  2. -1.]`, ridge `[ 0.25  0.5   1.5  -1.  ]`.

**Ví dụ 2 — hai biến, tính bằng công thức (5).** Hình 4 của Tibshirani 1996 dùng $\hat\beta^\circ=(6,3)$.
Khi cả hai hệ số còn dương, lasso trừ **cùng một lượng** $\gamma=(\hat\beta^\circ_1+\hat\beta^\circ_2-t)/2$ khỏi cả
hai, bất kể hai biến tương quan tới đâu:

- $t=5$: $\gamma=(6+3-5)/2=2$, nghiệm $(6-2,\ 3-2)=(4,\ 1)$. Tổng đúng bằng 5.
- $t=2$: công thức cho $\gamma=3{,}5$ và $\hat\beta_2=3-3{,}5<0$, nên $\hat\beta_2$ bị chặn về 0. Khi đó chỉ còn
  $\hat\beta_1$, và ràng buộc buộc $\hat\beta_1=t=2$. Nghiệm $(2,\ 0)$.

Đầu ra `v_lasso.py` mục 21: `t = 5 -> (4.0, 1.0) ; t = 2 -> (2.0, 0.0)`. Trường hợp thứ hai chính là chỗ
công thức (6) của bài báo in sai (mục 3.6).

### 0.6 Điểm dễ nhầm

1. **Cận $t$ và mức phạt $\lambda$ chạy ngược chiều nhau.** $t$ lớn = phạt yếu; $\lambda$ lớn = phạt mạnh.
2. **Đổi thang của một cột là đổi mô hình.** Mục 3.8 và 3.21 đo được hai lần.
3. **Hệ số chặn không bị phạt.** Mọi nguồn đều trừ trung bình của $y$ trước; nghiệm hệ số chặn là $\bar y$.
4. **Số bước của đường nghiệm có thể lớn hơn $p$**, vì một biến có thể rời mô hình rồi vào lại (mục 3.25).
5. **Lasso là mode hậu nghiệm, KHÔNG phải trung bình hậu nghiệm**; trung bình hậu nghiệm không bao giờ bằng
   đúng 0 (mục 3.11).
6. **Đổi dấu là có thể.** Với từ ba biến trở lên, một hệ số lasso có thể mang dấu **ngược** với hệ số OLS
   (mục 3.5 — 23 trên 3 000 bộ dữ liệu thử).

### 0.7 Đọc bài theo thứ tự nào

- **Cần dùng ngay:** 0.1 → 0.5 → mục 5 (thuật toán) → mục 7 (chọn mức phạt) → mục 8 (giá phải trả).
- **Cần hiểu vì sao có số 0:** mục 2 → 3.3 → 3.4 → mục 6.
- **Cần đối chiếu bài báo:** bảng đầu mục 3, rồi các mục 3.x theo trang, rồi mục 10.

### 0.8 Khi KHÔNG nên dùng

- **Mọi biến đều có ích một chút** (nhiều hiệu ứng nhỏ): ridge thắng rõ. Bảng 6 của bài báo 1996 đo
  được trung vị sai số 2,30 cho ridge so với 4,87 cho lasso tốt nhất; dựng lại ra 2,2205 so với 3,5591
  (mục 3.13).
- **Các biến đi theo nhóm tương quan cao và cần giữ cả nhóm:** lasso giữ một, bỏ phần còn lại gần như
  tuỳ tiện (mục 6.4). Dùng [Elastic Net](<../Elastic Net/Elastic Net.md>).
- **Mục tiêu là danh sách biến "thật":** chọn mức phạt bằng kiểm định chéo tối ưu dự đoán, không tối ưu
  việc chọn đúng biến (mục 6.3). Trong ví dụ 1 của bài báo, mô hình đúng $\{1,2,5\}$ chỉ được chọn
  **2,5 %** số lần.
- **Cần khoảng tin cậy cho hệ số:** không có công thức đóng; bản xấp xỉ (7) cho phương sai 0 ở mọi hệ số
  bằng 0 (mục 3.7).

---

## 1. Bài toán mà nó giải — và cách cũ tắc ở đâu

### 1.1 Bài toán

Có $N$ mẫu, mỗi mẫu $p$ biến và một đáp án. Cần một mô hình tuyến tính **dự đoán tốt** và **đọc được** —
tức dùng ít biến. Tibshirani 1996 tr.267 nêu đúng hai đòi hỏi đó: *"The first is prediction accuracy"* và
*"The second reason is interpretation"*.

Bình phương tối thiểu thường không đạt cả hai: nó dùng **mọi** biến, và khi biến nhiều hoặc tương quan thì
hệ số dao động mạnh giữa các mẫu dữ liệu.

### 1.2 Cách cũ tắc ở đâu, đo bằng số

Hai cách cũ, mỗi cách hỏng một kiểu (Tibshirani 1996 tr.267: *"subset selection and ridge regression, both
have drawbacks"*).

**Ridge không xoá được biến nào.** Chạy ridge trên `diabetes` (442 bệnh nhân, 10 biến):

| | Số hệ số bằng **đúng** 0 | Hệ số nhỏ nhất theo trị tuyệt đối |
|---|---:|---:|
| Ridge `alpha = 1` | **0/10** | $5{,}91$ |
| Lasso `alpha = 0.5` | **6/10** | $0$ |
| Lasso `alpha = 2` | **8/10** | $0$ |

Hệ số nhỏ nhất của ridge là $5{,}91$ — nhỏ so với các hệ số khác, nhưng **khác 0**. Không có mức phạt nào
làm nó bằng 0 mà không kéo tất cả về 0 cùng lúc. Đó là tính chất của khoản phạt bình phương (mục 6.1).

**Chọn tập con thì đúng về nguyên tắc nhưng không chạy nổi.** Muốn thử mọi tổ hợp biến:

| Số biến $p$ | Số tổ hợp phải thử |
|---:|---:|
| 10 | $1{,}02\cdot10^{3}$ |
| 20 | $1{,}05\cdot10^{6}$ |
| 40 | $1{,}1\cdot10^{12}$ |
| 60 | $1{,}15\cdot10^{18}$ |

Ở $p=60$ — một bài toán nhỏ theo tiêu chuẩn ngày nay — đó là $1{,}2$ tỉ tỉ tổ hợp. Tibshirani 1996 tr.283
tự ghi *"best subsets regression is generally considered impractical for p > 30"*, và vì vậy bỏ hẳn chọn tập
con khỏi ví dụ 4 ($p=40$).

**Chọn tập con còn hỏng theo kiểu thứ hai: nó nhảy.** Trong ví dụ 1 của bài báo (mục 3.13), dựng lại với
200 bộ dữ liệu và nhiễu $\sigma=2$ — mức nhiễu khớp Bảng 3: tập con tốt nhất chứa đủ ba biến thật $\{1,2,5\}$ ở **0,595** số lần, lasso chọn bằng GCV
chứa đủ ở **0,955** số lần. Bài báo in 0,535 và 0,955.

Mã sinh bảng ridge/lasso và bảng $2^p$: [`tools/v_gd1ss.py`](<../../../tools/v_gd1ss.py>), mục
`390  Lasso Regression`.

### 1.3 Nó đổi đúng cái gì

Lasso giữ nguyên hàm mục tiêu của bình phương tối thiểu và **đổi đúng một chữ** trong ràng buộc của ridge:
$\sum\beta_j^2\le t$ thành $\sum\lvert\beta_j\rvert\le t$. Hệ quả của chữ đó:

| | Ridge | Lasso | Chọn tập con |
|---|---|---|---|
| Ràng buộc | $\sum\beta_j^2\le t$ | $\sum\lvert\beta_j\rvert\le t$ | số biến $\le k$ |
| Bài toán lồi? | có | **có** | không |
| Hệ số bằng đúng 0 | không | **có** | có |
| Liên tục theo dữ liệu | có | **có** | không — nhảy |
| Nghiệm viết tay được | có | chỉ khi cột trực giao | chỉ khi cột trực giao |

---

## 2. Ý tưởng trung tâm — bằng lời, chưa công thức

Hãy hình dung mỗi hệ số là một núm vặn, và có một **ngân sách** tổng độ lớn cho mọi núm. Bình phương tối
thiểu muốn vặn mỗi núm tới vị trí riêng của nó; ngân sách không đủ cho tất cả.

- Nếu ngân sách tính bằng **tổng bình phương** (ridge), thì vặn một núm từ 0 lên một chút gần như **miễn
  phí** — bình phương của một số nhỏ là số rất nhỏ. Nên ridge luôn cho mọi núm nhích lên một ít.
- Nếu ngân sách tính bằng **tổng trị tuyệt đối** (lasso), thì mỗi đơn vị vặn đều **tốn như nhau**, kể cả
  đơn vị đầu tiên. Một núm chỉ được vặn nếu lợi ích của nó vượt một ngưỡng cố định. Núm nào không vượt thì
  nằm yên ở **đúng 0**.

Tibshirani 1996 tr.270 nói đúng điều đó bằng hình học: miền $\sum\lvert\beta_j\rvert\le t$ là một hình vuông
xoay 45° có **góc nằm trên trục**, và *"The lasso solution is the first place that the contours touch the
square, and this will sometimes occur at a corner, corresponding to a zero coefficient"*.

Ý tưởng này **không phải mới hoàn toàn** năm 1996. Bài báo tr.268 ghi rõ nó đến từ garotte không âm của
Breiman; tr.269 ghi dạng ngưỡng mềm đã có ở Donoho và Johnstone; bài hồi tưởng 2011 tr.273 viết *"The lasso
is just regression with an l1-norm penalty, and l1-norms have been around for a long time!"*. Đóng góp của
bài báo là đặt nó vào hồi quy **không trực giao**, đo nó bằng mô phỏng, và cho một cách tính.

---

## 3. Phần ruột — theo đúng năm nguồn gốc

Phần này đi qua **toàn bộ** phần phương pháp của các nguồn, theo đúng thứ tự in:

| Nguồn | Phần đưa vào | Vì sao là phần ruột |
|---|---|---|
| **Tibshirani 1996**, *JRSS B* 58(1):267–288 — **cả bài** | công thức (1)–(16) và các công thức không đánh số, Hình 1–9, Bảng 1–8, §1–§12 | bài báo định nghĩa lasso, đưa ví dụ thật, mô phỏng và thuật toán |
| **Tibshirani 2011**, *JRSS B* 73(3):273–282 — **cả bài** | §1–§5, Bảng 1, Hình 2, thảo luận của Holmes (2)–(3), phần trả lời | chính tác giả kể lại nguồn gốc và trả lời về cách nhìn Bayes |
| **ESL 2009**, chú thích tr.3, §3.4.2–3.4.3 tr.68–73, cùng ô Lasso của Hình 3.7 và cột Lasso của Bảng 3.3 | (3.51)–(3.54), Hình 3.7, 3.10–3.13, Bảng 3.3–3.4 | nguồn hiện đại có dữ liệu thật để dựng lại từng con số |
| **Efron và cộng sự 2004** — phần về lasso | (1.1)–(1.5), Bảng 1, Hình 1, §3.1 với (3.1)–(3.6) và định lý 1 | cách tính cả đường nghiệm lasso; bộ dữ liệu diabetes |
| **Friedman và cộng sự 2010** — §2 | (1)–(10), §2.1–2.6 | thuật toán mà scikit-learn và glmnet dùng hôm nay |

**Cách trích.** Bài 1996 là bản quét JSTOR có lớp chữ OCR; câu tiếng Anh dưới đây chép theo ảnh trang và
đối chiếu với lớp chữ đó, còn công thức thì lớp OCR vỡ nên **chỉ chép từ ảnh**. Bốn nguồn còn lại có lớp
chữ sạch. Khối công thức không có caption, nên với những khung cắt chỉ chứa văn bản và công thức, ô
"caption gốc" được thay bằng dòng **Khung chứa gì**.

**Hai mươi ba chỗ việc kiểm lại tìm ra.**

| # | Nguồn · chỗ | Nguồn in / nói | Kiểm lại cho thấy | Mục |
|---|---|---|---|---|
| 1 | [T] Bảng 1 tr.274 | hệ số và Z-score của LS, tập con, lasso trên dữ liệu prostate | chỉ khớp khi lweight của đối tượng 32 bằng $\ln 449$ — **giá trị nhập sai** mà ESL tr.3 kể lại; bản đã sửa khớp 4/9, 1/9 và 6/9 ô ở ba cột, bản có lỗi khớp 9/9 cả ba | 3.8 |
| 2 | [T] tr.268 và Bảng 1 | *"standardized so that Σᵢxᵢⱼ/N = 0, Σᵢxᵢⱼ²/N = 1"* | Bảng 1 khớp 9/9 khi chia cho $N-1$, chỉ 7/9 khi chia cho $N$ | 3.8 |
| 3 | [T] tr.272, công thức (6) | $\hat\beta_1=(t/2+(\hat\beta^\circ_1-\hat\beta^\circ_2)/2)^+$ | sai khi $t<\hat\beta^\circ_1-\hat\beta^\circ_2$: ở Hình 4, $t=2$ cho $(2{,}50;\ 0{,}00)$, tổng vượt $t$; nghiệm đúng $(2;\ 0)$ | 3.6 |
| 4 | [T] tr.273, công thức (7) | thừa số thứ hai $(X^\top X+\lambda X^-)^{-1}$ | $X^-$ có cỡ $8\times97$, không cộng được với ma trận $8\times8$; phải là $W^-$ | 3.7 |
| 5 | [T] tr.273, câu dưới (7) | *"it gives an estimated variance of 0 for predictors with β̂ⱼ = 0"* | chỉ đúng khi các cột có hệ số 0 bị **loại** hẳn; đọc $W^-$ đúng nghĩa nghịch đảo suy rộng thì các cột đó nhận phương sai 0,0233–0,1232 | 3.7 |
| 6 | [T] Bảng 1 so với Bảng 2 | cột SE của lasso ở Bảng 1 là *"bootstrap … fixing ŝ"*, tức cột "Fixed t" của Bảng 2 | hai cột lệch nhau ở **3/9** ô; và cột Z của lasso là thương của hai số **đã làm tròn** ($0{,}56/0{,}09=6{,}2222$) | 3.8, 3.9 |
| 7 | [T] tr.276, công thức rủi ro Stein | $\hat\tau^2\{p-2\#(\cdot)+\sum\max(\lvert\hat\beta^\circ_j/\hat\tau\rvert,\gamma)^2\}$ | phải là **min**: Monte Carlo ở $\gamma=1$ cho rủi ro thật 32,7539, công thức min 32,6381, công thức max 241,5024 | 3.10 |
| 8 | [T] tr.277, $\hat t=\sum(\lvert\hat\beta^\circ_j\rvert-\hat\gamma)^+$ | | $\hat\gamma$ đo theo đơn vị $\hat\tau$; trộn đơn vị cho $\hat s=0{,}1280$ thay vì $0{,}8586$ trên prostate | 3.10 |
| 9 | [T] tr.277, lập luận cho $\hat t$ | $Z=XV^{-1/2}$, $\theta=\beta V^{-1/2}$ | phải là $\theta=V^{1/2}\beta$: $\lVert Z\theta-X\beta\rVert$ bằng 9,0743 theo bản in, $1{,}373\cdot10^{-14}$ theo bản sửa | 3.10 |
| 10 | [T] tr.277 (và ESL tr.72) | tiên nghiệm Laplace với *"τ = 1/λ"* | chỉ đúng khi $\sigma^2=1/2$; tổng quát $\tau=2\sigma^2/\lambda$ | 3.11 |
| 11 | [T] tr.280, ví dụ 1 | *"We set σ = 3"* | Bảng 3 khớp $\sigma=2$: trung vị ME của LS dựng lại 2,8161 (in 2,79), với $\sigma=3$ là 6,3362; lasso GCV chứa $\{1,2,5\}$ ở 0,955 số lần — đúng bằng số in | 3.13 |
| 12 | [T] tr.280, ví dụ 1 | *"we simulated 50 data sets"* | mọi tỉ lệ của Bảng 4, 5 và văn bản là bội của 1/200 ($0{,}055\times50=2{,}75$ không phải số nguyên); tr.280 cũng viết *"over 200 simulations"* | 3.13 |
| 13 | [T] Bảng 3 so với Bảng 6 và 7 | trung vị ME của LS: 2,79 · 6,50 · 2,89 | ME của LS không phụ thuộc $\beta$; ví dụ 1 và 2 cùng $\sigma$ nên phải cùng phân phối (dựng lại 5,8295 cho cả hai), và ví dụ 3 có $\sigma$ nhỏ hơn mà in số lớn hơn | 3.13 |
| 14 | [T] Bảng 6 và 8 | SE của trung bình $\hat s$: 0,23 và 0,88 | $\hat s\in[0,1]$ nên SE không vượt được 0,0354 (200 lần) và 0,0707 (50 lần) | 3.13 |
| 15 | [T] tr.280 và §11 tr.286 | *"followed by the garotte and ridge regression"*; kịch bản (b) *"followed by ridge regression and then subset selection"* | Bảng 3: ridge 3,21 **tệ hơn cả LS** 2,79 và tệ hơn tập con 2,44 | 3.13 |
| 16 | [T] tr.283, ví dụ 4 | trung bình hệ số lasso theo bốn khối: 0,50 · 0,92 · 1,56 · 2,33 | tăng đơn điệu, trong khi $\beta$ theo khối là 0, 2, 0, 2; dựng lại: 0,3041 · 1,5353 · 0,4203 · 1,5767 | 3.13 |
| 17 | [T] tr.285, (14) | *"where zⱼ ∼ N(0, σ²)"* | với $\hat\beta^\circ_j=\beta_j+\sigma z_j$ thì $z_j$ phải có phương sai 1; bản in cho phương sai $\sigma^4$ | 3.15 |
| 18 | [R] tr.273 | *"a more general penalty of the form (Σβⱼ^q)^{1/q}"* | thiếu trị tuyệt đối: với $q=1$ và $\beta=(1,-1)$ biểu thức bằng 0 | 3.17 |
| 19 | [E] tr.69, chú thích Hình 3.10 | *"A value ŝ ≈ 0.36 was chosen by 10-fold cross-validation; this caused four coefficients to be set to zero"* | vạch đỏ của Hình 3.10 nằm ở $s=0{,}3750$, vạch tím của Hình 3.7 ở $0{,}3749$ — đúng điểm lưới $3/8$; tại $s=0{,}36$ có **năm** hệ số bằng 0, tại $0{,}375$ mới là bốn | 3.21, 3.22 |
| 20 | [E] Bảng 3.3 cột Lasso | $2{,}468\ 0{,}533\ 0{,}169\ 0{,}002\ 0{,}094$ | dựng lại **9/9 ô** và sai số kiểm tra 0,478601, chỉ khi bộ giải chuẩn hoá lại trên 67 mẫu huấn luyện rồi trả hệ số về thang chuẩn hoá trên 97 mẫu; giải thẳng trên một trong hai thang thì tập $\{$lcavol, lweight, lbph, svi$\}$ không bao giờ xuất hiện, hoặc chỉ khớp 6/9 | 3.21 |
| 21 | [E] tr.69 | *"for t = t0/2 say, then the least squares coefficients are shrunk by about 50% on average"* | trên chính dữ liệu của Hình 3.10, $\lvert\hat\beta_j(t_0/2)\rvert/\lvert\hat\beta^{\text{LS}}_j\rvert$ có trung bình 0,3514, và ba hệ số về 0 | 3.22 |
| 22 | [E] tr.71 so với tr.72 | chú thích Hình 3.11: $\beta_1^2+\beta_2^2\le t^2$; văn bản tr.72: $\beta_1^2+\beta_2^2\le t$ | hai trang dùng hai thang cho cùng một miền; và tr.72 gọi $\lvert\beta_j\rvert^q$ là *"log-prior density"* thiếu dấu trừ | 3.23 |
| 23 | [L] tr.409–410 | *"numerical results are expressed in the original units of the diabetes example"*, rồi *"to t = 3460.00, where β̂ equals the OLS regression vector"* | $\sum\lvert\hat\beta^{\text{OLS}}_j\rvert$ bằng 3459,9776 khi các cột chuẩn hoá độ dài 1 theo (1.1), nhưng chỉ 107,1213 theo đơn vị gốc của Bảng 1 — con số 3460 và trục của Hình 1 **không** theo đơn vị gốc | 3.24 |

**Bốn chỗ bỏ ngỏ**, không phải lỗi in:

- [T] tr.274: $\hat s=0{,}44$ *"selected by generalized cross-validation"*. Công thức (10) trên đúng bản dữ liệu
  khớp Bảng 1 cho cực tiểu ở $s=0{,}69$; đọc $W^-$ nguyên văn cho $0{,}75$; kiểm định chéo 5 mảnh trên 200 cách
  chia không lần nào chọn dưới $0{,}5714$ (mục 3.10).
- [T] Bảng 2, cột (7): công thức (7) dựng lại cho sai số chuẩn của lcavol 0,0536, bảng in 0,09 (mục 3.7).
- [T] §8.1: dữ liệu kyphosis 83 dòng không tìm được; bản 81 dòng của gói R `rpart` cho mô hình khác (mục 3.14).
- [L] §3.1: *"103 Lasso steps versus 64 for LARS"* trên mô hình bậc hai — số nút của scikit-learn không cùng
  nghĩa với "bước", nên phép so không kết luận được (mục 3.25).

---

### 3.1 Tibshirani 1996, tr.267 — tóm tắt và giới thiệu

![Tibshirani 1996 tr.267 — tóm tắt và §1](<paper_lasso_tib1996_tr267_tom_tat_gioi_thieu.png>)

**Hình 1 — Tibshirani 1996, tr.267: tóm tắt và §1 Introduction.** R. Tibshirani, *Regression Shrinkage and
Selection via the Lasso*, J. R. Statist. Soc. B 58(1):267–288, 1996, bản JSTOR
`_papers/1996 - Tibshirani - Regression Shrinkage and Selection via the Lasso.pdf` (chỉ có trên máy dựng
bài), trang PDF 2.

**Khung chứa gì.** Toàn bộ tóm tắt, dòng từ khoá, và ba đoạn của §1.

**Tác giả nói gì.** Tóm tắt: *"The 'lasso' minimizes the residual sum of squares subject to the sum of the
absolute value of the coefficients being less than a constant. Because of the nature of this constraint it
tends to produce some coefficients that are exactly 0 and hence gives interpretable models."* Và: *"It
produces interpretable models like subset selection and exhibits the stability of ridge regression."*

(Lasso cực tiểu tổng bình phương phần dư với điều kiện tổng trị tuyệt đối các hệ số nhỏ hơn một hằng số. Do
bản chất của ràng buộc này, nó có xu hướng sinh ra một số hệ số đúng bằng 0 và vì thế cho mô hình đọc được.
Nó cho mô hình đọc được như chọn tập con và có độ ổn định của ridge.)

§1 nêu hai lý do không hài lòng với OLS và hai cách cũ: *"Subset selection provides interpretable models but
can be extremely variable because it is a discrete process"*; *"Ridge regression is a continuous process that
shrinks coefficients and hence is more stable: however, it does not set any coefficients to 0"*. Tên gọi:
*"We propose a new technique, called the lasso, for 'least absolute shrinkage and selection operator'."*

(Chọn tập con cho mô hình đọc được nhưng có thể dao động cực mạnh vì nó là một quá trình rời rạc. Ridge là
quá trình liên tục, co hệ số nên ổn định hơn, nhưng không đặt hệ số nào về 0.)

**Cách đọc.** Toàn bộ động cơ của bài báo nằm trong hai câu cuối: chọn tập con **rời rạc** nên nhảy; ridge
**liên tục** nhưng không xoá. Lasso được đề xuất như thứ vừa liên tục vừa xoá được. Mục 3.13 đo xem lời hứa
*"stability of ridge regression"* đứng được tới đâu.

**Điều khung KHÔNG nói.** Tóm tắt dùng cụm *"Our simulation studies suggest"* — tức mọi so sánh với ridge và
chọn tập con ở bài này là **mô phỏng**, không có định lý nào về dự đoán trong dữ liệu tương quan. §10 tr.286
tự nói điều đó.

### 3.2 Tibshirani 1996, tr.268 — định nghĩa (1) và garotte không âm (2)

![Tibshirani 1996 tr.268 — (1) và (2)](<paper_lasso_tib1996_tr268_dinh_nghia_1_garotte_2.png>)

**Hình 2 — Tibshirani 1996, tr.268: §2.1 Definition, công thức (1) và (2).** Trang PDF 3.

**Khung chứa gì.** Giả thiết chuẩn hoá, định nghĩa lasso (1), vai trò của $t$, và garotte của Breiman (2).

$$(\hat\alpha,\hat\beta)=\arg\min\Bigl\{\sum_{i=1}^N\Bigl(y_i-\alpha-\sum_j\beta_jx_{ij}\Bigr)^2\Bigr\}\quad\text{subject to}\ \sum_j\lvert\beta_j\rvert\le t. \tag{1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $y_i$ | đáp án của mẫu $i$ | vô hướng |
| $x_{ij}$ | biến $j$ của mẫu $i$, **đã chuẩn hoá**: $\sum_ix_{ij}/N=0$, $\sum_ix_{ij}^2/N=1$ | vô hướng |
| $\alpha$ | hệ số chặn — nghiệm luôn là $\hat\alpha=\bar y$ | vô hướng |
| $\beta_j$ | hệ số của biến $j$ | vô hướng |
| $t$ | cận của tổng trị tuyệt đối, *"tuning parameter"* | $t\ge0$ |

**Ý nghĩa.** (1) là bình phương tối thiểu **với một ngân sách**. Nếu ngân sách $t$ không nhỏ hơn
$t_0=\sum\lvert\hat\beta^\circ_j\rvert$ thì ràng buộc không chạm tới và nghiệm là OLS. Nếu nhỏ hơn, nghiệm
nằm **trên biên** $\sum\lvert\beta_j\rvert=t$, và biên này có góc — chỗ các hệ số bằng 0. Vì hệ số chặn
không nằm trong ràng buộc, nó chỉ làm đúng việc khớp trung bình, nên được bỏ khỏi bài toán bằng cách trừ
$\bar y$.

$$\sum_{i=1}^N\Bigl(y_i-\alpha-\sum_jc_j\hat\beta^\circ_jx_{ij}\Bigr)^2\quad\text{subject to}\ c_j\ge0,\ \sum c_j\le t. \tag{2}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta^\circ_j$ | hệ số OLS của biến $j$ | vô hướng |
| $c_j$ | **hệ số co** của garotte, không âm | vô hướng |
| $t$ | cận của $\sum c_j$ — khác thang với $t$ của (1) | $0\le t\le p$ có nghĩa |

**Ý nghĩa.** Garotte không tự chọn hệ số mà chọn **mức giữ lại** của từng hệ số OLS. $c_j=1$ giữ nguyên,
$c_j=0$ xoá. Vì thế nó **không bao giờ đổi dấu** một hệ số OLS, và nó **cần OLS tồn tại** — hai điểm mà bài
báo dùng để phân biệt lasso. `v_lasso.py` mục 1: đường garotte giải bằng lasso **dương** trên các cột
$\hat\beta^\circ_jx_j$ kết thúc ở $\sum c_j=6{,}000000=p$ — tức $c=1$ là nghiệm không ràng buộc, đúng như định
nghĩa.

**Tác giả nói gì.** *"Values of t < t0 will cause shrinkage of the solutions towards 0, and some coefficients
may be exactly equal to 0. For example, if t = t0/2, the effect will be roughly similar to finding the best
subset of size p/2."* Về nguồn gốc: *"The motivation for the lasso came from an interesting proposal of
Breiman (1993)."* Và điểm yếu của garotte: *"A drawback of the garotte is that its solution depends on both
the sign and the magnitude of the OLS estimates. In overfit or highly correlated settings where the OLS
estimates behave poorly, the garotte may suffer as a result."*

(Giá trị $t<t_0$ làm nghiệm co về 0 và một số hệ số có thể đúng bằng 0. Ví dụ $t=t_0/2$ có tác dụng gần giống
tìm tập con tốt nhất cỡ $p/2$. Động lực của lasso đến từ một đề xuất của Breiman. Điểm yếu của garotte là
nghiệm phụ thuộc cả dấu lẫn độ lớn của OLS; khi OLS tệ, garotte tệ theo.)

**Cách đọc.** Câu *"$t=t_0/2$ … roughly similar to … best subset of size $p/2$"* là một câu **định lượng**
không kèm số. Trên chính dữ liệu prostate của bài báo, $s=0{,}44$ — gần $t_0/2$ — giữ **3 trên 8** biến
(`v_lasso.py` mục 4), tức ít hơn $p/2=4$. Mục 3.22 đo câu tương tự của ESL.

**Điều khung KHÔNG nói.** (1) không nói **tính duy nhất** của nghiệm. Câu *"the design matrix need not be of
full rank"* ngay dưới cho phép $X$ suy biến, và khi đó $\hat\beta$ có thể không duy nhất dù giá trị cực tiểu
thì duy nhất (mục 6.4).

### 3.3 Tibshirani 1996, tr.269 — thiết kế trực chuẩn: (3), (4), ridge, garotte, tập con

![Tibshirani 1996 tr.269 — (3), (4) và các công thức co](<paper_lasso_tib1996_tr269_truc_chuan_3_4.png>)

**Hình 3 — Tibshirani 1996, tr.269: §2.2 Orthonormal Design Case.** Trang PDF 4.

**Khung chứa gì.** Câu về bridge của Frank và Friedman, công thức (3), dạng ridge có phạt và dạng ràng buộc
(4), nghiệm ridge, nghiệm garotte, và đoạn so sánh bốn dạng co.

$$\hat\beta_j=\operatorname{sign}(\hat\beta^\circ_j)\,(\lvert\hat\beta^\circ_j\rvert-\gamma)^+ \tag{3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta^\circ_j$ | hệ số OLS; khi $X^\top X=I$ thì $\hat\beta^\circ=X^\top y$ | vô hướng |
| $\gamma$ | ngưỡng, chọn sao cho $\sum\lvert\hat\beta_j\rvert=t$ (*"determined by the condition"* tr.269) | $\gamma\ge0$ |
| $(\cdot)^+$ | phần dương, $\max(\cdot,0)$ | |

**Ý nghĩa.** Khi các cột vuông góc, bài toán tách thành $p$ bài toán một biến, và mỗi bài có đúng một cách
dùng ngân sách: **trừ đi cùng một lượng** $\gamma$ khỏi mọi hệ số, hệ số nào không đủ lớn thì dừng ở 0. Ngân
sách $t$ quyết định $\gamma$. Đây là hình dạng của mọi nghiệm lasso — trong trường hợp tổng quát nó chỉ bị
"xoay" bởi tương quan giữa các cột.

$$\sum_{i=1}^N\Bigl(y_i-\sum_j\beta_jx_{ij}\Bigr)^2+\lambda\sum_j\beta_j^2\quad\text{hoặc}\quad\sum_{i=1}^N\Bigl(y_i-\sum_j\beta_jx_{ij}\Bigr)^2\ \text{subject to}\ \sum\beta_j^2\le t \tag{4}$$

$$\hat\beta^{\text{ridge}}_j=\frac{1}{1+\gamma}\hat\beta^\circ_j,\qquad \hat\beta^{\text{garotte}}_j=\Bigl(1-\frac{\gamma}{\hat\beta^{\circ2}_j}\Bigr)^+\hat\beta^\circ_j,\qquad \hat\beta^{\text{subset}}_j=\hat\beta^\circ_j\cdot I(\lvert\hat\beta^\circ_j\rvert>\lambda)$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\lambda$ | mức phạt của ridge ở dạng thứ nhất; ngưỡng của chọn tập con ở dòng cuối — **hai nghĩa** | vô hướng |
| $\gamma$ | ở ridge: *"depends on λ or t"* (ở đây $\gamma=\lambda$); ở garotte: ngưỡng bình phương | vô hướng |
| $I(\cdot)$ | hàm chỉ thị: 1 nếu đúng, 0 nếu sai | |

**Ý nghĩa.** Bốn phương pháp là **bốn cách biến một hệ số OLS thành hệ số mới**: tập con giữ hoặc xoá; ridge
nhân với một hằng số; lasso trừ một hằng số rồi chặn; garotte trừ một lượng **giảm dần** theo độ lớn
($\gamma/\hat\beta^\circ_j$) rồi chặn — nên hệ số lớn gần như không bị co.

**Kiểm lại** (`v_lasso.py` mục 1, $Q$ cỡ $60\times6$ với $\lVert Q^\top Q-I\rVert=6{,}66\cdot10^{-16}$): (3) so với
nghiệm số của (1) ở năm giá trị $t$ lệch lớn nhất $8{,}882\cdot10^{-16}$; ridge $\hat\beta^\circ/(1+\lambda)$ ở
$\lambda=0{,}7$ lệch $2{,}220\cdot10^{-16}$; garotte so với nghiệm số của (2) ở ba giá trị $t$ lệch
$1{,}332\cdot10^{-15}$.

**Tác giả nói gì.** *"Interestingly, this has exactly the same form as the soft shrinkage proposals of Donoho
and Johnstone (1994)"*; và *"Ridge regression scales the coefficients by a constant factor, whereas the lasso
translates by a constant factor, truncating at 0. The garotte function is very similar to the lasso, with less
shrinkage for larger coefficients. As our simulations will show, the differences between the lasso and
garotte can be large when the design is not orthogonal."*

(Dạng này giống hệt các đề xuất co mềm của Donoho và Johnstone. Ridge nhân hệ số với một hằng số, còn lasso
tịnh tiến một lượng hằng rồi chặn tại 0. Garotte rất giống lasso nhưng co ít hơn với hệ số lớn. Mô phỏng sẽ
cho thấy khác biệt giữa lasso và garotte có thể lớn khi thiết kế không trực giao.)

**Cách đọc.** "translates by a constant factor" là cách nói lỏng — tịnh tiến không phải nhân với một
"factor"; ESL tr.69 chép lại nguyên cụm này. Công thức đúng là (3).

**Điều khung KHÔNG nói.** Dòng chọn tập con viết *"for some choice of λ this is equivalent to"* — tức ngưỡng
$\lambda$ phải được chọn **theo dữ liệu** để giữ đúng $k$ hệ số lớn nhất; nó không phải một hằng số cho sẵn
như $\gamma$ của lasso.

### 3.4 Tibshirani 1996, tr.270 — §2.3 hình học và Hình 1

![Tibshirani 1996 tr.270 — §2.3](<paper_lasso_tib1996_tr270_hinh_hoc.png>)

**Hình 4 — Tibshirani 1996, tr.270: §2.3 Geometry of Lasso.** Trang PDF 5.

**Khung chứa gì.** Câu hỏi vì sao có số 0 trong trường hợp không trực giao, dạng bậc hai của hàm mục tiêu,
và câu hỏi về dấu.

$$(\beta-\hat\beta^\circ)^\top X^\top X(\beta-\hat\beta^\circ)\quad(\text{cộng một hằng số})$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $X^\top X$ | ma trận hiệp phương sai (chưa chia $N$) của các biến chuẩn hoá | $p\times p$ |
| $\beta-\hat\beta^\circ$ | khoảng cách từ $\beta$ tới nghiệm OLS | $\mathbb{R}^p$ |

**Ý nghĩa.** Tổng bình phương phần dư là một **cái bát hình elip** có đáy ở $\hat\beta^\circ$. Đường mức của
nó là các elip đồng tâm. Hình dạng elip do $X^\top X$ quyết định: biến càng tương quan, elip càng dẹt và nghiêng.

**Tác giả nói gì.** *"The lasso solution is the first place that the contours touch the square, and this will
sometimes occur at a corner, corresponding to a zero coefficient. The picture for ridge regression is shown in
Fig. 2(b): there are no corners for the contours to hit and hence zero solutions will rarely result."* Về dấu:
*"when p = 2 the principal axes of the contours are at ±45° to the co-ordinate axes, and we can show that the
contours must contact the square in the same quadrant that contains β̂°. However, when p > 2 and there is at
least moderate correlation in the data, this need not be true."*

(Nghiệm lasso là chỗ đầu tiên các đường mức chạm hình vuông, và đôi khi đó là một góc — ứng với một hệ số
bằng 0. Hình tròn của ridge không có góc nên nghiệm bằng 0 hiếm khi xảy ra. Khi $p=2$ trục elip nghiêng
±45° và đường mức phải chạm hình vuông trong cùng góc phần tư với $\hat\beta^\circ$; khi $p>2$ và có tương
quan thì không nhất thiết.)

**Kiểm lại** (`v_lasso.py` mục 2). Với $p=2$: 3 000 bộ dữ liệu, tương quan ngẫu nhiên trong $[-0{,}95;\ 0{,}95]$,
lưới 25 giá trị $s$ — **0** bộ có hệ số lasso khác dấu hệ số OLS. Với $p=3$, tương quan AR với
$\rho\in[0{,}3;\ 0{,}95]$: **23/3000** bộ (0,0077). Ví dụ đầu tiên: $\rho=0{,}4719$,
$\hat\beta^\circ=(0{,}0788;\ -2{,}3278;\ 0{,}7897)$, và tại $s=0{,}46$ lasso cho $(-0{,}0028;\ -1{,}4675;\ 0)$ — hệ
số đầu mang dấu **âm** trong khi OLS dương.

**Điều khung KHÔNG nói.** "Sometimes occur at a corner" không cho tần suất. Và câu *"zero solutions will rarely
result"* cho ridge là đúng theo nghĩa xác suất: ridge có thể cho một hệ số đúng bằng 0 tại một mức phạt cô lập
khi đường của nó đổi dấu, nhưng không bao giờ **giữ** nó ở 0 trên một khoảng.

![Tibshirani 1996 tr.270 — Hình 1](<paper_lasso_tib1996_tr270_hinh1.png>)

**Hình 5 — Tibshirani 1996, Fig. 1, tr.270.** Trang PDF 5.

**Caption gốc.** *"Fig. 1. (a) Subset regression, (b) ridge regression, (c) the lasso and (d) the garotte: ———,
form of coefficient shrinkage in the orthonormal design case; ··········, 45°-line for reference"*

(Hình 1. (a) chọn tập con, (b) ridge, (c) lasso, (d) garotte: nét liền là dạng co hệ số trong thiết kế trực
chuẩn; nét chấm là đường 45° để so.)

**Nhận xét của tác giả.** Tr.269 (đã dẫn ở mục 3.3): *"Ridge regression scales the coefficients by a constant
factor, whereas the lasso translates by a constant factor, truncating at 0."* Và tr.270 mở §2.3: *"It is clear
from Fig. 1 why the lasso will often produce coefficients that are exactly 0."*

**Cách đọc hình.** Trục ngang là $\hat\beta^\circ_j$ từ 0 tới 5, trục dọc là hệ số sau khi co. Ba panel (a), (c),
(d) cùng có ngưỡng tại 2: dưới 2 thì hệ số về 0. Khác nhau ở phần trên ngưỡng. Tập con **nhảy** lên đường 45°.
Lasso là đường thẳng song song với đường 45° nhưng thấp hơn đúng 2 — tại $\hat\beta^\circ=5$ nó ở 3. Garotte cong
dần về đường 45° — tại 5 nó ở $5-4/5=4{,}2$. Các con số này ứng với $\gamma=2$ cho lasso và $\gamma=4$ cho
garotte (vì $1-4/\hat\beta^{\circ2}=0$ đúng tại $\hat\beta^\circ=2$); `v_lasso.py` mục 1 in
`lasso γ = 2 : [0. 0. 0.5 3. ]` và `garotte γ = 4 : [0. 0. 0.9 4.2]` tại $\hat\beta^\circ=(1;\ 2;\ 2{,}5;\ 5)$. Hình 51
panel (a) vẽ lại cả bốn.

**Điều hình KHÔNG nói.** Panel (b) không ghi $\gamma$ của ridge, nên không suy ngược được mức phạt; Hình 51 dùng
$\gamma=2$ (hệ số $1/3$) chỉ để minh hoạ. Và hình
chỉ đúng cho thiết kế trực chuẩn — tr.269 tự cảnh báo khác biệt lasso/garotte *"can be large"* khi thiết kế
không trực giao.

### 3.5 Tibshirani 1996, tr.271 — Hình 2, Hình 3, và garotte không đổi dấu

![Tibshirani 1996 tr.271 — Hình 2 và Hình 3](<paper_lasso_tib1996_tr271_hinh2_hinh3.png>)

**Hình 6 — Tibshirani 1996, Fig. 2 và Fig. 3, tr.271.** Trang PDF 6.

**Caption gốc.** *"Fig. 2. Estimation picture for (a) the lasso and (b) ridge regression"* và *"Fig. 3. (a)
Example in which the lasso estimate falls in an octant different from the overall least squares estimate; (b)
overhead view"*

(Hình 2. Bức tranh ước lượng cho (a) lasso và (b) ridge. Hình 3. (a) Ví dụ nghiệm lasso rơi vào một bát phân
khác với nghiệm bình phương tối thiểu; (b) nhìn từ trên xuống.)

**Nhận xét của tác giả.** Tr.270 (mục 3.4): *"The elliptical contours of this function are shown by the full
curves in Fig. 2(a); they are centred at the OLS estimates; the constraint region is the rotated square."* Và
*"Fig. 3 shows an example in three dimensions. The view in Fig. 3(b) confirms that the ellipse touches the
constraint region in an octant different from the octant in which its centre lies."* Lời cảm ơn tr.287:
*"David Andrews for producing Fig. 3 in MATHEMATICA"*.

**Cách đọc hình.** Hình 2: điểm $\hat\beta$ là tâm elip. Elip ngoài cùng chạm hình vuông đen ở **góc trên**
(trục đứng) — nên $\hat\beta_1=0$ — còn chạm hình tròn ở một điểm lệch khỏi trục, nên cả hai hệ số khác 0. Hình
3: khối bát diện là miền $\sum\lvert\beta_j\rvert\le t$ trong ba chiều; elipxoit chạm nó ở một mặt nằm ở bát phân
khác với tâm elipxoit. Mục 3.4 đo tần suất của hiện tượng đó: 23 trên 3 000.

**Điều hình KHÔNG nói.** Hình 3 không cho $\hat\beta^\circ$, $X^\top X$ hay $t$, nên không dựng lại được; ví dụ ở
mục 3.4 là một ví dụ khác cùng tính chất. Hình 2 vẽ elip nghiêng đúng 45° — tr.270 nói điều đó chỉ bắt buộc
khi $p=2$ và biến chuẩn hoá.

![Tibshirani 1996 tr.271 — garotte đổi dấu, mở §2.4](<paper_lasso_tib1996_tr271_garotte_hai_bien.png>)

**Hình 7 — Tibshirani 1996, tr.271: garotte giữ dấu, lasso thì không; mở đầu §2.4.** Trang PDF 6.

**Khung chứa gì.** Đoạn so sánh dấu và đoạn đầu §2.4.

**Tác giả nói gì.** *"Whereas the garotte retains the sign of each β̂ⱼ°, the lasso can change signs."* Và:
*"The model Σ cⱼβ̂ⱼ°xᵢⱼ with constraint Σ cⱼ ⩽ t can be written as Σ βⱼxᵢⱼ with constraint Σ βⱼ/β̂ⱼ° ⩽ t. If for
example p = 2 and β̂₁° > β̂₂° > 0 then the effect would be to stretch the square in Fig. 2(a) horizontally."*

(Garotte giữ dấu từng hệ số OLS, lasso thì có thể đổi dấu. Mô hình garotte viết lại được thành ràng buộc
$\sum\beta_j/\hat\beta^\circ_j\le t$; với $p=2$ và $\hat\beta^\circ_1>\hat\beta^\circ_2>0$, tác dụng là kéo giãn hình
vuông theo chiều ngang.)

**Cách đọc.** Đặt $\beta_j=c_j\hat\beta^\circ_j$: ràng buộc $\sum c_j\le t$ thành $\sum\beta_j/\hat\beta^\circ_j\le t$ cùng
điều kiện $\beta_j/\hat\beta^\circ_j\ge0$. Hệ số OLS càng lớn thì miền càng dài theo trục đó — garotte "ưu ái"
biến OLS đã thấy lớn. Đó là lý do nó phụ thuộc chất lượng của OLS (mục 3.2).

**Điều khung KHÔNG nói.** Miền của garotte **không đối xứng**: nó chỉ nằm trong góc phần tư cùng dấu với
$\hat\beta^\circ$, nên gọi là "hình vuông kéo giãn" chỉ đúng cho phần trong góc phần tư đó.

### 3.6 Tibshirani 1996, tr.272 — công thức (5), (6) và Hình 4

![Tibshirani 1996 tr.272 — Hình 4, (5), (6)](<paper_lasso_tib1996_tr272_hinh4_5_6.png>)

**Hình 8 — Tibshirani 1996, Fig. 4 và công thức (5)–(6), tr.272.** Trang PDF 7.

**Caption gốc.** *"Fig. 4. Lasso (———) and ridge regression (– – – –) for the two-predictor example: the curves
show the (β₁, β₂) pairs as the bound on the lasso or ridge parameters is varied; starting with the bottom
broken curve and moving upwards, the correlation ρ is 0, 0.23, 0.45, 0.68 and 0.90"*

(Hình 4. Lasso (nét liền) và ridge (nét đứt) cho ví dụ hai biến: mỗi đường là các cặp $(\beta_1,\beta_2)$ khi cận
thay đổi; từ đường đứt dưới cùng trở lên, tương quan $\rho$ là 0; 0,23; 0,45; 0,68 và 0,90.)

$$\hat\beta_j=(\hat\beta^\circ_j-\gamma)^+ \tag{5}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta^\circ_1,\hat\beta^\circ_2$ | hai hệ số OLS, giả sử **cùng dương** | vô hướng |
| $\gamma$ | *"chosen so that β̂₁ + β̂₂ = t"* | vô hướng |
| $t$ | cận, *"This formula holds for t ⩽ β̂₁° + β̂₂°"* | vô hướng |

**Ý nghĩa.** Với hai biến dương, lasso **trừ cùng một lượng** khỏi cả hai — **bất kể tương quan**. Lý do:
điều kiện tối ưu buộc hai hệ số còn hoạt động có cùng tương quan với phần dư, và với biến chuẩn hoá điều đó
tương đương trừ $\lambda/(1+\rho)$ khỏi cả hai. Tương quan chỉ đổi **tốc độ** theo $\lambda$, không đổi **đường**.

$$\hat\beta_1=\Bigl(\frac t2+\frac{\hat\beta^\circ_1-\hat\beta^\circ_2}{2}\Bigr)^+,\qquad \hat\beta_2=\Bigl(\frac t2-\frac{\hat\beta^\circ_1-\hat\beta^\circ_2}{2}\Bigr)^+ \tag{6}$$

**Ý nghĩa.** (6) là (5) sau khi giải $\gamma=(\hat\beta^\circ_1+\hat\beta^\circ_2-t)/2$. Nhưng phép chặn $(\cdot)^+$ đặt
**riêng cho từng hệ số** làm công thức sai ở một miền: khi $t<\hat\beta^\circ_1-\hat\beta^\circ_2$, dòng thứ hai bị
chặn về 0 còn dòng thứ nhất vẫn giữ giá trị $(t+\hat\beta^\circ_1-\hat\beta^\circ_2)/2>t$, tức vi phạm chính ràng
buộc $\hat\beta_1+\hat\beta_2=t$. Trong miền đó nghiệm đúng là $(t,\ 0)$.

**Kiểm lại** (`v_lasso.py` mục 3). Hai biến tương quan mẫu 0,6945, $\hat\beta^\circ=(2{,}099408;\ 1{,}209853)$: (5)/(6)
với $\lvert\hat\beta^\circ_1-\hat\beta^\circ_2\rvert\le t\le\hat\beta^\circ_1+\hat\beta^\circ_2$ lệch nghiệm số
$9{,}992\cdot10^{-16}$. Ngoài miền đó, với $\hat\beta^\circ=(6,3)$ của Hình 4:

| $t$ | (6) in ra | tổng | nghiệm đúng |
|---:|---|---:|---|
| 0,5 | $(1{,}75;\ 0)$ | 1,75 | $(0{,}5;\ 0)$ |
| 1,0 | $(2{,}00;\ 0)$ | 2,00 | $(1{,}0;\ 0)$ |
| 2,0 | $(2{,}50;\ 0)$ | 2,50 | $(2{,}0;\ 0)$ |
| 2,9 | $(2{,}95;\ 0)$ | 2,95 | $(2{,}9;\ 0)$ |

**Tác giả nói gì.** *"This formula holds for t ⩽ β̂₁° + β̂₂° and is valid even if the predictors are
correlated."* Và về Hình 4: *"For all values of ρ the lasso estimates follow the full curve. The ridge estimates
(broken curves) depend on ρ. When ρ = 0 ridge regression does proportional shrinkage. However, for larger
values of ρ the ridge estimates are shrunken differentially and can even increase a little as the bound is
decreased. As pointed out by Jerome Friedman, this is due to the tendency of ridge regression to try to make the
coefficients equal to minimize their squared norm."*

(Với mọi $\rho$, nghiệm lasso đi theo đường liền. Nghiệm ridge phụ thuộc $\rho$: khi $\rho=0$ ridge co theo tỉ lệ;
$\rho$ lớn hơn thì các hệ số bị co khác nhau và thậm chí **tăng** một chút khi cận giảm. Theo Jerome Friedman,
đó là vì ridge cố làm các hệ số bằng nhau để cực tiểu tổng bình phương.)

**Dựng lại Hình 4** (`v_lasso.py` mục 3): 100 điểm, $y=6x_1+3x_2$ không nhiễu, biến chuẩn hoá theo mẫu. Đường
lasso của năm giá trị $\rho$ lệch nhau lớn nhất $2{,}620\cdot10^{-14}$. Đỉnh của $\hat\beta_2$ ridge dọc đường:

| $\rho$ in | $\rho$ mẫu | $\hat\beta_2$ ridge lớn nhất | tại $\hat\beta_1=$ |
|---:|---:|---:|---:|
| 0,00 | 0,1030 | 3,000000 | 6,000000 |
| 0,23 | 0,3257 | 3,000000 | 6,000000 |
| 0,45 | 0,5292 | 3,003287 | 5,834791 |
| 0,68 | 0,7324 | 3,180837 | 5,053579 |
| 0,90 | 0,9182 | 3,646432 | 4,638886 |

**Cách đọc hình.** Trục ngang $\beta_1$, trục dọc $\beta_2$. Mọi đường xuất phát từ $(6,3)$ ở mép phải (cận lớn,
chưa co) và đi về gốc khi cận giảm. Đường liền của lasso đi thẳng xuống $(3,0)$ — trừ đều cả hai — rồi chạy
ngang theo trục $\beta_1$: $\beta_2$ đã về 0. Đường đứt trên cùng ($\rho=0{,}90$) **đi lên** trước khi đi xuống:
$\beta_2$ ridge tăng từ 3 lên 3,646432 trong khi $\beta_1$ giảm từ 6 xuống 4,638886. Bảng trên cho thấy hiện tượng
bắt đầu từ $\rho=0{,}45$ và rõ dần.

**Điều hình KHÔNG nói.** Đường liền trong bản quét bắt đầu cách gốc một đoạn chứ không từ gốc, nên hình không
cho biết đoạn $t$ nhỏ; đó đúng là đoạn mà (6) in sai. Hình 51 panel (b) vẽ lại, tô đỏ phần (6) sẽ cho ra nếu áp
nguyên văn. Và $\rho$ mẫu lệch khá xa $\rho$ danh nghĩa với $n=100$ (0,1030 cho $\rho=0$) — bài báo không nói nó
dùng tương quan mẫu hay tổng thể.

### 3.7 Tibshirani 1996, tr.272–273 — §2.5 sai số chuẩn và công thức (7)

![Tibshirani 1996 tr.272 — §2.4 cuối, §2.5](<paper_lasso_tib1996_tr272_sai_so_chuan.png>)

**Hình 9 — Tibshirani 1996, tr.272: phần chữ dưới (6) và đầu §2.5 Standard Errors.** Trang PDF 7.

**Khung chứa gì.** Đoạn ví dụ Hình 4 (đã dẫn ở mục 3.6) và hai đoạn đầu §2.5.

**Tác giả nói gì.** *"Since the lasso estimate is a non-linear and non-differentiable function of the response
values even for a fixed value of t, it is difficult to obtain an accurate estimate of its standard error. One
approach is via the bootstrap: either t can be fixed or we may optimize over t for each bootstrap sample."* Và
*"An approximate closed form estimate may be derived by writing the penalty Σ|βⱼ| as Σ βⱼ²/|βⱼ|."*

(Vì nghiệm lasso là hàm phi tuyến, không khả vi của $y$ ngay cả khi $t$ cố định, rất khó ước lượng chính xác sai
số chuẩn của nó. Một cách là bootstrap — cố định $t$ hoặc chọn lại $t$ cho từng mẫu. Một công thức xấp xỉ có
được bằng cách viết khoản phạt $\sum\lvert\beta_j\rvert$ thành $\sum\beta_j^2/\lvert\beta_j\rvert$.)

![Tibshirani 1996 tr.273 — công thức (7)](<paper_lasso_tib1996_tr273_cong_thuc_7.png>)

**Hình 10 — Tibshirani 1996, tr.273: phép xấp xỉ ridge và công thức (7).** Trang PDF 8.

**Khung chứa gì.** Nửa sau của câu xấp xỉ, công thức (7), và hai câu nhận xét.

$$\beta^*=(X^\top X+\lambda W^-)^{-1}X^\top y,\qquad \widehat{\operatorname{cov}}=(X^\top X+\lambda W^-)^{-1}X^\top X(X^\top X+\lambda X^-)^{-1}\hat\sigma^2 \tag{7}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $W$ | $\operatorname{diag}(\lvert\tilde\beta_j\rvert)$ tại nghiệm lasso | $p\times p$ |
| $W^-$ | *"the generalized inverse of W"* | $p\times p$ |
| $\lambda$ | chọn sao cho $\sum\lvert\beta^*_j\rvert=t$; bản in đặt dấu sao **sau** dấu trị tuyệt đối | vô hướng |
| $X^-$ | in ở thừa số thứ hai — **phải là $W^-$** | $X^-$ có cỡ $p\times N$ |
| $\hat\sigma^2$ | ước lượng phương sai nhiễu | vô hướng |

**Ý nghĩa.** Viết $\lvert\beta_j\rvert=\beta_j^2/\lvert\beta_j\rvert$ rồi **đóng băng** mẫu số tại nghiệm lasso, bài
toán thành một ridge có mức phạt khác nhau cho từng hệ số: hệ số nhỏ bị phạt mạnh, hệ số lớn bị phạt nhẹ.
Ridge là tuyến tính theo $y$ nên có công thức hiệp phương sai dạng "bánh kẹp" — (7). Cái giá: phép đóng băng bỏ
qua việc **tập biến khác 0 cũng thay đổi theo $y$**, mà đó chính là nguồn biến động lớn nhất.

**Kiểm lại** (`v_lasso.py` mục 5, prostate bản có lỗi, $s=0{,}44$):

- $\lambda$ thoả $(X^\top X+\lambda W^-)\beta=X^\top y$ bằng **17,892329**, và ba giá trị $\lvert x_j^\top(y-X\beta)\rvert$
  của ba biến khác 0 đều bằng đúng số đó. Kiểm $(X_A^\top X_A+\lambda W^-)^{-1}X_A^\top y=\hat\beta_A$: lệch
  $1{,}110\cdot10^{-16}$. Ở §5, mức phạt của **cùng** nghiệm này trong dạng $\text{RSS}+\lambda\sum\lvert\beta_j\rvert$
  là 35,784658 — đúng gấp đôi, vì đạo hàm của RSS mang hệ số 2.
- Thừa số thứ hai $X^\top X+\lambda X^-$: $X^-$ cỡ $8\times97$ còn $X^\top X$ cỡ $8\times8$ — phép cộng không xác
  định. Chỉ đọc là $W^-$ thì (7) mới có nghĩa.
- Đọc $W^-$ đúng nghĩa **nghịch đảo suy rộng** (ô 0 thì nghịch đảo là 0, tức **không phạt**): sai số chuẩn của
  tám hệ số là $0{,}061;\ 0{,}0233;\ 0{,}0812;\ 0{,}0779;\ 0{,}0302;\ 0{,}1044;\ 0{,}1111;\ 0{,}1232$ — các biến có hệ
  số 0 **không** nhận phương sai 0, trái với câu *"it gives an estimated variance of 0 for predictors with
  β̂ⱼ = 0"*. Câu đó chỉ đúng nếu các cột có hệ số 0 bị **loại** hẳn (phạt vô hạn).
- Theo cách loại cột, với $\hat\sigma^2=0{,}501849$ ($\text{RSS}/(N-p-1)$ của LS đầy đủ): sai số chuẩn
  $0{,}0719;\ 0{,}0536;\ 0{,}0245;\ 0;\ 0;\ 0{,}0312;\ 0;\ 0;\ 0$. Cột (7) của Bảng 2 in
  $0{,}07;\ 0{,}09;\ 0{,}06;\ 0;\ 0;\ 0{,}07;\ 0;\ 0;\ 0$ — khớp 6/9 ô, và ba ô lệch đều là ba biến khác 0.

**Tác giả nói gì.** *"A difficulty with this formula is that it gives an estimated variance of 0 for predictors
with β̂ⱼ = 0."* Và *"This approximation also suggests an iterated ridge regression algorithm for computing the
lasso estimate itself, but this turns out to be quite inefficient. However, it is useful for selection of the
lasso parameter t (Section 4)."*

(Khó khăn của công thức là nó cho phương sai 0 với biến có hệ số 0. Phép xấp xỉ gợi ý một thuật toán ridge
lặp để tính chính lasso, nhưng hoá ra rất kém hiệu quả; nó hữu ích cho việc chọn $t$.)

**Điều khung KHÔNG nói.** Không nói $\hat\sigma^2$ lấy từ mô hình nào, và không nói $\lambda$ ở (7) thuộc dạng phạt
nào. `v_lasso.py` mục 5 thử thêm các cách đọc khác, vẫn loại cột có hệ số 0:

| Cách đọc | lcavol | lweight | svi |
|---|---:|---:|---:|
| bánh kẹp (7), $\lambda\times1$ | 0,0536 | 0,0245 | 0,0312 |
| bánh kẹp, $\lambda\times0{,}25$ | 0,072 | 0,0491 | 0,0591 |
| bánh kẹp, $\lambda\times0{,}5$ | 0,0637 | 0,0368 | 0,0453 |
| bánh kẹp, $\lambda\times2$ | 0,0422 | 0,0148 | 0,0195 |
| bánh kẹp, $\lambda\times4$ | 0,0301 | 0,0082 | 0,0114 |
| bỏ thừa số giữa, $\lambda\times1$ | 0,0662 | 0,0425 | 0,0514 |
| bỏ thừa số giữa, $\lambda\times0{,}25$ | 0,0787 | 0,0601 | 0,0712 |
| **Bảng 2 in** | **0,09** | **0,06** | **0,07** |

Không cách nào cho đồng thời ba số in; gần nhất là cách cuối, lệch ở lcavol. Đây là chỗ bỏ ngỏ thứ hai trong bảng
đầu mục 3. Và mọi cách đều cho sai số chuẩn **nhỏ hơn** bootstrap (mục 3.9) — phép đóng băng tập biến bỏ qua đúng
nguồn dao động lớn nhất.

### 3.8 Tibshirani 1996, tr.273–274 — §3 dữ liệu prostate, Hình 5, Bảng 1

![Tibshirani 1996 tr.273 — §3 và Hình 5](<paper_lasso_tib1996_tr273_prostate_hinh5.png>)

**Hình 11 — Tibshirani 1996, §3 và Fig. 5, tr.273.** Trang PDF 8.

**Caption gốc.** *"Fig. 5. Lasso shrinkage of coefficients in the prostate cancer example: each curve represents
a coefficient (labelled on the right) as a function of the (scaled) lasso parameter s = t/Σ|β̂ⱼ°| (the intercept
is not plotted); the broken line represents the model for ŝ = 0.44, selected by generalized
cross-validation"*

(Hình 5. Đường co hệ số lasso trên dữ liệu ung thư tuyến tiền liệt: mỗi đường là một hệ số, đánh số ở bên
phải, theo tham số chuẩn hoá $s=t/\sum\lvert\hat\beta^\circ_j\rvert$; không vẽ hệ số chặn; nét đứt là mô hình ở
$\hat s=0{,}44$, chọn bằng kiểm định chéo tổng quát.)

**Nhận xét của tác giả.** Tr.273 mô tả dữ liệu: *"The prostate cancer data come from a study by Stamey et al.
(1989) that examined the correlation between the level of prostate specific antigen and a number of clinical
measures, in men who were about to receive a radical prostatectomy."* Tr.274: *"Notice that the absolute value
of each coefficient tends to 0 as s goes to 0. In this example, the curves decrease in a monotone fashion to 0,
but this does not always happen in general."* Và *"The vertical broken line represents the model for ŝ = 0.44,
the optimal value selected by generalized cross-validation. Roughly, this corresponds to keeping just under
half of the predictors."*

(Dữ liệu từ nghiên cứu của Stamey và cộng sự về tương quan giữa mức kháng nguyên đặc hiệu tuyến tiền liệt và
các chỉ số lâm sàng, trên bệnh nhân sắp cắt bỏ tuyến tiền liệt. Trị tuyệt đối của mọi hệ số tiến về 0 khi $s$
tiến về 0; ở ví dụ này các đường giảm đơn điệu, nhưng không phải lúc nào cũng vậy. Nét đứt là $\hat s=0{,}44$,
chọn bằng GCV — đại khái giữ gần một nửa số biến.)

**Dựng lại** (`v_lasso.py` mục 4, bản dữ liệu có lỗi — xem ngay dưới). Thứ tự biến rời khỏi 0 và giá trị $s$ tại
nút: lcavol 0,0000 · svi 0,2284 · lweight 0,3158 · lbph 0,4761 · pgg45 0,4845 · age 0,6135 · gleason 0,7065 ·
lcp 0,7460. $\lvert\hat\beta_j(s)\rvert$ không giảm theo $s$ ở **8/8** biến trên lưới 2 001 điểm. Tại $s=1$: hệ số
$0{,}6919;\ 0{,}2257;\ -0{,}1462;\ 0{,}1553;\ 0{,}3172;\ -0{,}1475;\ 0{,}0326;\ 0{,}1276$. Tại $s=0{,}44$: ba biến khác
0 — lcavol 0,5588, lweight 0,097, svi 0,1556.

**Cách đọc hình.** Trục ngang $s$ từ 0 (mọi hệ số bằng 0) tới 1 (OLS). Số ở mép phải là số thứ tự biến trong
Bảng 1 (1 là hệ số chặn): 2 lcavol, 3 lweight, 4 age, 5 lbph, 6 svi, 7 lcp, 8 gleason, 9 pgg45. Đường 2 rời 0 ngay
từ đầu và luôn lớn nhất. Đường 6 (svi) rời 0 **trước** đường 3 (lweight), khớp thứ tự dựng lại (0,2284 so với
0,3158). Nét đứt ở 0,44 cắt ba đường khác 0. Hai đường đi xuống dưới 0 là 4 và 7, bắt đầu quanh 0,61 và 0,75.

**Điều hình KHÔNG nói.** $s=0{,}44$ giữ **3/8 = 0,3750** số biến — "just under half" là cách nói rộng tay. Và
hình không cho biết GCV được tính theo cách nào; mục 3.10 cho thấy công thức (10) trên dữ liệu này **không** cho
0,44.

![Tibshirani 1996 tr.274 — Bảng 1 và Bảng 2](<paper_lasso_tib1996_tr274_bang1_bang2.png>)

**Hình 12 — Tibshirani 1996, Table 1 và Table 2, tr.274.** Trang PDF 9.

**Caption gốc.** *"TABLE 1. Results for the prostate cancer example"* và *"TABLE 2. Standard error estimates for
the prostate cancer example"*

(Bảng 1. Kết quả cho ví dụ ung thư tuyến tiền liệt. Bảng 2. Các ước lượng sai số chuẩn cho ví dụ đó.)

**Nhận xét của tác giả.** *"The lasso gave non-zero coefficients to lcavol, lweight and svi; subset selection
chose the same three predictors. Notice that the coefficients and Z-scores for the selected predictors from
subset selection tend to be larger than the full model values: this is common with positively correlated
predictors. However, the lasso shows the opposite effect, as it shrinks the coefficients and Z-scores from their
full model values."* Và *"The standard errors in the penultimate column were estimated by bootstrap resampling of
residuals from the full least squares fit. The standard errors were computed by fixing ŝ at its optimal value
0.44 for the original data set."*

(Lasso cho hệ số khác 0 ở lcavol, lweight và svi; chọn tập con chọn đúng ba biến đó. Hệ số và Z-score của tập
con có xu hướng lớn hơn mô hình đầy đủ — thường gặp khi biến tương quan dương. Lasso cho hiệu ứng ngược lại vì
nó co cả hệ số lẫn Z-score. Sai số chuẩn ở cột áp chót ước lượng bằng bootstrap phần dư của LS đầy đủ, với
$\hat s$ cố định ở 0,44.)

**Dựng lại Bảng 1 — và phát hiện bản dữ liệu.** ESL tr.3 (mục 3.20) kể rằng bản dữ liệu prostate đầu tiên có
một lỗi: lweight của đối tượng 32 là 6,1, tức tuyến tiền liệt nặng 449 gm, trong khi đúng là 44,9 gm. File
`tools/prostate.csv` là bản **đã sửa**. `v_lasso.py` mục 4 dựng lại Bảng 1 trên cả hai bản:

| Cột của Bảng 1 | Bản đã sửa (csv) | Bản có lỗi, lweight₃₂ $=\ln449$ |
|---|---|---|
| LS — hệ số | $2{,}4784\ 0{,}6651\ 0{,}2665\ -0{,}1582\ 0{,}1403\ 0{,}3153\ -0{,}1483\ 0{,}0355\ 0{,}1257$ · **4/9** ô | $2{,}4784\ 0{,}6919\ 0{,}2257\ -0{,}1462\ 0{,}1553\ 0{,}3172\ -0{,}1475\ 0{,}0326\ 0{,}1276$ · **9/9** ô |
| LS — Z-score | $34{,}8953\ 6{,}4251\ 3{,}0962\ -1{,}917\ 1{,}67\ 3{,}1582\ -1{,}1801\ 0{,}3169\ 1{,}0211$ · **1/9** | $34{,}4563\ 6{,}6767\ 2{,}6733\ -1{,}7577\ 1{,}8315\ 3{,}136\ -1{,}1589\ 0{,}2867\ 1{,}0236$ · **9/9** |
| lasso $s=0{,}44$ | $2{,}4784\ 0{,}535\ 0{,}1316\ 0\ 0\ 0{,}1496\ 0\ 0\ 0$ · **6/9** | $2{,}4784\ 0{,}5588\ 0{,}097\ 0\ 0\ 0{,}1556\ 0\ 0\ 0$ · **9/9** |

$\ln449=6{,}107023$; giá trị trong csv là $3{,}8044377947$, tức $e^{3{,}8044}=44{,}9000$ gm; hiệu hai giá trị
$2{,}302585=\ln10$ — đúng một lỗi **dấu thập phân**. Thêm một chi tiết: bài báo viết $\sum_ix_{ij}^2/N=1$, nhưng
hệ số LS khớp **9/9** ô khi chia cho $N-1$ và chỉ **7/9** khi chia cho $N$.

Cột tập con: OLS trên $\{$lcavol, lweight, svi$\}$ cho hệ số $2{,}4784;\ 0{,}6502;\ 0{,}2526;\ 0{,}2758$, SE
$0{,}0728;\ 0{,}088;\ 0{,}0746;\ 0{,}0868$, Z $34{,}0528;\ 7{,}3878;\ 3{,}3866;\ 3{,}1756$ — làm tròn ra đúng từng ô in.

Hai điều Bảng 1 tự lộ, không cần dữ liệu:

- **Z-score của cột lasso là thương hai số đã làm tròn.** $2{,}48/0{,}07=35{,}4286$, $0{,}56/0{,}09=6{,}2222$,
  $0{,}10/0{,}07=1{,}4286$, $0{,}16/0{,}09=1{,}7778$ — in 35,43 · 6,22 · 1,43 · 1,78. Cột LS thì không: $0{,}69/0{,}10=6{,}90$
  nhưng in 6,68.
- **"Hệ số … của tập con có xu hướng lớn hơn"** đúng cho Z-score (**3/3**: 7,39 > 6,68; 3,39 > 2,67; 3,18 > 3,14)
  nhưng chỉ đúng **1/3** cho hệ số (0,65 < 0,69; 0,25 > 0,23; 0,28 < 0,32).

**Cách đọc bảng.** Mỗi khối ba cột là một phương pháp: hệ số, sai số chuẩn, Z-score (hệ số chia sai số chuẩn).
Dòng 1 là hệ số chặn, luôn bằng $\bar y=2{,}48$. Dấu "—" ở cột Z của tập con là biến không được chọn. Ở cột lasso,
biến có hệ số 0 vẫn có sai số chuẩn khác 0 (0,01–0,04) — đó là độ dao động bootstrap, mục 3.9.

**Điều bảng KHÔNG nói.** Không bảng nào ghi bản dữ liệu. Không có dòng nào cho biết sai số dự đoán của ba mô
hình — Bảng 1 chỉ so **hệ số**, nên nó không trả lời được mô hình nào tốt hơn.

### 3.9 Tibshirani 1996, tr.274–275 — Bảng 2 và Hình 6: bootstrap

Bảng 2 nằm trong Hình 12 ở mục 3.8.

![Tibshirani 1996 tr.275 — Hình 6, §4, (8)](<paper_lasso_tib1996_tr275_hinh6_muc4_8.png>)

**Hình 13 — Tibshirani 1996, Fig. 6, đoạn bàn Bảng 2, và đầu §4 với công thức (8), tr.275.** Trang PDF 10.

**Caption gốc.** *"Fig. 6. Box plots of 200 bootstrap values of the lasso coefficient estimates for the eight
predictors in the prostate cancer example"*

(Hình 6. Hộp ria của 200 giá trị bootstrap của hệ số lasso cho tám biến trong ví dụ prostate.)

**Nhận xét của tác giả.** *"Table 2 compares the ridge approximation formula (7) with the fixed t bootstrap, and
the bootstrap in which t was re-estimated for each sample. The ridge formula gives a fairly good approximation to
the fixed t bootstrap, except for the zero coefficients. Allowing t to vary incorporates an additional source of
variation and hence gives larger standard error estimates."* Và *"The predictors whose estimated coefficient is 0
exhibit skewed bootstrap distributions. The central 90% percentile intervals (fifth and 95th percentiles of the
bootstrap distributions) all contained the value 0, with the exceptions of those for lcavol and svi."*

(Bảng 2 so công thức ridge (7) với bootstrap cố định $t$ và bootstrap chọn lại $t$ cho từng mẫu. Công thức ridge
xấp xỉ khá tốt bootstrap cố định $t$, trừ các hệ số bằng 0. Cho $t$ thay đổi thêm một nguồn dao động nên sai số
chuẩn lớn hơn. Các biến có hệ số 0 có phân phối bootstrap lệch. Khoảng phân vị 90 % đều chứa 0, trừ lcavol và
svi.)

**Dựng lại** (`v_lasso.py` mục 5): 200 bản bootstrap phần dư của LS đầy đủ trên bản dữ liệu có lỗi, hạt giống
11. Cột "cố định" giữ $s=0{,}44$ trên từng bản; cột "đổi $s$" chọn lại $s$ bằng GCV (10) cho từng bản.

| Biến | cố định | Bảng 2 "Fixed t" | Bảng 1 cột SE lasso | đổi $s$ | Bảng 2 "Varying t" | phân vị 5 % | phân vị 95 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| intcpt | 0,0643 | 0,07 | 0,07 | 0,0643 | 0,07 | 2,3862 | 2,5870 |
| lcavol | 0,0895 | 0,08 | 0,09 | 0,0976 | 0,10 | 0,4137 | 0,7025 |
| lweight | 0,0685 | 0,06 | 0,07 | 0,0791 | 0,08 | 0,0000 | 0,2106 |
| age | 0,0124 | 0,04 | 0,01 | 0,0809 | 0,05 | −0,0088 | 0,0000 |
| lbph | 0,0399 | 0,04 | 0,04 | 0,0818 | 0,07 | 0,0000 | 0,1215 |
| svi | 0,0835 | 0,09 | 0,09 | 0,0974 | 0,09 | 0,0040 | 0,2853 |
| lcp | 0,0252 | 0,03 | 0,03 | 0,0865 | 0,07 | 0,0000 | 0,0426 |
| gleason | 0,0174 | 0,02 | 0,02 | 0,0626 | 0,05 | 0,0000 | 0,0325 |
| pgg45 | 0,0365 | 0,03 | 0,03 | 0,0828 | 0,06 | 0,0000 | 0,1059 |

Làm tròn hai chữ số, cột "cố định" khớp **Bảng 1** ở 6/9 ô nhưng chỉ khớp cột "Fixed t" của **Bảng 2** ở 3/9 ô.
Cột "đổi $s$" khớp cột "Varying t" ở 2/9 ô — GCV ở đây chọn $s$ lớn hơn hẳn 0,44 (ngay dưới), nên mô hình trên từng bản
bootstrap khác với bài báo.

Khoảng 5–95 % **không** chứa 0 ở đúng hai biến lcavol và svi — khớp câu của tác giả. Số lần một hệ số bằng
đúng 0 trong 200 bản: lcavol 0, lweight 22, age 184, lbph 121, svi 9, lcp 178, gleason 174, pgg45 117. GCV chọn
lại $s$ cho các bản bootstrap với trung vị 0,77, trong khoảng $[0{,}58;\ 0{,}91]$ — không bản nào gần 0,44.

**Cách đọc hình.** Mỗi hộp là một biến. Hộp của lcavol nằm hẳn trên 0 quanh 0,5; hộp của svi cũng trên 0. Các
biến còn lại có hộp **dẹt sát 0** với vài chấm lẻ phía trên — đúng hình dạng "lệch" mà tác giả nói: phần lớn
bản bootstrap cho 0, một số ít cho giá trị dương. Bảng số trên cho con số: age bằng 0 ở 184/200 bản.

**Điều hình KHÔNG nói.** Bảng 1 nói cột SE của lasso là bootstrap với $\hat s$ cố định, tức cùng phép đo với cột
"Fixed t" của Bảng 2 — nhưng hai cột **lệch nhau ở 3/9 ô** (lcavol 0,09 so với 0,08; lweight 0,07 so với 0,06;
age 0,01 so với 0,04). Nhiều khả năng đó là hai lần chạy bootstrap khác nhau; bài báo không nói.

$$\text{PE}=E\{Y-\hat\eta(X)\}^2=\text{ME}+\sigma^2 \tag{8}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\eta(X)$ | hàm hồi quy thật, $Y=\eta(X)+\epsilon$ | hàm |
| $\hat\eta(X)$ | hàm ước lượng, **cố định** khi lấy kỳ vọng | hàm |
| ME | $E\{\hat\eta(X)-\eta(X)\}^2$ — sai số của **hàm** | vô hướng |
| PE | sai số khi **dự đoán** một $Y$ mới | vô hướng |
| $\sigma^2$ | phương sai nhiễu $\epsilon$ | vô hướng |

**Ý nghĩa.** Sai số dự đoán gồm phần mô hình có thể sửa (ME) và phần nhiễu không mô hình nào sửa được
($\sigma^2$). So hai phương pháp bằng PE hay ME cho cùng thứ hạng, nhưng ME làm khác biệt **rõ hơn** vì không
bị cộng thêm một hằng số lớn. Mọi bảng mô phỏng ở mục 3.13 dùng ME.

### 3.10 Tibshirani 1996, tr.275–277 — §4 chọn $t$: kiểm định chéo, GCV (9)–(10), Stein (11)

![Tibshirani 1996 tr.276 — ME, (9), (10), (11)](<paper_lasso_tib1996_tr276_9_10_11_stein.png>)

**Hình 14 — Tibshirani 1996, tr.276: ME tuyến tính, (9), $p(t)$, (10), (11), công thức rủi ro.** Trang PDF 11.

**Khung chứa gì.** Dạng ME cho mô hình tuyến tính, phép xấp xỉ ridge (9), số tham số hiệu dụng, GCV (10), bổ đề
Stein (11) và công thức rủi ro áp cho lasso.

$$\text{ME}=(\hat\beta-\beta)^\top V(\hat\beta-\beta)$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $V$ | *"the population covariance matrix of X"* | $p\times p$ |

**Ý nghĩa.** Với mô hình tuyến tính, sai số của hàm chỉ phụ thuộc **độ lệch hệ số**, đo bằng thước $V$: sai
theo hướng mà $X$ biến động nhiều thì bị tính nặng. Công thức này **không chứa $\beta$** ngoài phần hiệu — một
điểm mục 3.13 dùng để kiểm Bảng 3.

$$\tilde\beta=(X^\top X+\lambda W^-)^{-1}X^\top y,\qquad W=\operatorname{diag}(\lvert\tilde\beta_j\rvert) \tag{9}$$

$$p(t)=\operatorname{tr}\{X(X^\top X+\lambda W^-)^{-1}X^\top\},\qquad \operatorname{GCV}(t)=\frac1N\,\frac{\operatorname{rss}(t)}{\{1-p(t)/N\}^2} \tag{10}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\tilde\beta$ | nghiệm lasso viết lại dưới dạng ridge | $\mathbb{R}^p$ |
| $p(t)$ | *"the number of effective parameters"* | vô hướng |
| $\operatorname{rss}(t)$ | tổng bình phương phần dư của nghiệm có cận $t$ | vô hướng |

**Ý nghĩa.** GCV là sai số huấn luyện **phạt theo độ phức tạp**: chia cho $(1-p(t)/N)^2$ làm mô hình nhiều tham
số hiệu dụng bị đẩy lên. Chỗ gọn của nó: chỉ cần **một** lần khớp mỗi $t$. Chỗ mơ hồ: $p(t)$ phụ thuộc cách đọc
$W^-$ ở các cột có hệ số 0 — cùng câu hỏi như ở (7).

$$E_\mu\lVert\hat\mu-\mu\rVert^2=p+E_\mu\Bigl(\lVert g(z)\rVert^2+2\sum_{i=1}^p\frac{dg_i}{dz_i}\Bigr) \tag{11}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $z$ | vector chuẩn, trung bình $\mu$, hiệp phương sai $I$ | $\mathbb{R}^p$ |
| $\hat\mu=z+g(z)$ | ước lượng của $\mu$; $g$ khả vi *"almost"* | $\mathbb{R}^p$ |

**Ý nghĩa.** (11) biến **rủi ro** — thứ cần biết $\mu$ mới tính được — thành một biểu thức **chỉ chứa $z$**. Lấy
biểu thức bên trong ngoặc làm ước lượng thì có một ước lượng không chệch của rủi ro từ **một** mẫu. Với ngưỡng
mềm, $g_i=-\operatorname{sign}(z_i)\min(\lvert z_i\rvert,\gamma)$ và $dg_i/dz_i=-1$ khi $\lvert z_i\rvert<\gamma$, nên

$$R\{\hat\beta(\gamma)\}\approx\hat\tau^2\Bigl\{p-2\#(j:\lvert\hat\beta^\circ_j/\hat\tau\rvert<\gamma)+\sum_{j=1}^p\max(\lvert\hat\beta^\circ_j/\hat\tau\rvert,\gamma)^2\Bigr\}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\tau$ | $\hat\sigma/\sqrt N$, sai số chuẩn của $\hat\beta^\circ_j$; $\hat\sigma^2=\sum(y_i-\hat y_i)^2/(N-p)$ | vô hướng |
| $\#(\cdot)$ | số chỉ số thoả điều kiện | số nguyên |
| $\max(\cdot,\gamma)^2$ | **bản in** — suy ra từ (11) thì phải là $\min$ | |

**Ý nghĩa và chỗ in sai.** Với $\gamma=0$ (không co), ước lượng là chính $z$ và rủi ro đúng là $p$. Bản in cho
$p+\sum z_j^2$ — lớn hơn $p$ một lượng tuỳ dữ liệu. Bản $\min$ cho $p+0-0=p$. Monte Carlo (`v_lasso.py` mục 6,
$p=50$, 20 000 lần):

| $\gamma$ | rủi ro thật | công thức $\min$ | công thức $\max$ (bản in) |
|---:|---:|---:|---:|
| 0,5 | 32,6586 | 32,6407 | 248,6366 |
| 1,0 | 32,7539 | 32,6381 | 241,5024 |
| 1,5 | 43,7189 | 43,5706 | 262,9937 |
| 2,0 | 60,6023 | 60,4030 | 315,0525 |
| 3,0 | 99,1945 | 98,9493 | 505,3188 |

**Tác giả nói gì.** Về (9)–(10): *"This constraint is equivalent to adding a Lagrangian penalty λ Σ βⱼ²/|βⱼ| to
the residual sum of squares, with λ depending on t."* Về rủi ro: *"as an approximately unbiased estimate of the
risk or mean-square error"*. Về kiểm định chéo (tr.275): *"We estimate the prediction error for the lasso
procedure by fivefold cross-validation"* và *"The value ŝ yielding the lowest estimated PE is selected."*

![Tibshirani 1996 tr.277 — γ̂, t̂, lập luận, chi phí, §5](<paper_lasso_tib1996_tr277_stein_bayes.png>)

**Hình 15 — Tibshirani 1996, tr.277: $\hat\gamma$, $\hat t$, lập luận cho thiết kế không trực giao, chi phí, và §5.**
Trang PDF 12.

**Khung chứa gì.** Hai công thức không đánh số $\hat\gamma$, $\hat t$; đoạn lập luận; đoạn đếm chi phí; toàn bộ §5
(mục 3.11); câu đầu §6.

$$\hat\gamma=\arg\min_{\gamma\ge0}[R\{\hat\beta(\gamma)\}],\qquad \hat t=\sum(\lvert\hat\beta^\circ_j\rvert-\hat\gamma)^+$$

**Ý nghĩa và chỗ lệch đơn vị.** $\hat\gamma$ là ngưỡng **tính theo đơn vị $\hat\tau$** — nó so với
$\lvert\hat\beta^\circ_j/\hat\tau\rvert$. Trừ nó thẳng vào $\lvert\hat\beta^\circ_j\rvert$ là trộn hai đơn vị. Bản nhất quán:
$\hat t=\sum(\lvert\hat\beta^\circ_j\rvert-\hat\tau\hat\gamma)^+$.

**Tác giả nói gì.** *"Although the derivation of t̂ assumes an orthogonal design, we may still try to use it in
the usual non-orthogonal setting."* Rồi *"But we can offer only a heuristic argument in favour of it. Suppose that
XᵀX = V and let Z = XV^{-1/2}, θ = βV^{-1/2}."* Và về chi phí: *"In our experiments we optimized over a grid of
15 values of the lasso parameter t and used fivefold cross-validation. As a result, the cross-validation approach
required 75 applications of the model optimization procedure of Section 6 whereas the Stein method required only
one."*

(Dù $\hat t$ được suy ra cho thiết kế trực giao, vẫn có thể thử dùng cho thiết kế thường; chỉ có một lập luận
heuristic. Với lưới 15 giá trị và CV 5 mảnh, kiểm định chéo cần 75 lần tối ưu, Stein chỉ cần một.)

**Kiểm lại** (`v_lasso.py` mục 6, bản dữ liệu có lỗi):

- **Phép biến đổi của lập luận.** $Z=XV^{-1/2}$ đúng là cho $Z^\top Z=I$ (lệch $3{,}220\cdot10^{-15}$). Nhưng với
  $\theta=\beta V^{-1/2}$ như in, $\lVert Z\theta-X\beta\rVert=9{,}0743$; với $\theta=V^{1/2}\beta$ thì
  $1{,}373\cdot10^{-14}$. Phép đổi biến đúng phải là $V^{1/2}\beta$.
- **Chi phí.** $15\times5=75$ lần cho CV, 1 lần cho Stein, 15 lần cho GCV — đúng như bài báo.
- **GCV (10), cột có hệ số 0 bị loại:** cực tiểu ở $s=0{,}69$, $\operatorname{GCV}=0{,}516173$, $p(t)=4{,}2898$. Tại
  $s=0{,}44$: $\operatorname{GCV}=0{,}577495$, $p(t)=1{,}4509$. Đọc $W^-$ nguyên văn: cực tiểu ở $0{,}75$, tại
  0,44 là 0,639244. Lưới 15 điểm: cực tiểu ở $0{,}7143$.
- **CV 5 mảnh, lưới 15 điểm, 200 cách chia:**

| $s$ được chọn | 0,5714 | 0,6429 | 0,7143 | 0,7857 | 0,8571 | 0,9286 | 1,0000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| số lần | 38 | 49 | 43 | 21 | 13 | 23 | 13 |

- **Stein** ($\hat\tau=0{,}071523$; $\lvert\hat\beta^\circ_j/\hat\tau\rvert=9{,}6734;\ 3{,}1558;\ 2{,}0442;\ 2{,}1714;\ 4{,}4347;\ 2{,}062;\ 0{,}4558;\ 1{,}7845$):

| Cách tính | $\hat\gamma$ | $\hat s$ |
|---|---:|---:|
| rủi ro với $\min$, $\hat t$ nhất quán đơn vị | 0,455759 | **0,8586** |
| rủi ro với $\min$, $\hat t$ in nguyên văn | 0,455759 | 0,1280 |
| rủi ro với $\max$ (bản in), $\hat t$ nhất quán | 2,171415 | 0,4169 |

Chia cho $N$ hay $N-1$ khi chuẩn hoá không đổi hai chữ số đầu của $\hat s$.

**Cách đọc.** Không cách chọn nào dựng lại được $\hat s=0{,}44$ của Hình 5 bằng công thức của chính bài báo. Cách
gần nhất lại là công thức Stein **in sai** (0,4169). Kiểm định chéo và GCV đều chọn mô hình **lớn hơn** — giữ 5–7
biến thay vì 3.

**Điều khung KHÔNG nói.** Không nói lưới $s$ của GCV, không nói $W^-$ đọc thế nào ở các ô 0. Đó là hai chỗ đủ để
đổi kết quả, và là chỗ bỏ ngỏ thứ nhất trong bảng đầu mục 3.

### 3.11 Tibshirani 1996, tr.277–278 — §5 lasso là ước lượng Bayes, Hình 7

Phần chữ của §5 nằm trong Hình 15 ở mục 3.10.

$$f(\beta_j)=\frac1{2\tau}\exp\Bigl(-\frac{\lvert\beta_j\rvert}{\tau}\Bigr),\qquad \tau=1/\lambda$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $f(\beta_j)$ | mật độ tiên nghiệm Laplace của hệ số $j$, độc lập giữa các $j$ | hàm |
| $\tau$ | thang của tiên nghiệm | $\tau>0$ |
| $\lambda$ | mức phạt trong dạng $\text{RSS}+\lambda\sum\lvert\beta_j\rvert$ | vô hướng |

**Ý nghĩa.** $-\log f(\beta_j)=\lvert\beta_j\rvert/\tau+\text{hằng}$, nên cực đại hậu nghiệm với tiên nghiệm này là
cực tiểu $\text{RSS}/(2\sigma^2)+\sum\lvert\beta_j\rvert/\tau$. Nhân với $2\sigma^2$ ra
$\text{RSS}+(2\sigma^2/\tau)\sum\lvert\beta_j\rvert$ — tức $\lambda=2\sigma^2/\tau$. Bài báo viết $\tau=1/\lambda$, đúng
khi và chỉ khi $\sigma^2=1/2$.

**Kiểm lại** (`v_lasso.py` mục 7, prostate, $s=0{,}44$): $\lambda=35{,}784658$, $\hat\sigma^2=0{,}501849$. Tiên nghiệm
cho đúng nghiệm lasso có $\tau=2\hat\sigma^2/\lambda=0{,}028048$; công thức in cho $1/\lambda=0{,}027945$ — gần, chỉ vì
$\hat\sigma^2$ trên dữ liệu này tình cờ gần $1/2$. Cực đại hậu nghiệm tìm trực tiếp bằng thuật toán Powell (với
$\tau=2\hat\sigma^2/\lambda$) lệch nghiệm lasso $1{,}006\cdot10^{-8}$.

**Tác giả nói gì.** *"Now |βⱼ| is proportional to the (minus) log-density of the double-exponential
distribution. As a result we can derive the lasso estimate as the Bayes posterior mode under independent
double-exponential priors"*; và *"Notice how the double-exponential density puts more mass near 0 and in the
tails. This reflects the greater tendency of the lasso to produce estimates that are either large or 0."*

(Trị tuyệt đối tỉ lệ với trừ log mật độ Laplace, nên lasso là mode hậu nghiệm dưới tiên nghiệm Laplace độc
lập. Mật độ Laplace dồn nhiều khối lượng gần 0 và ở đuôi hơn — phản ánh xu hướng của lasso cho ước lượng hoặc
lớn hoặc bằng 0.)

![Tibshirani 1996 tr.278 — Hình 7 và thuật toán §6](<paper_lasso_tib1996_tr278_hinh7_thuat_toan.png>)

**Hình 16 — Tibshirani 1996, Fig. 7 và §6, tr.278.** Trang PDF 13.

**Caption gốc.** *"Fig. 7. Double-exponential density (———) and normal density (- - - - -): the former is the
implicit prior used by the lasso; the latter by ridge regression"*

(Hình 7. Mật độ Laplace (liền) và mật độ chuẩn (đứt): cái thứ nhất là tiên nghiệm ngầm của lasso, cái thứ hai
của ridge.)

**Nhận xét của tác giả.** Hai câu cuối §5 vừa dẫn.

**Kiểm lại "nhiều khối lượng gần 0 và ở đuôi"** (`v_lasso.py` mục 7). Hình không ghi tham số, nên đo cho hai
cách so:

| So với $N(0,1)$ | mật độ tại 0 | $P(\lvert\beta\rvert<0{,}5)$ | $P(\lvert\beta\rvert>3)$ |
|---|---:|---:|---:|
| chuẩn $N(0,1)$ | 0,3989 | 0,3829 | 0,0027 |
| Laplace $\tau=1$ (đỉnh 0,5) | 0,5000 | 0,3935 | 0,0498 |
| Laplace cùng phương sai 1, $\tau=1/\sqrt2$ | 0,7071 | 0,5069 | 0,0144 |

Cả hai cách đều cho Laplace nhiều khối lượng hơn gần 0 **và** ở đuôi — câu của tác giả đứng.

**Cách đọc hình.** Nét liền nhọn ở 0 và giảm theo hàm mũ; nét đứt tròn ở đỉnh và giảm nhanh hơn ở đuôi. Hai
đường cắt nhau hai lần mỗi bên.

**Điều hình KHÔNG nói.** Lasso là **mode**, không phải trung bình hậu nghiệm. Bài hồi tưởng 2011 tr.280 nhấn đúng
điểm này (mục 3.19): *"the lasso solution is not the posterior mean or median in that setting: these latter
solutions will not be sparse"*. Đo một chiều, $z\sim N(\beta,1)$, $\beta\sim$ Laplace($\tau=1$):

| $z$ | mode (= lasso) | trung bình hậu nghiệm | trung vị hậu nghiệm |
|---:|---:|---:|---:|
| 0,00 | 0 | 0,000000 | 0,000000 |
| 0,25 | 0 | 0,119166 | 0,090200 |
| 0,50 | 0 | 0,241019 | 0,190100 |
| 1,00 | 0 | 0,503223 | 0,428800 |
| 2,00 | 1 | 1,161089 | 1,106700 |
| 3,00 | 2 | 2,025812 | 2,012500 |

### 3.12 Tibshirani 1996, tr.277–279 — §6 thuật toán

Hình 16 ở mục 3.11 chứa phần lớn §6.

**Khung chứa gì.** Bài toán (1) viết thành $2^p$ ràng buộc tuyến tính, định nghĩa $\delta_i$, $E$, $S$, $G_E$, và các
bước (a)–(d).

$$\sum_j\lvert\beta_j\rvert\le t\iff\delta_i^\top\beta\le t\quad\text{với mọi } i=1,\dots,2^p,\qquad \delta_i\in\{-1,+1\}^p$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $g(\beta)$ | $\sum_i(y_i-\sum_j\beta_jx_{ij})^2$ | vô hướng |
| $\delta_i$ | một vector dấu | $\{-1,1\}^p$ |
| $E$, $S$ | *"equality set"* — ràng buộc đang chặt; *"slack set"* — đang lỏng | tập chỉ số |
| $G_E$ | ma trận có các dòng $\delta_i$, $i\in E$ | $\lvert E\rvert\times p$ |
| $\mathbf 1$ | vector toàn 1 dài $\lvert E\rvert$ | |

**Ý nghĩa.** Hình vuông xoay $\sum\lvert\beta_j\rvert\le t$ là giao của $2^p$ nửa không gian — mỗi mặt ứng với một tổ
hợp dấu. Thuật toán không đưa cả $2^p$ ràng buộc vào: nó bắt đầu với **một** mặt (dấu của OLS), giải, xem nghiệm
còn vượt ra ngoài không, và nếu có thì thêm đúng mặt mà nghiệm đang vượt — mặt $\delta=\operatorname{sign}(\hat\beta)$.

**Các bước in** (tr.278): *"(a) Start with E = {i₀} where δ_{i₀} = sign(β̂°), β̂° being the overall least squares
estimate. (b) Find β̂ to minimize g(β) subject to G_E β ⩽ t1. (c) While {Σ|β̂ⱼ| > t}, (d) add i to the set E where
δᵢ = sign(β̂). Find β̂ to minimize g(β) subject to G_E β ⩽ t1."*

**Tác giả nói gì.** *"This procedure must always converge in a finite number of steps since one element is added
to the set E at each step, and there is a total of 2^p elements."* Và tr.279: *"In practice we have found that
the average number of iterations required is in the range (0.5p, 0.75p) and is therefore quite acceptable for
practical purposes."* Về cách của David Gay: *"We write each βⱼ as βⱼ⁺ − βⱼ⁻, where βⱼ⁺ and βⱼ⁻ are non-negative"*
và *"In this way we transform the original problem (p variables, 2^p constraints) to a new problem with more
variables (2p) but fewer constraints (2p + 1)."*

(Thủ tục luôn dừng sau hữu hạn bước vì mỗi bước thêm một phần tử vào $E$ và có tổng cộng $2^p$ phần tử. Thực tế số
vòng trung bình nằm trong khoảng $0{,}5p$ tới $0{,}75p$. Cách của Gay viết mỗi hệ số thành hiệu hai số không âm,
đổi bài toán $p$ biến $2^p$ ràng buộc thành $2p$ biến $2p+1$ ràng buộc.)

**Dựng lại** (`v_lasso.py` mục 8). Cài đúng (a)–(d); bài toán con ở (b) và (d) giải bằng phương pháp tập hoạt
động với phép giải tuyến tính chính xác. Thiết kế tương quan AR 0,5, $n=100$, 40 bài toán mỗi ô. Cách của Gay
giải bằng SLSQP của SciPy nên chỉ chính xác tới sai số của bộ giải đó. Với $p=8$: $2^p=256$ ràng buộc, cách của Gay
có 16 biến và 17 ràng buộc.

| $p$ | $s$ | số vòng TB | TB $/p$ | [min; max] | lệch LARS | lệch Gay |
|---:|---:|---:|---:|---|---:|---:|
| 4 | 0,25 | 2,900 | 0,725 | [1; 4] | $3{,}272\cdot10^{-14}$ | $7{,}049\cdot10^{-7}$ |
| 4 | 0,50 | 2,175 | 0,544 | [1; 4] | $7{,}275\cdot10^{-14}$ | $6{,}083\cdot10^{-7}$ |
| 4 | 0,75 | 1,500 | 0,375 | [1; 4] | $3{,}642\cdot10^{-14}$ | $1{,}539\cdot10^{-7}$ |
| 8 | 0,25 | 6,175 | 0,772 | [4; 10] | $1{,}780\cdot10^{-13}$ | $4{,}319\cdot10^{-5}$ |
| 8 | 0,50 | 3,800 | 0,475 | [2; 6] | $6{,}694\cdot10^{-14}$ | $1{,}603\cdot10^{-5}$ |
| 8 | 0,75 | 2,150 | 0,269 | [1; 4] | $6{,}151\cdot10^{-14}$ | $4{,}922\cdot10^{-7}$ |
| 16 | 0,25 | 12,750 | 0,797 | [9; 18] | $3{,}334\cdot10^{-13}$ | — |
| 16 | 0,50 | 7,550 | 0,472 | [3; 12] | $5{,}348\cdot10^{-14}$ | — |
| 16 | 0,75 | 4,075 | 0,255 | [1; 10] | $3{,}786\cdot10^{-14}$ | — |

**Cách đọc.** Thuật toán cho **đúng** nghiệm lasso (lệch LARS dưới $4\cdot10^{-13}$). Số vòng chia cho $p$ **phụ
thuộc $t$**: 0,72–0,80 khi $s=0{,}25$, quanh 0,5 khi $s=0{,}5$, và 0,25–0,38 khi $s=0{,}75$. Khoảng
$(0{,}5p;\ 0{,}75p)$ của bài báo khớp phần giữa của bảng; bài báo không nói nó đo ở $t$ nào. Và số vòng tối đa
tăng gần tuyến tính theo $p$ — xa hẳn trần $2^p$.

**Điều khung KHÔNG nói.** Bước (d) lấy $\operatorname{sign}(\hat\beta)$, nhưng $\operatorname{sign}(0)=0$ không phải một
thành phần hợp lệ của $\delta_i$. Bản cài đặt ở đây quy ước $0\to+1$ và kiểm rằng $\delta$ mới chưa có trong $E$;
bài báo không nói gì về trường hợp đó. Câu *"A modification of this procedure removes elements from E … it is
not clear how to establish its convergence"* không được thử.

![Tibshirani 1996 tr.279 — Bảng 3, cách của Gay, §7.1](<paper_lasso_tib1996_tr279_bang3_gay_7_1.png>)

**Hình 17 — Tibshirani 1996, Table 3, phần cuối §6 và §7.1, tr.279.** Trang PDF 14. Bảng 3 được đọc ở mục 3.13.

### 3.13 Tibshirani 1996, tr.279–283 — §7 bốn mô phỏng, Bảng 3–8, Hình 8, công thức (12)

**Khung chứa gì (Hình 17, phần §7.1).** Danh sách phương pháp và cách kiểm định chéo cho chọn tập con.

**Tác giả nói gì.** *"We used fivefold cross-validation to estimate the regularization parameter in each case.
For best subset selection, we used the 'leaps' procedure in the S language, with fivefold cross-validation to
estimate the best subset size."*

![Tibshirani 1996 tr.280 — Bảng 4, (12), §7.2, Bảng 5](<paper_lasso_tib1996_tr280_bang4_12_bang5.png>)

**Hình 18 — Tibshirani 1996, Table 4, công thức (12), §7.2 và Table 5, tr.280.** Trang PDF 15.

$$\operatorname{PE}(J)=\frac15\sum_{v=1}^5\operatorname{PE}^v(J) \tag{12}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $S_J$, $S^v_J$ | tập con tốt nhất cỡ $J$ trên toàn tập $T$ và trên $T-T^v$ | tập chỉ số |
| $\operatorname{PE}^v(J)$ | sai số khi dùng $S^v_J$ trên mảnh kiểm tra $T^v$ | vô hướng |
| $J$ | cỡ tập con, $0\le J\le p$ | số nguyên |

**Ý nghĩa.** Kiểm định chéo ở đây chọn **cỡ** $J$, không chọn **tập**: ở mỗi mảnh, tập tốt nhất cỡ $J$ được tìm lại
từ đầu; cuối cùng mô hình là $S_{\hat J}$ trên toàn dữ liệu. Tác giả nói rõ điều đó khác với việc đánh giá các tập
cố định $S_0,\dots,S_p$ — cách sau *"can lead to inconsistent model selection"*.

**Mô hình của ví dụ 1–3** (tr.280): $y=\beta^\top x+\sigma\epsilon$, $N=20$, $p=8$, tương quan giữa $x_i$ và $x_j$ là
$\rho^{\lvert i-j\rvert}$ với $\rho=0{,}5$. Ví dụ 1: $\beta=(3;\ 1{,}5;\ 0;\ 0;\ 2;\ 0;\ 0;\ 0)$, *"We set σ = 3"*. Ví
dụ 2: $\beta_j=0{,}85$ mọi $j$, $\sigma=3$. Ví dụ 3: $\beta=(5,0,\dots,0)$, $\sigma=2$. Ví dụ 4 (tr.283): $N=100$,
$p=40$, $x_{ij}=z_{ij}+z_i$, $\beta$ gồm bốn khối 10 số $0,2,0,2$, $y=\beta^\top x+15\epsilon$, 50 bộ dữ liệu.

**Tác giả nói gì.** Ví dụ 1: *"The lasso performs the best, followed by the garotte and ridge regression."* Và
*"Estimation of the lasso parameter by generalized cross-validation seems to perform best, a trend that we find is
consistent through all our examples."* Về Bảng 4, 5 (tr.282): *"although the correct model (1, 2, 5) was chosen
only 2.5% of the time, the selected model contained (1, 2, 5) 95.5% of the time. … The correct model is chosen
more often (24% of the time), but subset selection can also underfit: the selected model contained (1, 2, 5) only
53.5% of the time."*

(Lasso tốt nhất, sau đó là garotte và ridge. Chọn tham số bằng GCV có vẻ tốt nhất, và xu hướng này nhất quán qua
mọi ví dụ. Mô hình đúng chỉ được lasso chọn 2,5 % số lần, nhưng mô hình được chọn chứa $(1,2,5)$ ở 95,5 % số lần.
Chọn tập con chọn đúng nhiều hơn (24 %) nhưng cũng hay thiếu: chỉ 53,5 % chứa đủ.)

![Tibshirani 1996 tr.281 — Hình 8](<paper_lasso_tib1996_tr281_hinh8.png>)

**Hình 19 — Tibshirani 1996, Fig. 8, tr.281.** Trang PDF 16.

**Caption gốc.** *"Fig. 8. Estimates for the eight coefficients in example 1, excluding the intercept: ··········,
true coefficients"*

(Hình 8. Ước lượng của tám hệ số trong ví dụ 1, không tính hệ số chặn; nét chấm là hệ số thật.)

**Nhận xét của tác giả.** Tr.282: *"Subset selection picks approximately the correct number of zero coefficients
(5), but suffers from too much variability as shown in the box plots of Fig. 8."*

**Cách đọc hình.** Tám hàng, mỗi hàng một hệ số; trong mỗi hàng năm hộp: LS, lasso, garotte, tập con, ridge.
Hàng 3, 4, 6, 7, 8 (hệ số thật 0): hộp của lasso, garotte và tập con **dẹt về một vạch** tại 0 — phần lớn lần chạy
cho đúng 0 — nhưng có một cột dài các chấm lẻ, và cột chấm của tập con kéo xa nhất. Hộp LS và ridge thì có bề
rộng thật. Hàng 1 (hệ số thật 3): hộp ridge **thấp hơn** vạch chấm rõ nhất — ridge co cả hệ số lớn.

**Điều hình KHÔNG nói.** Hình không có thang chung giữa các hàng, và không cho con số độ dao động nào — câu *"too
much variability"* không đi kèm số đo.

![Tibshirani 1996 tr.282 — Bảng 6, §7.3, §7.4, Bảng 7](<paper_lasso_tib1996_tr282_bang6_bang7.png>)

**Hình 20 — Tibshirani 1996, Table 6, §7.3, §7.4 và Table 7, tr.282.** Trang PDF 17.

![Tibshirani 1996 tr.283 — Bảng 8, §7.5, §8](<paper_lasso_tib1996_tr283_bang8_7_5_muc8.png>)

**Hình 21 — Tibshirani 1996, Table 8, cuối §7.4, §7.5 và §8, tr.283.** Trang PDF 18.

**Caption gốc của Bảng 3–8.** *"TABLE 3. Results for example 1†"*, *"TABLE 4. Most frequent models selected by the
lasso (generalized cross-validation) in example 1"*, *"TABLE 5. Most frequent models selected by all-subsets
regression in example 1"*, *"TABLE 6. Results for example 2†"*, *"TABLE 7. Results for example 3†"*, *"TABLE 8.
Results for example 4†"*; chú thích chung *"†Standard errors are given in parentheses."*

(Bảng 3, 6, 7, 8: kết quả ví dụ 1–4, sai số chuẩn trong ngoặc. Bảng 4, 5: các mô hình hay được lasso chọn bằng GCV
và được chọn tập con chọn ở ví dụ 1.)

**Nhận xét của tác giả.** Ví dụ 2 (tr.282): *"The results in Table 6 show that ridge regression does the best by a
good margin, with the lasso being the only other method to outperform the full least squares estimate."* Ví dụ 3
(tr.282–283): *"The results in Table 7 show that the garotte and subset selection perform the best, followed
closely by the lasso. Ridge regression does poorly and has a higher mean-squared error than do the full least
squares estimates."* Ví dụ 4 (tr.283): *"The results in Table 8 show that the ridge regression performs the best,
with the lasso (generalized cross-validation) a close second."* và *"Although the lasso only produced 14.4 zero
coefficients on average, the average value of ŝ (0.55) was close to the true proportion of 0s (0.5)."*

**Dựng lại bốn ví dụ** (`v_lasso.py` mục 9). Cài theo §7.1: chuẩn hoá biến trên từng bộ dữ liệu; lasso chọn $s$
bằng CV 5 mảnh, GCV (10) hoặc Stein (mục 3.10) trên lưới 15 điểm $s\in[0,1]$; garotte chọn $\sum c_j$ bằng CV 5 mảnh
trên cùng lưới nhân $p$; chọn tập con theo (12) với tìm kiếm vét cạn; ridge chọn bậc tự do trên lưới 15 điểm từ 0
tới $p$ bằng CV 5 mảnh. Sai số chuẩn của trung vị lấy bằng bootstrap 500 lần. Bài báo không nói lưới của garotte
và ridge, nên hai cột đó là lựa chọn của bản dựng lại.

**Ví dụ 1 — theo đúng văn bản, $\sigma=3$ (hạt giống 1001), và theo giả thuyết $\sigma=2$ (cùng hạt giống):**

| Phương pháp | Bảng 3 in: trung vị ME (SE) · số 0 | $\sigma=3$: trung vị ME (SE) · số 0 | $\sigma=2$: trung vị ME · số 0 |
|---|---|---|---|
| LS | 2,79 (0,12) · 0,0 | 6,3362 (0,3402) · 0,000 | **2,8161** · 0,000 |
| Lasso (CV) | 2,43 (0,14) · 3,3 | 4,0238 (0,3093) · 3,590 | 2,0262 · 3,105 |
| Lasso (Stein) | 2,07 (0,10) · 2,6 | 5,1690 (0,2669) · 1,395 | 2,4757 · 1,185 |
| Lasso (GCV) | 1,93 (0,09) · 2,4 | 3,4056 (0,2175) · 2,720 | 1,6731 · 2,330 |
| Garotte | 2,29 (0,16) · 3,9 | 4,7766 (0,4300) · 4,095 | 2,2176 · 3,535 |
| Best subset | 2,44 (0,16) · 4,8 | 6,1605 (0,3706) · 5,055 | 2,1657 · 4,500 |
| Ridge | 3,21 (0,12) · 0,0 | 3,9340 (0,1540) · 0,040 | 2,2736 · 0,000 |

Và Bảng 4, 5: mô hình đúng $\{1,2,5\}$ và mô hình chứa $\{1,2,5\}$:

| | in | $\sigma=3$ | $\sigma=2$ |
|---|---|---|---|
| lasso (GCV): đúng · chứa | 0,025 · **0,955** | 0,020 · 0,765 | 0,010 · **0,955** |
| tập con: đúng · chứa | 0,240 · 0,535 | 0,105 · 0,315 | 0,280 · 0,595 |

Trung bình $\hat s$ ở $\sigma=3$: CV 0,5061 (in 0,63), Stein 0,8140 (in 0,69), GCV 0,6307 (in 0,73).

**Ví dụ 2 (hạt giống 1002), 3 (1003), 4 (1004, 50 bộ):**

| Phương pháp | VD 2 in | VD 2 dựng lại | VD 3 in | VD 3 dựng lại | VD 4 in | VD 4 dựng lại |
|---|---|---|---|---|---|---|
| LS | 6,50 (0,64) | 5,7511 (0,3947) | 2,89 (0,04) | 2,5428 (0,2035) | 137,3 (7,3) | 143,8943 (5,8629) |
| Lasso (CV) | 5,30 (0,45) | 4,9179 (0,3070) | 0,89 (0,01) | 0,9844 (0,0966) | — | — |
| Lasso (Stein) | 5,85 (0,36) | 4,8568 (0,2159) | 1,26 (0,02) | 1,7704 (0,1666) | 80,2 (4,9) | 101,6297 (6,9804) |
| Lasso (GCV) | 4,87 (0,35) | 3,5591 (0,1172) | 1,02 (0,02) | 0,9090 (0,0807) | 64,9 (2,3) | 62,3732 (2,2398) |
| Garotte | 7,40 (0,48) | 6,3369 (0,2718) | 0,52 (0,01) | 0,2969 (0,0375) | 94,8 (3,2) | 104,4199 (6,5655) |
| Best subset | 9,05 (0,78) | 7,9131 (0,4766) | 0,64 (0,02) | 0,1803 (0,0442) | — | — |
| Ridge | 2,30 (0,22) | 2,2205 (0,1644) | 3,53 (0,05) | 2,6200 (0,1623) | 57,4 (1,4) | 32,8945 (1,4445) |

Số hệ số 0 trung bình ở ví dụ 4: lasso Stein 5,940 (in 14,4), GCV 15,280 (in 13,6), garotte 23,820 (in 22,9).
Trung bình hệ số lasso GCV theo bốn khối 10 biến: $0{,}3041;\ 1{,}5353;\ 0{,}4203;\ 1{,}5767$ (SE
$0{,}0419;\ 0{,}0598;\ 0{,}0406;\ 0{,}0444$).

**Sáu phép kiểm không cần mô phỏng** (`v_lasso.py` mục 9):

1. **ME của LS không phụ thuộc $\beta$.** $\hat\beta^{\text{LS}}-\beta$ chỉ là phép chiếu của nhiễu. 20 000 lần: trung vị
   5,8295 cho ví dụ 1 và 2 (cùng $\sigma=3$), 2,5880 cho ví dụ 3, 149,3084 cho ví dụ 4. Bảng in 2,79 · 6,50 · 2,89 ·
   137,3. Ví dụ 2, 3, 4 khớp trong độ dao động; **ví dụ 1 thì không**, và nó thấp hơn cả ví dụ 3 dù văn bản cho nó
   nhiễu **lớn hơn**. $2{,}79\times4/9=1{,}2400$ mới là số của ví dụ 3 nếu ví dụ 1 thật có $\sigma=3$.
2. **Giả thuyết $\sigma=2$ cho ví dụ 1** giải thích cả bảng: LS 2,8161 so với 2,79; thứ tự và độ lớn của mọi dòng
   trừ ridge gần bản in; tỉ lệ "chứa $\{1,2,5\}$" của lasso trùng **đúng** 0,955. Với $\sigma=2$, tỉ số tín hiệu/nhiễu
   $\beta^\top V\beta/\sigma^2$ là 5,3125 — gần hơn nhiều so với con số 2,3611 của $\sigma=3$ khi so với "khoảng 5,7" in ở
   văn bản.
3. **50 hay 200 lần?** Mọi tỉ lệ in ở Bảng 4, 5 và văn bản đều là bội của $1/200$: $0{,}055\times200=11$ còn
   $0{,}055\times50=2{,}75$; $0{,}955\times200=191$ còn $\times50=47{,}75$; $0{,}535\times50=26{,}75$. Và tr.280 viết
   *"Table 3 shows the mean-squared errors over 200 simulations"*. Câu *"we simulated 50 data sets"* ở cùng đoạn sai.
4. **Sai số chuẩn không thể có.** $\hat s\in[0,1]$ nên độ lệch chuẩn của nó không vượt 0,5, và sai số chuẩn của trung
   bình không vượt $0{,}5/\sqrt{200}=0{,}0354$ hay $0{,}5/\sqrt{50}=0{,}0707$. Bảng 6 in **0,23** cho GCV, Bảng 8 in
   **0,88**.
5. **Tỉ số tín hiệu/nhiễu không có định nghĩa**, và không cách đọc nào khớp cả bốn số in:

| Ví dụ | $\beta^\top V\beta/\sigma^2$ | căn của nó | $1+\beta^\top V\beta/\sigma^2$ | căn của nó | in |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,3611 | 1,5366 | 3,3611 | 1,8333 | 5,7 |
| 2 | 1,6068 | 1,2676 | 2,6068 | 1,6146 | 1,8 |
| 3 | 6,2500 | 2,5000 | 7,2500 | 2,6926 | 7,0 |
| 4 | 7,4667 | 2,7325 | 8,4667 | 2,9098 | 9,0 |

6. **Thứ hạng trong văn bản trái với bảng.** Bảng 3: ridge 3,21 **tệ hơn LS** 2,79 và tệ hơn tập con 2,44 — không
   thể "followed by the garotte and ridge regression". Cùng chỗ đó, §11 kịch bản (b) nói ridge đứng thứ hai.

**Cách đọc.** Kết luận định tính của bài báo **đứng** trên bản dựng lại: lasso GCV tốt nhất trong các phương pháp
chọn biến ở ví dụ 1 (với $\sigma=2$: 1,6731, thấp nhất bảng) và ở ví dụ 2; ridge thắng rõ ở ví dụ 2 và 4; tập con và
garotte thắng ở ví dụ 3. Con số thì không đứng: ridge ở ví dụ 4 dựng lại tốt hơn bản in 1,7450 lần (32,8945 so với
57,4), và ví dụ 4 cho thấy các khối hệ số theo đúng mẫu $0,2,0,2$ chứ không tăng đơn điệu như bản in.

**Điều các bảng KHÔNG nói.** Bảng 8 không có dòng lasso CV và dòng tập con; văn bản chỉ giải thích dòng tập con
(*"impractical for p > 30"*). Và câu *"the average value of ŝ (0.55) was close to the true proportion of 0s (0.5)"*
so hai thứ khác đơn vị: $\hat s$ là tỉ lệ **ngân sách**, không phải tỉ lệ **hệ số 0**.

### 3.14 Tibshirani 1996, tr.283–284 — §8 mô hình hồi quy tổng quát, §8.1 logistic, §9

Phần đầu §8 nằm trong Hình 21 ở mục 3.13.

![Tibshirani 1996 tr.284 — §8.1](<paper_lasso_tib1996_tr284_muc8_1.png>)

**Hình 22 — Tibshirani 1996, §8.1 Logistic Regression, tr.284.** Trang PDF 19.

**Khung chứa gì.** Dữ liệu kyphosis, ba mô hình được in, và tiêu chí hội tụ.

**Tác giả nói gì.** §8 tr.283: *"we consider here models for which a quadratic approximation to l(β) leads to an
iteratively reweighted least squares (IRLS) procedure for computation of β. … Using this approach, we can solve the
constrained problem by iterative application of the lasso algorithm, within an IRLS loop. Convergence of this
procedure is not ensured in general, but in our limited experience it has behaved quite well."* §8.1: *"There are 83
observations. Since the predictor effects are known to be non-linear, we included squared terms in the model after
centring each of the variables."*

(Với các mô hình mà phép xấp xỉ bậc hai của hàm hợp lý dẫn tới bình phương tối thiểu có trọng số lặp lại, ta giải
bài toán có ràng buộc bằng cách chạy thuật toán lasso bên trong vòng IRLS; hội tụ không được bảo đảm nhưng thực tế
khá ổn. Có 83 quan sát; vì hiệu ứng phi tuyến nên thêm các số hạng bình phương sau khi trừ trung bình.)

**Dựng lại, và vì sao không khớp** (`v_lasso.py` mục 10). Bản kyphosis công khai (gói R `rpart`, qua Rdatasets,
lưu ở `tools/kyphosis.csv`) có **81** dòng, không phải 83. Trên 81 dòng:

| Mô hình | Dựng lại (81 dòng) | Bài báo in (83 dòng) |
|---|---|---|
| đầy đủ | $-2{,}9819+1{,}1128x_1+1{,}0031x_2-2{,}6504x_3-1{,}2485x_1^2-0{,}5867x_2^2-1{,}3896x_3^2$ | $-2{,}64+0{,}83x_1+0{,}77x_2-2{,}28x_3-1{,}55x_1^2+0{,}03x_2^2-1{,}17x_3^2$ |
| lùi theo AIC | bỏ $x_2^2$ (AIC 61,8324 → 61,4550) **rồi bỏ cả $x_2$** (→ 61,2975) | chỉ bỏ $x_2^2$ |
| lasso $s=0{,}33$ | $-1{,}8112+0{,}3583x_1+0{,}3895x_2-0{,}9856x_3-0{,}6877x_1^2-0{,}2160x_3^2$, 5 vòng IRLS | $-1{,}42+0{,}03x_1+0{,}31x_2-0{,}48x_3-0{,}28x_1^2$, 5 vòng |

**Cách đọc.** Hai dòng dữ liệu đủ đổi hẳn hệ số của $x_2^2$ (0,03 so với −0,5867). Điều duy nhất khớp là bước AIC
đầu tiên bỏ $x_2^2$, và số vòng IRLS. Không kết luận gì được về bảng in.

**Điều khung KHÔNG nói.** Không nói ngân sách $s$ tính trên hệ số của mô hình nào; bản dựng lại dùng
$t=0{,}33\sum\lvert\hat\beta^{\text{MLE}}_j\rvert$.

**§9 (Hình 22, phần dưới).** Hai hướng mở rộng — cây hồi quy (LeBlanc và Tibshirani 1994) và MARS — cùng câu
*"The lasso idea can also be applied to ill-posed problems, in which the predictor matrix is not full rank."* Không có
số hay công thức nào; mục 10 ghi lý do không dựng lại.

### 3.15 Tibshirani 1996, tr.285–286 — §10 kết quả về ngưỡng mềm, (13)–(16)

![Tibshirani 1996 tr.285 — §10](<paper_lasso_tib1996_tr285_muc10.png>)

**Hình 23 — Tibshirani 1996, §10 Results on Soft Thresholding, tr.285.** Trang PDF 20.

**Khung chứa gì.** (13), mô hình trực chuẩn (14), rủi ro, họ phép chiếu chéo (15), và cận (16).

$$\hat\beta_j=\operatorname{sign}(\hat\beta^\circ_j)(\lvert\hat\beta^\circ_j\rvert-\gamma)^+ \tag{13}$$

$$\hat\beta^\circ_j=\beta_j+\sigma z_j,\quad z_j\sim N(0,\sigma^2)\ \text{(bản in)} \tag{14}$$

$$T_{DP}(\hat\beta^\circ,\delta)=(\delta_j\hat\beta_j)_{j=1}^p,\qquad \delta\in\{0,1\} \tag{15}$$

$$R(\tilde\beta,\beta)\le(2\log p+1)(\sigma^2+R_{DP}) \tag{16}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $R(\hat\beta,\beta)$ | $E\lVert\hat\beta-\beta\rVert^2$ | vô hướng |
| $\delta_j$ | 1 giữ, 0 xoá | $\{0,1\}$ |
| $R_{DP}$ | rủi ro của phép chiếu **lý tưởng**, chọn $\delta_j=I(\lvert\beta_j\rvert>\sigma)$; bằng $\sum_j\min(\beta_j^2,\sigma^2)$ | vô hướng |
| $\tilde\beta_j$ | ngưỡng cứng $\hat\beta^\circ_jI(\lvert\hat\beta^\circ_j\rvert>\gamma)$ | vô hướng |
| $\gamma$ | $\sigma(2\log n)^{1/2}$ | vô hướng |

**Ý nghĩa.** $R_{DP}$ là rủi ro của một "người chọn biến biết trước đáp án": giữ hệ số nào lớn hơn nhiễu, xoá hệ số
nào nhỏ hơn. Không ai đạt được nó. (16) nói ngưỡng cứng — và theo tác giả, cả ngưỡng mềm tức lasso — chỉ tệ hơn
người đó một thừa số cỡ $2\log p$. Với $p$ lớn, đó là một cái giá rất nhỏ cho việc không biết đáp án.

**Chỗ in sai ở (14).** Nếu $z_j\sim N(0,\sigma^2)$ thì $\sigma z_j$ có phương sai $\sigma^4$ — với $\sigma=2$ là 16
thay vì 4. Cả đoạn sau (*"we incur a risk of σ² if we use β̂ⱼ°"*) chỉ đúng khi $z_j\sim N(0,1)$.

**Kiểm lại (16)** (`v_lasso.py` mục 11): $p=n=1024$, $\sigma=1$, 20 hệ số bằng 5 và 40 hệ số bằng 0,5, còn lại 0;
$R_{DP}=30{,}0000$; $\gamma=\sigma\sqrt{2\log p}=3{,}7233$; 4 000 lần.

| Ước lượng | Rủi ro | Rủi ro $/R_{DP}$ |
|---|---:|---:|
| LS (không co) | 1024,3246 | — |
| ngưỡng cứng | 77,1494 | 2,5716 |
| ngưỡng mềm (lasso) | 296,7782 | 9,8926 |
| cận (16) | 460,7513 | $2\log p+1=14{,}8629$ |

Cả hai nằm dưới cận. Ngưỡng mềm tệ hơn ngưỡng cứng 3,8468 lần ở ví dụ này, vì nó trừ 3,7233 khỏi cả 20 hệ số lớn.

**Tác giả nói gì.** *"It shows that asymptotically the soft threshold estimator (lasso) comes as close as subset
selection to the performance of an ideal subset selector"* và tr.286: *"However, the important differences between
the various approaches tend to occur for correlated predictors, and theoretical results such as those given here seem
to be more difficult to obtain in that case."*

**Điều khung KHÔNG nói.** (16) viết $\log p$ nhưng $\gamma$ viết $\log n$; trong thiết kế trực chuẩn của Donoho–Johnstone
hai số bằng nhau, và bản dựng lại đặt $p=n$. Cận là **tiệm cận**; ở $p=1024$ nó còn lỏng: rủi ro thật của ngưỡng mềm
bằng 0,6441 lần cận, của ngưỡng cứng 0,1674 lần.

### 3.16 Tibshirani 1996, tr.286–287 — §11 thảo luận, Hình 9, §12

![Tibshirani 1996 tr.286 — §11](<paper_lasso_tib1996_tr286_muc11.png>)

**Hình 24 — Tibshirani 1996, §11 Discussion, tr.286.** Trang PDF 21.

**Tác giả nói gì.** Ba kịch bản: *"(a) small number of large effects—subset selection does best here, the lasso not
quite as well and ridge regression does quite poorly; (b) small to moderate number of moderate-sized effects—the lasso
does best, followed by ridge regression and then subset selection; (c) large number of small effects—ridge regression
does best by a good margin, followed by the lasso and then subset selection."* Và *"Breiman's garotte does a little
better than the lasso in the first scenario, and a little worse in the second two scenarios."* Về $q$: *"The value
q = 1 has the advantage of being closer to subset selection than is ridge regression (q = 2) and is also the smallest
value of q giving a convex region."*

(Ba kịch bản: ít hiệu ứng lớn — tập con tốt nhất, lasso kém hơn chút, ridge khá tệ; vài hiệu ứng vừa — lasso tốt
nhất, rồi ridge, rồi tập con; nhiều hiệu ứng nhỏ — ridge tốt nhất rõ rệt, rồi lasso, rồi tập con. Garotte tốt hơn
lasso một chút ở kịch bản đầu, kém một chút ở hai kịch bản sau. $q=1$ gần chọn tập con hơn ridge và là giá trị nhỏ
nhất cho miền lồi.)

**Đối chiếu với chính các bảng.** (a) là ví dụ 3: Bảng 7 cho garotte 0,52 **tốt hơn** tập con 0,64 — "subset selection
does best" không khớp, dù §7.4 viết đúng *"the garotte and subset selection perform the best"*. (b) là ví dụ 1: như
phép kiểm 6 ở mục 3.13, ridge không đứng thứ hai. (c) là ví dụ 2 và 4: khớp. Về garotte: ở ví dụ 1, garotte 2,29
**tốt hơn** lasso CV 2,43 — hai phương pháp dùng cùng cách chọn tham số — và chỉ kém lasso GCV 1,93.

**Kiểm "nhỏ nhất cho miền lồi"** (`v_lasso.py` mục 12): trung điểm của $(1,0)$ và $(0,1)$ có $\sum\lvert\beta_j\rvert^q$
bằng 0,1250 ($q=4$), 0,5000 ($q=2$), 1,0000 ($q=1$), 1,4142 ($q=0{,}5$), 1,8661 ($q=0{,}1$). Với $q<1$ trung điểm
nằm **ngoài** miền, nên miền không lồi; $q=1$ là ranh giới.

![Tibshirani 1996 tr.287 — Hình 9](<paper_lasso_tib1996_tr287_hinh9.png>)

**Hình 25 — Tibshirani 1996, Fig. 9, tr.287.** Trang PDF 22.

**Caption gốc.** *"Fig. 9. Contours of constant value of Σⱼ|βⱼ|^q for given values of q: (a) q = 4; (b) q = 2; (c)
q = 1; (d) q = 0.5; (e) q = 0.1"*

(Hình 9. Đường mức của $\sum\lvert\beta_j\rvert^q$ với $q$ = 4; 2; 1; 0,5; 0,1.)

**Nhận xét của tác giả.** *"Fig. 9 depicts the situation in two dimensions. Subset selection corresponds to q → 0."*

**Cách đọc hình.** $q=4$ gần hình vuông; $q=2$ là hình tròn; $q=1$ là hình thoi có góc trên trục; $q=0{,}5$ lõm vào
thành hình sao; $q=0{,}1$ gần như chỉ còn hai trục. Từ trái sang phải, miền càng "dồn" về các trục — tức càng ưu
tiên nghiệm có hệ số 0 — nhưng từ (d) trở đi thì mất tính lồi.

**Điều hình KHÔNG nói.** Góc nhọn trên trục **không** độc quyền của $q=1$: mọi $q\le1$ đều có. Thứ $q=1$ có mà các
$q<1$ không có là **tính lồi**, và đó mới là lý do chọn nó.

**§12 Software** (Hình 25, phần dưới): hàm S-PLUS trên Statlib — không có nội dung phương pháp.

### 3.17 Tibshirani 2011, tr.273–274 — §1 lasso, §2 lịch sử, §3 tiến bộ tính toán

![Tibshirani 2011 tr.273 — §1, §2](<paper_lasso_tib2011_tr273_muc1_2.png>)

**Hình 26 — Tibshirani 2011, tr.273: §1 The lasso và đầu §2 History of the idea.** R. Tibshirani, *Regression
shrinkage and selection via the lasso: a retrospective*, J. R. Statist. Soc. B 73(3):273–282, 2011,
`_papers/2011 - Tibshirani - Regression shrinkage and selection via the lasso, a retrospective.pdf`, trang PDF 1.

**Khung chứa gì.** Dạng phạt của lasso, câu so với ridge, họ phạt tổng quát, garrotte của Breiman.

$$\sum_{i=1}^N\Bigl(y_i-\sum_jx_{ij}\beta_j\Bigr)^2+\lambda\sum_{j=1}^p\lvert\beta_j\rvert$$

**Ý nghĩa.** Đây là dạng **phạt** của (1): thay vì giới hạn ngân sách $t$, mỗi đơn vị trị tuyệt đối bị tính một giá
$\lambda$. Hai dạng cho **cùng một họ nghiệm** — mỗi $t$ ứng với một $\lambda$ (mục 4.1). Bài hồi tưởng gọi cận là
$s$, khác nghĩa $s$ của bài 1996 (chỗ trùng tên 1 ở mục 0.3).

**Tác giả nói gì.** *"Because of the form of the l1-penalty, the lasso does variable selection and shrinkage,
whereas ridge regression, in contrast, only shrinks. If we consider a more general penalty of the form
(Σ_{j=1}^p β_j^q)^{1/q}, then the lasso uses q = 1 and ridge regression has q = 2."* Về nguồn gốc: *"My most direct
influence was Leo Breiman's non-negative garrotte (Breiman, 1995)."* và *"This is undefined when p>N (which was not a
hot topic in 1995!) so I just combined the two stages into one (as a Canadian I also wanted a gentler name)."*

(Vì dạng của phạt $\ell_1$, lasso vừa chọn biến vừa co, còn ridge chỉ co. Nếu xét phạt tổng quát
$(\sum\beta_j^q)^{1/q}$ thì lasso là $q=1$, ridge là $q=2$. Ảnh hưởng trực tiếp nhất là garrotte của Breiman. Garrotte
không xác định khi $p>N$ — điều năm 1995 chưa ai quan tâm — nên tác giả gộp hai giai đoạn làm một, và "là người
Canada nên cũng muốn một cái tên dịu hơn".)

**Chỗ thiếu trị tuyệt đối** (`v_lasso.py` mục 13). Biểu thức $(\sum\beta_j^q)^{1/q}$ in không có $\lvert\cdot\rvert$. Với
$q=1$ và $\beta=(1,-1)$ nó bằng **0**, trong khi $\sum\lvert\beta_j\rvert=2$ — không còn là một chuẩn. Đúng phải là
$(\sum\lvert\beta_j\rvert^q)^{1/q}$, như Hình 9 của bài 1996.

**Cách đọc.** Câu *"undefined when p>N"* là lý do **kỹ thuật** duy nhất bài hồi tưởng nêu cho việc bỏ garrotte: garrotte
cần nghiệm OLS, mà OLS không duy nhất khi số biến vượt số mẫu. Lasso không cần.

![Tibshirani 2011 tr.274 — §2, §3](<paper_lasso_tib2011_tr274_muc2_3.png>)

**Hình 27 — Tibshirani 2011, tr.274: cuối §2 và §3 Computational advances.** Trang PDF 2.

**Tác giả nói gì.** *"After publication, the paper did not receive much attention until years later. Why?: my guesses
are that (a) the computation in 1996 was slow compared with today, (b) the algorithms for the lasso were black boxes
and not statistically motivated (until the LARS algorithm in 2002), (c) the statistical and numerical advantages of
sparsity were not immediately appreciated (by me or the community), (d) large data problems (in N, p or both) were
rare and (e) the community did not have the R language for fast, easy sharing of new software tools."* Và §3: *"The
original lasso paper used an off-the-shelf quadratic program solver. This does not scale well and is not
transparent."* cùng *"Co-ordinate descent algorithms are extremely simple and fast, and exploit the assumed sparsity
of the model to great advantage."*

(Sau khi đăng, bài báo không được chú ý nhiều cho tới nhiều năm sau. Năm lý do tác giả đoán: tính toán năm 1996
chậm; thuật toán là hộp đen cho tới LARS năm 2002; lợi ích của tính thưa chưa được nhận ra; bài toán dữ liệu lớn còn
hiếm; cộng đồng chưa có R để chia sẻ phần mềm. Bài báo gốc dùng một bộ giải quy hoạch toàn phương có sẵn, không mở
rộng được và không minh bạch. Hạ theo toạ độ đơn giản, nhanh, và tận dụng tính thưa.)

**Cách đọc.** "LARS algorithm in 2002" là năm của **báo cáo kỹ thuật** (tài liệu tham khảo của bài hồi tưởng ghi *Efron
et al. (2002) … Technical Report*); bài đăng tạp chí là 2004 — nguồn thứ tư của bài này.

**Điều khung KHÔNG nói.** "Not much attention until years later" không kèm con số; số đếm trích dẫn nằm ở Hình 2 (mục
3.18), và hình đó không in giá trị.

### 3.18 Tibshirani 2011, tr.274–276 — §4 mở rộng, Bảng 1, Hình 2, §5

![Tibshirani 2011 tr.275 — Bảng 1](<paper_lasso_tib2011_tr275_bang1.png>)

**Hình 28 — Tibshirani 2011, Table 1, tr.275.** Trang PDF 3.

**Caption gốc.** *"Table 1. A sampling of generalizations of the lasso"*

(Bảng 1. Một số mở rộng của lasso.)

**Nhận xét của tác giả.** Tr.274: *"There has been much work in recent years, applying and generalizing the lasso and
l1-like penalties to a variety of problems. Table 1 gives a partial list."*

**Cách đọc bảng.** Ba cột: tên phương pháp, tài liệu, và **khoản phạt** hoặc bài toán. Mười dòng: grouped lasso
($\sum_g\lVert\beta_g\rVert_2$ — phạt theo nhóm), elastic net ($\lambda_1\sum\lvert\beta_j\rvert+\lambda_2\sum\beta_j^2$), fused
lasso (phạt hiệu hai hệ số kề nhau), adaptive lasso (trọng số $w_j$ riêng), graphical lasso (phạt ma trận nghịch đảo
hiệp phương sai), Dantzig selector, near isotonic, matrix completion (chuẩn nhân), compressive sensing
($\min\lVert\beta\rVert_1$ với $y=X\beta$), và các phương pháp nhiều biến thưa.

**Điều bảng KHÔNG nói.** Không có số đo nào; bảng là danh sách. Nó cũng không phân biệt mở rộng nào giữ được tính
lồi — tất cả mười dòng đều lồi, trong khi các phạt phi lồi (SCAD, MCP) chỉ được nhắc ở phần thảo luận.

![Tibshirani 2011 tr.278 — Hình 2](<paper_lasso_tib2011_tr278_hinh2.png>)

**Hình 29 — Tibshirani 2011, Fig. 2, tr.278** (trong phần thảo luận của Bühlmann). Trang PDF 6.

**Caption gốc.** *"Fig. 2. Cumulative citation counts (on a log-scale) from the Thomson ISI Web of Knowledge (the
largest abscissa on the x-axis corresponds to August 31st, 2010): (a) the lasso (○) (Tibshirani, 1996), false
discovery rate (Δ) (Benjamini and Hochberg, 1995), reversible jump Markov chain Monte Carlo sampling (+) (Green,
1995) and wavelet shrinkage (×) (Donoho and Johnstone, 1994), published between 1994 and 1996; (b) the bootstrap (Δ)
(Efron, 1979), published earlier"*

(Hình 2. Số trích dẫn cộng dồn, thang log, theo Web of Knowledge tới 31/8/2010: (a) lasso (tròn), tỉ lệ phát hiện sai
(tam giác), MCMC nhảy ngược (cộng), co sóng con (chéo) — bốn bài đăng 1994–1996; (b) bootstrap, đăng sớm hơn.)

**Nhận xét của tác giả hình.** Bühlmann tr.277: *"Fig. 2 shows that its frequency of citation continues to be in the
exponential growth regime, together with the false discovery rate paper from Benjamini and Hochberg (1995)"*.

**Cách đọc hình.** Trục dọc thang log nên đường thẳng nghĩa là tăng theo hàm mũ. Chuỗi tròn (lasso) nằm **dưới
cùng** trong bốn chuỗi ở panel (a) suốt cả trục thời gian, dù dốc lên đều; chuỗi chéo (co sóng con) đi trước
ngay từ những năm đầu. Đó là bằng chứng hình ảnh cho câu "not much attention until years later" ở mục 3.17.

**Điều hình KHÔNG nói.** Các điểm là ký tự văn bản trong PDF, không có giá trị số kèm theo; bài này không số hoá chúng
nên không trích con số trích dẫn nào.

**§4 (phần chữ ở Hình 27) và §5.** §4 kể graphical lasso và hồi quy gần đơn điệu (Hình 1 của bài hồi tưởng, một
phương pháp khác — mục 10 ghi lý do không lấy). §5 kết: *"I conclude with a challenge for statisticians. This is an
enjoyable area to work in, but we should not invent new models and algorithms just for the sake of it. We should focus
on developing tools and understanding their properties, to help us and our collaborators to solve important scientific
problems."*

(Một lời thách: đây là lĩnh vực thú vị, nhưng không nên đặt ra mô hình và thuật toán mới chỉ để có cái mới; nên tập
trung phát triển công cụ, hiểu tính chất của chúng, để giải bài toán khoa học quan trọng.)

### 3.19 Tibshirani 2011, tr.279–281 — thảo luận của Holmes, công thức (2)–(3), và phần trả lời

![Tibshirani 2011 tr.279 — Holmes](<paper_lasso_tib2011_tr279_holmes.png>)

**Hình 30 — Tibshirani 2011, tr.279: thảo luận của Chris Holmes, (2) và (3).** Trang PDF 7.

$$\hat\beta=\arg\max_\beta\Bigl\{l(\beta)-\lambda\sum_j\lvert\beta_j\rvert^q\Bigr\} \tag{2}$$

$$\{\tilde\beta,\tilde\eta\}=\arg\max_{\beta,\eta}\Bigl\{l(\beta)-\sum_j\eta_j^{-1}\beta_j^2\Bigr\}\quad\text{subject to}\ \sum_j\eta_j=t \tag{3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $l(\beta)$ | log-hợp lý, *"recording fidelity to the data"* | vô hướng |
| $\eta_j$ | phương sai tiên nghiệm chuẩn riêng cho hệ số $j$ | $\eta_j\ge0$ |
| $t$ | tổng các phương sai, *"a monotone function of λ"* | vô hướng |

**Ý nghĩa.** (3) nói lasso là **ridge với mức phạt riêng từng hệ số**, trong đó các mức phạt tự điều chỉnh theo dữ
liệu dưới một tổng cố định. Cực đại theo $\eta$ với $\sum\eta_j=t$ cho $\eta_j\propto\lvert\beta_j\rvert$, và khoản phạt thành
$(\sum\lvert\beta_j\rvert)^2/t$ — bình phương chuẩn một, nên có **cùng tập nghiệm** với lasso. Hệ số nào không liên quan thì
$\eta_j\to0$, bị phạt vô hạn, về 0.

**Kiểm lại** (`v_lasso.py` mục 13, prostate bản có lỗi, $l=-\tfrac12\text{RSS}$, $t=0{,}05$). Lặp ridge với
$\eta_j=t\lvert\beta_j\rvert/\sum\lvert\beta_k\rvert$ hội tụ sau 260 vòng. Mức phạt lasso tương đương
$\lambda=2\sum\lvert\beta_j\rvert/t=25{,}889748$. Hệ số theo (3): $0{,}517902;\ 0{,}027483;\ 0;\ 0;\ 0{,}101858;\ 0;\ 0;\ 0$;
lasso cùng $\lambda$: y hệt, lệch lớn nhất $2{,}620\cdot10^{-14}$.

**Tác giả (Holmes) nói gì.** *"one interpretation of the lasso is as a Bayesian maximum a posteriori (MAP) estimate
under a double-exponential prior on β"*; và *"In terms of the MAP interpretation of expression (2) we see that the
tails of the double-exponential prior are too light."*

(Một cách hiểu lasso là ước lượng MAP dưới tiên nghiệm Laplace. Nhìn theo (2), đuôi của tiên nghiệm Laplace là quá
nhẹ — tức lasso co cả hệ số lớn, cái giá mà các phạt phi lồi cố tránh.)

![Tibshirani 2011 tr.280 — trả lời](<paper_lasso_tib2011_tr280_tra_loi.png>)

**Hình 31 — Tibshirani 2011, tr.280: phần trả lời của tác giả.** Trang PDF 8.

**Tác giả nói gì.** *"In my original paper I mentioned that the lasso solution can be viewed as the Bayesian maximum a
posteriori estimate when the parameters are a priori independent, each having a Laplacian (double-exponential) prior
distribution. However, the lasso solution is not the posterior mean or median in that setting: these latter solutions
will not be sparse."* Về cách viết của Holmes: *"It interesting that the solution to this problem gives the lasso
estimate. But I am not sure what to make of it"*. Và *"if you want to obtain sparse solutions from a standard Bayesian
model, you need to specify a prior that puts some mass at zero."*

(Lasso là mode hậu nghiệm với tiên nghiệm Laplace, nhưng không phải trung bình hay trung vị hậu nghiệm — hai thứ đó
không thưa. Cách viết của Holmes thú vị, nhưng không rõ nên hiểu nó thế nào. Muốn nghiệm thưa từ một mô hình Bayes
chuẩn thì tiên nghiệm phải đặt khối lượng tại 0.)

**Kiểm lại "not sparse".** Bảng một chiều ở mục 3.11: với $z=1$, mode bằng 0 còn trung bình và trung vị hậu nghiệm là
0,503223 và 0,428800. Không giá trị $z\ne0$ nào trong bảng cho trung bình hay trung vị bằng 0.

**Điều khung KHÔNG nói.** Phần thảo luận của Bühlmann (tr.277–279) nêu các tốc độ hội tụ và điều kiện chọn đúng biến
— *"I suggest that we interpret the second 's' in lasso as 'screening' rather than 'selection'"* — nhưng không có phép
đo nào; mục 10 ghi lý do không dựng lại.

### 3.20 ESL tr.3 — lỗi trong dữ liệu prostate

![ESL tr.3 — chú thích](<paper_lasso_esl2009_tr3_chu_thich_loi_du_lieu.png>)

**Hình 32 — ESL, tr.3: đoạn Example 2 và chú thích 1.** T. Hastie, R. Tibshirani, J. Friedman, *The Elements of
Statistical Learning*, bản 2, Springer 2009, bản in 12,
`_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf`, trang
PDF 22.

**Khung chứa gì.** Câu mở ví dụ prostate và chú thích cuối trang.

**Tác giả nói gì.** *"There was an error in these data in the first edition of this book. Subject 32 had a value of 6.1
for lweight, which translates to a 449 gm prostate! The correct value is 44.9 gm. We are grateful to Prof. Stephen W.
Link for alerting us to this error."*

(Bản đầu của sách có một lỗi dữ liệu: đối tượng 32 có lweight bằng 6,1, tức tuyến tiền liệt 449 gm; giá trị đúng là
44,9 gm.)

Tài liệu của gói R `ElemStatLearn` (bản 2015.6.26.2) ghi thêm: *"There was an error in this dataset in earlier versions
of the package, as indicated in a footnote on page 3 of the second edition of the book. As of version 2012.04-0 this
was corrected."*

**Cách đọc.** Chú thích nói về **bản đầu của sách** (2001), nhưng lỗi có từ trước: bài báo 1996 dùng cùng bộ dữ liệu, và
mục 3.8 cho thấy Bảng 1 của bài báo chỉ dựng lại được khi đặt lweight$_{32}=\ln449=6{,}107023$. Hai giá trị cách nhau
đúng $\ln10=2{,}302585$ (`v_lasso.py` mục 4) — lỗi dấu thập phân khi nhập khối lượng.

**Điều khung KHÔNG nói.** Chú thích không nói các bảng nào của bản đầu bị ảnh hưởng, và không nhắc bài báo 1996.

### 3.21 ESL tr.62–63 — ô Lasso của Hình 3.7 và cột Lasso của Bảng 3.3

![ESL tr.62 — Hình 3.7](<paper_lasso_esl2009_tr62_hinh3_7.png>)

**Hình 33 — ESL, Figure 3.7, tr.62.** Trang PDF 81.

**Caption gốc.** *"FIGURE 3.7. Estimated prediction error curves and their standard errors for the various selection
and shrinkage methods. Each curve is plotted as a function of the corresponding complexity parameter for that method.
The horizontal axis has been chosen so that the model complexity increases as we move from left to right. The
estimates of prediction error and their standard errors were obtained by tenfold cross-validation; full details are
given in Section 7.10. The least complex model within one standard error of the best is chosen, indicated by the purple
vertical broken lines."*

(Hình 3.7. Đường sai số dự đoán ước lượng và sai số chuẩn của chúng cho các phương pháp. Mỗi đường vẽ theo tham số độ
phức tạp của phương pháp đó, trục ngang chọn sao cho độ phức tạp tăng từ trái sang phải. Ước lượng bằng kiểm định chéo
10 mảnh. Mô hình đơn giản nhất nằm trong một sai số chuẩn của mô hình tốt nhất được chọn, đánh dấu bằng vạch tím đứng.)

**Nhận xét của tác giả.** Tr.61: *"We have used the "one-standard-error" rule—we pick the most parsimonious model within
one standard error of the minimum"*. Tr.69: *"A value ŝ ≈ 0.36 was chosen by 10-fold cross-validation; this caused
four coefficients to be set to zero (fifth column of Table 3.3)."*

**Đọc toạ độ vectơ của ô Lasso** (`v_lasso.py` mục 16). Hình trong PDF là đồ hoạ vectơ: khung ô Lasso từ
$x=158{,}36$ tới $285{,}10$ pt; trục ngang có 6 vạch chia (0,0 … 1,0), trục dọc 7 vạch (0,6 … 1,8).

| $s$ đọc được | −0,0006 | 0,124 | 0,2497 | 0,3743 | 0,5 | 0,6246 | 0,7491 | 0,8748 | 0,9994 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CV đọc được | 1,4687 | 1,041 | 0,7915 | 0,6509 | 0,5836 | 0,5955 | 0,6172 | 0,6192 | 0,6172 |

Vạch tím **ngang** ở CV = 0,6753; vạch tím **đứng** ở $s=0{,}3749$. Cực tiểu của đường ở $s=0{,}5000$, CV 0,5836; khoảng
cách từ cực tiểu lên vạch ngang là 0,0917 — một sai số chuẩn. Các điểm nằm dưới vạch ngang có $s$ nhỏ nhất là
**0,3743**.

**Cách đọc hình.** Ô Lasso là ô giữa bên trái. Chín điểm cam cách đều $1/8$ trên trục $s$. Đường giảm nhanh tới
$s=0{,}5$ rồi phẳng. Quy tắc một sai số chuẩn chọn điểm **đầu tiên** rơi xuống dưới vạch ngang — điểm thứ tư,
$s=3/8=0{,}375$ — và vạch tím đứng nằm đúng ở đó.

**Điều hình KHÔNG nói.** Lưới chỉ có chín giá trị; $0{,}36$ **không phải** một điểm của lưới. Văn bản tr.69 viết
"≈ 0.36" cho một điểm mà chính hình đặt ở $0{,}375$.

![ESL tr.63 — Bảng 3.3](<paper_lasso_esl2009_tr63_bang3_3.png>)

**Hình 34 — ESL, Table 3.3, tr.63.** Trang PDF 82.

**Caption gốc.** *"TABLE 3.3. Estimated coefficients and test error results, for different subset and shrinkage methods
applied to the prostate data. The blank entries correspond to variables omitted."*

(Bảng 3.3. Hệ số ước lượng và sai số kiểm tra của các phương pháp trên dữ liệu prostate. Ô trống là biến bị bỏ.)

**Nhận xét của tác giả.** Tr.69: *"The resulting model has the second lowest test error, slightly lower than the full
least squares model, but the standard errors of the test error estimates (last line of Table 3.3) are fairly large."*

**Dựng lại cột Lasso** (`v_lasso.py` mục 14). Cột LS của cùng bảng khớp dữ liệu chuẩn hoá trên **cả 97 mẫu** (bài
[Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) đo điều đó). Giữ đầu vào như vậy và giải lasso
thẳng trên 67 mẫu huấn luyện thì **không** ra cột in:

| Cách giải | ô khớp tốt nhất | thứ tự biến vào đường |
|---|---:|---|
| chuẩn hoá trên 97 mẫu, giải thẳng | 5/9 | lcavol, lweight, svi, **pgg45, lbph**, age, lcp, gleason |
| chuẩn hoá trên 67 mẫu, giải thẳng | 6/9 | lcavol, lweight, svi, **lbph, pgg45**, age, lcp, gleason |
| đầu vào chuẩn hoá trên 97 mẫu, bộ giải **chuẩn hoá lại** trên 67 mẫu rồi trả hệ số về thang đầu vào | **9/9** | như dòng thứ hai |

Cột in có đúng bốn biến lcavol, lweight, **lbph**, svi. Với thang 97 mẫu, pgg45 vào đường **trước** lbph, nên tập đó không
bao giờ xuất hiện. Dòng thứ ba là cách các gói như `lars` làm theo mặc định: chuẩn hoá bên trong trên dữ liệu được
đưa vào (67 mẫu), rồi trả hệ số theo thang của dữ liệu đầu vào. Theo cách đó, mọi $s$ trong $[0{,}36658;\ 0{,}36691]$
(tính trên thang đầu vào) cho **9/9** ô. Tại $s=0{,}36675$: $2{,}4683;\ 0{,}5328;\ 0{,}1695;\ 0;\ 0{,}0022;\ 0{,}0936;\ 0;\ 0;\ 0$;
sai số kiểm tra **0,478601** (in 0,479), sai số chuẩn **0,163806** (in 0,164). Quy về thang 67 mẫu của Hình 3.10, cùng
nghiệm đó có $s=0{,}37489$ ($t_0=2{,}260491$).

Hai dãy nút của đường theo hai thang: thang đầu vào $0;\ 0{,}1865;\ 0{,}249;\ 0{,}3642;\ 0{,}3683;\ 0{,}5501;\ 0{,}5926;\ 0{,}9421;\ 1$;
thang 67 mẫu $0;\ 0{,}1893;\ 0{,}2544;\ 0{,}3724;\ 0{,}3765;\ 0{,}557;\ 0{,}5992;\ 0{,}9433;\ 1$.

**Cách đọc bảng.** Sáu cột phương pháp, chín dòng hệ số, hai dòng sai số. Cột Lasso có bốn ô trống. Sai số kiểm tra
0,479 đứng thứ hai sau PCR 0,449; nhưng sai số chuẩn 0,164 lớn hơn mọi khoảng cách giữa các cột.

**Điều bảng KHÔNG nói.** Bảng không nói quy ước chuẩn hoá, và các cột **không cùng quy ước**: cột LS khớp chuẩn hoá trên 97
mẫu, cột Lasso chỉ khớp khi bộ giải chuẩn hoá lại trên 67 mẫu, và cột Ridge không khớp quy ước nào đã thử (bài Ridge).

### 3.22 ESL tr.68–70 — (3.51), (3.52), chọn $s$, Hình 3.10

![ESL tr.68 — (3.51), (3.52)](<paper_lasso_esl2009_tr68_3_51_3_52.png>)

**Hình 35 — ESL, §3.4.2 The Lasso, tr.68.** Trang PDF 87.

$$\hat\beta^{\text{lasso}}=\arg\min_\beta\sum_{i=1}^N\Bigl(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\Bigr)^2\quad\text{subject to}\ \sum_{j=1}^p\lvert\beta_j\rvert\le t \tag{3.51}$$

$$\hat\beta^{\text{lasso}}=\arg\min_\beta\Bigl\{\frac12\sum_{i=1}^N\Bigl(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\Bigr)^2+\lambda\sum_{j=1}^p\lvert\beta_j\rvert\Bigr\} \tag{3.52}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\beta_0$ | hệ số chặn; sau khi chuẩn hoá thì $\hat\beta_0=\bar y$ | vô hướng |
| $t$ | cận — cùng nghĩa với (1) của bài 1996 | vô hướng |
| $\lambda$ | mức phạt — với hệ số $\tfrac12$ trước RSS, **bằng nửa** $\lambda$ của bài 1996 | vô hướng |

**Ý nghĩa.** (3.51) là (1) với tên khác; (3.52) là dạng phạt. Hệ số $\tfrac12$ không đổi nghiệm, chỉ đổi thang của
$\lambda$ — và làm $\lambda$ bằng đúng giá trị $\lvert x_j^\top(y-X\hat\beta)\rvert$ của mọi biến đang hoạt động (mục 4.2).

**Tác giả nói gì.** *"The lasso is a shrinkage method like ridge, with subtle but important differences."*; *"In the
signal processing literature, the lasso is also known as basis pursuit (Chen et al., 1998)."*; và *"This latter
constraint makes the solutions nonlinear in the yi, and there is no closed form expression as in ridge regression."*

![ESL tr.69 — chọn s](<paper_lasso_esl2009_tr69_chon_s.png>)

**Hình 36 — ESL, tr.69: $t_0$, chọn $s$, Hình 3.10, mở §3.4.3.** Trang PDF 88.

**Tác giả nói gì.** *"Thus the lasso does a kind of continuous subset selection."*; *"On the other hand, for t = t0/2
say, then the least squares coefficients are shrunk by about 50% on average."*; và về Hình 3.10: *"At s = 1.0 these are
the least squares estimates; they decrease to 0 as s → 0. This decrease is not always strictly monotonic, although it is
in this example. A vertical line is drawn at s = 0.36, the value chosen by cross-validation."*

**Kiểm "co khoảng 50 % trung bình"** (`v_lasso.py` mục 15, dữ liệu của Hình 3.10). Tại $t=t_0/2$, tỉ số
$\lvert\hat\beta_j\rvert/\lvert\hat\beta^{\text{LS}}_j\rvert$ của tám biến: $0{,}8014;\ 0{,}7782;\ 0;\ 0{,}4684;\ 0{,}5408;\ 0;\ 0;\ 0{,}2222$ —
trung bình **0,3514**, trung vị 0,3453, ba biến về 0. Chỉ tổng $\sum\lvert\hat\beta_j\rvert/t_0$ mới bằng đúng 0,5000, và đó
là định nghĩa của $t_0/2$ chứ không phải một phát hiện. Hai biến lớn nhất giữ lại 0,8014 và 0,7782 độ lớn OLS; phần
co dồn vào các biến nhỏ.

![ESL tr.70 — Hình 3.10](<paper_lasso_esl2009_tr70_hinh3_10.png>)

**Hình 37 — ESL, Figure 3.10, tr.70.** Trang PDF 89.

**Caption gốc.** *"FIGURE 3.10. Profiles of lasso coefficients, as the tuning parameter t is varied. Coefficients are
plotted versus s = t/Σ|β̂j|. A vertical line is drawn at s = 0.36, the value chosen by cross-validation. Compare Figure
3.8 on page 65; the lasso profiles hit zero, while those for ridge do not. The profiles are piece-wise linear, and so are
computed only at the points displayed; see Section 3.4.4 for details."*

(Hình 3.10. Đường hệ số lasso khi $t$ thay đổi, vẽ theo $s$. Vạch đứng ở $s=0{,}36$, giá trị chọn bằng kiểm định chéo.
So với Hình 3.8: đường lasso chạm 0, đường ridge thì không. Các đường tuyến tính từng khúc nên chỉ cần tính tại các
điểm vẽ.)

**Nhận xét của tác giả.** Đoạn tr.69 vừa dẫn.

**Đọc toạ độ vectơ** (`v_lasso.py` mục 15). Vạch chia trục ngang ở $x=180{,}41;\ 225{,}38;\ 270{,}35;\ 315{,}33;\ 360{,}29;\ 405{,}26$
pt, trục dọc ở $y=343{,}03;\ 395{,}49;\ 447{,}94;\ 500{,}39;\ 552{,}85$ pt. Trong hình có:

- **Chín vạch xám đứng** — chính là các nút của đường — ở $s=0;\ 0{,}1893;\ 0{,}2544;\ 0{,}3724;\ 0{,}3765;\ 0{,}5571;\ 0{,}5992;\ 0{,}9433;\ 1$.
  Đường lasso trên dữ liệu chuẩn hoá theo 67 mẫu huấn luyện có nút ở cùng các giá trị, lệch lớn nhất $2{,}20\cdot10^{-5}$.
- **Tám đường xanh** với đầu mút $0{,}7164;\ 0{,}2926;\ -0{,}1426;\ 0{,}212;\ 0{,}3096;\ -0{,}289;\ -0{,}0209;\ 0{,}2774$ — trùng LS chuẩn
  hoá theo 67 mẫu tới chữ số thứ tư. Mọi điểm gãy của cả tám đường lệch đường dựng lại $0{,}0000$ khi làm tròn bốn chữ số.
- $\lvert\hat\beta_j(s)\rvert$ không giảm theo $s$ trên cả tám đường — câu *"although it is in this example"* đúng.
- **Một vạch đỏ đứng ở $s=0{,}3750$**, không phải 0,36.

Tại $s=0{,}36$ đường có ba biến khác 0 (lcavol, lweight, svi) và **năm** biến bằng 0. Tại $s=0{,}3750$ có bốn biến khác 0
(thêm lbph) và **bốn** bằng 0 — đúng câu *"four coefficients to be set to zero"* và đúng cột Lasso của Bảng 3.3.

**Cách đọc hình.** Trục ngang $s$ từ 0 tới 1; tên biến ghi ở mép phải. Vạch xám mảnh là các nút; giữa hai nút, mọi
đường là đoạn thẳng. lcavol rời 0 đầu tiên và luôn lớn nhất; lbph và pgg45 rời 0 ở hai nút rất gần nhau (0,3724 và
0,3765) — vạch đỏ nằm **giữa** hai nút đó. Hình 52 panel (a) chồng điểm đọc từ PDF lên đường dựng lại.

**Điều hình KHÔNG nói.** Hình không nói chuẩn hoá theo tập nào; phép đọc cho thấy là theo 67 mẫu huấn luyện, **khác**
cột LS của Bảng 3.3 ngay trang trước. Và vạch đỏ nằm trong khe giữa hai nút 0,3724 và 0,3765 — dời nó tới 0,36 như văn bản là
đổi mô hình.

### 3.23 ESL tr.69–73 — §3.4.3: Bảng 3.4, Hình 3.11, (3.53), Hình 3.12, Hình 3.13, (3.54)

![ESL tr.71 — Bảng 3.4 và Hình 3.11](<paper_lasso_esl2009_tr71_bang3_4_hinh3_11.png>)

**Hình 38 — ESL, Table 3.4 và Figure 3.11, tr.71.** Trang PDF 90.

**Caption gốc.** *"TABLE 3.4. Estimators of βj in the case of orthonormal columns of X. M and λ are constants chosen by
the corresponding techniques; sign denotes the sign of its argument (±1), and x+ denotes "positive part" of x. Below the
table, estimators are shown by broken red lines. The 45° line in gray shows the unrestricted estimate for reference."*
và *"FIGURE 3.11. Estimation picture for the lasso (left) and ridge regression (right). Shown are contours of the error
and constraint functions. The solid blue areas are the constraint regions |β1| + |β2| ≤ t and β1² + β2² ≤ t²,
respectively, while the red ellipses are the contours of the least squares error function."*

(Bảng 3.4. Ước lượng $\beta_j$ khi các cột trực chuẩn; $M$, $\lambda$ là hằng số do từng phương pháp chọn. Hình 3.11. Bức
tranh ước lượng của lasso và ridge; vùng xanh là miền ràng buộc $\lvert\beta_1\rvert+\lvert\beta_2\rvert\le t$ và
$\beta_1^2+\beta_2^2\le t^2$.)

**Nhận xét của tác giả.** Tr.69: *"Ridge regression does a proportional shrinkage. Lasso translates each coefficient by a
constant factor λ, truncating at zero. This is called "soft thresholding," … Best-subset selection drops all variables
with coefficients smaller than the Mth largest; this is a form of "hard-thresholding.""* Tr.72: *"the region for ridge
regression is the disk β1² + β2² ≤ t, while that for lasso is the diamond |β1| + |β2| ≤ t."*

**Ba công thức của Bảng 3.4:**

$$\text{Best subset: }\hat\beta_j\cdot I(\lvert\hat\beta_j\rvert\ge\lvert\hat\beta_{(M)}\rvert),\qquad\text{Ridge: }\hat\beta_j/(1+\lambda),\qquad\text{Lasso: }\operatorname{sign}(\hat\beta_j)(\lvert\hat\beta_j\rvert-\lambda)^+$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta_{(M)}$ | hệ số LS lớn thứ $M$ theo trị tuyệt đối | vô hướng |
| $M$ | số biến giữ lại | số nguyên |

**Ý nghĩa.** Ba dòng là (3) và hai công thức ridge, tập con của bài 1996 ở dạng gọn — cùng kết luận: ridge nhân, lasso
trừ rồi chặn, tập con giữ hoặc xoá. `v_lasso.py` mục 17 thử dòng tập con: $\hat\beta=(3;\ -0{,}4;\ 1{,}7;\ -2{,}2;\ 0{,}1)$ với
$M=2$ cho $(3;\ 0;\ 0;\ -2{,}2;\ 0)$.

**Hai chỗ lệch.** Chú thích Hình 3.11 viết miền ridge là $\beta_1^2+\beta_2^2\le t^2$, văn bản tr.72 viết $\le t$ — hai
thang cho cùng một miền, trong hai trang liền nhau. Và câu *"translates each coefficient by a constant factor λ"* dùng
chữ "factor" cho một phép **trừ**, chép lại cụm của bài 1996 (mục 3.3).

**Cách đọc hình.** Hai hình giống Hình 2 của bài 1996, thêm màu: vùng xanh là miền ràng buộc, elip đỏ là đường mức.
Elip chạm hình thoi ở một góc — một hệ số bằng 0.

![ESL tr.72 — (3.53), Hình 3.12](<paper_lasso_esl2009_tr72_3_53_hinh3_12.png>)

**Hình 39 — ESL, tr.72: (3.53) và Figure 3.12.** Trang PDF 91.

$$\tilde\beta=\arg\min_\beta\Bigl\{\sum_{i=1}^N\Bigl(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\Bigr)^2+\lambda\sum_{j=1}^p\lvert\beta_j\rvert^q\Bigr\} \tag{3.53}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $q$ | số mũ; $q\ge0$ | vô hướng |
| $\lambda$ | mức phạt — ở đây RSS **không** có hệ số $\tfrac12$, khác (3.52) | vô hướng |

**Ý nghĩa.** Một họ phạt nối tập con ($q=0$, đếm số hệ số khác 0), lasso ($q=1$) và ridge ($q=2$). Đọc theo Bayes, mỗi $q$
là một tiên nghiệm với $-\log$ mật độ tỉ lệ $\lvert\beta_j\rvert^q$.

**Caption gốc của Hình 3.12.** *"FIGURE 3.12. Contours of constant value of Σj |βj|^q for given values of q."*

**Nhận xét của tác giả.** *"Thinking of |βj|^q as the log-prior density for βj, these are also the equi-contours of the
prior distribution of the parameters."*; *"The prior corresponding to the q = 1 case is an independent double exponential
(or Laplace) distribution for each input, with density (1/2τ)exp(−|β|/τ) and τ = 1/λ."*; *"Ridge regression is also the
posterior mean, but the lasso and best subset selection are not."*

**Hai chỗ cần sửa khi đọc.** $\lvert\beta_j\rvert^q$ là **trừ** log mật độ tiên nghiệm, không phải log mật độ — bài 1996 viết
đúng *"(minus) log-density"*. Và $\tau=1/\lambda$ có cùng điều kiện ẩn $\sigma^2=1/2$ như ở mục 3.11, vì (3.53) cũng dùng RSS
không nhân $\tfrac12$. Câu về trung bình hậu nghiệm được mục 3.11 đo: lasso là mode, không phải trung bình.

**Cách đọc hình.** Giống Hình 9 của bài 1996: năm đường mức với $q=4;\ 2;\ 1;\ 0{,}5;\ 0{,}1$.

![ESL tr.73 — Hình 3.13, (3.54)](<paper_lasso_esl2009_tr73_hinh3_13_3_54.png>)

**Hình 40 — ESL, Figure 3.13 và (3.54), tr.73.** Trang PDF 92.

**Caption gốc.** *"FIGURE 3.13. Contours of constant value of Σj |βj|^q for q = 1.2 (left plot), and the elastic-net
penalty Σj(αβj²+(1−α)|βj|) for α = 0.2 (right plot). Although visually very similar, the elastic-net has sharp
(non-differentiable) corners, while the q = 1.2 penalty does not."*

(Hình 3.13. Đường mức của $\sum\lvert\beta_j\rvert^{1{,}2}$ và của phạt elastic net với $\alpha=0{,}2$. Nhìn rất giống nhau,
nhưng elastic net có góc nhọn không khả vi, còn $q=1{,}2$ thì không.)

$$\lambda\sum_{j=1}^p\bigl(\alpha\beta_j^2+(1-\alpha)\lvert\beta_j\rvert\bigr) \tag{3.54}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\alpha$ | tỉ lệ trộn — **nhân với phần bình phương**: $\alpha=0$ là lasso | $[0,1]$ |

**Ý nghĩa.** Một cách khác để đứng giữa lasso và ridge: cộng hai khoản phạt thay vì chọn một số mũ ở giữa. Vì vẫn còn
phần $\lvert\beta_j\rvert$, góc nhọn trên trục vẫn còn, nên vẫn có hệ số bằng 0. Chiều của $\alpha$ **ngược** với Friedman 2010
(2) và scikit-learn (chỗ trùng tên 4 ở mục 0.3).

**Nhận xét của tác giả.** *"Figure 3.13 compares the Lq penalty with q = 1.2 and the elastic-net penalty with α = 0.2; it
is hard to detect the difference by eye."*

**Đo "khó phân biệt bằng mắt"** (`v_lasso.py` mục 17). Bán kính của đường mức 1 theo góc: hai đường cùng đi qua điểm
bán kính 1,0000 trên trục, và lệch nhau lớn nhất **0,01860** (tương đối 0,0215) ở góc 16,10°. Còn độ dốc sát trục thì khác
hẳn: $q=1{,}2$ có độ dốc 0 (khả vi), elastic net có $-(1-\alpha)/(2\alpha r+1-\alpha)=-0{,}6667$ — một góc thật.

**Điều hình KHÔNG nói.** Góc nhọn là điều kiện để có số 0, không phải bảo đảm; và hình không nói gì về việc chọn $\alpha$.
Phần đó thuộc bài [Elastic Net](<../Elastic Net/Elastic Net.md>).

### 3.24 Efron và cộng sự 2004, tr.408–410 — (1.1)–(1.5), Bảng 1, Hình 1

![Efron 2004 tr.409 — (1.1)–(1.4) và Bảng 1](<paper_lasso_efron2004_tr409_1_1_bang1.png>)

**Hình 41 — Efron và cộng sự 2004, (1.1)–(1.4) và Table 1.** B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, *Least
Angle Regression*, Ann. Statist. 32(2):407–451, 2004, bản arXiv math/0406456,
`_papers/2004 - Efron et al - Least Angle Regression.pdf`, trang PDF 3.

**Caption gốc.** *"Table 1. Diabetes study: 442 diabetes patients were measured on 10 baseline variables; a prediction
model was desired for the response variable, a measure of disease progression one year after baseline"*

(Bảng 1. Nghiên cứu tiểu đường: 442 bệnh nhân đo 10 biến ban đầu; cần mô hình dự đoán mức tiến triển bệnh sau một năm.)

$$\sum_{i=1}^ny_i=0,\qquad\sum_{i=1}^nx_{ij}=0,\qquad\sum_{i=1}^nx_{ij}^2=1\quad\text{for }j=1,2,\dots,m \tag{1.1}$$

$$\hat\mu=\sum_{j=1}^mx_j\hat\beta_j=X\hat\beta \tag{1.2}$$

$$S(\hat\beta)=\lVert y-\hat\mu\rVert^2=\sum_{i=1}^n(y_i-\hat\mu_i)^2 \tag{1.3}$$

$$T(\hat\beta)=\sum_{j=1}^m\lvert\hat\beta_j\rvert \tag{1.4}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $m$ | số biến (bài này gọi là $p$) — $m=10$ | số nguyên |
| $n$ | số mẫu — $n=442$ | số nguyên |
| $x_j$ | cột $j$, trung bình 0 và **độ dài 1** | $\mathbb{R}^n$ |
| $\hat\mu$ | vector dự đoán | $\mathbb{R}^n$ |
| $S$, $T$ | tổng bình phương sai số và chuẩn một của hệ số | vô hướng |

**Ý nghĩa.** (1.1) chuẩn hoá theo **độ dài** chứ không theo độ lệch chuẩn — khác bài 1996 và ESL một hệ số $\sqrt n$, nên $t$
của Efron có thang khác. (1.2)–(1.4) chỉ đặt tên cho hai đại lượng mà lasso đánh đổi: sai số $S$ và ngân sách $T$.

**Nhận xét của tác giả.** *"Table 1 shows a small part of the data for our main example."* và *"This is assumed to be the
case in the theory which follows, except that numerical results are expressed in the original units of the diabetes
example."*

**Kiểm lại** (`v_lasso.py` mục 18). Tám dòng in của Bảng 1 so với `load_diabetes(scaled=False)` của scikit-learn: **88/88**
ô khớp sau khi làm tròn như bản in (ví dụ cột $x_9$ của bệnh nhân 1 là 4,8598, in 4,9). Chuẩn hoá (1.1):
$\sum x$ lệch $1{,}0\cdot10^{-13}$, $\sum x^2-1$ lệch $7{,}5\cdot10^{-15}$, $\sum y$ lệch $3{,}4\cdot10^{-12}$.

**Cách đọc bảng.** Mười cột biến: tuổi, giới (1 hoặc 2), BMI, huyết áp, sáu chỉ số huyết thanh; cột cuối là đáp án. Chỉ
in sáu dòng đầu và hai dòng cuối.

**Điều bảng KHÔNG nói.** Tên của sáu chỉ số huyết thanh; scikit-learn gọi chúng `s1`–`s6`.

![Efron 2004 tr.410 — (1.5)](<paper_lasso_efron2004_tr410_1_5.png>)

**Hình 42 — Efron và cộng sự 2004, tr.410: định nghĩa (1.5).** Trang PDF 4.

$$\text{Lasso: minimize } S(\hat\beta)\ \text{subject to } T(\hat\beta)\le t \tag{1.5}$$

**Ý nghĩa.** Cùng bài toán (1) của bài 1996, viết bằng tên đã đặt ở (1.3)–(1.4).

**Tác giả nói gì.** *"The left panel of Figure 1 shows all Lasso solutions β̂(t) for the diabetes study, as t increases
from 0, where β̂ = 0, to t = 3460.00, where β̂ equals the OLS regression vector, the constraint in (1.5) no longer
binding."* và *"At t = 1000, for example, only variables 3, 9, 4 and 7 enter the Lasso regression model (1.2)."*

![Hình 1 của Efron 2004 — đường lasso và stagewise trên bộ diabetes](<paper_efron2004_fig1_duong_lasso_diabetes.png>)

**Hình 43 — Efron và cộng sự 2004, Fig. 1.** Trang PDF 4.

**Caption gốc.** *"Fig. 1. Estimates of regression coefficients β̂j, j = 1, 2, . . . , 10, for the diabetes study. (Left
panel) Lasso estimates, as a function of t = Σj|β̂j|. The covariates enter the regression equation sequentially as t
increases, in order j = 3, 9, 4, 7, . . . , 1. (Right panel) The same plot for Forward Stagewise Linear Regression. The two
plots are nearly identical, but differ slightly for large t as shown in the track of covariate 8."*

(Hình 1. Hệ số hồi quy của 10 biến trên bộ tiểu đường. Trái: lasso theo $t$; các biến vào lần lượt theo thứ tự 3, 9, 4,
7, …, 1. Phải: cùng loại hình cho Forward Stagewise. Hai hình gần như giống hệt, khác nhau chút ít khi $t$ lớn ở đường của
biến 8.)

**Nhận xét của tác giả.** Hai câu tr.410 vừa dẫn, và §3.1 tr.418 (mục 3.25).

**Kiểm lại** (`v_lasso.py` mục 18, và mục 6 của `v_reg.py`): $t$ ở bước cuối **3459,9776** — bằng $\sum\lvert\hat\beta^{\text{OLS}}_j\rvert$ —
làm tròn ra 3460,00, **khi các cột chuẩn hoá theo (1.1)**. Theo đơn vị gốc của Bảng 1, cùng tổng đó chỉ là 107,1213. Vậy
câu *"numerical results are expressed in the original units"* không đúng cho $t$: trục ngang của Hình 1 và con số 3460 dùng
cột chuẩn hoá, còn $y$ giữ đơn vị gốc (chỉ trừ trung bình). Tại $t=1000$ các biến khác 0 là $\{3,4,7,9\}$. Thứ tự vào: 3, 9, 4, 7, 2, 10, 5, 8, 6, 1 — khớp chú
thích ở cả bốn biến đầu lẫn biến cuối.

**Cách đọc hình.** Trục ngang $t$ từ 0 tới 3460, tính trên cột chuẩn hoá độ dài 1; trục dọc là hệ số trên cùng các
cột đó, với $y$ theo đơn vị gốc. Số ở
mép phải là số thứ tự biến. Mỗi đường là các đoạn thẳng. Đường của biến 7 chạm lại 0 ở gần mép phải rồi đi tiếp — đó là
lần duy nhất một biến **rời** mô hình (mục 3.25).

**Điều hình KHÔNG nói.** Không đánh dấu các nút. Và câu "nearly identical" về stagewise thuộc phần LARS, không kiểm ở bài
này.

### 3.25 Efron và cộng sự 2004, tr.417–418 — §3.1 quan hệ LARS–lasso, (3.1)–(3.6), định lý 1

![Efron 2004 tr.417 — §3.1](<paper_lasso_efron2004_tr417_3_1.png>)

**Hình 44 — Efron và cộng sự 2004, §3.1 The LARS–Lasso relationship.** Trang PDF 11.

$$\operatorname{sign}(\hat\beta_j)=\operatorname{sign}(\hat c_j)=s_j \tag{3.1}$$

$$w_A=A_AG_A^{-1}1_A \tag{3.2}$$

$$\beta_j(\gamma)=\hat\beta_j+\gamma\hat d_j\quad\text{for }j\in A \tag{3.3}$$

$$\gamma_j=-\hat\beta_j/\hat d_j \tag{3.4}$$

$$\tilde\gamma=\min_{\gamma_j>0}\{\gamma_j\} \tag{3.5}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat c_j$ | *"current correlation"* $x_j^\top(y-\hat\mu)$ | vô hướng |
| $s_j$ | dấu của $\hat c_j$ | $\pm1$ |
| $A$ | tập biến đang hoạt động | tập chỉ số |
| $G_A$ | ma trận $(s_jx_j)^\top(s_kx_k)$ trên $A$ | $\lvert A\rvert\times\lvert A\rvert$ |
| $A_A$ | $(1_A^\top G_A^{-1}1_A)^{-1/2}$ — hằng số chuẩn hoá | vô hướng |
| $w_A$ | trọng số của hướng "góc đều" | $\mathbb{R}^{\lvert A\rvert}$ |
| $\hat d$ | $s_jw_{Aj}$ trên $A$, 0 ngoài $A$ — hướng đi của hệ số | $\mathbb{R}^m$ |
| $\gamma$ | độ dài bước dọc hướng đó | vô hướng |

**Ý nghĩa.** (3.1) là điều kiện tối ưu của lasso viết gọn: **hệ số khác 0 phải cùng dấu với tương quan hiện tại**. LARS đi
theo một hướng thẳng (3.3); nếu trên đường đi một hệ số chạm 0 (3.4) **trước** khi một biến mới đuổi kịp, thì đi tiếp sẽ
làm hệ số đó đổi dấu mà tương quan thì không — vi phạm (3.1). (3.5) tìm thời điểm đó.

**Tác giả nói gì.** *"Let β̂ be a Lasso solution (1.5), with μ̂ = Xβ̂. Then it is easy to show that the sign of any nonzero
coordinate β̂j must agree with the sign sj of the current correlation"* và *"The LARS algorithm does not enforce
restriction (3.1), but it can easily be modified to do so."*

![Efron 2004 tr.418 — lasso modification, định lý 1](<paper_lasso_efron2004_tr418_lasso_modification.png>)

**Hình 45 — Efron và cộng sự 2004, tr.418: *Lasso modification*, (3.6) và định lý 1.** Trang PDF 12.

$$\hat\mu_{A_+}=\hat\mu_A+\tilde\gamma u_A\quad\text{and}\quad A_+=A-\{\tilde j\} \tag{3.6}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $u_A$ | vector đơn vị "góc đều", $X_Aw_A$ | $\mathbb{R}^n$ |
| $\tilde j$ | chỉ số của hệ số chạm 0 ở (3.5) | chỉ số |

**Ý nghĩa.** Khi một hệ số chạm 0, **dừng bước**, bỏ biến đó khỏi tập hoạt động, rồi tính hướng mới. Đó là toàn bộ sửa đổi
biến LARS thành một thuật toán cho **cả đường nghiệm lasso**.

**Tác giả nói gì.** *"Theorem 1. Under the Lasso modification, and assuming the "one at a time" condition discussed below,
the LARS algorithm yields all Lasso solutions."* Về Hình 1: *"Modification (3.6) came into play only once, at the arrowed
point in the left panel. There A contained all 10 indices while A+ = A − {7}. Variable 7 was restored to the active set one
LARS step later, the next and last step then taking β̂ all the way to the full OLS solution."* và *"The price of using Lasso
instead of unmodified LARS comes in the form of added steps, 12 instead of 10 in this example. For the more complicated
"quadratic model" of Section 4, the comparison was 103 Lasso steps versus 64 for LARS."*

(Định lý 1: với sửa đổi lasso và điều kiện "mỗi lần một biến", LARS cho mọi nghiệm lasso. Sửa đổi chỉ xảy ra một lần ở
Hình 1: khi tập hoạt động có đủ 10 biến, biến 7 bị bỏ, một bước sau được đưa lại, và bước cuối đưa hệ số tới OLS. Cái giá
là thêm bước: 12 thay vì 10; với mô hình bậc hai của §4, 103 bước lasso so với 64 bước LARS.)

**Kiểm lại** (`v_lasso.py` mục 18): đường lasso có **12** bước, LARS **10**. Tập biến khác 0 đổi ở từng bước:
`+3, +9, +4, +7, +2, +10, +5, +8, +6, +1−7, (không đổi), +7` — ở bước 10 biến 1 rời 0 và biến 7 chạm 0; bước 11 đi với 9
biến; bước 12 đưa biến 7 trở lại và tới OLS. Đúng câu của tác giả.

Mô hình bậc hai (3.15): 10 biến chính, 45 tích đôi, 9 bình phương (bỏ bình phương của giới tính) — 64 cột, hạng 64.

| Cách tạo tích | số nút lasso (lần rời) | số nút LAR | bài báo |
|---|---:|---:|---|
| tích của biến gốc | 146 (11) | 87 | 103 và 64 |
| tích của biến đã trừ trung bình | 104 (3) | 96 | 103 và 64 |

**Cách đọc.** Về lý thuyết LAR cần **đúng 64** bước khi $n>m$. scikit-learn trả nhiều nút hơn thế, nên "số nút" của nó
không phải "số bước" theo nghĩa của bài báo; con số 104 gần 103 không chứng minh gì. Chỗ bỏ ngỏ thứ tư ở bảng đầu mục 3.

**Điều khung KHÔNG nói.** Điều kiện "one at a time" chỉ được mô tả là *"the usual case for quantitative data"*; bài báo
không nói tần suất vi phạm với dữ liệu có giá trị lặp — bộ diabetes có biến giới tính chỉ hai giá trị.

### 3.26 Friedman và cộng sự 2010, tr.3 — (1)–(3): elastic net chứa lasso

![Friedman 2010 tr.3 — (1)–(3)](<paper_lasso_glmnet2010_tr3_1_3.png>)

**Hình 46 — Friedman, Hastie, Tibshirani 2010, §2, công thức (1)–(3).** J. Friedman, T. Hastie, R. Tibshirani,
*Regularization Paths for Generalized Linear Models via Coordinate Descent*, J. Stat. Softw. 33(1), 2010,
`_papers/2010 - Friedman, Hastie, Tibshirani - Regularization Paths for GLM via Coordinate Descent (glmnet).pdf`, trang
PDF 3.

$$\min_{(\beta_0,\beta)\in\mathbb{R}^{p+1}}R_\lambda(\beta_0,\beta)=\min_{(\beta_0,\beta)\in\mathbb{R}^{p+1}}\Bigl[\frac1{2N}\sum_{i=1}^N(y_i-\beta_0-x_i^\top\beta)^2+\lambda P_\alpha(\beta)\Bigr] \tag{1}$$

$$P_\alpha(\beta)=(1-\alpha)\tfrac12\lVert\beta\rVert^2_{\ell_2}+\alpha\lVert\beta\rVert_{\ell_1} \tag{2}$$

$$=\sum_{j=1}^p\Bigl[\tfrac12(1-\alpha)\beta_j^2+\alpha\lvert\beta_j\rvert\Bigr] \tag{3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $x_i$ | **một dòng** dữ liệu, chuẩn hoá $\sum_ix_{ij}=0$, $\tfrac1N\sum_ix_{ij}^2=1$ | $\mathbb{R}^p$ |
| $\lambda$ | mức phạt, thang **chia $N$** | vô hướng |
| $\alpha$ | tỉ lệ trộn: $\alpha=1$ là lasso, $\alpha=0$ là ridge | $[0,1]$ |

**Ý nghĩa.** Với $\alpha=1$, (1) là lasso với RSS chia $2N$ — đúng thang của `alpha` trong scikit-learn. Chia cho $N$ làm
$\lambda$ không phụ thuộc cỡ mẫu: nhân đôi dữ liệu không đổi mức phạt thích hợp.

**Tác giả nói gì.** *"Ridge regression is known to shrink the coefficients of correlated predictors towards each other,
allowing them to borrow strength from each other. In the extreme case of k identical predictors, they each get identical
coefficients with 1/kth the size that any single one would get if fit alone."*; *"Lasso, on the other hand, is somewhat
indifferent to very correlated predictors, and will tend to pick one and ignore the rest."*; và *"As α increases from 0
to 1, for a given λ the sparsity of the solution to (1) (i.e., the number of coefficients equal to zero) increases
monotonically from 0 to the sparsity of the lasso solution."*

(Ridge co các hệ số của biến tương quan về gần nhau; với $k$ biến giống hệt, mỗi hệ số bằng $1/k$ hệ số khi chỉ có một biến.
Lasso thì gần như không quan tâm tới tương quan, chọn một biến và bỏ phần còn lại. Khi $\alpha$ tăng từ 0 tới 1, với $\lambda$
cố định, độ thưa tăng đơn điệu.)

**Kiểm lại** (`v_lasso.py` mục 19):

- **"1/kth the size" chỉ đúng khi $\lambda\to0$.** Với $k$ cột trùng nhau, mỗi hệ số ridge bằng $(1+\lambda)/(k+\lambda)$ lần hệ số
  khi đứng một mình:

| $k$ | $\lambda=0{,}01$ | $\lambda=0{,}1$ | $\lambda=1$ | $1/k$ |
|---:|---:|---:|---:|---:|
| 2 | 0,5025 | 0,5238 | 0,6667 | 0,5000 |
| 3 | 0,3355 | 0,3548 | 0,5000 | 0,3333 |
| 5 | 0,2016 | 0,2157 | 0,3333 | 0,2000 |

  Số đo khớp $(1+\lambda)/(k+\lambda)$ ở mọi ô.
- **Độ thưa đơn điệu theo $\alpha$** — số hệ số bằng 0 tại $\alpha=0;\ 0{,}05;\ \dots;\ 1$:

| Dữ liệu | $\lambda$ | số hệ số 0 | không giảm? |
|---|---|---|---|
| prostate | $0{,}1\lambda_{\max}$ | 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 | có |
| prostate | $0{,}3\lambda_{\max}$ | 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5 | có |
| $n=60$, $p=200$, tương quan 0,9 | $0{,}1\lambda_{\max}$ | 0, 56, 88, 109, 118, 126, 136, 147, 150, 155, 155, 156, 160, 163, 166, 168, 168, 169, 169, 170, 171 | có |
| như trên | $0{,}3\lambda_{\max}$ | 0, 76, 104, 125, 142, 145, 156, 161, 167, 168, 174, 177, 179, 181, 184, 187, 188, 189, 190, 192, 193 | có |

**Điều khung KHÔNG nói.** "Increases monotonically" không kèm chứng minh; bốn phép thử trên không bác được nó, nhưng không
chứng minh nó.

### 3.27 Friedman và cộng sự 2010, tr.5–6 — (4)–(9): một bước hạ theo toạ độ

![Friedman 2010 tr.5 — (4)–(6)](<paper_lasso_glmnet2010_tr5_4_6.png>)

**Hình 47 — Friedman 2010, tr.5: (4), (5), (6).** Trang PDF 5.

$$\frac{\partial R_\lambda}{\partial\beta_j}\Big|_{\beta=\tilde\beta}=-\frac1N\sum_{i=1}^Nx_{ij}(y_i-\tilde\beta_0-x_i^\top\tilde\beta)+\lambda(1-\alpha)\beta_j+\lambda\alpha \tag{4}$$

$$\tilde\beta_j\leftarrow\frac{S\bigl(\frac1N\sum_{i=1}^Nx_{ij}(y_i-\tilde y_i^{(j)}),\ \lambda\alpha\bigr)}{1+\lambda(1-\alpha)} \tag{5}$$

$$S(z,\gamma)=\operatorname{sign}(z)(\lvert z\rvert-\gamma)_+=\begin{cases}z-\gamma&z>0,\ \gamma<\lvert z\rvert\\ z+\gamma&z<0,\ \gamma<\lvert z\rvert\\ 0&\gamma\ge\lvert z\rvert\end{cases} \tag{6}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\tilde\beta$ | ước lượng hiện tại | $\mathbb{R}^p$ |
| $\tilde y_i^{(j)}$ | dự đoán **bỏ đi** đóng góp của biến $j$ | vô hướng |
| $S(z,\gamma)$ | ngưỡng mềm | vô hướng |

**Ý nghĩa.** (4) là đạo hàm khi $\tilde\beta_j>0$. Đặt nó bằng 0 và xử lý riêng $\tilde\beta_j=0$ cho ra (5): lấy hệ số OLS **một
biến** của biến $j$ trên phần dư riêng phần, **ngưỡng mềm** nó ở $\lambda\alpha$ (phần lasso), rồi **chia** cho
$1+\lambda(1-\alpha)$ (phần ridge). Với $\alpha=1$, một bước chính là (3) của bài 1996 áp cho một biến.

**Tác giả nói gì.** *"Thus we compute the simple least-squares coefficient on the partial residual, apply
soft-thresholding to take care of the lasso contribution to the penalty, and then apply a proportional shrinkage for the
ridge penalty."*

![Friedman 2010 tr.5 — (7)](<paper_lasso_glmnet2010_tr5_7.png>)

**Hình 48 — Friedman 2010, tr.5: §2.1 Naive updates, (7).** Trang PDF 5.

$$y_i-\tilde y_i^{(j)}=y_i-\hat y_i+x_{ij}\tilde\beta_j=r_i+x_{ij}\tilde\beta_j \tag{7}$$

**Ý nghĩa.** Phần dư riêng phần không cần tính lại từ đầu: nó là phần dư hiện tại $r_i$ cộng lại đóng góp của chính biến
$j$.

![Friedman 2010 tr.6 — (8), (9)](<paper_lasso_glmnet2010_tr6_8_9.png>)

**Hình 49 — Friedman 2010, tr.6: (8) và §2.2 Covariance updates, (9).** Trang PDF 6.

$$\frac1N\sum_{i=1}^Nx_{ij}(y_i-\tilde y_i^{(j)})=\frac1N\sum_{i=1}^Nx_{ij}r_i+\tilde\beta_j \tag{8}$$

$$\sum_{i=1}^Nx_{ij}r_i=\langle x_j,y\rangle-\sum_{k:\lvert\tilde\beta_k\rvert>0}\langle x_j,x_k\rangle\tilde\beta_k \tag{9}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $r_i$ | phần dư hiện tại $y_i-\hat y_i$ | vô hướng |
| $\langle x_j,y\rangle$ | $\sum_ix_{ij}y_i$ | vô hướng |

**Ý nghĩa.** (8) dùng chuẩn hoá $\tfrac1N\sum x_{ij}^2=1$ để gộp. (9) là cách tính **khác** cho cùng một số: thay vì cập nhật
vector phần dư dài $N$, giữ sẵn các tích vô hướng $\langle x_j,x_k\rangle$ và chỉ cộng trên các biến khác 0. Với $m$ biến khác
0, một vòng tốn $O(pm)$ thay vì $O(pN)$.

**Tác giả nói gì.** *"Such a step costs O(N) operations— the sum to compute the gradient."*; *"Thus a complete cycle
through all p variables costs O(pN) operations."*; và *"Hence with m non-zero terms in the model, a complete cycle costs
O(pm) operations if no new variables become non-zero, and costs O(Np) for each new variable entered."*

**Kiểm lại** (`v_lasso.py` mục 19, prostate bản có lỗi, chuẩn hoá chia $N$):

| $\lambda/\lambda_{\max}$ | naive (8) so với hiệp phương sai (9) | số lượt quét | so với LARS | $\max_j\lvert x_j^\top r\rvert/N$ |
|---:|---:|---|---:|---:|
| 0,5 | $2{,}2\cdot10^{-16}$ | 21 = 21 | $1{,}8\cdot10^{-14}$ | 0,421714 |
| 0,1 | $4{,}4\cdot10^{-16}$ | 27 = 27 | $1{,}6\cdot10^{-14}$ | 0,084343 |
| 0,01 | $1{,}0\cdot10^{-15}$ | 48 = 48 | $8{,}0\cdot10^{-14}$ | 0,008434 |

Hai cách cập nhật cho **cùng dãy lặp**; nghiệm trùng LARS; và cột cuối bằng đúng $\lambda$ (vì $\lambda_{\max}=0{,}843427$) —
điều kiện tối ưu của biến đang hoạt động.

### 3.28 Friedman và cộng sự 2010, tr.7 — (10), §2.5 đường nghiệm, §2.6 chi tiết

![Friedman 2010 tr.7 — (10), §2.5, §2.6](<paper_lasso_glmnet2010_tr7_10_2_5_2_6.png>)

**Hình 50 — Friedman 2010, tr.7: §2.4 (10), §2.5, §2.6.** Trang PDF 7.

$$\tilde\beta_j\leftarrow\frac{S\bigl(\sum_{i=1}^Nw_ix_{ij}(y_i-\tilde y_i^{(j)}),\ \lambda\alpha\bigr)}{\sum_{i=1}^Nw_ix_{ij}^2+\lambda(1-\alpha)} \tag{10}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $w_i$ | trọng số của quan sát $i$ (thay cho $1/N$) | $w_i\ge0$ |

**Ý nghĩa.** Khi quan sát có trọng số — như bên trong vòng IRLS của logistic (§8 của bài 1996) — mẫu số không còn là 1 nữa
mà là tổng bình phương có trọng số của cột. Đó cũng là dạng dùng khi cột **không** chuẩn hoá.

**Tác giả nói gì.** §2.5: *"We compute the solutions for a decreasing sequence of values for λ, starting at the smallest
value λmax for which the entire vector β̂ = 0. Apart from giving us a path of solutions, this scheme exploits warm starts,
and leads to a more stable algorithm. We have examples where it is faster to compute the path down to λ (for small λ) than
the solution only at that value for λ."* Và *"Hence Nαλmax = max_ℓ |⟨xℓ, y⟩|. Our strategy is to select a minimum value
λmin = ελmax, and construct a sequence of K values of λ decreasing from λmax to λmin on the log scale. Typical values are
ε = 0.001 and K = 100."* §2.6: *"Since the intercept is not regularized, this means that β̂0 = ȳ, the mean of the yi, for
all values of α and λ."*

(Tính nghiệm cho một dãy $\lambda$ giảm dần, bắt đầu từ $\lambda_{\max}$ — giá trị nhỏ nhất làm mọi hệ số bằng 0. Ngoài việc
cho cả đường, cách này tận dụng khởi động ấm và ổn định hơn. Có ví dụ mà tính cả đường xuống tới $\lambda$ nhanh hơn tính
riêng tại $\lambda$ đó. $N\alpha\lambda_{\max}=\max\lvert\langle x_\ell,y\rangle\rvert$; dãy $K=100$ giá trị theo thang log từ
$\lambda_{\max}$ xuống $0{,}001\lambda_{\max}$. Hệ số chặn không bị phạt nên luôn bằng $\bar y$.)

**Kiểm lại** (`v_lasso.py` mục 19):

- $\lambda_{\max}=0{,}843427$ trên prostate. Tại $\lambda_{\max}$ có **0** hệ số khác 0; tại $0{,}999\lambda_{\max}$ có **1**.
- **Khởi động ấm.** Prostate, 100 giá trị $\lambda$: tổng số lần cập nhật toạ độ với khởi động ấm 23 680, khởi động lạnh
  28 256; riêng giá trị nhỏ nhất, khởi động lạnh: 384. $n=60$, $p=200$, tương quan 0,9, 100 giá trị xuống $0{,}01\lambda_{\max}$:
  cả đường với khởi động ấm 28 676 400 lần cập nhật, riêng giá trị nhỏ nhất khởi động lạnh 8 580 400.

**Cách đọc.** Khởi động ấm rẻ hơn khởi động lạnh **cho cả đường** (23 680 so với 28 256). Nhưng ở cả hai ví dụ, tính **một**
điểm từ đầu vẫn rẻ hơn tính cả đường tới điểm đó. Câu *"We have examples where …"* là câu **tồn tại**; hai ví dụ ở đây không
phải những ví dụ đó, và không bác được câu ấy.

**Điều khung KHÔNG nói.** "Faster" trong bài báo có thể đo bằng thời gian, còn ở đây đo bằng số lần cập nhật — hai thứ khác
nhau khi có tập hoạt động (§2.6 tr.7: *"we iterate on only the active set till convergence"*), mà bản cài đặt ở đây không
làm.

### 3.29 Hai hình tự vẽ: bốn phép dựng lại của bài 1996 và hai phép đọc vectơ của ESL

![Bốn phép dựng lại Tibshirani 1996](<lasso_tib1996_dung_lai.png>)

**Hình 51 — Dựng lại Hình 1, Hình 4, Hình 5 và Bảng 3 của Tibshirani 1996 (hình tự vẽ).** Mọi mảng do
[`tools/v_lasso.py`](<../../../tools/v_lasso.py>) in ra (mục 3, 4, 9, 20); khối mã dưới chỉ vẽ.

Hình này bù cho bốn chỗ bản quét không cho đọc được:

- **(a)** Hình 1 vẽ từ công thức, cùng ngưỡng 2 cho ba panel có ngưỡng. Đường lasso song song đường 45°; garotte cong dần
  về đường 45°; ridge là một đường thẳng qua gốc với độ dốc tuỳ mức phạt.
- **(b)** Hình 4 với đường lasso tính bằng (5) **có miền đúng** — mọi $t$, cả năm $\rho$ trùng nhau. Dải đỏ mờ là thứ (6) cho
  ra nếu áp nguyên văn khi $t<3$: nó nằm trên trục ngang nhưng lệch sang phải so với đường đúng. Đường ridge với
  $\rho=0{,}90$ vượt vạch $\beta_2=3$ rõ nhất.
- **(c)** Hình 5 trên **hai** bản dữ liệu: nét liền là bản có lỗi (khớp Bảng 1), nét đứt là bản đã sửa. Khác biệt lớn nhất
  là đường lweight (xanh dương): bản đã sửa rời 0 sớm hơn và lên cao hơn — lỗi dữ liệu nằm đúng ở biến đó.
- **(d)** Bảng 3: kim cương đen là số in, chấm xanh là dựng lại theo văn bản ($\sigma=3$), vuông cam là dựng lại với
  $\sigma=2$. Hàng LS đặt vuông cam gần như chồng lên kim cương, còn chấm xanh ở tận bên phải.

<details>
<summary><b>Code sinh Hình 51</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
MAU = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


def nhan_phai(a, ys, x, gap):
    """Ghi tên biến ở mép phải, đẩy các nhãn gần nhau ra cho khỏi chồng."""
    thu_tu = np.argsort(ys)
    vi_tri = np.array(ys, dtype=float)[thu_tu]
    for i in range(1, len(vi_tri)):
        vi_tri[i] = max(vi_tri[i], vi_tri[i - 1] + gap)
    for j, v in zip(thu_tu, vi_tri):
        a.annotate(TEN[j], (x, ys[j]), xytext=(x + 0.03, v), textcoords="data", fontsize=7,
                   color=MAU[j], va="center")

# Mọi mảng dưới đây do tools/v_lasso.py in ra (mục 4, 9 và 20); khối này chỉ vẽ.
S_LOI = np.array([0, 0.2283801674, 0.3158224504, 0.4760973639, 0.4844883101, 0.6134985707, 0.7065472527, 0.7459998282, 1])
B_LOI = np.array([[0, 0.4211311879, 0.5017526205, 0.5753305642, 0.5782094339, 0.6082966537, 0.6239226518, 0.6286205146, 0.6918722977], [0, 0, 0, 0.1252127624, 0.1279517184, 0.1716733829, 0.1947184486, 0.2049275644, 0.2257160472], [0, 0, 0, 0, 0, 0, -0.0581457138, -0.0823370198, -0.1462060263], [0, 0, 0, 0, 0.0051942117, 0.0739202551, 0.1094600263, 0.1236325501, 0.1553065124], [0, 0, 0.0806214325, 0.177376375, 0.1820371748, 0.2348739212, 0.2486293464, 0.255055723, 0.3171829304], [0, 0, 0, 0, 0, 0, 0, 0, -0.14747849], [0, 0, 0, 0, 0, 0, 0, 0.0090542714, 0.0325973634], [0, 0, 0, 0, 0, 0.0425222065, 0.0679912522, 0.0719900271, 0.1276322725]])
S_SUA = np.array([0, 0.2270228333, 0.2706411284, 0.4917470828, 0.5123525536, 0.5996824157, 0.6932909827, 0.7472112829, 1])
B_SUA = np.array([[0, 0.4211311879, 0.4615875248, 0.5574525141, 0.5624760014, 0.5828329231, 0.5972691681, 0.6029232795, 0.6651466713], [0, 0, 0, 0.1718051892, 0.1860763342, 0.2100962793, 0.2326862311, 0.2462930422, 0.2664802617], [0, 0, 0, 0, 0, 0, -0.0598626448, -0.0937414612, -0.1581952246], [0, 0, 0, 0, 0, 0.0511503136, 0.088065044, 0.1081242209, 0.1403111691], [0, 0, 0.0404563369, 0.1829414359, 0.1912867675, 0.2292324519, 0.2435999107, 0.2526922409, 0.3153288818], [0, 0, 0, 0, 0, 0, 0, 0, -0.1482856849], [0, 0, 0, 0, 0, 0, 0, 0.0123116155, 0.035549169], [0, 0, 0, 0, 0.0105835323, 0.0391090357, 0.0645834769, 0.0700036823, 0.1257198175]])
RIDGE = {
    0.00: np.array([[6, 3], [5.99942485, 2.99975925], [5.99918039, 2.99965692], [5.99883203, 2.9995111], [5.99833567, 2.99930332], [5.99762844, 2.99900725], [5.99662085, 2.99858542], [5.99518553, 2.99798446], [5.99314125, 2.99712843], [5.99023037, 2.9959093], [5.98608698, 2.99417354], [5.98019222, 2.9917032], [5.97181179, 2.98818941], [5.95990967, 2.98319542], [5.94303038, 2.97610583], [5.91914163, 2.96605765], [5.88543047, 2.95184906], [5.83805228, 2.93182306], [5.77184724, 2.90372788], [5.6800717, 2.86456737], [5.55425045, 2.81047714], [5.38434477, 2.73670092], [5.1595329, 2.63779725], [4.86994195, 2.50825871], [4.50950392, 2.34371629], [4.07955829, 2.14272149], [3.59192247, 1.90866968], [3.06946717, 1.65089719], [2.54275136, 1.38387421], [2.04333211, 1.12423663], [1.59660947, 0.8868098], [1.2174085, 0.68151788], [0.90959093, 0.51238968], [0.66867011, 0.37849203], [0.48544301, 0.27577437], [0.34911614, 0.1988583], [0.2493321, 0.14229645], [0.17716824, 0.10125296], [0.12543204, 0.07175684], [0.08857233, 0.05070614], [0.06242838, 0.0357571]]),
    0.23: np.array([[6, 3], [5.99943816, 2.99988302], [5.99919936, 2.99983329], [5.99885907, 2.99976241], [5.99837421, 2.9996614], [5.99768337, 2.99951742], [5.99669917, 2.9993122], [5.99529721, 2.99901966], [5.99330052, 2.9986026], [5.9904576, 2.99800793], [5.98641132, 2.99715982], [5.98065546, 2.99594989], [5.97247401, 2.99422301], [5.96085753, 2.99175685], [5.94438934, 2.98823208], [5.9210943, 2.98318894], [5.88824415, 2.97596356], [5.84212048, 2.96559486], [5.77775157, 2.95068944], [5.68867121, 2.92923042], [5.56680155, 2.89832243], [5.4026385, 2.85389137], [5.18599549, 2.79042798], [4.90757368, 2.70099646], [4.56146135, 2.57789405], [4.14822205, 2.41437914], [3.67757807, 2.20749158], [3.16921263, 1.96104495], [2.65048729, 1.6869626], [2.15118307, 1.40343602], [1.69711978, 1.13027813], [1.30532791, 0.88381557], [0.98251136, 0.6737957], [0.72661657, 0.50312939], [0.52998459, 0.36959873], [0.38250384, 0.26813659], [0.27389725, 0.19272221], [0.19499822, 0.13757388], [0.13824721, 0.09772093], [0.09771872, 0.06916622], [0.06892379, 0.04883135]]),
    0.45: np.array([[6, 3], [5.99938721, 3.00002428], [5.99912676, 3.00003458], [5.99875566, 3.00004922], [5.9982269, 3.00007002], [5.99747359, 3.00009954], [5.9964005, 3.00014136], [5.99487216, 3.00020044], [5.99269598, 3.0002836], [5.98959848, 3.00040001], [5.9851919, 3.00056166], [5.97892756, 3.00078345], [5.97003154, 3.00108226], [5.95741683, 3.00147345], [5.93956615, 3.00196182], [5.91437996, 3.0025209], [5.87898889, 3.00304932], [5.82953964, 3.00328415], [5.76098448, 3.00263878], [5.66693915, 2.99991922], [5.53972214, 2.99287171], [5.37073329, 2.97755303], [5.15133437, 2.94764272], [4.87430597, 2.89406046], [4.53575889, 2.80553146], [4.13714788, 2.67076385], [3.68688054, 2.48227505], [3.20092811, 2.24063088], [2.70176776, 1.95678979], [2.21519913, 1.65062455], [1.76551933, 1.34579309], [1.37079643, 1.06344588], [1.04024567, 0.81777788], [0.77448945, 0.6148517], [0.56789552, 0.45407034], [0.41152254, 0.33073665], [0.29556484, 0.23841803], [0.21088907, 0.17055977], [0.1497523, 0.12133915], [0.10597235, 0.08597824], [0.07480638, 0.06074849]]),
    0.68: np.array([[6, 3], [5.99917991, 3.00030062], [5.99883142, 3.00042829], [5.99833493, 3.00061012], [5.99762766, 3.000869], [5.99662031, 3.00123743], [5.9951859, 3.00176146], [5.99314409, 3.00250622], [5.9902391, 3.00356342], [5.9861089, 3.00506164], [5.98024255, 3.00717981], [5.97192191, 3.0101643], [5.9601435, 3.01434909], [5.94351668, 3.02017638], [5.92013675, 3.02821072], [5.88743778, 3.03913122], [5.84204263, 3.05367235], [5.77964922, 3.07246239], [5.69502068, 3.09568517], [5.58217098, 3.12248307], [5.43482652, 3.15006604], [5.24715701, 3.17264816], [5.0146056, 3.18061203], [4.73452165, 3.16058107], [4.40643082, 3.09707654], [4.0322196, 2.97591903], [3.6168437, 2.78864805], [3.16975351, 2.53645371], [2.70611083, 2.2318581], [2.2461584, 1.89690404], [1.81192387, 1.55803389], [1.42240816, 1.23963861], [1.08967684, 0.9590529], [0.81759803, 0.72473255], [0.60316759, 0.53739621], [0.43911878, 0.39266294], [0.31648046, 0.28373094], [0.22638738, 0.20333184], [0.16105383, 0.1448385], [0.11412052, 0.10272378], [0.08063409, 0.07262789]]),
    0.90: np.array([[6, 3], [5.99793387, 3.00159696], [5.99705692, 3.00227449], [5.99580857, 3.00323865], [5.99403238, 3.00460987], [5.99150686, 3.00655832], [5.98791936, 3.00932358], [5.98283022, 3.0132412], [5.9756248, 3.01877761], [5.96545078, 3.02657419], [5.95113994, 3.0374991], [5.93111765, 3.0527005], [5.90331167, 3.07364514], [5.86508748, 3.10210932], [5.81326003, 3.14006436], [5.74425588, 3.18937291], [5.65450418, 3.25120562], [5.54107849, 3.32514411], [5.40245321, 3.40809734], [5.23900773, 3.49340186], [5.05278532, 3.57063062], [4.84625414, 3.62645408], [4.6204489, 3.6463668], [4.37346467, 3.61664586], [4.10030361, 3.52602534], [3.79451654, 3.36713933], [3.45134102, 3.13813929], [3.07137414, 2.84453499], [2.66327279, 2.50042586], [2.24385608, 2.12770503], [1.83488325, 1.75242731], [1.45762404, 1.3992861], [1.12777921, 1.08660441], [0.85296382, 0.82395253], [0.63318217, 0.61276594], [0.46315066, 0.44879537], [0.33497412, 0.32488656], [0.24023269, 0.2331471], [0.17122133, 0.16624584], [0.12148682, 0.11799378], [0.08592044, 0.08346851]]),
}
# (phương pháp, trung vị ME in ở Bảng 3, dựng lại với σ = 3, dựng lại với σ = 2)
BANG3 = [
    ('LS', 2.79, 6.336191, 2.816085),
    ('Lasso (CV)', 2.43, 4.023847, 2.026167),
    ('Lasso (Stein)', 2.07, 5.16904, 2.475672),
    ('Lasso (GCV)', 1.93, 3.405607, 1.673076),
    ('Garotte', 2.29, 4.776572, 2.217645),
    ('Best subset', 2.44, 6.160472, 2.165651),
    ('Ridge', 3.21, 3.934005, 2.273567),
]

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.8))

# (a) Hình 1 tr.270: bốn phép co trong thiết kế trực chuẩn
b = np.linspace(0, 5, 501)
a = ax[0, 0]
a.plot(b, b, ":", color="#9ca3af", lw=1, label="đường 45°")
a.plot(b, np.where(b > 2, b, 0), color="#111827", lw=1.3, label="tập con, ngưỡng 2")
a.plot(b, b / 3, color="#2563eb", lw=1.3, label="ridge, γ = 2 (hình gốc không ghi γ)")
a.plot(b, np.maximum(b - 2, 0), color="#dc2626", lw=1.8, label="lasso (3), γ = 2")
a.plot(b, np.maximum(1 - 4 / np.maximum(b, 1e-9) ** 2, 0) * b, color="#059669", lw=1.3, label="garotte, γ = 4")
a.set_xlabel(r"$\hat\beta^\circ_j$ (bình phương tối thiểu)")
a.set_ylabel(r"$\hat\beta_j$")
a.set_title("(a) Hình 1 tr.270 dựng lại từ công thức", fontsize=9)
a.legend(fontsize=7.5, loc="upper left")

# (b) Hình 4 tr.272: β̂° = (6, 3); lasso theo (5) đã sửa miền, ridge theo năm ρ
a = ax[0, 1]
t = np.linspace(0, 9, 91)
lasso = np.c_[np.where(t <= 3, t, (t + 3) / 2), np.where(t <= 3, 0, (t - 3) / 2)]
for (rho, B), m in zip(RIDGE.items(), ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]):
    a.plot(B[:, 0], B[:, 1], "--", color=m, lw=1.2, label="ridge, ρ = %.2f" % rho)
a.plot(lasso[:, 0], lasso[:, 1], color="#111827", lw=2.2, label="lasso, cả năm ρ (trùng nhau)")
tt = np.linspace(0, 2.9, 30)
a.plot((tt + 3) / 2, np.zeros_like(tt), color="#dc2626", lw=5, alpha=.35,
       label="(6) in nguyên văn khi t < 3: β̂₁ = (t+3)/2 > t")
a.axhline(3, color="#9ca3af", lw=.8, ls=":")
a.set_xlabel(r"$\hat\beta_1$")
a.set_ylabel(r"$\hat\beta_2$")
a.set_title("(b) Hình 4 tr.272: đường lasso không phụ thuộc ρ; ridge thì có", fontsize=9)
a.legend(fontsize=7, loc="upper left")

# (c) Hình 5 tr.273: prostate, dữ liệu CÓ LỖI (liền) và ĐÃ SỬA (đứt)
a = ax[1, 0]
for j in range(8):
    a.plot(S_LOI, B_LOI[j], color=MAU[j], lw=1.5)
    a.plot(S_SUA, B_SUA[j], color=MAU[j], lw=1.0, ls="--")
nhan_phai(a, B_LOI[:, -1], 1.0, 0.035)
a.axvline(0.44, color="#111827", ls=":", lw=1)
a.text(0.45, 0.62, "s = 0,44", fontsize=7.5)
a.axhline(0, color="#111827", lw=.6)
a.set_xlim(0, 1.18)
a.set_xlabel(r"$s=t/\sum_j|\hat\beta^\circ_j|$")
a.set_ylabel("hệ số (biến chuẩn hoá)")
a.set_title("(c) Hình 5 tr.273: liền = lweight đối tượng 32 bằng ln 449 (khớp Bảng 1),\n"
            "đứt = bản đã sửa ln 44,9", fontsize=9)

# (d) Bảng 3 tr.279: trung vị ME của ví dụ 1
a = ax[1, 1]
y = np.arange(len(BANG3))[::-1]
a.scatter([r[1] for r in BANG3], y, marker="D", s=46, color="#111827", label="Bảng 3 in", zorder=3)
a.scatter([r[2] for r in BANG3], y + 0.15, marker="o", s=34, color="#2563eb", label="dựng lại, σ = 3 (như văn bản)")
a.scatter([r[3] for r in BANG3], y - 0.15, marker="s", s=34, color="#d97706", label="dựng lại, σ = 2")
a.set_yticks(y)
a.set_yticklabels([r[0] for r in BANG3])
a.set_xlabel("trung vị sai số ME trên 200 bộ dữ liệu")
a.set_title("(d) Bảng 3 tr.279 khớp σ = 2, không khớp σ = 3", fontsize=9)
a.legend(fontsize=7.5, loc="lower right")

fig.tight_layout()
fig.savefig("lasso_tib1996_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("đã ghi lasso_tib1996_dung_lai.png")
```

</details>

![Hai phép đọc vectơ ESL](<lasso_esl_dung_lai.png>)

**Hình 52 — Hình 3.10 và ô Lasso của Hình 3.7 trong ESL, đọc thẳng từ lệnh vẽ của PDF (hình tự vẽ).** Mảng do
`tools/v_lasso.py` in ra (mục 15, 16).

- **(a)** Vòng tròn là **các điểm gãy của tám đường xanh** trong PDF; đường liền là đường lasso dựng lại trên dữ liệu chuẩn
  hoá theo 67 mẫu huấn luyện. Mọi vòng tròn nằm trên đường. Vạch đỏ liền là vạch đỏ **trong** Hình 3.10 ($s=0{,}3750$); vạch
  đỏ đứt là $s=0{,}36$ mà chú thích viết. Giữa hai vạch, đường lbph (cam) rời trục ngang — đó là chỗ số biến khác 0 đổi từ
  ba sang bốn.
- **(b)** Chín điểm CV của ô Lasso, vạch tím ngang (cực tiểu cộng một sai số chuẩn) và vạch tím đứng (lựa chọn). Điểm đầu
  tiên dưới vạch ngang là điểm thứ tư của lưới, $s=3/8$.

<details>
<summary><b>Code sinh Hình 52</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TEN = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
MAU = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


def nhan_phai(a, ys, x, gap):
    """Ghi tên biến ở mép phải, đẩy các nhãn gần nhau ra cho khỏi chồng."""
    thu_tu = np.argsort(ys)
    vi_tri = np.array(ys, dtype=float)[thu_tu]
    for i in range(1, len(vi_tri)):
        vi_tri[i] = max(vi_tri[i], vi_tri[i - 1] + gap)
    for j, v in zip(thu_tu, vi_tri):
        a.annotate(TEN[j], (x, ys[j]), xytext=(x + 0.03, v), textcoords="data", fontsize=7,
                   color=MAU[j], va="center")

# Toạ độ vectơ đọc thẳng từ PDF ESL (tools/v_lasso.py mục 15–16) và đường dựng lại (mục 15, mục 20).
NUT = np.array([1.3e-05, 0.189335, 0.254405, 0.372386, 0.376482, 0.557056, 0.599226, 0.943313, 1.000009])
DUONG_IN = {
    'lcavol': np.array([[1.3e-05, 4e-06], [0.189335, 0.427944], [0.254401, 0.5015], [0.372386, 0.561007], [0.376482, 0.562185], [0.557051, 0.579689], [0.599222, 0.58643], [0.943305, 0.699384], [1.000005, 0.716397]]),
    'lweight': np.array([[0.189335, 4e-06], [0.254401, 0.073541], [0.372386, 0.187769], [0.376482, 0.188985], [0.557051, 0.245624], [0.599222, 0.257222], [0.943305, 0.291], [1.000005, 0.292636]]),
    'age': np.array([[0.557056, 4e-06], [0.599222, -0.032085], [0.943309, -0.133741], [1.000009, -0.142553]]),
    'lbph': np.array([[0.372386, 4e-06], [0.376482, 0.003573], [0.557051, 0.143491], [0.599222, 0.163862], [0.943309, 0.206226], [1.000005, 0.211995]]),
    'svi': np.array([[0.254405, 4e-06], [0.372386, 0.092948], [0.376482, 0.09625], [0.557051, 0.200301], [0.599222, 0.20822], [0.943309, 0.300307], [1.000005, 0.30961]]),
    'lcp': np.array([[0.599226, 4e-06], [0.943313, -0.256456], [1.000009, -0.289002]]),
    'gleason': np.array([[0.943313, 4e-06], [1.000009, -0.020921]]),
    'pgg45': np.array([[0.376482, 4e-06], [0.557051, 0.090081], [0.599222, 0.106655], [0.943309, 0.245204], [1.000005, 0.27735]]),
}
DO = 0.375001
S67 = np.array([0, 0.1893167014, 0.2543884563, 0.3723704365, 0.3764680298, 0.5570445635, 0.5992041056, 0.9433024977, 1])
B67 = np.array([[0, 0.4279487498, 0.5014958165, 0.561008306, 0.5621810998, 0.579691685, 0.5864363211, 0.6993856607, 0.7164070125], [0, 0, 0.0735470667, 0.187775267, 0.1889809686, 0.2456216279, 0.2572313691, 0.2909974619, 0.2926424008], [0, 0, 0, 0, 0, 0, -0.0320869097, -0.13374448, -0.142549626], [0, 0, 0, 0, 0.0035792824, 0.1435012811, 0.1638641747, 0.2062244718, 0.2120076045], [0, 0, 0, 0.0929565461, 0.096261342, 0.2002992337, 0.2082280615, 0.3003101264, 0.3096195331], [0, 0, 0, 0, 0, 0, 0, -0.2564584594, -0.2890056157], [0, 0, 0, 0, 0, 0, 0, 0, -0.0209135198], [0, 0, 0, 0, 0, 0.0900805423, 0.1066488106, 0.2452063958, 0.2773459525]])
CAM_S = np.array([-0.0006, 0.124, 0.2497, 0.3743, 0.5, 0.6246, 0.7491, 0.8748, 0.9994])
CAM_CV = np.array([1.4687, 1.041, 0.7915, 0.6509, 0.5836, 0.5955, 0.6172, 0.6192, 0.6172])
TIM_NGANG, TIM_DUNG = 0.6753, 0.3749

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
for j, ten in enumerate(TEN):
    a.plot(S67, B67[j], color=MAU[j], lw=1.3)
    P = DUONG_IN[ten]
    a.plot(P[:, 0], P[:, 1], "o", ms=4, mfc="none", mec=MAU[j])
nhan_phai(a, B67[:, -1], 1.0, 0.045)
for x in NUT:
    a.axvline(x, color="#d1d5db", lw=.7, zorder=0)
a.axvline(DO, color="#dc2626", lw=1.4, label="vạch đỏ trong Hình 3.10: s = %.4f" % DO)
a.axvline(0.36, color="#dc2626", lw=1, ls="--", label="s = 0,36 theo chú thích và tr.69")
a.axhline(0, color="#111827", lw=.6)
a.set_xlim(0, 1.16)
a.set_xlabel("s")
a.set_ylabel("hệ số")
a.set_title("(a) Hình 3.10 tr.70: vòng tròn = điểm gãy đọc từ PDF, đường = dựng lại\n"
            "(chuẩn hoá trên 67 mẫu huấn luyện)", fontsize=9)
a.legend(fontsize=7.5, loc="upper left")
a = ax[1]
a.plot(CAM_S, CAM_CV, "o-", color="#e69f00", lw=1.2, ms=5)
a.axhline(TIM_NGANG, color="#a020f0", lw=1, label="cực tiểu + 1 sai số chuẩn = %.4f" % TIM_NGANG)
a.axvline(TIM_DUNG, color="#a020f0", lw=1, ls="--", label="lựa chọn trong hình: s = %.4f" % TIM_DUNG)
k = int(np.argmin(CAM_CV))
a.plot(CAM_S[k], CAM_CV[k], "*", ms=12, color="#111827", label="cực tiểu: s = %.4f" % CAM_S[k])
a.set_xlabel("s")
a.set_ylabel("sai số kiểm định chéo 10 mảnh")
a.set_title("(b) Hình 3.7 tr.62, ô Lasso — lưới 9 điểm, bước 1/8", fontsize=9)
a.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("lasso_esl_dung_lai.png", dpi=130, bbox_inches="tight", facecolor="white")
print("đã ghi lasso_esl_dung_lai.png")
```

</details>

---

## 4. Công thức hiện đại — một bảng ký hiệu cho cả năm nguồn

### 4.1 Hai cách viết, một đường nghiệm

$$\text{(ràng buộc)}\quad \min_\beta\ \lVert y-X\beta\rVert_2^2\ \ \text{với}\ \ \lVert\beta\rVert_1\le t
\qquad\Longleftrightarrow\qquad
\text{(phạt)}\quad \min_\beta\ \tfrac12\lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_1$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $X$ | các cột đã chuẩn hoá, $y$ đã trừ trung bình | $n\times p$ |
| $t$ | cận, $0\le t\le t_0$ | vô hướng |
| $\lambda$ | mức phạt, thang $\tfrac12\text{RSS}$ của ESL (3.52) | $\lambda\ge0$ |

**Ý nghĩa.** Mỗi $t$ trong $(0,t_0)$ ứng với đúng một $\lambda>0$ và ngược lại: $\lambda$ là "giá bóng" của ngân sách — một đơn
vị $t$ thêm vào làm RSS giảm bao nhiêu. Vì thế mọi nguồn vẽ đường nghiệm theo $t$ (hoặc $s=t/t_0$) hay theo $\lambda$ là vẽ
**cùng một đường**, chỉ đổi trục. Khi đọc một con số $\lambda$, phải biết thang: thang RSS của bài 1996 gấp đôi thang
$\tfrac12\text{RSS}$ (mục 3.7 đo đúng hệ số 2), thang $\tfrac1{2n}\text{RSS}$ của scikit-learn nhỏ hơn $n$ lần.

### 4.2 Điều kiện tối ưu

$$x_j^\top(y-X\hat\beta)=\lambda\operatorname{sign}(\hat\beta_j)\ \ \text{nếu}\ \hat\beta_j\ne0,\qquad \lvert x_j^\top(y-X\hat\beta)\rvert\le\lambda\ \ \text{nếu}\ \hat\beta_j=0$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $x_j^\top(y-X\hat\beta)$ | tương quan của cột $j$ với phần dư | vô hướng |

**Ý nghĩa.** Mọi biến đang dùng có **cùng** độ lớn tương quan với phần dư — đúng bằng $\lambda$ — và cùng dấu với hệ số của
nó (đó là (3.1) của Efron). Biến nào có tương quan nhỏ hơn $\lambda$ thì nằm yên ở 0: kéo nó ra khỏi 0 tốn $\lambda$ mỗi đơn
vị mà chỉ lợi được ít hơn thế. Hệ quả tức thì:
$$\lambda_{\max}=\max_j\lvert x_j^\top y\rvert$$
là mức phạt nhỏ nhất làm **mọi** hệ số bằng 0 — Friedman 2010 §2.5 viết đúng điều này theo thang của nó, và
`v_lasso.py` mục 19 kiểm: tại $\lambda_{\max}$ không có hệ số nào khác 0, tại $0{,}999\lambda_{\max}$ có một. Ở mục 3.7, ba
tương quan của ba biến khác 0 trên prostate đều bằng đúng $17{,}892329$.

### 4.3 Trường hợp trực chuẩn

$$X^\top X=I\ \Longrightarrow\ \hat\beta_j=\operatorname{sign}(x_j^\top y)\,(\lvert x_j^\top y\rvert-\lambda)^+$$

**Ý nghĩa.** Điều kiện ở 4.2 tách thành $p$ điều kiện một biến, và mỗi điều kiện giải ra ngưỡng mềm. Đây là (3) của bài
1996, (13) ở §10, dòng Lasso của Bảng 3.4 ESL, và (5)–(6) của Friedman 2010 khi $\alpha=1$.

### 4.4 Bảng đổi ký hiệu giữa năm nguồn

| Đại lượng | Bài này | Tibshirani 1996 | Tibshirani 2011 | ESL | Efron 2004 | Friedman 2010 | scikit-learn |
|---|---|---|---|---|---|---|---|
| số mẫu, số biến | $n$, $p$ | $N$, $p$ | $N$, $p$ | $N$, $p$ | $n$, $m$ | $N$, $p$ | `n_samples`, `n_features` |
| chuẩn hoá cột | theo từng nguồn | $\sum x^2/N=1$ (bảng dùng $N-1$) | chuẩn hoá | chuẩn hoá | $\sum x^2=1$ | $\sum x^2/N=1$ | `StandardScaler` |
| cận | $t$ | $t$ | $s$ | $t$ | $t$ | — | — |
| cận chuẩn hoá | $s$ | $s=t/t_0$ | — | $s=t/\sum\lvert\hat\beta_j\rvert$ | — | — | — |
| mức phạt | $\lambda$ (thang ½RSS) | $\lambda$ (thang RSS) | $\lambda$ (thang RSS) | $\lambda$ (thang ½RSS) | — | $\lambda$ (thang $\tfrac1{2N}$RSS) | `alpha` (thang $\tfrac1{2n}$RSS) |
| tỉ lệ trộn với ridge | — | — | — | $\alpha$ nhân $\beta^2$ | — | $\alpha$ nhân $\lvert\beta\rvert$ | `l1_ratio` nhân $\lvert\beta\rvert$ |
| ngưỡng mềm | $\operatorname{sign}(z)(\lvert z\rvert-\gamma)^+$ | cùng dạng, $\gamma$ | — | $(\lvert\hat\beta\rvert-\lambda)_+$ | — | $S(z,\gamma)$ | — |

---

## 5. Thuật toán từng bước

### 5.1 Lasso gồm những gì, ghép theo thứ tự nào

Ridge có một công thức và giải xong trong một dòng. Lasso không có công thức đóng (ESL tr.68: *"there is no closed form
expression as in ridge regression"*), nên bộ giải hiện đại là **ba vòng lặp lồng nhau** — cách của Friedman 2010:

```
   X (n×p) đã chuẩn hoá, y đã trừ trung bình, lưới λ giảm dần
        │
        ▼
   ┌──────────────────────────────────────────────────┐
   │  VÒNG NGOÀI — đi dọc lưới λ, từ LỚN xuống NHỎ    │
   │  λ_max = max_j |xⱼᵀy|  ⟹  β̂ = 0 hoàn toàn        │
   │                                                  │
   │   ┌────────────────────────────────────────────┐ │
   │   │  VÒNG GIỮA — lặp tới khi β̂ thôi đổi        │ │
   │   │                                            │ │
   │   │   ┌──────────────────────────────────────┐ │ │
   │   │   │  VÒNG TRONG — duyệt từng toạ độ j    │ │ │
   │   │   │                                      │ │ │
   │   │   │  1. rⱼ = y − Σ_{k≠j} xₖ β̂ₖ           │ │ │
   │   │   │     (phần dư khi TẠM BỎ cột j) — (7) │ │ │
   │   │   │  2. z  = xⱼᵀ rⱼ                       │ │ │
   │   │   │  3. β̂ⱼ ← soft(z, λ) / ‖xⱼ‖²   — (5)  │ │ │
   │   │   │     ↑ ĐÂY là chỗ số 0 sinh ra        │ │ │
   │   │   └──────────────────────────────────────┘ │ │
   │   └────────────────────────────────────────────┘ │
   │                                                  │
   │  khởi tạo λ kế tiếp bằng β̂ vừa tìm ("warm start")│
   └──────────────────────────────────────────────────┘
        │
        ▼
   đường nghiệm β̂(λ) cho MỌI λ trên lưới
```

| Thành phần | Nhận vào | Trả ra | Bỏ nó đi thì hỏng gì |
|---|---|---|---|
| **Ngưỡng mềm** `soft(z,λ)` | một số $z$ | $\operatorname{sign}(z)\max(\lvert z\rvert-\lambda,0)$ | Đây là **toàn bộ** lasso. Thay bằng phép co tỉ lệ $z/(1+\lambda)$ thì ra ridge, và không hệ số nào bằng 0 nữa. |
| **Vòng trong (từng toạ độ)** | $z,\lambda$ | một $\hat\beta_j$ | Không có nó thì phải giải bài toán $p$ chiều không trơn cùng lúc — cách của bài 1996, với tới $2^p$ ràng buộc (mục 3.12). Tách từng toạ độ biến nó thành $p$ bài toán một chiều có nghiệm đóng. |
| **Vòng giữa (lặp tới hội tụ)** | $\hat\beta$ hiện tại | $\hat\beta$ đã ổn định | Sửa toạ độ $j$ làm phần dư của mọi toạ độ khác đổi theo, nên một lượt là chưa đủ: trên prostate cần 21 tới 48 lượt (mục 3.27). Dừng sớm sinh ra "số 0 giả" (mục 6.4). |
| **Vòng ngoài (lưới $\lambda$)** | lưới $\lambda$ | đường nghiệm | Không có nó thì chỉ có **một** mô hình ứng với **một** $\lambda$ đoán bừa. Kiểm định chéo cần cả đường. |
| **Khởi động ấm** | $\hat\beta$ của $\lambda$ trước | điểm khởi tạo | Chỉ ảnh hưởng tốc độ, không ảnh hưởng nghiệm: trên prostate cả đường tốn 23 680 lần cập nhật với khởi động ấm, 28 256 lần không có (mục 3.28). |

Điểm đáng nhớ: **chỉ một dòng trong toàn bộ sơ đồ khác với ridge** — dòng `soft(z, λ)`.

### 5.2 Mã tự chứa: hạ theo toạ độ và đường giá đỡ

Dữ liệu: $n=80$, $p=12$, chỉ 3 hệ số thật khác 0 (ở cột 0, 3, 7). Hàm mục tiêu $\tfrac12\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_1$.

```python
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)

def lasso_cd(X, y, lam, max_iter=200000, tol=1e-12):
    b = np.zeros(X.shape[1])
    r = y.copy()
    norm2 = (X**2).sum(axis=0)
    for _ in range(max_iter):
        largest = 0.0
        for j in range(X.shape[1]):
            r += X[:, j] * b[j]
            new = soft(X[:, j] @ r, lam) / norm2[j]
            largest = max(largest, abs(new - b[j]))
            b[j] = new
            r -= X[:, j] * b[j]
        if largest < tol:
            return b
    raise RuntimeError("coordinate descent chưa hội tụ")

rng = np.random.default_rng(0)
n, p = 80, 12
X = rng.normal(size=(n, p))
X = (X - X.mean(0)) / X.std(0)
beta = np.zeros(p)
beta[[0, 3, 7]] = [3.0, -2.0, 1.5]
y = X @ beta + rng.normal(size=n)
y -= y.mean()

for lam in (0.5, 2, 8, 32, 128, 512):
    b = lasso_cd(X, y, lam)
    support = np.flatnonzero(np.abs(b) > 1e-8).tolist()
    print("λ = %-5g -> %2d biến %s"
          % (lam, len(support), support if len(support) <= 6 else ""))
```

```text
λ = 0.5   -> 11 biến
λ = 2     -> 10 biến
λ = 8     ->  6 biến [0, 1, 3, 4, 6, 7]
λ = 32    ->  3 biến [0, 3, 7]
λ = 128   ->  2 biến [0, 3]
λ = 512   ->  0 biến []
```

Ở $\lambda=32$ trong hạt giống này, lasso khôi phục **đúng** giá đỡ thật. Đây là một ca minh hoạ, không phải bảo đảm. Đọc cả
dòng $\lambda=8$: 6 biến, tức **3 dương tính giả**; và $\lambda=128$: 2 biến, tức **1 âm tính giả**. Không có cách nào biết
$\lambda=32$ là đúng nếu không biết trước đáp án.

### 5.3 Cách tính cả đường một lượt: LARS có sửa đổi lasso

Thay vì một lưới $\lambda$, LARS (mục 3.25) đi thẳng từ nút này sang nút kia của đường nghiệm — giữa hai nút mọi hệ số là
hàm tuyến tính của $t$. Bốn bước lặp: tìm biến có tương quan lớn nhất; đi theo hướng "góc đều" (3.2)–(3.3) của các biến
đang hoạt động; dừng khi một biến mới đuổi kịp **hoặc** một hệ số chạm 0 (3.4)–(3.5); thêm hoặc bỏ biến (3.6). Trong
scikit-learn là `lars_path(X, y, method="lasso")`; mọi đường nghiệm trong bài này (Hình 51, 52, 53) tính bằng hàm đó.

### 5.4 Dùng thật: pipeline scikit-learn trên dữ liệu prostate

```python
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
# tools/prostate.csv: 97 bệnh nhân, cột 'train' chia 67/30 đúng như ESL tr.50 (bản dữ liệu đã sửa lỗi)
df = pd.read_csv("prostate.csv")
ten = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
tr = df["train"].astype(str) == "True"
Xtr, ytr = df.loc[tr, ten].to_numpy(), df.loc[tr, "lpsa"].to_numpy()
Xte, yte = df.loc[~tr, ten].to_numpy(), df.loc[~tr, "lpsa"].to_numpy()

mo_hinh = make_pipeline(StandardScaler(),
                        LassoCV(cv=KFold(10, shuffle=True, random_state=0), alphas=100, max_iter=100000))
mo_hinh.fit(Xtr, ytr)
las = mo_hinh[-1]
e_las = (yte - mo_hinh.predict(Xte)) ** 2
print("alpha chọn = %.6f  (λ dạng ½RSS = n·alpha = %.4f)" % (las.alpha_, len(ytr) * las.alpha_))
print("hệ số (thang chuẩn hoá):", np.round(las.coef_, 4))
print("biến giữ lại:", [t for t, b in zip(ten, las.coef_) if b != 0])
print("hệ số chặn = %.4f" % las.intercept_)
print("sai số kiểm tra lasso = %.6f  (SE %.6f)" % (e_las.mean(), e_las.std(ddof=1) / np.sqrt(len(e_las))))

ls = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
e_ls = (yte - ls.predict(Xte)) ** 2
print("sai số kiểm tra LS    = %.6f  (SE %.6f)" % (e_ls.mean(), e_ls.std(ddof=1) / np.sqrt(len(e_ls))))
d = e_ls - e_las
print("LS − lasso, cặp trên 30 mẫu: %.6f (SE %.6f)" % (d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
```

```text
alpha chọn = 0.010105  (λ dạng ½RSS = n·alpha = 0.6770)
hệ số (thang chuẩn hoá): [ 0.6799  0.2846 -0.1198  0.1993  0.2863 -0.222  -0.      0.2258]
biến giữ lại: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'pgg45']
hệ số chặn = 2.4523
sai số kiểm tra lasso = 0.498541  (SE 0.167213)
sai số kiểm tra LS    = 0.521274  (SE 0.178724)
LS − lasso, cặp trên 30 mẫu: 0.022733 (SE 0.013767)
```

Kiểm định chéo 10 mảnh **chọn cực tiểu** giữ 7 biến và cho sai số kiểm tra 0,498541; ESL chọn bằng **quy tắc một sai số
chuẩn** giữ 4 biến và cho 0,479 (mục 3.21). Cả hai đều thấp hơn LS 0,521274, nhưng chênh lệch cặp giữa LS và lasso ở đây là
0,022733 với sai số chuẩn 0,013767 — chưa tới hai sai số chuẩn.

---

## 6. Vì sao $\ell_1$ cho hệ số bằng đúng 0 — cơ chế và phép đo

### 6.1 Ba lời giải thích, cùng một sự thật

**Hình học** (Tibshirani 1996 §2.3, ESL Hình 3.11). Miền $\sum\lvert\beta_j\rvert\le t$ có **góc nằm trên trục**; đường mức elip
của RSS chạm miền lần đầu ở một góc với xác suất dương. Miền $\sum\beta_j^2\le t$ tròn đều, không có điểm nào ưu tiên. Đây là
lời giải thích hay được vẽ nhất, nhưng nó chỉ **gợi ý**.

**Đạo hàm.** $\beta^2$ có đạo hàm $2\beta\to0$ khi $\beta\to0$: khi hệ số đã nhỏ, lực kéo về 0 cũng nhỏ theo, nên nó tiến tới 0
mà không tới nơi. $\lvert\beta\rvert$ có đạo hàm $\pm1$ **không đổi**, và tại 0 dưới vi phân là cả đoạn $[-1,1]$. Lực kéo về 0
không yếu đi. Nếu độ dốc của RSS tại 0 nhỏ hơn $\lambda$, thì 0 là điểm dừng thật.

**Điều kiện tối ưu** (mục 4.2–4.3). Chính là chứng minh: với cột trực chuẩn, nghiệm là ngưỡng mềm, và chữ $(\cdot)^+$ là chỗ
hệ số bị đặt về **đúng** 0 khi $\lvert z_j\rvert\le\lambda$. Ridge trên cùng giả định là $z_j/(1+\lambda)$ — chỉ bằng 0 khi $z_j=0$.

| $\lvert z_j\rvert$, $\lambda=1$ | Lasso | Ridge |
|---|---|---|
| 0,5 | **0** | 0,25 |
| 1,0 | **0** | 0,50 |
| 3,0 | 2,0 | 1,50 |

Hàng cuối cho thêm một khác biệt: lasso trừ một lượng cố định khỏi hệ số vượt ngưỡng, ridge co theo tỉ lệ. Hệ số lớn vì thế
bị lasso **co tuyệt đối** nhiều như hệ số vừa qua ngưỡng — nguồn của chệch mà mục 3.15 đo (ngưỡng mềm tệ hơn ngưỡng cứng
3,8468 lần khi có nhiều hệ số lớn).

### 6.2 Đo trên dữ liệu thật: đường lasso và đường ridge trên bộ diabetes

![Đường lasso và ridge trên bộ diabetes](<lasso_vs_ridge_path.png>)

**Hình 53 — Đường nghiệm lasso và ridge trên cùng bộ diabetes (hình tự vẽ).** Hình 43 là bản gốc của Efron; hình này bù
đúng chỗ bản gốc **không** có: nó vẽ ridge trên **cùng dữ liệu, cùng trục** để so trực tiếp. Bản gốc vẽ lasso và stagewise.

Hai panel dùng cùng dữ liệu (442 bệnh nhân, 10 biến, chuẩn hoá theo (1.1)) và cùng trục hoành $\sum_j\lvert\hat\beta_j\rvert$. Bên
trái nhiều đoạn bị **ghim đúng ở 0** rồi mới rời trục; bên phải ridge không có vùng nào bị ghim ở 0. Một hệ số ridge vẫn có
thể tình cờ bằng 0 tại một giá trị $\lambda$ cô lập khi đường của nó đổi dấu.

<details>
<summary><b>Code sinh Hình 53</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import sys
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Chuan hoa DUNG quy uoc Efron muc 2: cot trung binh 0 va chuan 2 bang 1, y tru trung binh.
d = load_diabetes()
Xd = d.data - d.data.mean(0)
Xd = Xd / np.sqrt((Xd ** 2).sum(0))
yd = d.target - d.target.mean()

alphas, active, coefs = lars_path(Xd, yd, method="lasso")   # các nút đường nghiệm theo số học máy

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

# --- bang trai: duong LASSO, truc hoanh t = sum|beta| giong Efron Hinh 1 ---
tt = np.abs(coefs).sum(0)
order, prev = [], set()
for k in range(coefs.shape[1]):
    cur = set(np.nonzero(coefs[:, k])[0])
    order += [int(j) + 1 for j in sorted(cur - prev)]
    prev = cur
for j in range(coefs.shape[0]):
    axes[0].plot(tt, coefs[j], lw=1.2)
    axes[0].annotate(str(j + 1), (tt[-1], coefs[j, -1]), fontsize=6.5,
                     xytext=(3, -2), textcoords="offset points")
axes[0].axvline(1000, color="#111827", ls=":", lw=.9)
axes[0].text(1020, coefs.max() * .9, "t = 1000", fontsize=7)
axes[0].axhline(0, color="#111827", lw=.6)
axes[0].set_xlabel(r"$t=\sum_j|\hat\beta_j|$"); axes[0].set_ylabel(r"$\hat\beta_j$")
axes[0].set_title("Lasso qua LARS — bộ diabetes (Efron 2004, Hình 1 trái)\n"
                  "thứ tự vào: %s" % ", ".join(map(str, order[:6])), fontsize=9)

# --- bang phai: RIDGE tren CUNG du lieu, cung truc hoanh de so sanh cong bang ---
lam3 = np.logspace(-4, 2, 150)
Bp = np.array([np.linalg.solve(Xd.T @ Xd + l * np.eye(10), Xd.T @ yd) for l in lam3])
tr_ = np.abs(Bp).sum(1)
for j in range(10):
    axes[1].plot(tr_, Bp[:, j], lw=1.2)
    axes[1].annotate(str(j + 1), (tr_[0], Bp[0, j]), fontsize=6.5,
                     xytext=(3, -2), textcoords="offset points")
axes[1].axhline(0, color="#111827", lw=.6)
axes[1].set_xlabel(r"$\sum_j|\hat\beta_j|$"); axes[1].set_ylabel(r"$\hat\beta_j$")
axes[1].set_title("Ridge trên cùng dữ liệu — không có đoạn\n"
                  "bị ghim ở 0 bởi một ngưỡng mềm", fontsize=9)

fig.savefig("lasso_vs_ridge_path.png", dpi=130, bbox_inches="tight", facecolor="white")
print("thứ tự biến vào:", order)
print("t cuối = %.2f" % tt[-1])
```

Bảng trái dùng `lars_path` để lấy các nút của đường đi. Coordinate descent cũng giải được lasso
đến dung sai yêu cầu, nhưng các hệ số rất nhỏ quanh ngưỡng cần một quy tắc tolerance rõ ràng
nếu mục tiêu là đếm biến.

</details>

### 6.3 Mô phỏng: một lần tìm đúng giá đỡ — và vì sao chưa đủ

Dữ liệu của mục 5.2, cùng $\lambda=8$, ba bộ ước lượng cùng một vòng lặp toạ độ, khác đúng một dòng
([`tools/demo_shrink.py`](<../../../tools/demo_shrink.py>)):

```
Cùng λ = 8, ba bộ cho ba nghiệm khác hẳn nhau:
                               0       1       2       3       4       5       6       7       8       9      10      11
   hệ số thật               3.00    0.00    0.00   -2.00    0.00    0.00    0.00    1.50    0.00    0.00    0.00    0.00
   Ridge   (α = 0)          2.57   -0.13    0.10   -1.69    0.12    0.07   -0.29    1.15    0.03   -0.04   -0.01    0.01   khác 0: 12/12  (21 vòng)
   Elastic Net (α=0.5)      2.66   -0.09    0.05   -1.75    0.07    0.01   -0.24    1.18    0.00   -0.00   -0.00    0.00   khác 0:  8/12  (14 vòng)
   Lasso   (α = 1)          2.75   -0.05    0.00   -1.81    0.01    0.00   -0.19    1.19    0.00   -0.00    0.00    0.00   khác 0:  6/12  (13 vòng)
```

Ba hệ số thật lớn bị co ở cả ba bộ, lasso co ít nhất (2,75 so với 3,00). Nhưng lasso vẫn giữ **ba** hệ số không có thật ở
mức phạt này. Trong ví dụ 1 của bài 1996 (mục 3.13), mô hình đúng $\{1,2,5\}$ được lasso GCV chọn với tỉ lệ **0,025** theo bản
in, 0,020 và 0,010 theo hai bản dựng lại ($\sigma=3$ và $\sigma=2$).

> **Mục tiêu dự đoán và mục tiêu chọn đúng biến khác nhau.** Kiểm định chéo tối ưu sai số dự đoán, không tối ưu xác suất
> khôi phục đúng tập biến. Bühlmann trong bài hồi tưởng 2011 tr.278 đề nghị đọc chữ "s" thứ hai của lasso là *"screening"*
> chứ không phải *"selection"*: lasso thường giữ một **tập chứa** các biến thật — ví dụ 1 dựng lại với $\sigma=2$ cho tỉ lệ
> chứa đủ 0,955 — chứ không phải đúng tập đó.

### 6.4 Hai giới hạn đo được

**Không quá $n$ biến, và "số 0 giả".** Với $n=20$, $p=60$ ([`tools/v_reg.py`](<../../../tools/v_reg.py>) mục 4):

```
   (a) n = 20, p = 60 — 'lasso chọn nhiều nhất n biến' (Zou & Hastie mục 2.1)
       LARS (nghiệm chính xác)         : nhiều nhất 20 biến  -> KHỚP
       Lasso(coordinate descent, sklearn): nhiều nhất 27 biến  -> KHÔNG khớp — xem ghi chú
       Ghi chú: ở α = 0.001, 27 hệ số 'khác 0' của coordinate descent có biên độ
       nhỏ nhất 1.77e-03 — đó là ngưỡng hội tụ chưa siết, không phải nghiệm lasso thật.
       Siết tol xuống 1e-14: 18 biến.
```

Trong vị trí tổng quát, nghiệm lasso có không quá $\operatorname{rank}(X)\le n$ hệ số khác 0. Đếm bằng `coef_ != 0` với dung sai
mặc định biến một sai số số học thành "biến được chọn". Nếu cần báo giá đỡ, phải công bố ngưỡng và siết dung sai.

**Nhóm biến tương quan.** Ba cột gần trùng nhau (tương quan 0,9999), đóng góp thật chia đều (`demo_shrink.py`):

```
      Ridge   hệ số ba cột nhóm = [ 0.9505  0.9517  0.9531]   hội tụ sau    241 vòng
      ElNet   hệ số ba cột nhóm = [ 0.9497  0.9537  0.9520]   hội tụ sau    388 vòng
      Lasso   hệ số ba cột nhóm = [ 0.0000  2.2915  0.5577]   hội tụ sau 120718 vòng
```

Ba cột gần như không phân biệt được, mà lasso **xoá hẳn cột 0** rồi chia phần còn lại không đều — đúng câu *"will tend to pick
one and ignore the rest"* của Friedman 2010 (mục 3.26). Hàm mục tiêu gần như phẳng dọc hướng "chuyển hệ số giữa các cột gần
trùng", nên nghiệm vừa tuỳ tiện vừa làm thuật toán bò chậm: 120 718 vòng so với 241 của ridge với cùng dung sai.

### 6.5 Một kết quả âm: tương quan cao chưa đủ để ridge thắng

Zou & Hastie (2005) mô tả tình huống biến tương quan nơi ridge dự báo tốt hơn lasso. Thử lần đầu, siêu tham số chọn bằng
kiểm định chéo 5 mảnh (`v_reg.py` mục 4):

```
   (c) n=120 > p=10, ρ = 0.995, 60 lần lặp, siêu tham số chọn bằng CV 5 mảnh
       Lasso       2.0746 ± 0.2033   (chênh với lasso: +0.0000 ± 0.0000)
       Ridge       2.0746 ± 0.1996   (chênh với lasso: +0.0000 ± 0.0055)
       Ridge thắng lasso ở 31/60 lần lặp.
       -> KHÔNG tái hiện được.
```

Hoà. Nguyên nhân: bộ sinh dữ liệu chỉ có 2 nhân tố ẩn, nên hệ số thật **thưa** (8/10 bằng 0) — sân nhà của lasso. Làm lại
với hệ số thật **dày**:

```
       Làm lại với hệ số thật DÀY (10/10 khác 0), ρ giảm dần theo khoảng cách 0.9^|i−j|:
       Lasso       3.2321 ± 0.3339   (chênh với lasso: +0.0000 ± 0.0000)
       Ridge       3.1479 ± 0.3239   (chênh với lasso: -0.0842 ± 0.0135)
       ElasticNet  3.1647 ± 0.3291   (chênh với lasso: -0.0674 ± 0.0119)
       Ridge thắng lasso ở 49/60 lần lặp -> khớp Zou & Hastie
```

Đúng khuôn §11 của bài 1996: **số lượng và độ lớn** của hiệu ứng quyết định, không phải riêng tương quan. Ví dụ 2 của bài
báo (tám hệ số 0,85) là trường hợp dày; ridge thắng ở đó, trong bản in lẫn bản dựng lại (mục 3.13).

---

## 7. Chọn mức phạt — năm nguồn, năm cách

| Cách | Nguồn | Tính gì | Trên prostate cho gì | Chi phí |
|---|---|---|---|---|
| kiểm định chéo 5 mảnh, lấy cực tiểu | [T] §4 | sai số dự đoán ước lượng | $s$ từ 0,5714 tới 1,0000 tuỳ cách chia (mục 3.10) | 75 lần giải với lưới 15 điểm |
| GCV (10) | [T] §4 | RSS phạt theo $p(t)$ | $s=0{,}69$; bài báo in 0,44 (mục 3.10) | 15 lần |
| Stein, $\hat t$ | [T] §4 | rủi ro không chệch, **giả định cột trực giao** | $\hat s=0{,}8586$ (mục 3.10) | 1 lần |
| kiểm định chéo 10 mảnh + **quy tắc một sai số chuẩn** | [E] tr.61 | mô hình đơn giản nhất trong một sai số chuẩn của cực tiểu | $s=0{,}375$, 4 biến, sai số kiểm tra 0,479 (mục 3.21) | 10 lần giải cho mỗi điểm của lưới 9 điểm |
| kiểm định chéo 10 mảnh, lấy cực tiểu | scikit-learn `LassoCV` | như trên, không có quy tắc một sai số chuẩn | 7 biến, sai số kiểm tra 0,498541 (mục 5.4) | 10 lần giải cho mỗi điểm của lưới 100 điểm, có khởi động ấm |

Ba điều rút ra:

1. **Cực tiểu và quy tắc một sai số chuẩn cho mô hình khác nhau** trên cùng dữ liệu: 7 biến so với 4. Quy tắc một sai số
   chuẩn cố ý chọn mô hình nhỏ hơn vì đường CV phẳng gần cực tiểu (Hình 52 panel b).
2. **Cả năm cách chọn cho mô hình lớn hơn mô hình của Bảng 1.** Không cách nào dựng lại $\hat s=0{,}44$.
3. **Không cách nào tối ưu việc chọn đúng biến** (mục 6.3).

Hai điều thực hành: đặt chuẩn hoá bên trong pipeline để kiểm định chéo không nhìn thấy dữ liệu kiểm tra; và khi báo một
mức phạt, ghi rõ thang của nó (mục 4.1).

---

## 8. Giá phải trả — nó hỏng khi nào, và khi nào đừng dùng

### 8.1 Bảng giá

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
|---|---|---|---|
| Xoá hẳn biến | ridge 0/10 hệ số bằng 0 → lasso 6/10 ở `alpha = 0.5` | **mất nghiệm đóng**: phải giải lặp | mục 1.2 |
| Thay được chọn tập con | $1{,}15\cdot10^{18}$ tổ hợp ở $p=60$ → một bài toán lồi | nghiệm là **một** tập hợp lý, không phải tập đúng: mô hình đúng được chọn 2,5 % số lần | mục 1.2, 3.13 |
| Ổn định hơn chọn tập con | chứa đủ biến thật 0,955 so với 0,595 | vẫn **chệch** ở hệ số lớn: rủi ro ngưỡng mềm gấp 3,8468 lần ngưỡng cứng khi có nhiều hệ số lớn | mục 3.13, 3.15 |
| Dự đoán tốt khi ít hiệu ứng vừa | ví dụ 1 ($\sigma=2$): lasso GCV 1,6731, thấp nhất bảng | **thua ridge** khi nhiều hiệu ứng nhỏ: ví dụ 2 ridge 2,2205 so với lasso 3,5591 | mục 3.13 |
| Chạy được khi $p>n$ | có | chọn **nhiều nhất $n$ biến**, kể cả khi sự thật cần nhiều hơn | mục 6.4 |
| Có sai số chuẩn | bootstrap cho con số | không có công thức đóng đáng tin: (7) cho sai số chuẩn **nhỏ hơn** bootstrap (lcavol 0,0536 so với 0,0895) | mục 3.7, 3.9 |
| Kết quả tính được nhanh | cả đường 23 680 lần cập nhật trên prostate | nhóm biến tương quan làm thuật toán bò: 120 718 vòng so với 241 của ridge | mục 3.28, 6.4 |

### 8.2 Khi nào đừng dùng

- **Nhiều hiệu ứng nhỏ, không có lý do tin là thưa** → ridge (mục 3.13 ví dụ 2 và 4; mục 6.5).
- **Biến đi theo nhóm và cần giữ cả nhóm** → [Elastic Net](<../Elastic Net/Elastic Net.md>) (mục 6.4).
- **Cần danh sách biến có kiểm soát sai** → lasso chỉ là bước sàng lọc; cần thêm bước chọn lại (mục 6.3).
- **Hệ số lớn cần ít chệch** → các phạt phi lồi hoặc lasso thích nghi; bài hồi tưởng 2011 tr.281 nhắc cả hai (mục 3.19).

### 8.3 So với các cách khác, trên cùng các bài toán của bài 1996

Trung vị ME dựng lại (mục 3.13; ví dụ 1 với $\sigma=2$):

| Phương pháp | VD 1: ít hiệu ứng vừa | VD 2: nhiều hiệu ứng nhỏ | VD 3: một hiệu ứng lớn | VD 4: $p=40$, nửa khác 0 |
|---|---:|---:|---:|---:|
| LS | 2,8161 | 5,7511 | 2,5428 | 143,8943 |
| Lasso (GCV) | **1,6731** | 3,5591 | 0,9090 | 62,3732 |
| Garotte | 2,2176 | 6,3369 | 0,2969 | 104,4199 |
| Chọn tập con | 2,1657 | 7,9131 | **0,1803** | — |
| Ridge | 2,2736 | **2,2205** | 2,6200 | **32,8945** |

Không phương pháp nào thắng cả bốn cột. Lasso không bao giờ tệ nhất, và đó là lời hứa thật của nó.

---

## 9. Ứng dụng thật

- **Tin sinh học.** Hàng nghìn gen, vài trăm bệnh nhân, cần một danh sách ngắn. Lasso làm bước sàng lọc; danh sách cuối cần
  kiểm tra độ ổn định (mục 6.3).
- **Khôi phục tín hiệu thưa** (*compressed sensing*, *basis pursuit*). ESL tr.68 ghi lasso còn có tên *basis pursuit* trong xử
  lý tín hiệu. Bảo đảm khôi phục chính xác cần điều kiện về ma trận đo và độ thưa, không chỉ "ít phép đo".
- **Mô hình hồi quy tổng quát.** Bài 1996 §8 đưa lasso vào vòng IRLS cho logistic; Friedman 2010 làm điều đó cho logistic
  và đa lớp với trọng số (10).
- **Đồ thị thưa.** Graphical lasso phạt $\ell_1$ ma trận nghịch đảo hiệp phương sai; một ô 0 là một cạnh bị bỏ (bài hồi tưởng
  2011 §4).
- **Mô hình tài chính.** Phạt $\ell_1$ kiểm soát tổng độ lớn vị thế. Với ràng buộc chỉ mua và tổng trọng số bằng 1,
  $\lVert w\rVert_1=1$ là hằng số, nên phạt này **không tự tạo danh mục thưa**.

---

## 10. Đối chiếu với nguồn gốc — đã lấy gì, bỏ gì, vì sao

### 10.1 Tibshirani 1996 — theo hình và bảng

Bài báo có **9 hình và 8 bảng**.

| Hình / bảng | Trang | Nhóm | Lấy? | Nhận xét của tác giả đã dẫn | Ở đâu |
|---|---|---|---|---|---|
| Fig. 1 — bốn dạng co, thiết kế trực chuẩn | tr.270 | RUỘT | **có, vẽ lại từ công thức** | tr.269, tr.270 | Hình 5, 51(a); mục 3.4 |
| Fig. 2 — bức tranh ước lượng lasso/ridge | tr.271 | RUỘT | có | tr.270 | Hình 6; mục 3.5 |
| Fig. 3 — nghiệm lasso khác bát phân với OLS | tr.271 | RUỘT | **có, đo tần suất** | tr.270 | Hình 6; mục 3.4, 3.5 |
| Fig. 4 — lasso và ridge, hai biến | tr.272 | RUỘT | **có, dựng lại** | tr.272 | Hình 8, 51(b); mục 3.6 |
| Fig. 5 — đường lasso trên prostate | tr.273 | RUỘT | **có, dựng lại trên hai bản dữ liệu** | tr.274 | Hình 11, 51(c); mục 3.8 |
| Fig. 6 — hộp ria bootstrap | tr.275 | NGOÀI | **có, dựng lại bằng số** | tr.275 | Hình 13; mục 3.9 |
| Fig. 7 — mật độ Laplace và chuẩn | tr.278 | RUỘT | có, đo khối lượng | tr.277 | Hình 16; mục 3.11 |
| Fig. 8 — hộp ria hệ số ví dụ 1 | tr.281 | NGOÀI | có, đọc bằng mắt | tr.282 | Hình 19; mục 3.13 |
| Fig. 9 — đường mức $\sum\lvert\beta_j\rvert^q$ | tr.287 | RUỘT | **có, kiểm tính lồi** | tr.286 | Hình 25; mục 3.16 |
| Table 1 — kết quả prostate | tr.274 | NGOÀI | **có, dựng lại 9/9 cả ba cột** | tr.274 | Hình 12; mục 3.8 |
| Table 2 — sai số chuẩn | tr.274 | NGOÀI | **có, dựng lại; cột (7) bỏ ngỏ** | tr.275 | Hình 12; mục 3.7, 3.9 |
| Table 3 — ví dụ 1 | tr.279 | NGOÀI | **có, dựng lại với $\sigma=3$ và $\sigma=2$** | tr.280 | Hình 17; mục 3.13 |
| Table 4 — mô hình lasso hay chọn | tr.280 | NGOÀI | **có, dựng lại** | tr.282 | Hình 18; mục 3.13 |
| Table 5 — mô hình tập con hay chọn | tr.280 | NGOÀI | **có, dựng lại** | tr.282 | Hình 18; mục 3.13 |
| Table 6 — ví dụ 2 | tr.282 | NGOÀI | **có, dựng lại** | tr.282 | Hình 20; mục 3.13 |
| Table 7 — ví dụ 3 | tr.282 | NGOÀI | **có, dựng lại** | tr.282–283 | Hình 20; mục 3.13 |
| Table 8 — ví dụ 4 | tr.283 | NGOÀI | **có, dựng lại** | tr.283 | Hình 21; mục 3.13 |

### 10.2 Tibshirani 1996 — theo mục và công thức

| Mục | Trang | Nội dung | Công thức | Lấy? | Ở đâu |
|---|---|---|---|---|---|
| Summary | tr.267 | tóm tắt | — | **có** | mục 3.1 |
| §1 Introduction | tr.267–268 | hai lý do, hai cách cũ, tên gọi | — | **có** | mục 3.1, 1.2 |
| §2 The lasso — §2.1 Definition | tr.268–269 | định nghĩa, $t_0$, garotte | (1), (2) | **có** | mục 3.2 |
| §2.2 Orthonormal design case | tr.269 | bốn dạng co | (3), (4), ridge, garotte, tập con | **có, kiểm bằng số** | mục 3.3 |
| §2.3 Geometry of lasso | tr.270–271 | hình học, dấu, garotte | dạng bậc hai | **có, đo** | mục 3.4, 3.5 |
| §2.4 More on two-predictor case | tr.271–272 | hai biến tương quan | (5), (6) | **có — (6) in sai một miền** | mục 3.6 |
| §2.5 Standard errors | tr.272–273 | bootstrap, xấp xỉ ridge | (7) | **có — thừa số in sai** | mục 3.7 |
| §3 Example — prostate cancer data | tr.273–275 | ví dụ thật | — | **có, dựng lại** | mục 3.8, 3.9 |
| §4 Prediction error and estimation of t | tr.275–277 | CV, GCV, Stein | (8), ME, (9), $p(t)$, (10), (11), $R$, $\hat\gamma$, $\hat t$ | **có — ba chỗ in sai** | mục 3.9, 3.10 |
| §5 Lasso as Bayes estimate | tr.277 | tiên nghiệm Laplace | $f(\beta_j)$ | **có** | mục 3.11 |
| §6 Algorithms | tr.277–279 | ràng buộc dấu, cách của Gay | các bước (a)–(d) | **có, cài lại** | mục 3.12 |
| §7 Simulations — §7.1 Outline | tr.279–280 | cách so sánh | (12) | **có** | mục 3.13 |
| §7.2 Example 1 | tr.280–282 | ít hiệu ứng vừa | — | **có, dựng lại** | mục 3.13 |
| §7.3 Example 2 | tr.282 | nhiều hiệu ứng nhỏ | — | **có, dựng lại** | mục 3.13 |
| §7.4 Example 3 | tr.282–283 | một hiệu ứng lớn | — | **có, dựng lại** | mục 3.13 |
| §7.5 Example 4 | tr.283 | $p=40$ | — | **có, dựng lại** | mục 3.13 |
| §8 Application to generalized regression models | tr.283 | IRLS | — | có | mục 3.14 |
| §8.1 Logistic regression | tr.284 | kyphosis | ba mô hình | **có, thử lại — dữ liệu 83 dòng không có** | mục 3.14 |
| §9 Some further extensions | tr.284 | cây, MARS, bài toán suy biến | — | có, chỉ tóm tắt | mục 3.14 — không có số hay công thức để dựng lại |
| §10 Results on soft thresholding | tr.285–286 | ngưỡng mềm, rủi ro lý tưởng | (13)–(16) | **có — (14) in sai; (16) đo** | mục 3.15 |
| §11 Discussion | tr.286 | ba kịch bản, bridge | — | **có, đối chiếu với bảng** | mục 3.16 |
| §12 Software | tr.287 | Statlib | — | có | mục 3.16 |
| Acknowledgements | tr.287 | lời cảm ơn | — | có một câu | mục 3.5 (ai vẽ Hình 3) |
| References | tr.287–288 | 20 tài liệu | — | không | danh mục; các bài được dẫn trong thân đã ghi tên tại chỗ |

### 10.3 Tibshirani 2011 — theo hình và bảng

| Hình / bảng | Trang | Nhóm | Lấy? | Ở đâu |
|---|---|---|---|---|
| Table 1 — các mở rộng của lasso | tr.275 | RUỘT (danh mục) | **có** | Hình 28; mục 3.18 |
| Fig. 1 — hồi quy gần đơn điệu | tr.275 | NGOÀI | không | minh hoạ một phương pháp khác (nearly isotonic regression), không phải lasso chuẩn |
| Fig. 2 — số trích dẫn cộng dồn | tr.278 | NGOÀI | có, đọc định tính | Hình 29; mục 3.18 — điểm là ký tự văn bản trong PDF, không số hoá |

### 10.4 Tibshirani 2011 — theo mục

| Mục | Trang | Nội dung | Lấy? | Ở đâu |
|---|---|---|---|---|
| Summary | tr.273 | tóm tắt | có | mục 3.17 |
| §1 The lasso | tr.273 | dạng phạt, họ $q$ | **có — thiếu trị tuyệt đối** | mục 3.17 |
| §2 History of the idea | tr.273–274 | garrotte, năm lý do | **có** | mục 3.17 |
| §3 Computational advances | tr.274 | QP, LARS, hạ theo toạ độ | **có** | mục 3.17 |
| §4 Some generalizations and variants | tr.274–276 | Bảng 1, graphical lasso, near isotonic | có | mục 3.18; phần near isotonic không lấy — phương pháp khác |
| §5 Discussion | tr.276 | lời thách | **có** | mục 3.18 |
| Acknowledgements, References | tr.276–277 | | không | danh mục |
| Comments — Bühlmann | tr.277–279 | tốc độ, điều kiện chọn đúng, "screening", Hình 2 | có một phần | mục 3.18, 3.19, 6.3 — các định lý tiệm cận không có phép đo để dựng lại |
| Comments — Holmes | tr.279–280 | (2), (3), phê bình MAP | **có, kiểm (3) bằng số** | mục 3.19 |
| Author's response | tr.280–281 | trung bình hậu nghiệm không thưa | **có, đo** | mục 3.19, 3.11 |
| References in the comments | tr.281–282 | | không | danh mục |

### 10.5 ESL — theo hình và bảng

Phạm vi: chú thích tr.3, §3.4.2–3.4.3 (tr.68–73), cùng ô Lasso của Hình 3.7 và cột Lasso của Bảng 3.3. Phần ridge của
§3.3.4–3.4.1 thuộc bài [Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>).

| Hình / bảng | Trang | Nhóm | Lấy? | Ở đâu |
|---|---|---|---|---|
| Figure 1.1 — ma trận tán xạ prostate | tr.3 | NGOÀI | không | chỉ mô tả dữ liệu; lấy chú thích cùng trang |
| Figure 3.7 — đường CV | tr.62 | RUỘT (ô Lasso) | **có, đọc toạ độ vectơ** | Hình 33, 52(b); mục 3.21. Bốn ô khác thuộc bài Ridge, PCR, PLS và chọn tập con |
| Table 3.3 — hệ số và sai số kiểm tra | tr.63 | RUỘT (cột Lasso) | **có, dựng lại 9/9** | Hình 34; mục 3.21. Cột LS và Ridge đo ở bài Ridge |
| Figure 3.10 — đường lasso | tr.70 | RUỘT | **có, đọc toạ độ vectơ** | Hình 37, 52(a); mục 3.22 |
| Table 3.4 — ba ước lượng trực chuẩn | tr.71 | RUỘT | **có** | Hình 38; mục 3.23 |
| Figure 3.11 — bức tranh ước lượng | tr.71 | RUỘT | **có — chú thích lệch văn bản** | Hình 38; mục 3.23 |
| Figure 3.12 — đường mức $L_q$ | tr.72 | RUỘT | có | Hình 39; mục 3.23 |
| Figure 3.13 — $L_{1,2}$ và elastic net | tr.73 | RUỘT | **có, đo khoảng cách** | Hình 40; mục 3.23 |

### 10.6 ESL — theo mục và công thức

| Mục | Trang | Công thức | Lấy? | Ở đâu |
|---|---|---|---|---|
| §1 Introduction, Example 2 | tr.3 | — | **có: chú thích lỗi dữ liệu** | mục 3.20 |
| §3.3.4 Prostate cancer data (continued) | tr.61–63 | — | có phần về lasso | mục 3.21; phần còn lại ở bài Ridge |
| §3.4.1 Ridge regression | tr.61–68 | (3.41)–(3.50) | không | thuộc bài Ridge |
| §3.4.2 The lasso | tr.68–69 | (3.51), (3.52) | **có** | mục 3.22 |
| §3.4.3 Discussion: subset selection, ridge regression and the lasso | tr.69–73 | (3.53), (3.54) | **có** | mục 3.23 |
| §3.4.4 Least angle regression và phần còn lại của chương 3 | tr.73–… | (3.55) trở đi | không | thuộc bài [Least Angle Regression](<../Least Angle Regression (LARS)/Least Angle Regression (LARS).md>) và các bài PCR/PLS |
| các chương khác | | | không | ngoài phạm vi bài lasso |

### 10.7 Efron và cộng sự 2004 — theo hình và bảng

| Hình / bảng | Trang PDF | Nhóm | Lấy? | Ở đâu |
|---|---|---|---|---|
| Table 1 — dữ liệu diabetes | 3 | RUỘT (dữ liệu) | **có, kiểm 88/88 ô** | Hình 41; mục 3.24 |
| Fig. 1 — đường lasso và stagewise | 4 | RUỘT (bảng trái) | **có, dựng lại** | Hình 43, 53; mục 3.24 |
| Fig. 2 — hình học LARS, $m=2$ | 6 | NGOÀI | không | thuộc bài LARS |
| Fig. 3 — LARS trên diabetes | 9 | NGOÀI | không | thuộc bài LARS |
| Fig. 4 — LARS tiến gần OLS | 10 | NGOÀI | không | thuộc bài LARS |
| Fig. 5 — mô phỏng LARS/lasso/stagewise | 15 | NGOÀI | không | thuộc bài LARS |
| Fig. 6 — bậc tự do | 19 | NGOÀI | không | thuộc bài LARS |
| Fig. 7 — ước lượng $C_p$ | 20 | NGOÀI | không | thuộc bài LARS |
| Fig. 8 — đường $(T,S)$ của lasso | 30 | NGOÀI | không | thuộc bài LARS (chứng minh định lý 1) |
| Fig. 9 — hình học sửa đổi stagewise | 37 | NGOÀI | không | thuộc bài LARS |

### 10.8 Efron và cộng sự 2004 — theo mục

| Mục | Trang PDF | Nội dung | Lấy? | Ở đâu |
|---|---|---|---|---|
| §1 Introduction | 2–5 | (1.1)–(1.7), Bảng 1, Hình 1 | **có phần về lasso**: (1.1)–(1.5) | mục 3.24; (1.6)–(1.7) là stagewise, thuộc bài LARS |
| §2 The LARS algorithm | 5–11 | (2.1)–(2.22) | không | thuộc bài LARS; các ký hiệu $A_A$, $G_A$, $u_A$ dùng ở (3.2) được định nghĩa lại tại chỗ ở mục 3.25 |
| §3 Modified versions — §3.1 The LARS–Lasso relationship | 11–12 | (3.1)–(3.6), định lý 1 | **có** | mục 3.25 |
| §3.2 The LARS–Stagewise relationship | 12–14 | (3.7)–(3.14), định lý 2 | không | thuộc bài LARS |
| §3.3 Simulation study | 14–16 | (3.15)–(3.17) | có mô hình (3.15) | mục 3.25 |
| §3.4 Other LARS modifications | 16–17 | (3.18)–(3.20) | không | thuộc bài LARS |
| §4 Degrees of freedom and $C_p$ estimates, §4.1–4.3 | 17–25 | (4.1)–(4.24) | không | thuộc bài LARS |
| §5 LARS and Lasso properties | 25–34 | (5.1)–(5.44), bổ đề 4–10, chứng minh định lý 1 | không | chứng minh định lý 1; bài này chỉ dùng phát biểu và kiểm bằng số ở mục 3.25 |
| §6 Stagewise properties | 34–39 | (6.1)–(6.22) | không | thuộc bài LARS |
| §7 Computations | 39–40 | | không | thuộc bài LARS |
| §8 Boosting procedures | 40–41 | | không | ngoài phạm vi |
| Appendix A.1–A.3 | 41–47 | chứng minh bổ đề | không | thuộc bài LARS |
| Acknowledgments, References | 47–48 | | không | danh mục |

### 10.9 Friedman và cộng sự 2010 — theo hình và bảng

| Hình / bảng | Trang PDF | Nhóm | Lấy? | Ở đâu |
|---|---|---|---|---|
| Figure 1 — đường hệ số leukemia, ba phạt | 4 | NGOÀI | không | minh hoạ elastic net trên dữ liệu 3 571 gen không có trong repo; thuộc bài Elastic Net |
| Figure 2 — kiểm định chéo 10 mảnh | 17 | NGOÀI | không | minh hoạ phần mềm, dữ liệu mô phỏng không in đủ tham số |
| Table 1 — thời gian hồi quy lasso | 13 | NGOÀI | không | thời gian máy năm 2009, không tái lập được |
| Table 2 — thời gian logistic | 14 | NGOÀI | không | như trên |
| Table 3 — thời gian logistic, đặc trưng thưa | 15 | NGOÀI | không | như trên |
| Table 4 — thời gian trên dữ liệu thật | 16 | NGOÀI | không | như trên |
| Table 5 — so với các phần mềm khác | 16 | NGOÀI | không | như trên |

### 10.10 Friedman và cộng sự 2010 — theo mục

| Mục | Trang PDF | Công thức | Lấy? | Ở đâu |
|---|---|---|---|---|
| §1 Introduction | 1–3 | — | có một câu | mục 3.26 |
| §2 Algorithms for the lasso, ridge regression and elastic net | 3–5 | (1)–(6) | **có** | mục 3.26, 3.27 |
| §2.1 Naive updates | 5–6 | (7), (8) | **có, cài lại** | mục 3.27 |
| §2.2 Covariance updates | 6 | (9) | **có, cài lại** | mục 3.27 |
| §2.3 Sparse updates | 6–7 | — | không | ma trận thưa dạng cột; không đổi nghiệm, chỉ đổi cách lưu |
| §2.4 Weighted updates | 7 | (10) | **có** | mục 3.28 |
| §2.5 Pathwise coordinate descent | 7 | $\lambda_{\max}$ | **có, đo** | mục 3.28 |
| §2.6 Other details | 7 | — | **có** | mục 3.28 |
| §3 Regularized logistic regression | 8–9 | (11)–(18) | không | thuộc bài Logistic Regression |
| §4 Regularized multinomial regression, §4.1, §4.2 | 9–12 | (19)–(31) | không | thuộc bài hồi quy đa lớp |
| §5 Timings, §5.1–5.4 | 12–16 | — | không | thời gian máy, không tái lập được |
| §6 Selecting the tuning parameters | 16–18 | — | có một ý | mục 7 (chọn mức phạt) |
| §7 Discussion | 18–19 | — | không | tổng kết phần mềm |
| Appendix A Proof of Theorem 1 | 22 | (32)–(34) | không | định lý về đa lớp, thuộc mục 4 |

---

## 11. Tài liệu tham khảo và điều kiện tái lập

```
[1] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso",
    J. R. Statist. Soc. B 58(1):267–288, 1996.  https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
    → BÀI GỐC. Không có bản truy cập mở hợp pháp (Unpaywall: is_oa = false).
      Bản dùng để viết bài này là bản tải JSTOR, đặt tại
      AI algorithms/_papers/1996 - Tibshirani - Regression Shrinkage and Selection via the Lasso.pdf
      CHỈ trên máy dựng bài; file được liệt kê trong .gitignore và KHÔNG có trong repo công khai.
      Người đọc cần tự lấy bản của mình qua DOI (Wiley) hoặc JSTOR (stable/2346178).

[2] R. Tibshirani, "Regression shrinkage and selection via the lasso: a retrospective",
    J. R. Statist. Soc. B 73(3):273–282, 2011.
    AI algorithms/_papers/2011 - Tibshirani - Regression shrinkage and selection via the lasso, a retrospective.pdf

[3] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.,
    Springer, 2009 (bản in 12).
    AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf

[4] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression",
    The Annals of Statistics 32(2):407–451, 2004.  arXiv:math/0406456
    AI algorithms/_papers/2004 - Efron et al - Least Angle Regression.pdf

[5] J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized Linear Models
    via Coordinate Descent", Journal of Statistical Software 33(1), 2010.
    AI algorithms/_papers/2010 - Friedman, Hastie, Tibshirani - Regularization Paths for GLM via Coordinate Descent (glmnet).pdf

[6] H. Zou, T. Hastie, "Regularization and variable selection via the elastic net",
    J. R. Statist. Soc. B 67(2):301–320, 2005.   (nguồn của hai mô phỏng ở mục 6.4–6.5)
    AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf
```

**Dữ liệu.**

- `tools/prostate.csv` — 97 bệnh nhân, trích từ gói R `ElemStatLearn` 2015.6.26.2 (bản **đã sửa** lỗi lweight). Bản có
  lỗi dùng ở mục 3.7–3.12 được dựng lại trong script bằng cách đặt lweight của dòng 32 bằng $\ln449$.
- `tools/kyphosis.csv` — 81 dòng, gói R `rpart` qua Rdatasets. Không phải bộ 83 dòng mà bài 1996 dùng.
- Bộ `diabetes` — `sklearn.datasets.load_diabetes`, cả bản chuẩn hoá lẫn `scaled=False`.

**Script.**

| Script | Phần của bài | Thời gian |
|---|---|---|
| [`tools/v_lasso.py`](<../../../tools/v_lasso.py>) | mọi phép kiểm của mục 3, mục 0.1, 0.5, 4.2, 7, 8; mảng cho Hình 51–52 | vài phút, phần lớn ở mục 9 |
| [`tools/v_gd1ss.py`](<../../../tools/v_gd1ss.py>) mục 390 | mục 1.2 | vài giây |
| [`tools/demo_shrink.py`](<../../../tools/demo_shrink.py>) | mục 6.3, 6.4 | vài giây |
| [`tools/v_reg.py`](<../../../tools/v_reg.py>) mục 4 và 6 | mục 3.24, 6.4, 6.5 | vài phút |
| [`tools/fig_lasso.py`](<../../../tools/fig_lasso.py>) | Hình 51, 52, 53 — trích nguyên văn từ ba khối mã trong bài | vài giây |

Thư viện: numpy 2.4.4, scikit-learn 1.9.1, scipy 1.17.1. Mọi phép ngẫu nhiên có hạt giống ghi trong script. Các khung
cắt `paper_lasso_*.png` do `pdffig.py` sinh từ năm PDF trên.

---

## 12. Kiểm tra trước khi đóng bài

- [x] **Năm nguồn gốc** đều có phần ruột cắt vào bài theo Luật 6 và quy tắc E.1d; 50 khung cắt, mỗi khung có dòng caption
      gốc hoặc "Khung chứa gì", nhận xét nguyên văn của tác giả kèm trang, cách đọc, và điều khung không nói
- [x] **Mọi công thức đánh số** của phần đã lấy có bảng ký hiệu và đoạn ý nghĩa: [T] (1)–(16) cùng các công thức không
      đánh số; [R] (2)–(3) của Holmes; [E] (3.51)–(3.54) và Bảng 3.4; [L] (1.1)–(1.5), (3.1)–(3.6); [G] (1)–(10)
- [x] **Mọi ví dụ số và bảng số** của các nguồn được dựng lại hoặc ghi rõ vì sao không: Bảng 1 và 2, Hình 4, 5, 6, bốn
      mô phỏng, kyphosis, Bảng 3.3 cột Lasso, Hình 3.7, 3.10, Bảng 1 và Hình 1 của Efron, các khẳng định của Friedman
- [x] **Hai bảng đối chiếu cho mỗi nguồn** — mục 10.1–10.10
- [x] **Bảng ký hiệu có cột "Đọc là"** và bảy chỗ trùng tên giữa các nguồn — mục 0.3; bảng đổi ký hiệu — mục 4.4
- [x] **`kiemtrich.py`** trên năm PDF (ESL cắt còn 14 trang): 191 đoạn trích tiếng Anh — 35 khớp theo cụm, 28 khớp theo
      thứ tự từ, 19 không khớp. Đọc tay cả 19: 16 là lớp chữ OCR hoặc công thức vỡ của bản quét 1996 (so với ảnh trang thì
      khớp), 1 là câu của tài liệu gói `ElemStatLearn` (không nằm trong PDF nào), 1 là lớp chữ của Friedman 2010 in
      `O(N p)` có dấu cách, 1 là câu hồi tưởng 2011 có công thức. Một lỗi thật đã sửa: đoạn trích *"standardized so
      that …"* ở bảng đầu mục 3 bỏ mất vế $\sum_ix_{ij}/N=0$
- [x] Bế tắc của cách cũ nêu bằng số — mục 1.2
- [x] Bảng giá bốn cột — mục 8.1
- [x] Hình tự vẽ có khối mã sinh ra nó ngay trong bài — Hình 51, 52, 53
- [x] Mã có đầu ra thật — mục 0.1, 5.2, 5.4
- [ ] **Bốn chỗ bỏ ngỏ** chưa giải: $\hat s=0{,}44$ của Hình 5; cột (7) của Bảng 2; dữ liệu kyphosis 83 dòng; số bước của
      mô hình bậc hai ở Efron §3.1
- [ ] **Bản PDF 1996 không có trong repo công khai** — mọi câu trích từ nó chỉ kiểm được trên máy có bản JSTOR
- [ ] Chưa có người đọc hết bài — nhãn vẫn là `Đang viết`
