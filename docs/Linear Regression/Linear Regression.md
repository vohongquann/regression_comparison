# Linear Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 360
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Adrien-Marie Legendre
Date: January 1, 1805
url: https://doi.org/10.3931/e-rara-35115

> **Linear Regression — hồi quy tuyến tính** học một đường thẳng (hoặc một mặt phẳng khi có
> nhiều biến) để dự đoán **một giá trị dạng số**: từ tuổi một chiếc máy dự đoán chi phí sửa
> chữa, từ diện tích và số phòng dự đoán giá nhà. Cách chọn đường thẳng — **làm tổng bình
> phương các sai lệch nhỏ nhất** — được in lần đầu năm 1805 trong một phụ lục chín trang của
> Adrien-Marie Legendre, rồi được Carl Friedrich Gauss nối với lý thuyết xác suất năm 1809.
>
> Phần ruột của bài (mục 3) đi theo **đúng hai bản in gốc đó**: mọi hệ phương trình đánh số,
> mọi ví dụ số tác giả tự làm và mọi câu tác giả tự đánh giá kết quả đều được trích nguyên văn
> rồi **dựng lại bằng máy**. Việc dựng lại tìm ra ba điều mà không bản tóm tắt nào nói: Legendre
> tự bác kết quả chính của phần ứng dụng; Gauss in sai một hệ số trong ví dụ số duy nhất của
> mình; và một bài báo lịch sử năm 1981 kết luận ngược với cách chuyện tranh chấp thường được kể.

---

## 0. Bắt đầu từ đây

### 0.1 Nhận gì, trả gì

| | Nói bằng lời | Kiểu và kích thước |
|---|---|---|
| **Đầu vào khi học** | một bảng $n$ dòng dữ liệu, mỗi dòng có các đặc trưng | ma trận $X$, số thực, cỡ $(n, p)$ |
| **Đáp án khi học** | một con số cho mỗi dòng | vectơ $y$, cỡ $(n,)$ |
| **Đầu ra của việc học** | một hệ số cho mỗi cột | vectơ $\hat\beta$, cỡ $(p,)$ |
| **Dùng để dự đoán** | nhân dòng mới với các hệ số rồi cộng lại | $\hat y = x_{\text{mới}}^\top\hat\beta$, một số |
| **Tiêu chuẩn chọn $\hat\beta$** | tổng bình phương phần dư nhỏ nhất | $\min_\beta \lVert y - X\beta\rVert_2^2$ |

$p$ đếm **cả cột hệ số chặn** (cột toàn số 1). Có $d$ đặc trưng và có hệ số chặn thì $p = d+1$.

Ví dụ chạy được ngay, dùng đúng bốn chiếc máy của mục 0.5:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 4.0, 5.0, 8.0])
X = np.column_stack([np.ones_like(x), x])  # cột 1 tạo hệ số chặn

beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
x_new = 2.5
y_new = np.array([1.0, x_new]) @ beta

print("beta =", beta, "rank =", rank)
print("dự đoán tại x = 2.5:", y_new)
```

Đầu ra thật:

```text
beta = [1.  1.6] rank = 2
dự đoán tại x = 2.5: 5.0
```

`rank = 2` nghĩa là hai cột của $X$ độc lập nên hai hệ số được xác định duy nhất. Dòng mới
phải là `[1.0, x_new]` chứ không chỉ `[x_new]`, vì mô hình đã học cả hệ số chặn.

### 0.2 Từ mới trong bài này

| Từ | Nghĩa, nói cho người chưa biết | Xuất hiện ở |
|---|---|---|
| **mẫu** (sample) | một dòng dữ liệu — một chiếc máy, một ngôi nhà, một lần quan sát | khắp bài |
| **đặc trưng** (feature) | một cột thông tin dùng để dự đoán — tuổi máy, diện tích | mục 0, 4 |
| **biến đích** (target) | con số muốn dự đoán — chi phí, giá nhà | mục 0, 4 |
| **hệ số chặn** (intercept) | giá trị đường thẳng dự đoán tại $x = 0$; đến từ cột toàn số 1 | mục 0.5, 4 |
| **hệ số góc** (slope) | $x$ tăng một đơn vị thì dự đoán tăng bao nhiêu | mục 0.5 |
| **phần dư** (residual) | giá trị thật trừ giá trị dự đoán, $e = y - \hat y$; quan sát được | khắp bài |
| **sai số** (error) | độ lệch thật giữa quan sát và giá trị đúng, $\varepsilon$; **không** quan sát được | mục 0.6, 3.5 |
| **bình phương tối thiểu** (least squares) | chọn tham số làm tổng bình phương phần dư nhỏ nhất. Legendre gọi là *méthode des moindres quarrés* | khắp bài |
| **OLS** (ordinary least squares) | bình phương tối thiểu "thường": mọi dòng cùng trọng số, không phạt hệ số | mục 4, 6 |
| **phương trình chuẩn** (normal equations) | hệ $X^\top X\hat\beta = X^\top y$ — đạo hàm của tổng bình phương đặt bằng 0 | mục 3, 4, 6 |
| **ma trận thiết kế** (design matrix) | chính bảng $X$, thêm cột toàn 1 nếu có hệ số chặn | mục 4 |
| **hạng** (rank) | số cột thật sự độc lập của $X$; nhỏ hơn $p$ thì nghiệm không duy nhất | mục 0.1, 4.3 |
| **số điều kiện** (condition number) $\kappa$ | đo ma trận khuếch đại sai số làm tròn bao nhiêu lần; càng lớn càng khó giải chính xác | mục 6 |
| **leverage** $h_{ii}$ | mức một điểm tự kéo dự đoán của chính nó; nằm giữa 0 và 1 | mục 4.4, 5 |
| **$R^2$** | tỉ lệ dao động của $y$ mà mô hình giải thích được; 1 là khớp hoàn hảo | mục 5, 9 |
| **phân phối chuẩn** (normal / Gaussian) | phân phối hình chuông; Gauss suy ra nó ở điều 177 | mục 3.5 |
| **hợp lý cực đại** (maximum likelihood) | chọn tham số làm dữ liệu đã thấy **có xác suất cao nhất** | mục 3.6, 8 |
| **trọng số** (weight) | hệ số nhân vào bình phương phần dư của từng dòng; dòng kém tin cậy nhận trọng số nhỏ | mục 3.7 |
| **độ chính xác** (*praecisio*, $h$) | đại lượng của Gauss, tỉ lệ **nghịch** với độ lệch chuẩn: $h = 1/(\sigma\sqrt2)$ | mục 3.5, 3.7 |
| **độ dẹt** (*aplatissement*, $\alpha$) | Trái Đất phình ở xích đạo bao nhiêu phần so với bán kính; 0 là hình cầu | mục 3.3 |
| **cung kinh tuyến** | một đoạn đường bắc–nam đo trên mặt đất, kèm vĩ độ hai đầu | mục 3.3, 3.9 |
| **module**, **toise** | đơn vị dài thời 1805: một module bằng hai toise, khoảng 12,78 foot (Stigler tr.467) | mục 3.3 |
| **phút cung** ($'$), **giây cung** ($''$) | một phần 60 và một phần 3600 của một độ góc | mục 3.1, 3.3 |
| **QR**, **SVD** | hai cách phân rã ma trận để giải bình phương tối thiểu mà không lập $X^\top X$ | mục 6 |
| **hạ gradient** (gradient descent) | đi từng bước ngược hướng đạo hàm tới khi hết giảm được | mục 6, 7 |
| **L1**, **nguyên lý Boscovich / Laplace** | chọn tham số làm tổng **trị tuyệt đối** phần dư nhỏ nhất, thay vì tổng bình phương | mục 3.8, 8 |
| **ngoại lai** (outlier) | một điểm lệch hẳn khỏi phần còn lại | mục 8, 9 |
| **đa cộng tuyến** (collinearity) | hai cột gần như trùng nhau | mục 9 |
| **Ridge**, **Huber** | hai biến thể: Ridge cộng phạt $\lambda\lVert\beta\rVert_2^2$; Huber đổi bình phương thành hàm tăng tuyến tính ở phần dư lớn | mục 8, 9 |
| **BLAS** | thư viện nhân–giải ma trận mà numpy gọi xuống; khác máy thì có thể khác chữ số cuối | mục 6.2 |

### 0.3 Bảng ký hiệu — và chỗ trùng tên giữa ba nguồn

Bài dùng **ba hệ ký hiệu**: hiện đại (mục 4–9), của Legendre 1805 và của Gauss 1809 (mục 3).
Mục 3 giữ ký hiệu của từng tác giả để đối chiếu được với ảnh trang in.

**Hiện đại**

| Ký hiệu | Đọc là gì | Nghĩa | Chiều / kiểu |
|---|---|---|---|
| $n$, $p$, $d$ | "en", "pê", "đê" | số mẫu · số tham số kể cả chặn · số đặc trưng | số nguyên |
| $X$ | "ích lớn" | ma trận thiết kế | $\mathbb R^{n\times p}$ |
| $y$, $\hat y$, $\bar y$ | "y", "y mũ", "y ngang" | đáp án · dự đoán · trung bình của $y$ | $\mathbb R^n$, $\mathbb R^n$, vô hướng |
| $\beta$, $\hat\beta$ | "bê-ta", "bê-ta mũ" | hệ số thật · hệ số ước lượng | $\mathbb R^p$ |
| $\varepsilon$, $e$ | "ép-xi-lon", "e" | sai số thật · phần dư quan sát được | $\mathbb R^n$ |
| $\sigma^2$, $\hat\sigma^2$ | "xích-ma bình phương" | phương sai sai số · ước lượng của nó | vô hướng |
| $H$, $h_{ii}$ | "hát", "hát i i" | ma trận chiếu $y\mapsto\hat y$ · phần tử đường chéo | $\mathbb R^{n\times n}$, vô hướng |
| $\kappa(A)$ | "kap-pa" | số điều kiện của $A$ | vô hướng $\ge 1$ |
| $X^\top$ | "ích chuyển vị" | lật dòng thành cột | $\mathbb R^{p\times n}$ |
| $A^{-1}$ | "a nghịch đảo" | ma trận với $A^{-1}A = I$ | vuông |
| $\lVert v\rVert_2^2$ | "chuẩn hai bình phương" | $\sum_i v_i^2$ | vô hướng |
| $\arg\min_\beta f$ | "ác-min theo bê-ta" | **giá trị $\beta$** làm $f$ nhỏ nhất, không phải giá trị nhỏ nhất của $f$ | $\mathbb R^p$ |
| $\operatorname{tr}(A)$ | "vết" | tổng đường chéo | vô hướng |
| $\mathbb E[\cdot\mid X]$ | "kỳ vọng khi biết ích" | trung bình qua các lần lấy mẫu có cùng $X$ | — |

**Legendre 1805**

| Ký hiệu | Đọc là gì | Nghĩa trong bản in | Kiểu |
|---|---|---|---|
| $E, E', E'', E''', E^{iv}, E^{v}$ | "e", "e phẩy"… | sai số (*erreur*) của từng phương trình | vô hướng, đơn vị của bài toán |
| $a, b, c, f$ | — | hệ số **đã biết** trong $E = a + bx + cy + fz + \&c.$ | vô hướng |
| $x, y, z, u$ | — | **ẩn** cần tìm — không phải dữ liệu | vô hướng |
| $\int ab$ | "tổng a b" | $ab + a'b' + a''b'' + \&c.$ — Legendre dùng dấu tích phân cho **tổng** | vô hướng |
| $\alpha$ | "an-pha" | độ dẹt: trục Trái Đất tỉ lệ $1 : 1+\alpha$ | vô hướng nhỏ |
| $\zeta$ | "dê-ta" (bản in là một chữ giống c có đuôi) | hiệu chỉnh độ dài một độ: $1/D = (1+\zeta)/28500$ | vô hướng nhỏ |
| $D$ | "đê" | độ dài của độ vĩ thứ 45, tính bằng module | vô hướng |
| $S$ | "ét" | độ dài một cung, module | vô hướng |
| $L, L'$ | "e-lờ" | vĩ độ hai đầu cung, độ | vô hướng |

**Gauss 1809**

| Ký hiệu | Đọc là gì | Nghĩa | Kiểu |
|---|---|---|---|
| $p, q, r, s$ | — | các **ẩn** | vô hướng |
| $V, V', V''$ | "vê" | hàm của các ẩn — giá trị tính ra | vô hướng |
| $M, M', M''$ | "em" | giá trị quan sát được | vô hướng |
| $v = V - M$ | "vê nhỏ" | sai lệch tính ra trừ quan sát — chính là phần dư, **đổi dấu** so với $e$ | vô hướng |
| $\Delta$ | "đen-ta" | sai số của một quan sát | vô hướng |
| $\varphi(\Delta)$ | "phi của đen-ta" | xác suất (mật độ) của sai số $\Delta$ | hàm |
| $\Omega$ | "ô-mê-ga" | tích các $\varphi$ — xác suất để mọi quan sát cùng xảy ra | vô hướng |
| $h$ | "hát" | *mensura praecisionis*, độ chính xác | vô hướng $>0$ |
| $P, Q, R, S$ | — | vế trái của phương trình chuẩn — **ký hiệu đặt tên cho một biểu thức**, không phải ẩn | biểu thức tuyến tính |

**Bốn chỗ trùng tên phải nhớ**, vì chúng gây đọc sai thật:

1. **$S$** là **độ dài cung** ở Legendre, nhưng là **vế trái phương trình chuẩn thứ tư** ở Gauss.
2. **$x, y$** là **ẩn** ở cả Legendre lẫn Gauss, trong khi ở mục 4–9 chúng là **dữ liệu**.
   Ngược vai hoàn toàn: người 1805 biết hệ số, tìm ẩn; người nay biết dữ liệu, tìm hệ số.
3. **$v$ của Gauss $= V - M$ = tính trừ quan sát**, còn $e$ hiện đại $= y - \hat y$ = quan sát trừ
   tính. Hai đại lượng **ngược dấu**; bình phương lên thì như nhau.
4. **$h$** là **độ chính xác** của Gauss, **ngược chiều** với độ lệch chuẩn: $h$ gấp đôi là
   $\sigma$ chỉ còn một nửa. Còn $h_{ii}$ ở mục 4 là **leverage** — không liên quan gì.

### 0.4 Luồng xử lý

1. **Lập bảng** $X$ (thêm cột toàn 1 nếu cần hệ số chặn) và vectơ đáp án $y$.
2. **Chọn tiêu chuẩn**: tổng bình phương phần dư $S(\beta) = \sum_i (y_i - x_i^\top\beta)^2$.
3. **Lấy đạo hàm theo từng hệ số, đặt bằng 0** — ra $p$ phương trình tuyến tính cho $p$ ẩn. Đây
   chính là thao tác Legendre mô tả bằng lời năm 1805 (mục 3.1).
4. **Giải** hệ đó — trên máy thì bằng QR hoặc SVD, **không** lập nghịch đảo (mục 6).
5. **Kiểm**: phần dư tổng bằng 0 khi có hệ số chặn, trực giao với mọi cột, rồi mới dùng.

### 0.5 Một vòng tính tay, bằng số thật

Tuổi và chi phí sửa chữa hằng năm của bốn chiếc máy:

| Máy | Tuổi $x$ (năm) | Chi phí $y$ (triệu đồng) |
|---|---:|---:|
| A | 1 | 3 |
| B | 2 | 4 |
| C | 3 | 5 |
| D | 4 | 8 |

**So ba đường thẳng** bằng đúng tiêu chuẩn của bài — tổng bình phương phần dư:

| Đường thử | Tổng bình phương phần dư |
|---|---:|
| $\hat y = 0 + 2{,}0x$ — đi qua gốc, dốc | 2,000000 |
| $\hat y = 1{,}5 + 1{,}4x$ — thoải hơn | 1,400000 |
| $\hat y = 1 + 1{,}6x$ — nghiệm bình phương tối thiểu | **1,200000** |

**Lập phương trình chuẩn** $X^\top X\hat\beta = X^\top y$:

$$
X^\top X = \begin{pmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2\end{pmatrix}
= \begin{pmatrix} 4 & 10 \\ 10 & 30\end{pmatrix},\qquad
X^\top y = \begin{pmatrix}\sum y_i \\ \sum x_iy_i\end{pmatrix}
= \begin{pmatrix} 20 \\ 58 \end{pmatrix}
$$

**Ý nghĩa.** Hàng đầu của hệ đòi tổng phần dư bằng 0, tức đường đi qua điểm trung bình
$(2{,}5;\ 5)$; hàng sau đòi phần dư không còn đồng biến với tuổi máy. Hai điều kiện cho đúng
hai ẩn. Định thức $4\cdot30 - 10^2 = 20$, nên

$$
\hat\beta = \frac{1}{20}\begin{pmatrix} 30 & -10\\ -10 & 4\end{pmatrix}\begin{pmatrix}20\\58\end{pmatrix}
= \begin{pmatrix}1{,}5 & -0{,}5\\ -0{,}5 & 0{,}2\end{pmatrix}\begin{pmatrix}20\\58\end{pmatrix}
= \begin{pmatrix}1\\1{,}6\end{pmatrix}.
$$

**Ý nghĩa.** Hệ số góc 1,6: trong vùng tuổi 1–4 năm, thêm một năm tuổi thì chi phí dự đoán
tăng 1,6 triệu đồng. Hệ số chặn 1 là giá trị đường thẳng tại tuổi 0 — nằm ngoài vùng dữ liệu,
nên không được đọc nó như chi phí sửa chữa của một máy mới.

**Phần dư và leverage**:

| Máy | $x_i$ | $y_i$ | $\hat y_i$ | $e_i$ | $h_{ii}$ |
|---|---:|---:|---:|---:|---:|
| A | 1 | 3 | 2,6 | +0,4 | 0,7 |
| B | 2 | 4 | 4,2 | −0,2 | 0,3 |
| C | 3 | 5 | 5,8 | −0,8 | 0,3 |
| D | 4 | 8 | 7,4 | +0,6 | 0,7 |

Ba phép kiểm: $\sum e_i = 0$; $\sum e_i x_i = 0{,}4 - 0{,}4 - 2{,}4 + 2{,}4 = 0$; $\sum h_{ii} = 2 = p$.

**Thống kê**: tổng bình phương phần dư 1,2; tổng bình phương quanh trung bình
$\sum(y_i - 5)^2 = 14$; nên $R^2 = 1 - 1{,}2/14 = 0{,}914286$. Phương sai sai số ước lượng
$\hat\sigma^2 = 1{,}2/(4-2) = 0{,}6$; sai số chuẩn của hệ số góc
$\sqrt{0{,}6\times0{,}2} = 0{,}346410$, thống kê $t = 1{,}6/0{,}346410 = 4{,}618802$.

Mọi con số trên in ra từ `tools/v_linreg.py` mục 1.

### 0.6 Điểm dễ nhầm

| Nhầm | Đúng | Hệ quả đo được |
|---|---|---|
| "Tuyến tính" nghĩa là chỉ vẽ được đường thẳng | tuyến tính **theo hệ số**; cột $x^2$ hay $\log x$ vẫn là hồi quy tuyến tính | trên dữ liệu cong, thêm cột $x^2, x^3$ đưa RMSE từ 0,2962 xuống 0,0937 (`v_linreg.py` mục 10) |
| Dòng "chính xác một nửa" thì trọng số một nửa | độ chính xác tỉ lệ $1/\sigma$, trọng số tỉ lệ $1/\sigma^2$ — **một nửa độ chính xác là trọng số 1/4** | dùng 1/2, ma trận chuẩn của ví dụ Gauss ra $\begin{psmallmatrix}28&3&-3\\3&24&10\\-3&10&63\end{psmallmatrix}$, lệch hẳn bản in; dùng 1/4 thì khớp từng số (mục 3.7) |
| $e$ và $\varepsilon$ là một | $e$ tính được sau khi khớp; $\varepsilon$ không bao giờ thấy | chia $e^\top e$ cho $n$ thay vì $n-p$ thì $\hat\sigma^2$ chệch xuống 8,0 % ở ví dụ 25 điểm (mục 7.2) |
| $R^2$ cao chứng minh $x$ gây ra $y$ | hồi quy đo đồng biến, không đo nhân quả | không có con số nào sửa được điều này; thiếu là thiếu thiết kế nghiên cứu |

### 0.7 Đọc bài theo thứ tự nào

* Lần đầu học hồi quy: mục **0 → 4 → 5 → 7.1 → 9**. Đủ để chạy và biết khi nào đừng dùng.
* Muốn biết phương pháp **từ đâu ra và vì sao là bình phương**: mục **1 → 2 → 3 → 8**.
* Muốn cài đặt đúng trên máy: mục **6 → 7.2**.

### 0.8 Khi KHÔNG nên dùng

| Tình huống | Dấu hiệu nhận ra | Số đo trong bài |
|---|---|---|
| quan hệ cong rõ rệt | phần dư cong hình chữ U (Hình 22, bảng 2) | đường thẳng giữ RMSE 0,2938 ở $n = 200\,000$, gấp 2,9378 lần mức nhiễu 0,10 |
| có ngoại lai thật | vài điểm bay xa và không xoá được | 1 điểm trên 60 kéo hệ số góc lệch 16,2 %; Huber lệch 0,1 % |
| hai cột gần trùng | hệ số đổi dấu khi thêm bớt vài dòng | hệ số ngược dấu ở 48,2 % số lần lấy mẫu |
| cần khoảng chứ không chỉ trung bình | câu hỏi dạng "90 % trường hợp dưới mức nào" | hồi quy phân vị trả lời thẳng câu đó |
| câu hỏi nhân quả | — | không thuật toán nào thay được thiết kế nghiên cứu |

---

## 1. Bài toán mà nó giải — và cách cũ tắc ở đâu

### 1.1 Nhiều quan sát hơn số ẩn, và mọi quan sát đều sai một chút

Năm 1805 chưa có "học máy". Bài toán là **thiên văn và trắc địa**: tìm vài đại lượng chưa biết —
yếu tố quỹ đạo một sao chổi, độ dẹt Trái Đất — từ **nhiều** quan sát. Mỗi quan sát cho một
phương trình, mỗi phương trình sai một chút, nên hệ **nhiều phương trình hơn ẩn** gần như chắc
chắn **vô nghiệm**.

Legendre mô tả đúng tình thế đó ở trang in đầu phụ lục (tr.72, PDF 84), nguyên văn:

> « Si l'on a autant d'équations que d'inconnues x, y, z, &c., il n'y a aucune difficulté pour
> la détermination de ces inconnues, et on peut rendre les erreurs E absolument nulles. Mais le
> plus souvent, le nombre des équations est supérieur à celui des inconnues, et il est
> impossible d'anéantir toutes les erreurs. »
>
> (Nếu số phương trình bằng số ẩn x, y, z…, việc tìm ẩn không có gì khó, và có thể làm các sai
> số E bằng không tuyệt đối. Nhưng thường thì số phương trình lớn hơn số ẩn, và không thể xoá
> hết mọi sai số.)

### 1.2 Cách cũ và bế tắc của nó, đo bằng số

Ngay trước câu quyết định ở mục LVII (tr.64, PDF 76), Legendre làm đúng cách cũ trên chính ví dụ
của mình: ông **ép hai sai số $E''$ và $E^{iv}$ bằng 0** để khử hai ẩn $\tau$ và $\varpi$ — tức
chọn ra những phương trình được giải **đúng tuyệt đối** — rồi xem ba sai số còn lại theo hai ẩn
$z$, $u$. Nếu tiếp tục cách đó bằng cách cho luôn $z = u = 0$, ba sai số là 2,248 · 0,006 · 1,418
phút cung. Ông gọi chúng là *« erreurs très-tolérables dans la théorie des comètes »* (sai số hoàn
toàn chấp nhận được trong lý thuyết sao chổi), rồi làm tiếp:

| | Cách cũ: chọn số phương trình bằng số ẩn | Bình phương tối thiểu trên **cả ba** |
|---|---:|---:|
| sai số lớn nhất (phút cung) | 2,248 | **0,868** |
| tổng bình phương sai số | 7,0643 | **1,4329** |
| giảm | — | **4,9302 lần** |

Nguồn: `tools/v_linreg.py` mục 2, dựng lại từ các hệ số in ở tr.64.

Bế tắc của cách cũ không phải "kém chính xác" chung chung. Nó có **ba chỗ đo được**:

1. **Vứt dữ liệu.** Chọn phương trình nào để giữ là một quyết định tuỳ ý; phần còn lại không góp
   gì vào nghiệm.
2. **Kết quả phụ thuộc lựa chọn đó.** Hai người chọn hai bộ phương trình khác nhau thì ra hai
   nghiệm khác nhau, và không có tiêu chuẩn nào nói ai đúng hơn. Legendre gọi thẳng tên điều đó ở
   tr.72: *« il entre nécessairement de l'arbitraire dans la distribution des erreurs »* (sự tuỳ
   tiện tất yếu len vào cách phân bổ sai số).
3. **Không có con số nào so được.** Thiếu một hàm mục tiêu thì không có cách nói nghiệm này
   **tốt hơn** nghiệm kia bao nhiêu. Bình phương tối thiểu cho một con số duy nhất để so: 7,0643
   so với 1,4329.

### 1.3 Nó đổi đúng cái gì

**Một câu:** thay vì chọn ra một số phương trình để giải đúng, hãy **giữ tất cả** và chọn ẩn làm
**tổng bình phương mọi sai số nhỏ nhất**.

| | Trước | Sau |
|---|---|---|
| số phương trình dùng | đúng bằng số ẩn | **tất cả** |
| tiêu chuẩn | không có — tuỳ người chọn | $\min \sum E^2$, **duy nhất** |
| cách tính | khử trực tiếp | lập $p$ phương trình chuẩn rồi khử — **vẫn chỉ là giải hệ tuyến tính** |
| so hai nghiệm | không so được | so bằng tổng bình phương |

Hàng thứ ba là lý do phương pháp lan nhanh: nó **không đòi phép tính mới nào**. Người tính năm
1805 đã quen khử hệ tuyến tính; bình phương tối thiểu chỉ cho họ **hệ nào** để khử.

---

## 2. Ý tưởng trung tâm — bằng lời, chưa công thức

**Chọn đường làm tổng bình phương các khoảng lệch nhỏ nhất.** Legendre tự giải thích vì sao ông
chọn tiêu chuẩn đó, ở tr.72–73 (PDF 84–85), nguyên văn:

> « Par ce moyen, il s'établit entre les erreurs une sorte d'équilibre qui empêchant les extrêmes
> de prévaloir, est très-propre à faire connoître l'état du système le plus proche de la vérité. »
>
> (Bằng cách này, giữa các sai số hình thành một kiểu **cân bằng**, ngăn không cho các sai số cực
> đoan lấn át, rất thích hợp để cho biết trạng thái của hệ gần với sự thật nhất.)

Chữ **cân bằng** là ẩn dụ đúng, và nó giải thích được **hành vi** chứ không chỉ nghe hay. Legendre
tự chỉ ra hai hệ quả ở tr.74–75 (PDF 86–87):

* Với **một** đại lượng đo nhiều lần, nghiệm bình phương tối thiểu **chính là trung bình cộng**:
  *« La règle par laquelle on prend le milieu entre les résultats de différentes observations,
  n'est qu'une conséquence très-simple de notre méthode générale »* (quy tắc lấy trung bình giữa
  kết quả của nhiều quan sát chỉ là một hệ quả rất đơn giản của phương pháp tổng quát của
  [Legendre]).
* Với nhiều điểm trong không gian, nghiệm là **trọng tâm**: *« si on divise la masse d'un corps en
  molécules égales et assez petites pour être considérées comme des points, la somme des quarrés
  des distances des molécules au centre de gravité sera un minimum »* (chia một vật thành những
  phân tử bằng nhau đủ nhỏ để coi là điểm, thì tổng bình phương khoảng cách từ các phân tử tới
  trọng tâm là nhỏ nhất).

Trọng tâm là ẩn dụ tốt nhất cho người mới: **mỗi điểm dữ liệu kéo nghiệm về phía nó bằng một lực
tỉ lệ với khoảng lệch**. Điểm ở xa kéo mạnh — và đó cũng chính là điểm yếu (mục 9): một điểm lệch
gấp 10 lần góp vào tổng bình phương gấp 100 lần.

### 2.1 Ba cách nhìn cùng một nghiệm

![Ba cách nhìn OLS](<ols_three_views.png>)

**Hình 1 — Ba cách nhìn cùng một nghiệm bình phương tối thiểu (hình tự vẽ).** Legendre và Gauss
không vẽ hình nào cho phương pháp — hai bản in chỉ có chữ và hệ phương trình (mục 11). Hình này bù
đúng chỗ đó: nó cho thấy nghiệm **trông như thế nào** trong ba không gian khác nhau.

**Cách đọc hình.** Ba bảng, đọc từ trái sang:

- **Bảng 1 — không gian dữ liệu.** Chấm xám là dữ liệu, đường xám là đường hồi quy, **đoạn đỏ
  thẳng đứng** là phần dư $e_i$. Nghiệm làm tổng *bình phương* các đoạn đỏ nhỏ nhất. Đoạn đỏ vẽ
  **thẳng đứng**, không vuông góc với đường: mô hình đặt sai số vào $y$ và coi $x$ là đo đúng.
- **Bảng 2 — không gian tham số.** Trục là $\beta_0$ và $\beta_1$; các vòng là đường mức của
  $S(\beta)$; ngôi sao là nghiệm. Chỉ có **một** tâm vì $S$ là parabôlôit lồi. Elip càng dẹt thì
  bài toán càng khó cho hạ gradient.
- **Bảng 3 — hình học.** Mặt phẳng xám là mọi $X\beta$ đạt được; mũi tên đen là $y$, nằm **ngoài**
  mặt phẳng; mũi tên xám là $\hat y$, hình chiếu vuông góc của $y$. Mũi tên đỏ là $e$, và **ký hiệu
  góc vuông** là toàn bộ nội dung phương trình chuẩn: $e$ vuông góc với mọi cột của $X$.

**Điều hình KHÔNG nói ra.** Bảng 3 là sơ đồ, toạ độ đặt tay, không mã hoá số liệu. Và cả ba bảng
đều im lặng về **vì sao** chọn bình phương thay vì trị tuyệt đối — câu hỏi đó Gauss trả lời ở mục 3.6.

<details>
<summary><b>Code sinh Hình 3</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mp

plt.rcParams.update({                          # kiểu chữ, lưới và màu nền của hình
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

rng = np.random.default_rng(3)                 # cùng seed với mục 7, nên cùng dữ liệu
n = 25
x = np.sort(rng.uniform(0, 10, n))
y = 2.0 + 1.3 * x + rng.normal(0, 2.0, n)
X = np.c_[np.ones(n), x]
b = np.linalg.lstsq(X, y, rcond=None)[0]
yhat = X @ b
OLS = "#6b7280"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

# --- bảng 1: không gian dữ liệu, phần dư vẽ THẲNG ĐỨNG ---
ax = axes[0]
ax.scatter(x, y, s=22, color="#374151", zorder=3)
ax.plot(x, yhat, color=OLS, lw=1.8, zorder=2)
for xi, yi, yh in zip(x, y, yhat):
    ax.plot([xi, xi], [yi, yh], color="#dc2626", lw=1.0, alpha=.75, zorder=1)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.text(.5, max(y) * .97, r"$\hat\beta$ = (%.3f, %.3f)" % (b[0], b[1]), fontsize=8)
ax.set_title("1. Trong không gian dữ liệu\nOLS cực tiểu TỔNG BÌNH PHƯƠNG đoạn đỏ", fontsize=9)

# --- bảng 2: đường mức của S(β) trên lưới 200x200 ---
ax = axes[1]
b0 = np.linspace(b[0] - 6, b[0] + 6, 200)
b1 = np.linspace(b[1] - 1.2, b[1] + 1.2, 200)
B0, B1 = np.meshgrid(b0, b1)
S = np.array([[np.sum((y - (i + j * x)) ** 2) for i in b0] for j in b1])
ax.contour(B0, B1, S, levels=np.percentile(S, [1, 3, 7, 14, 25, 40, 60, 85]),
           cmap="Greys", linewidths=.7)
ax.plot(b[0], b[1], "*", color=OLS, ms=16)
ax.set_xlabel(r"$\beta_0$"); ax.set_ylabel(r"$\beta_1$")
ax.set_title("2. Trong không gian tham số\n"
             r"$S(\beta)$ là parabôlôit lồi — đúng MỘT cực tiểu", fontsize=9)

# --- bảng 3: sơ đồ hình chiếu, toạ độ đặt tay ---
ax = axes[2]
ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
ax.add_patch(mp.Polygon([[0, 0], [3.4, .55], [4.7, 2.45], [1.3, 1.9]], closed=True,
                        fc="#e5e7eb", ec="#9ca3af", lw=.8, alpha=.7, zorder=0))
ax.plot([0, 3.4], [0, 0.55], color="#9ca3af", lw=1.4)     # cột x1
ax.plot([0, 1.3], [0, 1.9],  color="#9ca3af", lw=1.4)     # cột x2
ax.annotate("", (2.6, 1.35), (0, 0), arrowprops=dict(arrowstyle="->", color=OLS, lw=1.8))
ax.annotate("", (2.6, 3.3),  (0, 0), arrowprops=dict(arrowstyle="->", color="#111827", lw=1.8))
ax.annotate("", (2.6, 3.3), (2.6, 1.35), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6))
ax.plot([2.6, 2.36], [1.35, 1.58], color="#dc2626", lw=.9)   # ký hiệu góc vuông
ax.plot([2.36, 2.6], [1.58, 1.82], color="#dc2626", lw=.9)
ax.text(2.72, 3.32, r"$y$", fontsize=10)
ax.text(2.05, 1.05, r"$\hat y=X\hat\beta$", fontsize=10, color=OLS)
ax.text(2.70, 2.30, r"$e=y-\hat y$", fontsize=10, color="#dc2626")
ax.text(3.55, 1.50, r"$\mathrm{col}(X)$", fontsize=9, color="#4b5563")  # tránh đè nhãn e
ax.text(3.45, 0.26, r"$x_1$", fontsize=9, color="#4b5563")
ax.text(1.02, 2.00, r"$x_2$", fontsize=9, color="#4b5563")
ax.set_xlim(-.35, 5.1); ax.set_ylim(-.35, 3.9)
ax.set_title("3. Trong hình học\ne vuông góc mọi cột của X — đó là phương trình chuẩn", fontsize=9)

fig.savefig("ols_three_views.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

Bảng 1 và 2 vẽ **dữ liệu thật** (cùng hạt giống 3 với mục 7.2, nên $\hat\beta$ trong hình bằng
đúng $\hat\beta$ in ra ở đó).

### 2.2 Cả ba cách nhìn cho cùng phương trình chuẩn

$S(\beta) = \lVert y - X\beta\rVert^2$ là một hàm bậc hai của $\beta$. Đạo hàm theo $\beta$ đặt
bằng 0 cho

$$-2X^\top(y - X\hat\beta) = 0 \quad\Longleftrightarrow\quad X^\top X\hat\beta = X^\top y.$$

**Ý nghĩa:** phần dư $e = y - X\hat\beta$ **không còn tương quan với bất kỳ cột nào** của dữ liệu.
Nếu còn một cột tương quan với $e$, dịch hệ số của cột đó một chút theo hướng tương quan sẽ làm
tổng bình phương giảm tiếp — tức chưa phải cực tiểu. Đó cũng là lý do nghiệm đi qua điểm trung
bình $(\bar x, \bar y)$ khi có hệ số chặn: cột toàn số 1 trực giao với $e$ nghĩa là $\sum e_i = 0$.

| Tính chất của nghiệm | Vì sao | Chỉ đúng khi |
|---|---|---|
| $\sum_i e_i = 0$ | cột toàn 1 trực giao với $e$ | **có hệ số chặn** |
| $\sum_i e_i x_{ij} = 0$ với mọi cột $j$ | chính là phương trình chuẩn | luôn đúng |
| đường đi qua $(\bar x, \bar y)$ | hệ quả của hai dòng trên | có hệ số chặn |

Bỏ hệ số chặn thì phần dư **không** tổng bằng 0 nữa, và $R^2$ tính theo công thức thường có thể **âm**.

---

## 3. Phần ruột — theo đúng hai bản in gốc

Hồi quy tuyến tính **không có một bài báo gốc có hình**. Nguồn gốc của nó là hai bản in, và
phần ruột của mỗi bản được xác định theo quy tắc E.1d của `Dàn ý mẫu`: đoạn phát biểu phương
pháp, mọi hệ phương trình đánh số, mọi ví dụ số tác giả tự làm, và mọi câu tác giả tự đánh giá
kết quả.

| Nguồn | Phần ruột | Trang PDF | Mục của bài này |
|---|---|---|---|
| Legendre 1805, *Nouvelles méthodes pour la détermination des orbites des comètes* | mục LVII–LVIII (lần dùng đầu tiên) · phụ lục *Sur la méthode des moindres quarrés* gồm cả phần kinh tuyến | 76–77 · 84–92 | 3.1–3.4 |
| Gauss 1809, *Theoria motus corporum coelestium*, quyển II mục III | điều 172–187 | 221–238 | 3.5–3.8 |
| Stigler 1981, *Gauss and the Invention of Least Squares* (trọng tài lịch sử) | mục 2, 3 và Bảng 1–2 | 1–8 | 3.9 |

**Về các đoạn trích.** Lớp text của hai bản in là OCR vỡ — chẳng hạn câu quyết định của Legendre
đọc ra thành *"Biais il est possible de les attàioer encore en cherphant le miniaum"*. Mọi đoạn
tiếng Pháp và tiếng Latinh dưới đây vì thế được **chép từ ảnh trang**, giữ nguyên chính tả năm
1805 (*quarrés*, *connoître*, *devroit*), và `tools/kiemtrich.py` **không** đối chiếu được chúng
với PDF. Đoạn tiếng Anh của Stigler cũng vậy: PDF là bản quét, lớp text chỉ có 153 ký tự.

### 3.1 Legendre mục LVII–LVIII — lần đầu phương pháp được dùng

![Mục LVII–LVIII, trang in 64 bản 1805 — ví dụ bằng số của Legendre](<paper_legendre1805_muc_LVII_LVIII_trang64.png>)

**Hình 2 — Trang in 64 của Legendre (1805), mục LVII–LVIII.** A.-M. Legendre, *Nouvelles méthodes
pour la détermination des orbites des comètes*, Firmin Didot, Paris, 1805, **trang in 64, PDF trang
76** — cắt từ `AI algorithms/_papers/1805 - Legendre - Nouvelles methodes pour la determination des orbites des cometes.pdf`.

**Caption gốc.** Không có. Bản in 1805 không đánh số hình và không chú thích trang; đây là một trang
văn bản, không phải một hình.

**Tác giả nói gì.** Nguyên văn trang 64, cuối mục LVII và đầu mục LVIII:

> « Mais il est possible de les atténuer encore en cherchant le *minimum* de la somme des quarrés
> des quantités E′, E″, E‴. LVIII. Pour cela, il faut multiplier tous les termes de E′ par 1.499
> coëfficient de z, tous ceux de E″ par 0.059, tous ceux de E‴ par 4.780, et ajouter les trois
> produits. Opérant ensuite de la même manière par rapport aux coëfficiens de u, on formera les
> deux équations »
>
> (Nhưng còn có thể làm chúng nhỏ hơn nữa bằng cách tìm cực tiểu của tổng bình phương các đại
> lượng E′, E″, E‴. LVIII. Muốn thế, phải nhân mọi số hạng của E′ với 1,499 là hệ số của z, mọi số
> hạng của E″ với 0,059, mọi số hạng của E‴ với 4,780, rồi cộng ba tích lại. Làm tương tự với các
> hệ số của u, ta lập được hai phương trình)

Và kết luận ở trang 65: *« de sorte que la plus grande ne va pas à une minute »* (thành ra sai số
lớn nhất không tới một phút).

**Cách đọc hình.** Bản in dùng vài quy ước khác bây giờ:

- **Dấu chấm là dấu thập phân**, và dấu phẩy trên cao (`2′.248`) là **phút cung**.
- Chữ `o` in nghiêng ở đầu hai dòng `o = 10.148 + z(25.099) − u(0.908)` là **số 0**.
- Ba dòng $E', E'', E'''$ là ba sai số viết theo hai ẩn $z, u$. Hai dòng `o = …` là **phương
  trình chuẩn**; hai dòng cuối là nghiệm $z = -0.4144$, $u = -0.2796$.
- Thủ tục *"nhân mọi số hạng với hệ số của z rồi cộng lại"* **chính là** $\partial/\partial z
  \sum E^2 = 0$ chia 2. Legendre thực hiện đạo hàm bằng lời, trước khi ký hiệu ma trận ra đời.

**Điều hình KHÔNG nói ra.** Trang này không nói **vì sao** chọn bình phương — lý do chỉ xuất hiện
ở phụ lục (mục 2 và 3.2), và lý do xác suất phải đợi Gauss (mục 3.6). Trang cũng không nói con số
$E''' = -0.236$ được làm tròn hay cắt.

Viết các sai số thành $E = a + A\theta$ với $\theta = (z, u)^\top$, cực tiểu $\lVert a + A\theta
\rVert^2$ cho hệ $A^\top A\,\theta = -A^\top a$. Khối mã dưới dựng lại bằng numpy, copy ra chạy được:

```python
import numpy as np

# E = a + X @ theta, với theta = [z, u]
a = np.array([2.248, 0.006, 1.418])
X = np.array([
    [1.499,  2.996],
    [0.059,  3.040],
    [4.780, -1.167],
])

# Tối thiểu hóa ||a + X theta||² tương đương giải X theta ≈ -a.
theta = np.linalg.lstsq(X, -a, rcond=None)[0]
E = a + X @ theta

print("X.T @ a =", X.T @ a)
print("X.T @ X =\n", X.T @ X)
print("z, u =", theta)
print("E =", E)
print("RSS trước/sau =", a @ a, E @ E)
```

Đầu ra thật:

```text
X.T @ a = [10.148146  5.098442]
X.T @ X =
 [[25.098882 -0.907896]
 [-0.907896 19.579505]]
z, u = [-0.41444104 -0.27961439]
E = [ 0.78902816 -0.86847977 -0.23671818]
RSS trước/sau = 7.0642640000000005 1.4328580522919465
```

| Đại lượng | Legendre in (tr.64–65) | Dựng lại |
|---|---|---|
| phương trình chuẩn 1 | $0 = 10.148 + z(25.099) - u(0.908)$ | $0 = 10.148 + 25.099z - 0.908u$ |
| phương trình chuẩn 2 | $0 = 5.098 - z(0.908) + u(19.580)$ | $0 = 5.098 - 0.908z + 19.580u$ |
| $z$, $u$ | −0,4144 · −0,2796 | −0,414441 · −0,279614 |
| $E'$, $E''$, $E'''$ | 0,789 · −0,868 · −0,236 | 0,78903 · −0,86848 · **−0,23672** |

Mười một con số đầu khớp tới số chữ số Legendre in. Con số thứ mười hai khớp nếu Legendre **cắt**
sau ba chữ số (−0,236) chứ không làm tròn (−0,237). Hai sai số kia cho cùng ba chữ số dù cắt hay
làm tròn, nên không đủ để khẳng định quy ước — chỉ nói được là cách đọc "cắt" giải thích được
chênh lệch 0,001.

### 3.2 Phụ lục — phát biểu tổng quát của phương pháp (tr.72–75)

![Trang mở đầu phụ lục Sur la Méthode des moindres quarrés, bản in 1805](<paper_legendre1805_appendice.png>)

**Hình 3 — Trang in 72, mở đầu phụ lục.** Legendre (1805), **trang in 72, PDF trang 84**, cùng file PDF.

![Trang in 73 — quy tắc lập phương trình cực tiểu](<paper_linreg_legendre1805_tr73_phuong_trinh_minimum.png>)

**Hình 4 — Trang in 73: quy tắc lập phương trình cực tiểu.** Legendre (1805), **trang in 73, PDF trang 85**.

![Trang in 74 — loại phương trình có sai số quá lớn, và trung bình cộng](<paper_linreg_legendre1805_tr74_trung_binh_cong.png>)

**Hình 5 — Trang in 74: loại phương trình sai quá lớn; trung bình cộng là trường hợp riêng.** Legendre
(1805), **trang in 74, PDF trang 86**.

![Trang in 75 — trọng tâm](<paper_linreg_legendre1805_tr75_trong_tam.png>)

**Hình 6 — Trang in 75: trọng tâm là trường hợp riêng.** Legendre (1805), **trang in 75, PDF trang 87**.

**Caption gốc.** Không có — bốn trang văn bản.

**Tác giả nói gì.** Ngoài hai câu đã trích ở mục 1.1 và mục 2, phụ lục có **ba** câu mang nội dung
kỹ thuật, nguyên văn:

1. Quy tắc tổng quát, in nghiêng ở tr.73: *« En général, pour former l'équation du minimum par
   rapport à l'une des inconnues, il faut multiplier tous les termes de chaque équation proposée par
   le coëfficient de l'inconnue dans cette équation, pris avec son signe, et faire une somme de ces
   produits. »* (Tổng quát, muốn lập phương trình cực tiểu theo một ẩn, phải nhân mọi số hạng của
   từng phương trình với hệ số của ẩn đó trong phương trình ấy, **giữ nguyên dấu**, rồi cộng các
   tích lại.)
2. Được phép **loại dữ liệu**, tr.74: *« Si parmi ces erreurs il s'en trouve que l'on juge trop
   grandes pour être admises, alors on rejettera les équations qui ont produit ces erreurs, comme
   venant d'expériences trop défectueuses, et on déterminera les inconnues par le moyen des
   équations restantes »* (Nếu trong các sai số có cái bị cho là quá lớn để chấp nhận, thì loại bỏ
   các phương trình sinh ra chúng như đến từ những thí nghiệm quá lỗi, và tìm ẩn bằng các phương
   trình còn lại).
3. Làm tròn theo độ chính xác cần thiết, tr.73–74: *« en n'admettant dans chaque opération que le
   nombre de chiffres entiers ou décimaux que peut exiger le degré d'approximation dont la question
   est susceptible »* (chỉ giữ trong mỗi phép tính số chữ số mà mức gần đúng của bài toán đòi hỏi).

**Cách đọc hình.** Trang 72 có dòng công thức duy nhất $\mathrm E = a + bx + cy + fz + \&c.$: $a, b, c,
f$ là **số đã biết**, $x, y, z$ là **ẩn**. Trang 73 viết tổng bình phương rồi đạo hàm theo $x$:

$$o = \int ab + x\int b^2 + y\int bc + z\int bf + \&c.$$

Dấu $\int$ ở đây **không phải tích phân** — chính Legendre định nghĩa ngay dưới: *« par $\int ab$
on entend la somme des produits semblables $ab + a'b' + a''b'' + \&c.$ »*. Viết lại hiện đại: gom các
hàng $(b, c, f, \ldots)$ thành ma trận $A$, các ẩn thành $\theta$, thì dòng trên là hàng đầu của
$A^\top a + A^\top A\,\theta = 0$.

**Ý nghĩa của dòng công thức đó** — không phải đọc lại nó: cực tiểu đạt được khi **phần dư không
còn "nghiêng" về phía cột nào**. Tổng $\int bE$ đo mức sai số còn đồng biến với cột hệ số $b$; chừng
nào tổng đó khác 0 thì dịch ẩn $x$ theo hướng đó còn giảm được tổng bình phương.

**Điều hình KHÔNG nói ra.** Câu số 2 — loại phương trình sai quá lớn — **không kèm ngưỡng nào**.
Legendre để ngưỡng cho người tính phán đoán. Đó là lỗ hổng mà hơn một thế kỷ sau các phương pháp
chống ngoại lai như Huber mới lấp bằng một quy tắc (mục 9). Và phụ lục **không có lập luận xác suất
nào**: lý do duy nhất Legendre đưa ra là tính "cân bằng" (mục 2) và việc tính được.

### 3.3 Ứng dụng kinh tuyến — bốn hệ đánh số (a)–(d) (tr.76–78)

Phần cuối phụ lục dùng phương pháp cho bài toán đã quyết định **độ dài của mét**: đo chiếc cung kinh
tuyến Dunkerque–Barcelona để suy ra hình dạng Trái Đất.

![Trang in 76 — công thức (a) và bảng bốn cung của Delambre–Méchain](<paper_linreg_legendre1805_tr76_cong_thuc_a_bang_cung.png>)

**Hình 7 — Trang in 76: công thức (a) và bảng số đo.** Legendre (1805), **trang in 76, PDF trang 88**.

![Trang in 77 — hệ (b)](<paper_linreg_legendre1805_tr77_he_b.png>)

**Hình 8 — Trang in 77: hệ (b).** Legendre (1805), **trang in 77, PDF trang 89**.

![Trang in 78 — hệ (c), hệ (d) và kết quả](<paper_linreg_legendre1805_tr78_he_c_d.png>)

**Hình 9 — Trang in 78: hệ (c), hệ (d) và kết quả.** Legendre (1805), **trang in 78, PDF trang 90**.

**Caption gốc.** Tiêu đề in nghiêng ở tr.76: *« Application à la mesure des degrés du méridien »*
(Áp dụng vào việc đo các độ của kinh tuyến). Bảng không có chú thích; ba tiêu đề cột là *« Lieu de
l'observation »*, *« Sa latitude »*, *« Arcs compris exprimés en modules »*, cộng hai cột $L' - L$ và
$L' + L$.

**Tác giả nói gì.** Trang 77 nêu vì sao phải dùng bình phương tối thiểu ở đây: *« Mais comme ces
quatre équations ne peuvent pas être satisfaites toutes à-la-fois, nous supposerons qu'elles ont lieu,
en attribuant une certaine erreur à la latitude de chaque lieu »* (Nhưng vì bốn phương trình này không
thể cùng thoả mãn, ta coi chúng đúng bằng cách gán một sai số nào đó cho vĩ độ từng nơi). Và câu chọn
ẩn trung gian: *« Comme il importe de considérer les erreurs séparément, on regardera comme une nouvelle
inconnue l'erreur E‴, par exemple »* (Vì cần xét riêng từng sai số, ta coi chẳng hạn sai số E‴ là một ẩn mới).

**Cách đọc hình.** Ba ẩn: độ dẹt $\alpha$, hiệu chỉnh độ dài $\zeta$, và sai số vĩ độ ở Evaux $E'''$.
Năm vĩ độ cho bốn cung; mỗi cung cho một phương trình (a); năm sai số vĩ độ $E', \ldots, E^{v}$ nối
bốn phương trình đó thành một bài toán bình phương tối thiểu.

#### Công thức (a) — độ dài một cung trên Trái Đất dẹt

$$L' - L = \frac{S}{28500} + \zeta\cdot\frac{S}{28500} + \alpha\cdot\frac{270}{\pi}\sin(L'-L)\cos(L'+L) \tag{a}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $L, L'$ | vĩ độ hai đầu cung, **độ** | vô hướng |
| $S$ | độ dài cung, **module** | vô hướng |
| $28500$ | độ dài xấp xỉ của một độ vĩ ở 45°, module | hằng số |
| $\zeta$ | hiệu chỉnh nhỏ: $1/D = (1+\zeta)/28500$ | vô hướng, không đơn vị |
| $\alpha$ | độ dẹt | vô hướng, không đơn vị |

**Ý nghĩa.** Nếu Trái Đất tròn, số độ vĩ của một cung chỉ là độ dài chia một hằng số — hạng
$S/28500$. Hai hạng sau là **hai hiệu chỉnh tuyến tính**: $\zeta$ sửa hằng số đó, còn $\alpha$ sửa
theo **vĩ độ**, vì trên Trái Đất dẹt một độ vĩ ở gần cực dài hơn ở gần xích đạo — dấu của
$\cos(L'+L)$ đổi đúng ở vĩ độ trung bình 45°. Cả bài toán hình học phức tạp được **tuyến tính hoá**
thành hai hệ số cần tìm: đó là điều kiện để bình phương tối thiểu áp dụng được.

**Một chỗ bản in không đọc chắc được.** Dòng suy ra (a) in phân số rất nhỏ trước $\alpha D$, trên ảnh
không phân biệt được là ½ hay ³⁄₂. Con số quyết định: $\tfrac32\cdot\tfrac{180}{\pi} = \tfrac{270}{\pi}$
đúng là hệ số in trong (a), và hệ số của $\alpha$ tính theo ³⁄₂ khớp bảng hệ (a) Legendre in — theo ½
thì lệch ba lần:

| Cung | hệ số $\alpha$ in | tính với ³⁄₂ | tính với ½ |
|---|---:|---:|---:|
| Dunkerque–Panthéon | −0,563 | −0,5635 | −0,1878 |
| Panthéon–Evaux | −0,351 | −0,3505 | −0,1168 |
| Evaux–Carcassonne | 0,047 | 0,0470 | 0,0157 |
| Carcassonne–Montjouy | 0,263 | 0,2626 | 0,0875 |

#### Hệ (b), (c), (d) — ba bước khử

| Hệ | Legendre làm gì | Ý nghĩa |
|---|---|---|
| **(b)** tr.77 | viết cả năm sai số qua $E'''$ bằng cách cộng dồn bốn phương trình (a) | bốn phương trình **hiệu** $E_k - E_{k+1}$ chỉ xác định sai số **sai khác một hằng số**; $E'''$ chính là hằng số đó |
| cực tiểu theo $E'''$ | $0 = 5E''' + 0.013123 - \zeta(0.239) - \alpha(1.622)$ | hệ số 5 vì $E'''$ có mặt với hệ số 1 trong cả năm sai số — đây là **trung bình cộng** của mục 2 |
| **(c)** tr.78 | thay $E'''$ trở lại → năm sai số theo $\zeta, \alpha$ | năm phương trình hai ẩn, đã khử hằng số chung |
| **(d)** tr.78 | nhân với hệ số của $\zeta$ rồi cộng; tương tự với $\alpha$ | chính quy tắc tr.73 — hai phương trình chuẩn |

$$0 = 0.020983 + \zeta(62.726) - \alpha(3.830), \qquad 0 = -0.003287 - \zeta(3.830) + \alpha(0.531) \tag{d}$$

**Ý nghĩa của (d).** Hai phương trình là hai điều kiện của quy tắc tr.73: phần dư không còn "nghiêng" về cột $\zeta$, và không còn nghiêng về cột $\alpha$. Phần tử đường chéo của $\alpha$ (0,531) nhỏ hơn của $\zeta$ (62,726) hai bậc: bốn cung chỉ trải 9,67380 độ vĩ (Stigler Bảng 1), nên đổi độ dẹt chỉ đổi rất ít số độ vĩ tính ra. Tức $\alpha$ được bốn cung này xác định **kém hơn hẳn** $\zeta$ — và đó là chỗ kết quả dễ lệch xa giá trị thật (mục 3.4).

Khối mã dưới dựng lại **cả bốn hệ từ bảng số đo**, không dùng hệ số nào Legendre đã tính:

```python
import math
import numpy as np


def dms(d, m, s):
    return d + m / 60 + s / 3600


# năm vĩ độ và bốn cung, chép từ bảng trang in 76 của Legendre 1805
lat = [dms(51, 2, 10.50), dms(48, 50, 49.75), dms(46, 10, 42.50),
       dms(43, 12, 54.40), dms(41, 21, 44.80)]
S = [62472.59, 76145.74, 84424.55, 52749.48]

# công thức (a): mỗi cung cho  E_k - E_{k+1} = hằng + zeta*(S/28500) + alpha*(270/pi) sin(L'-L) cos(L'+L)
C = []
for k in range(4):
    dL, sL = lat[k] - lat[k + 1], lat[k] + lat[k + 1]
    C.append((S[k] / 28500 - dL,
              S[k] / 28500,
              270 / math.pi * math.sin(math.radians(dL)) * math.cos(math.radians(sL))))
C = np.array(C)
print("hệ (a), mỗi hàng [hằng, hệ số zeta, hệ số alpha]:")
print(C.round(6))

# hệ (b): viết năm sai số qua E''' (Evaux)
d = np.array([C[0] + C[1], C[1], 0 * C[0], -C[2], -C[2] - C[3]])
# cực tiểu tổng bình phương theo E''' -> trừ trung bình -> hệ (c)
e = d - d.mean(0)
M, v = e[:, 1:], e[:, 0]
# cực tiểu theo zeta và alpha -> hệ (d)
N, r = M.T @ M, M.T @ v
zeta, alpha = np.linalg.solve(N, -r)
E = v + M @ [zeta, alpha]
print("hệ (d): N =", N.round(4).tolist(), " r =", r.round(6).tolist())
print("alpha = %.6f -> độ dẹt 1/%.2f ; zeta = %.8f ; D = %.2f"
      % (alpha, 1 / alpha, zeta, 28500 / (1 + zeta)))
print("sai số (giây):", (3600 * E).round(2), " trung bình |.| = %.2f" % np.abs(3600 * E).mean())

# ghim độ dẹt 1/320 như Legendre làm ở trang in 79, chỉ còn tìm zeta
a0 = 1 / 320
z0 = -(r[0] + N[0, 1] * a0) / N[0, 0]
E0 = v + M @ [z0, a0]
print("ghim 1/320: zeta = %.7f ; D = %.2f ; sai số (giây):" % (z0, 28500 * (1 - z0)), (3600 * E0).round(2))
print("tổng bình phương (giây²): tự do %.2f ; ghim %.2f" % ((3600 * E) @ (3600 * E), (3600 * E0) @ (3600 * E0)))
```

Đầu ra thật:

```text
hệ (a), mỗi hàng [hằng, hệ số zeta, hệ số alpha]:
[[ 2.923000e-03  2.192021e+00 -5.634800e-01]
 [ 3.100000e-03  2.671780e+00 -3.505430e-01]
 [-1.096000e-03  2.962265e+00  4.702400e-02]
 [-1.808000e-03  1.850859e+00  2.625650e-01]]
hệ (d): N = [[62.7246, -3.8306], [-3.8306, 0.5307]]  r = [0.020983, -0.003288]
alpha = 0.006759 -> độ dẹt 1/147.95 ; zeta = 0.00007826 ; D = 28497.77
sai số (giây): [-0.73  1.84 -1.55  0.42  0.02]  trung bình |.| = 0.91
ghim 1/320: zeta = -0.0001437 ; D = 28504.10 ; sai số (giây): [ 3.06  0.01 -5.83 -0.88  3.64]
tổng bình phương (giây²): tự do 6.48 ; ghim 57.28
```

| Đại lượng | Legendre in (tr.78–79) | Giải (d) bằng **số in** | Dựng lại **từ bảng** |
|---|---|---|---|
| $\alpha$ | 0,00675 | 0,006750 | 0,006759 |
| độ dẹt | 1/148 | 1/148,14 | 1/147,95 |
| $\zeta$ | 0,0000778 | 0,00007765 | 0,00007826 |
| $D$ (module) | 28497,78 | 28497,7872 | 28497,7699 |
| sai số vĩ độ (giây) | −0,73 · 1,83 · −1,55 · 0,42 · 0,03 | — | −0,7327 · 1,8366 · −1,5454 · 0,4221 · 0,0194 |
| trung bình trị tuyệt đối | 0,91 | — | 0,9112 |

Nguồn cột 3–4: `tools/v_linreg.py` mục 3. Chênh lệch nhỏ giữa hai cột cuối đến từ việc Legendre làm
tròn hệ số của (b)–(d) còn ba chữ số — đúng như quy tắc tr.73–74 ông tự đặt. Chênh lệch đó không đổi
kết luận nào ở chữ số Legendre in.

Tác giả chốt ở tr.79: *« La plus grande de toutes ne monte pas à 2″, et la moyenne, sans égard aux
signes, n'est que de 0″91. »* (Lớn nhất không tới 2 giây, và trung bình, không kể dấu, chỉ 0,91 giây.)

### 3.4 Legendre tự bác kết quả của chính mình (tr.78–80)

![Trang in 79 — giải lại khi ghim độ dẹt 1/320](<paper_linreg_legendre1805_tr79_ghim_do_det.png>)

**Hình 10 — Trang in 79: giải lại khi ghim độ dẹt.** Legendre (1805), **trang in 79, PDF trang 91**.

**Caption gốc.** Không có.

**Tác giả nói gì.** Bản trước của chính bài này bỏ hẳn đoạn này, và đó là đoạn dạy nhiều nhất. Nguyên văn tr.78:

> « L'aplatissement déterminé par la longueur du pendule et par quelques phénomènes astronomiques,
> n'est que de 1/320, et le 45ème degré tel qu'on l'a déduit de la comparaison des mesures faites en
> France avec les mesures faites au Pérou, est de 28504.10. C'est sur ce dernier résultat qu'est
> fondée la détermination définitive du mètre : il devroit être diminué d'environ un 4500ème, si on
> s'en tenoit aux seules mesures exécutées en France ; mais l'aplatissement 1/148 trop peu d'accord
> avec celui que l'on connoît par d'autres phénomènes, ne permet pas d'adopter ce dernier parti. »
>
> (Độ dẹt xác định từ chiều dài con lắc và vài hiện tượng thiên văn chỉ là 1/320, và độ vĩ thứ 45
> suy ra khi so số đo ở Pháp với số đo ở Peru là 28504,10. Mét được xác định dứt khoát dựa trên kết
> quả sau này: nếu chỉ dựa vào số đo ở Pháp thì mét phải giảm khoảng một phần 4500; nhưng độ dẹt
> 1/148 **quá không khớp** với độ dẹt biết từ các hiện tượng khác, nên không thể chọn cách đó.)

Rồi tr.79 giải lại với $\alpha$ ghim ở 1/320, ra $\zeta = -0.0001436$, $D = 28504.09$ và năm sai số
3″06 · 0″00 · −5″83 · −0″88 · 3″62, kèm nhận xét: *« Ces erreurs sont plus fortes qu'elles n'étoient
dans le cas du minimum absolu ; la plus grande tombe sur la latitude d'Evaux »* (Các sai số này lớn
hơn trường hợp cực tiểu tuyệt đối; lớn nhất rơi vào vĩ độ Evaux). Và chẩn đoán ở tr.79–80: *« les
anomalies dans les latitudes, qui sans aucun doute ne doivent point être attribuées aux observations,
tiennent vraisemblablement à des attractions locales qui agissent irrégulièrement sur le fil à
plomb »* (các dị thường về vĩ độ, chắc chắn không phải do quan sát, nhiều khả năng là do **lực hút
cục bộ** tác động thất thường lên dây dọi).

**Cách đọc hình.** Hệ (e) ở đầu trang là hệ (c) với $\alpha$ đã thay bằng số, nên chỉ còn một ẩn $\zeta$.

**Điều hình KHÔNG nói ra — và phép dựng lại tìm ra.**

| Kiểm | Kết quả |
|---|---|
| độ dẹt ngầm chứa trong từng hằng số của hệ (e) | $E'$: 1/320,0 · $E''$: 1/321,4 · $E'''$: 1/319,8 · $E^{iv}$: 1/319,9 · $E^{v}$: **1/241,4** |
| ghim 1/320 và giải lại từ bảng | $\zeta = -0.0001437$, $D = 28504.10$; sai số 3,06 · 0,01 · −5,83 · −0,88 · 3,64 |
| tổng bình phương sai số (giây²) | tự do **6,48** · ghim **57,28** |
| "mét phải giảm khoảng 1/4500" | $28504{,}10 / 28497{,}78$ ứng với **1/4509,1** |

Bốn trong năm hằng số của hệ (e) cùng chỉ về 1/320; hằng số của $E^{v}$ lệch — một chỗ in hoặc tính
lệch nhỏ, làm sai số cuối chênh 0,02 giây (3,62 in so với 3,64 dựng lại), không đổi kết luận.

**Vì sao đoạn này là đoạn dạy nhiều nhất.** Nghiệm bình phương tối thiểu **khớp số liệu tốt hơn** theo đúng tiêu chuẩn của nó (6,48 so với 57,28), vậy mà chính người phát minh ra phương
pháp **từ chối** nó. Lý do: tham số tìm được mâu thuẫn với bằng chứng độc lập, và ông tìm ra nguyên
nhân nằm **ngoài mô hình** — lực hút cục bộ làm lệch dây dọi, một sai số **có hệ thống** chứ không
phải ngẫu nhiên. Bình phương tối thiểu không có cách nào tự phát hiện điều đó: nó chỉ làm sai số nhỏ
nhất, không kiểm mô hình đúng hay sai. Mục 9 quay lại đúng bài học này với số đo hiện đại.

### 3.5 Gauss điều 175–177 — từ tiên đề trung bình cộng tới luật sai số

Legendre cho **tiêu chuẩn** và **cách tính**, không cho lý do xác suất. Gauss 1809 đi ngược
chiều: bắt đầu từ xác suất của sai số, rồi **suy ra** bình phương tối thiểu.

![Gauss điều 175 — tích Ω](<paper_linreg_gauss1809_dieu175_tich_omega.png>)

**Hình 11 — Gauss (1809), điều 175, trang in 210.** C. F. Gauss, *Theoria motus corporum coelestium
in sectionibus conicis solem ambientium*, Perthes & Besser, Hamburg, 1809, quyển II mục III, **trang in
210, PDF trang 226** — cắt từ `AI algorithms/_papers/1809 - Gauss - Theoria motus corporum coelestium.pdf`.

![Gauss điều 177 — cực đại của Ω](<paper_linreg_gauss1809_dieu177_cuc_dai.png>)

**Hình 12 — Gauss (1809), điều 177, trang in 211.** **PDF trang 227**, cùng file.

![Gauss điều 177 — suy ra luật sai số](<paper_linreg_gauss1809_dieu177_luat_sai_so.png>)

**Hình 13 — Gauss (1809), điều 177, trang in 212: suy ra luật sai số.** **PDF trang 228**, cùng file.

**Caption gốc.** Không có — trang văn bản, các điều được đánh số bằng chữ số Ả Rập ở giữa dòng.

**Tác giả nói gì.** Ba bước, nguyên văn tiếng Latinh:

1. Điều 175, tr.210 — quan sát độc lập thì xác suất cùng xảy ra là **tích**: *« Quamobrem
   quandoquidem omnes obseruationes tamquam euentus ab inuicem independentes spectare licet, productum
   φ(M−V).φ(M′−V′).φ(M″−V″) etc. = Ω exprimet exspectationem seu probabilitatem, omnes istos valores
   simul ex obseruationibus prodituros esse. »* (Vì mọi quan sát được coi là các biến cố độc lập với
   nhau, tích … = Ω biểu thị xác suất để mọi giá trị ấy **cùng** xuất hiện từ các quan sát.)
2. Điều 177, tr.211 — hệ ẩn khả dĩ nhất là hệ làm Ω **cực đại**: *« systema maxime probabile valorum
   quantitatum p, q, r, s etc. id fore, in quo Ω valorem maximum obtineat »* (hệ giá trị khả dĩ nhất của
   p, q, r, s… là hệ làm Ω đạt giá trị lớn nhất).
3. Điều 177, tr.212 — **tiên đề trung bình cộng** quyết định dạng của φ: *« Axiomatis scilicet loco
   haberi solet hypothesis, si quae quantitas per plures obseruationes immediatas, sub aequalibus
   circumstantiis aequalique cura institutas, determinata fuerit, medium arithmeticum inter omnes
   valores obseruatos exhibere valorem maxime probabilem »* (Người ta thường coi như tiên đề giả thuyết
   rằng: nếu một đại lượng được xác định qua nhiều quan sát trực tiếp, trong cùng điều kiện và cùng sự
   cẩn thận, thì **trung bình cộng** của mọi giá trị quan sát là giá trị khả dĩ nhất). Từ đó ông ra

$$\varphi\Delta = \frac{h}{\sqrt\pi}\,e^{-hh\Delta\Delta}.$$

| Ký hiệu | Nghĩa | Kiểu |
|---|---|---|
| $\Delta$ | sai số của một quan sát | vô hướng |
| $\varphi\Delta$ | mật độ xác suất của sai số đó (Gauss viết không có ngoặc) | hàm |
| $h$ | *mensura praecisionis* — độ chính xác | vô hướng $> 0$ |
| $hh$, $\Delta\Delta$ | cách viết năm 1809 của $h^2$, $\Delta^2$ | — |
| $\pi$ | Gauss định nghĩa là *"semicircumferentiam circuli cuius radius 1"* | hằng số |

**Ý nghĩa.** Công thức không phải một giả định rơi từ trời xuống mà là **cái giá của tiên đề trung
bình cộng**: nếu muốn trung bình cộng luôn là ước lượng tốt nhất, thì sai số **bắt buộc** phải phân
phối hình chuông. Đảo lại, sai số có luật khác thì trung bình cộng không còn tối ưu. Và $h$ ở vị trí
của $1/(\sigma\sqrt2)$: $h$ gấp đôi nghĩa là sai số điển hình còn một nửa.

**Cách đọc hình.** Hình 12: dòng $\frac{d\Omega}{dp} = 0$, $\frac{d\Omega}{dq} = 0$… là điều kiện cực đại;
Gauss đặt $V - M = v$ và $\frac{d\varphi\Delta}{\varphi\Delta\,d\Delta} = \varphi'\Delta$. Hình 13: bước quyết
định là dòng *"$\frac{\varphi'\Delta}{\Delta}$ quantitatem constantem"* — tỉ số đó phải là hằng số, và lấy
tích phân ra ngay luật hình chuông. Tích phân $\int e^{-hh\Delta\Delta}d\Delta = \sqrt\pi/h$ Gauss ghi công cho
*"ill. Laplace"*.

**Điều hình KHÔNG nói ra.** Gauss **không chứng minh** tiên đề trung bình cộng; ông nói rõ nó chỉ được
*"thường coi như tiên đề"*. Toàn bộ lý do xác suất của bình phương tối thiểu vì thế đứng trên một giả
thuyết về cách người đo làm việc. Phép đo nhỏ ở `tools/v_linreg.py` mục 4 cho thấy lựa chọn luật sai
số đổi ước lượng thật: trên 11 quan sát, hợp lý cực đại theo luật Gauss ra **5,04** (trung bình cộng
5,040641), còn theo luật Laplace $e^{-|\Delta|}$ ra **5,21** (trung vị 5,209800).

### 3.6 Gauss điều 179–180 — nguyên lý, trọng số, và phương trình chuẩn

![Gauss điều 179 — nguyên lý bình phương tối thiểu](<paper_linreg_gauss1809_dieu179_nguyen_ly.png>)

**Hình 14 — Gauss (1809), điều 179, trang in 213.** **PDF trang 229**, cùng file.

![Gauss điều 179–180 — trọng số và phương trình chuẩn](<paper_linreg_gauss1809_dieu179_180_phuong_trinh_chuan.png>)

**Hình 15 — Gauss (1809), điều 179–180, trang in 214.** **PDF trang 230**, cùng file.

**Caption gốc.** Không có.

**Tác giả nói gì.** Nguyên lý, in nghiêng ở điều 179 tr.213: *« Systema itaque maxime probabile
valorum incognitarum p, q, r, s etc. id erit, in quo quadrata differentiarum inter functionum V, V′,
V″ etc. valores obseruatos et computatos summam minimam efficiunt, siquidem in omnibus obseruationibus
idem praecisionis gradus praesumendus est. »* (Vậy hệ giá trị khả dĩ nhất của các ẩn là hệ làm **tổng
bình phương hiệu giữa giá trị quan sát và giá trị tính** của các hàm V, V′, V″… **nhỏ nhất**, với điều
kiện mọi quan sát được coi là có cùng độ chính xác.) Ngay sau đó Gauss ví nguyên lý này với chính tiên
đề ban đầu: *« quo medium arithmeticum inter plures valores obseruatos eiusdem quantitatis tamquam
valor maxime probabilis adoptatur »* (như việc lấy trung bình cộng của nhiều giá trị quan sát cùng một
đại lượng làm giá trị khả dĩ nhất).

Quan sát **không** cùng độ chính xác, cũng điều 179 tr.214: tổng phải là *« summa quadratorum
differentiarum inter valores reuera obseruatos et computatos per numeros qui praecisionis gradum
metiuntur multiplicatarum »* — tổng bình phương hiệu **đã nhân với các số đo độ chính xác**. Tức mỗi
dòng nhân với $h$ **trước khi** bình phương, và trọng số là $h^2$.

**Ý nghĩa.** Từ $\Omega = h^\mu\pi^{-\frac12\mu}e^{-hh(vv + v'v' + v''v'' + \cdots)}$ ở đầu điều 179: làm
Ω lớn nhất **đồng nghĩa** làm tổng bình phương nhỏ nhất, vì tổng đó nằm trong số mũ với dấu trừ.
Bình phương tối thiểu vì thế là **hợp lý cực đại dưới sai số chuẩn** — không phải một lựa chọn tiện
tính như Legendre trình bày, mà là hệ quả của một mô hình xác suất cụ thể.

**Phương trình chuẩn, điều 180 tr.214.** Khi các hàm tuyến tính, $v = -m + ap + bq + cr + ds + \cdots$,
Gauss đặt

$$av + a'v' + a''v'' + \cdots = P,\quad bv + b'v' + b''v'' + \cdots = Q,\quad \ldots$$

và hệ cần giải là $P = 0$, $Q = 0$, $R = 0$, $S = 0$. **Ý nghĩa:** $P$ đo mức phần dư còn đồng biến
với cột hệ số của ẩn $p$ — **cùng điều kiện** Legendre viết bằng $\int ab$ bốn năm trước, giờ có tên.

**Cách đọc hình.** Hình 14: số mũ của Ω chứa $vv + v'v' + v''v''$ — nhìn thấy ngay vì sao cực đại
tích biến thành cực tiểu tổng. Hình 15: bốn dòng $V - M = v = -m + ap + bq + cr + ds + $etc. là bảng
dữ liệu hiện đại $X$ viết theo hàng; bốn dòng $P, Q, R, S$ là $X^\top v$.

**Điều hình KHÔNG nói ra.** Gauss giả định các hàm **tuyến tính** hoặc đã tuyến tính hoá quanh một
nghiệm gần đúng; điều 180 tr.215 nói cách làm với hàm phi tuyến là lặp lại phép tính quanh nghiệm mới
— thứ ngày nay gọi là **Gauss–Newton**.

### 3.7 Gauss điều 182–184 — độ chính xác của từng ẩn, và ví dụ số duy nhất

Điều 182–183 chỉ ra: khử các ẩn lần lượt, **hệ số đứng trước ẩn cuối cùng** cho biết ẩn đó được xác
định chính xác tới đâu. Viết hiện đại, độ chính xác của ẩn thứ $j$ (so với độ chính xác 1 của mỗi
quan sát) là $1/\sqrt{(N^{-1})_{jj}}$ với $N$ là ma trận của phương trình chuẩn. Điều 184 minh hoạ
bằng số.

![Gauss điều 184 — ví dụ số](<paper_linreg_gauss1809_dieu184_vi_du.png>)

**Hình 16 — Gauss (1809), điều 184, trang in 219.** **PDF trang 235**, cùng file.

![Gauss điều 184 — độ chính xác](<paper_linreg_gauss1809_dieu184_do_chinh_xac.png>)

**Hình 17 — Gauss (1809), điều 184, trang in 220: độ chính xác của ba ẩn.** **PDF trang 236**, cùng file.

**Caption gốc.** Không có.

**Tác giả nói gì.** Nguyên văn điều 184: *« per quartam vero, cui praecisio dimidia tantum tribuenda
est, prodiisse −2p + 6q + 6r = 28 Loco aequationis vltimae itaque hanc substituemus −p + 3q + 3r = 14
hancque ex obseruatione prioribus praecisione aequali prouenisse supponemus. »* (còn từ phương trình thứ
tư, vốn chỉ được coi là có **một nửa độ chính xác**, ta được −2p + 6q + 6r = 28. Thay phương trình cuối
bằng −p + 3q + 3r = 14, và coi nó như đến từ một quan sát cùng độ chính xác với các phương trình trước.)

**Cách đọc hình.** Bốn dòng đầu là dữ liệu; dòng thay thế là **phép chia cho 2**; ba dòng $P, Q, R$
là phương trình chuẩn; ba dòng *"19899 p = …"* là kết quả khử, viết sao cho đọc được cả nghiệm lẫn
hàng của $N^{-1}$; ba dòng *"pro p… pro q… pro r…"* ở Hình 17 là độ chính xác.

**Vì sao "một nửa độ chính xác" là phép chia cho 2, không phải trọng số 1/2.** Độ chính xác $h$ tỉ lệ
$1/\sigma$; trọng số trong tổng bình phương là $h^2$. Một nửa $h$ là trọng số $1/4$ — đúng bằng việc
chia phương trình cho 2 rồi bình phương. Khối mã dưới thử cả hai:

```python
import numpy as np

# bốn phương trình của Gauss 1809 điều 184; phương trình thứ tư chỉ có "praecisio dimidia"
G = np.array([[1, -1, 2], [3, 2, -5], [4, 1, 4], [-2, 6, 6]], float)
g = np.array([3, 5, 21, 28], float)

for w4 in (0.5, 0.25):
    w = np.array([1, 1, 1, w4])
    N = G.T @ np.diag(w) @ G
    rhs = G.T @ (w * g)
    print("trọng số %.2f: N =" % w4, N.tolist(), " vế phải =", rhs.tolist())

w = np.array([1, 1, 1, 0.25])             # độ chính xác một nửa = sigma gấp đôi = trọng số 1/4
N = G.T @ np.diag(w) @ G
rhs = G.T @ (w * g)
p, q, r = np.linalg.solve(N, rhs)
Ninv = np.linalg.inv(N)
print("p, q, r = %.6f %.6f %.6f" % (p, q, r))
print("det N = %.1f" % np.linalg.det(N))
print("19899·(N⁻¹) hàng p =", (19899 * Ninv[0]).round(6))
print("  737·(N⁻¹) hàng q =", (737 * Ninv[1]).round(6))
print("39798·(N⁻¹) hàng r =", (39798 * Ninv[2]).round(6))
print("độ chính xác 1/sqrt(diag N⁻¹) = %.4f %.4f %.4f" % tuple(1 / np.sqrt(np.diag(Ninv))))
```

Đầu ra thật:

```text
trọng số 0.50: N = [[28.0, 3.0, -3.0], [3.0, 24.0, 10.0], [-3.0, 10.0, 63.0]]  vế phải = [74.0, 112.0, 149.0]
trọng số 0.25: N = [[27.0, 6.0, 0.0], [6.0, 15.0, 1.0], [0.0, 1.0, 54.0]]  vế phải = [88.0, 70.0, 107.0]
p, q, r = 2.470174 3.550882 1.915724
det N = 19899.0
19899·(N⁻¹) hàng p = [ 809. -324.    6.]
  737·(N⁻¹) hàng q = [-12.  54.  -1.]
39798·(N⁻¹) hàng r = [ 12. -54. 738.]
độ chính xác 1/sqrt(diag N⁻¹) = 4.9595 3.6943 7.3435
```

| Đại lượng | Gauss in (tr.219–220) | Dựng lại | Khớp |
|---|---|---|---|
| $P$ · $Q$ · $R$ | $27p + 6q - 88$ · $6p + 15q + r - 70$ · $q + 54r - 107$ | trọng số 1/4: đúng từng hệ số | **có** |
| hàng p của phép khử | $19899p = 49154 + 809P - 324Q + 6R$ | $809 \cdot -324 \cdot 6$ | **có** |
| hàng q | $737q = 2617 - 12P + 54Q - R$ | $-12 \cdot 54 \cdot -1$ | **có** |
| hàng r | $39798r = 76242 + 12P - 54Q + \mathbf{1473}R$ | $12 \cdot -54 \cdot \mathbf{738}$ | **không** |
| $p$, $q$, $r$ | 2,470 · 3,551 · 1,916 | 2,470174 · 3,550882 · 1,915724 | **có** |
| độ chính xác của $p$, $q$ | $\sqrt{19899/809} = 4{,}96$ · $\sqrt{737/54} = 3{,}69$ | 4,9595 · 3,6943 | **có** |
| độ chính xác của $r$ | $\sqrt{13266/491} = \mathbf{5{,}20}$ | **7,3435** | **không** |

**Một hệ số in sai trong ví dụ số duy nhất của Gauss.** Hàng r của phép khử phải có hệ số **738**
đứng trước $R$; bản in ghi 1473. Kiểm bằng phân số chính xác: phần phụ đại số của $N_{rr}$ là
$27\cdot15 - 6\cdot6 = 369$, định thức 19899, nên $(N^{-1})_{rr} = 369/19899 = 738/39798$. Hai hàng p và
q khớp tuyệt đối, và hai số hạng $12P$, $-54Q$ của chính hàng r cũng khớp — chỉ riêng đường chéo sai.
Hệ quả: **nghiệm $r = 1{,}916$ vẫn đúng**, nhưng độ chính xác của $r$ là **7,34 chứ không phải 5,20**.
Phân số 13266/491 mà Gauss in rút gọn từ đúng $39798/1473$, nên con số 5,20 là hệ quả nhất quán của
hệ số in sai, không phải một lỗi thứ hai. Thứ hạng giữa ba ẩn **không đổi** — ngay trong bản in, $r$ (5,20) đã chính xác hơn $p$ (4,96) và $q$ (3,69) — nhưng khoảng cách lớn hơn hẳn: 7,3435 so với 4,9595.

**Điều hình KHÔNG nói ra.** Gauss không nói cách chọn trọng số trong thực tế — "một nửa độ chính
xác" được cho sẵn. Không có trọng số, nghiệm đổi: bỏ phép chia cho 2 thì $p = 2{,}465273$,
$q = 3{,}563718$, $r = 1{,}918560$ (`tools/v_linreg.py` mục 5).

### 3.8 Gauss điều 186 — vì sao không dùng trị tuyệt đối, và câu về Legendre

![Gauss điều 186 — Legendre và Laplace](<paper_linreg_gauss1809_dieu186_legendre_laplace.png>)

**Hình 18 — Gauss (1809), điều 186, trang in 221.** **PDF trang 237**, cùng file.

**Caption gốc.** Không có.

**Tác giả nói gì.** Ba câu, nguyên văn:

1. Mọi luỹ thừa chẵn đều cho một nguyên lý, bình phương chỉ là cái đơn giản nhất: *« Sed ex omnibus
   his principiis nostrum simplicissimum est, dum in reliquis ad calculos complicatissimos
   deferremur. »* (Nhưng trong mọi nguyên lý ấy, nguyên lý của [Gauss] là đơn giản nhất, còn với
   các nguyên lý khác ta sẽ bị dẫn tới những phép tính cực kỳ phức tạp.)
2. Câu tranh chấp quyền tác giả: *« Ceterum principium nostrum, quo iam inde ab anno 1795 vsi sumus,
   nuper etiam a clar. Legendre in opere Nouvelles methodes pour la determination des orbites des cometes, Paris 1806 prolatum est »* (Vả lại, nguyên lý của [Gauss], **đã dùng từ năm 1795**, gần
   đây cũng được ngài Legendre trình bày trong tác phẩm Nouvelles méthodes…, **Paris 1806**).
3. Nguyên lý của Laplace và Boscovich — tổng **trị tuyệt đối** nhỏ nhất, kèm điều kiện tổng sai lệch
   bằng 0: *« systema valorum incognitarum, quod ex hoc solo principio erutum sit, necessario \*) tot aequationibus e propositarum numero exacte satisfacere debere, quot sint incognitae »* (hệ giá trị ẩn
   suy ra chỉ từ nguyên lý này **nhất thiết phải thoả mãn đúng tuyệt đối** số phương trình **bằng số
   ẩn**); và với điều kiện thêm của Laplace thì *« multitudo aequationum exacte repraesentatarum
   vnitate minor fiat quam multitudo quantitatum incognitarum »* (số phương trình được thoả mãn đúng
   **ít hơn số ẩn một đơn vị**).

**Cách đọc hình.** Đoạn trên của trang là phần cuối lập luận "luỹ thừa chẵn"; đoạn giữa là câu về
Legendre; đoạn dưới cùng chú thích chân trang *"Casibus specialibus exceptis"* là phần về Laplace.

**Điều hình KHÔNG nói ra — ba điều phép đo và đối chiếu tìm ra.**

| Kiểm | Kết quả |
|---|---|
| năm xuất bản của Legendre Gauss ghi | **1806** — nhưng trang cuối phụ lục Legendre (tr.80, PDF 92) ký *« Paris, le 15 ventôse an 13. 6 mars 1805 »* |
| L1 tự do trên 25 điểm của mục 7.2 (hai ẩn) | **2** phần dư bằng 0 — đúng *"bằng số ẩn"* |
| L1 cộng điều kiện tổng phần dư bằng 0 | **1** phần dư bằng 0, tổng phần dư $1{,}04\times10^{-14}$ — đúng *"ít hơn một đơn vị"* |
| OLS trên cùng dữ liệu | **0** phần dư bằng 0; tổng trị tuyệt đối 39,129625 so với 38,888514 của L1 |

Nguồn ba dòng cuối: `tools/v_linreg.py` mục 7. Dòng cuối nói điều người mới hay bỏ qua: L1 và bình
phương tối thiểu **tối ưu hai thứ khác nhau**, nên mỗi cái thắng theo thước đo của chính nó. Tính chất
"đi qua đúng $p$ điểm" của L1 là lý do nó bền trước ngoại lai: vài điểm lệch nằm ngoài tập $p$ điểm đó
thì không kéo được nghiệm.

### 3.9 Stigler 1981 — trọng tài của vụ tranh chấp, trên chính bốn cung của Legendre

Câu *"Gauss dùng từ năm 1795"* ở mục 3.8 là câu duy nhất gây tranh chấp. Stephen Stigler kiểm nó bằng
một cách lạ: dựng lại **lá thư năm 1799** trong đó Gauss báo đã dùng *"phương pháp của tôi"* trên đúng
bốn cung kinh tuyến Pháp mà Legendre dùng ở mục 3.3.

![Stigler 1981 Bảng 1](<paper_linreg_stigler1981_bang1.png>)

**Hình 19 — Bảng 1 của Stigler (1981), trang 467.** S. M. Stigler, "Gauss and the Invention of Least
Squares", *The Annals of Statistics* 9(3):465–474, **trang 467, PDF trang 3** — cắt từ
`AI algorithms/_papers/1981 - Stigler - Gauss and the Invention of Least Squares.pdf`.

![Stigler 1981 Hình 1–2](<paper_linreg_stigler1981_hinh1_2.png>)

**Hình 20 — Hình 1 và Hình 2 của Stigler (1981), trang 468.** **PDF trang 4**, cùng file.

![Stigler 1981 Bảng 2](<paper_linreg_stigler1981_bang2.png>)

**Hình 21 — Bảng 2 của Stigler (1981), trang 470.** **PDF trang 6**, cùng file.

**Caption gốc.** Bảng 1: *"French arc measurements, from Allgemeine Geographische Ephemeriden, 4, 1799,
page xxxv. The number 76545.74 is a misprint; the correct number is 76145.74."* (Số đo cung ở Pháp, từ
AGE tập 4, 1799, trang xxxv. Số 76545.74 là in sai; số đúng là 76145.74.) Hình 1: *"The French meridian
arc, through Dunkirk (D), the Pantheon (P) in Paris, Evaux (E), Carcassone (C), and Barcelona (B)."*
Hình 2: *"A meridian quadrant, from the equator (E) to the north pole (N), showing an arc segment of d
degrees latitude and length S modules, centered at latitude L."* Bảng 2: *"The results of several
first-order fits, compared with Gauss's results."*

**Tác giả nói gì.** Kết luận của Stigler **ngược với cách chuyện thường được kể**. Tóm tắt, tr.465:
*"It is argued (though not conclusively) that Gauss probably possessed the method well before Legendre,
but that he was unsuccessful in communicating it to his contemporaries."* (Bài lập luận — dù không dứt
khoát — rằng Gauss **có lẽ đã nắm phương pháp từ lâu trước Legendre**, nhưng không truyền đạt được nó
cho người cùng thời.) Về bằng chứng kinh tuyến, tr.471: *"Examination of the French arc data must, I
feel, be taken as supportive of the view that Gauss treated these data with least squares in 1799, more
than five years before Legendre published on the subject. The case is not unchallengable, but
considering the complexity of the problem, it is as strong as we have a right to expect."* Và câu kết,
tr.472: *"Gauss may well have been the first of these, but he was no Henry Ford of statistics. If there
was any single scientist who first put the method within the reach of the common man, it was Legendre."*

**Cách đọc hình.** Hình 2 là hình học của công thức (a) ở mục 3.3: cung dài $S$, chắn $d$ độ vĩ, đặt
giữa vĩ độ $L$. Bảng 2 đặt kết quả của Gauss (1/187) cạnh mọi cách khớp bậc một — tất cả nằm trong
khoảng 141–165, **không cách nào ra 187**.

**Điều hình KHÔNG nói ra — phép dựng lại.** Công thức (3.1) tr.469 của Stigler, $a = S/d = z + y\sin^2L$,
**chính là một hồi quy tuyến tính hai hệ số**:

```python
import numpy as np

# Bảng 1 trang 467 của Stigler 1981: độ dài cung S (module), số độ vĩ d, vĩ độ điểm giữa L
S = np.array([62472.59, 76145.74, 84424.55, 52749.48])
d = np.array([2.18910, 2.66868, 2.96336, 1.85266])
L = np.radians([49 + 56 / 60 + 30 / 3600, 47 + 30 / 60 + 46 / 3600,
                44 + 41 / 60 + 48 / 3600, 42 + 17 / 60 + 20 / 3600])

for nhan, S2 in (("số liệu đã sửa", 76145.74), ("số liệu in sai", 76545.74)):
    SS = S.copy()
    SS[1] = S2
    # công thức (3.1): độ dài một độ a = S/d = z + y sin²L — một hồi quy tuyến tính hai hệ số
    z, y = np.linalg.lstsq(np.c_[np.ones(4), np.sin(L) ** 2], SS / d, rcond=None)[0]
    print("%s: z = %.5f  y = %.7f  1/độ dẹt = 3z/y + 3/2 = %.2f  góc phần tư = 90(z + y/2) = %.2f"
          % (nhan, z, y, 3 * z / y + 1.5, 90 * (z + y / 2)))
```

Đầu ra thật:

```text
số liệu đã sửa: z = 28227.16205  y = 541.2639335  1/độ dẹt = 3z/y + 3/2 = 157.95  góc phần tư = 90(z + y/2) = 2564801.46
số liệu in sai: z = 28074.82697  y = 906.7901274  1/độ dẹt = 3z/y + 3/2 = 94.38  góc phần tư = 90(z + y/2) = 2567539.98
```

| | Stigler in (tr.469–470) | Dựng lại |
|---|---|---|
| số liệu đã sửa: $z$ · 1/độ dẹt · góc phần tư | 28227,16205 · 157,95 · 2564801,46 | 28227,16205 · 157,95 · 2564801,46 |
| số liệu in sai: $z$ · 1/độ dẹt · góc phần tư | 28074,82697 · 94,38 · 2567539,98 | 28074,82697 · 94,38 · 2567539,98 |
| $y$ (đã sửa) | 541,2639**353** | 541,2639**335** |
| cột $d$ của Bảng 1, so với vĩ độ Legendre | 2,18910 · 2,66868 · 2,96336 · 1,85266 | 2,18910 · 2,66868 · 2,96336 · 1,8526**7** |

Khớp tới chữ số in, trừ chữ số thứ chín của $y$ — chênh $1{,}8\times10^{-6}$, cỡ sai số làm tròn.

**Ba nguồn, một bộ số đo, ba độ dẹt khác nhau** — và đây là bài học về **mô hình**, không về thuật toán:

| Ai | Cách viết mô hình | 1/độ dẹt |
|---|---|---:|
| Legendre 1805 | sai số nằm ở **vĩ độ**, ẩn $\alpha, \zeta, E'''$ | 147,95 |
| Stigler 1981, (3.1) | sai số nằm ở **độ dài một độ**, không trọng số | 157,95 |
| Gauss 1799 (Stigler tr.466) | không rõ; Stigler phỏng đoán khai triển bậc hai | 187 |

Cùng bình phương tối thiểu, cùng bốn cung, **đặt sai số ở đâu** là đổi đáp án. Stigler giải thích vì
sao ở tr.469: các quan hệ *"are nonlinear, … there are many ways of converting the problem into a linear
least squares problem, and … all are nearly singular with respect to the ellipticity (and thus very
sensitive to round off error and the precise mode of calculation employed)"*.

---

## 4. Công thức và ký hiệu hiện đại

Mục 3 giữ ký hiệu của từng tác giả. Từ đây trở đi dùng ký hiệu hiện đại ở bảng thứ nhất của mục 0.3.
Bảng đổi ký hiệu, để đọc hai phần cạnh nhau:

| Hiện đại | Legendre 1805 | Gauss 1809 |
|---|---|---|
| hệ số cần tìm $\beta$ | ẩn $x, y, z, u$ | ẩn $p, q, r, s$ |
| hàng dữ liệu $x_i$ | hệ số $b, c, f$ của một phương trình | hệ số $a, b, c, d$ của một phương trình |
| đáp án $y_i$ | $-a$ | $m$ |
| phần dư $e_i = y_i - x_i^\top\beta$ | $-E$ | $-v$ |
| $X^\top X\hat\beta = X^\top y$ | $o = \int ab + x\int b^2 + \ldots$ | $P = 0, Q = 0, \ldots$ |
| $1/\sigma^2$ làm trọng số | — | $h^2$ |

### 4.1 Từ bảng bốn máy sang ma trận

$$
X=\begin{pmatrix}1&1\\1&2\\1&3\\1&4\end{pmatrix},\qquad
\beta=\begin{pmatrix}\beta_0\\\beta_1\end{pmatrix},\qquad
y=\begin{pmatrix}3\\4\\5\\8\end{pmatrix}.
$$

Cột đầu toàn 1 tạo hệ số chặn; cột sau là tuổi máy. $X\beta$ tạo **cùng lúc** bốn dự đoán. Ma trận
không phải một thuật toán mới — nó gói bốn phương trình giống nhau vào một dòng, đúng việc Legendre
làm bằng dấu $\int$.

### 4.2 Bài toán và nghiệm

$$\hat\beta = \arg\min_\beta \lVert y - X\beta\rVert_2^2 = (X^\top X)^{-1}X^\top y$$

**Ý nghĩa.** Vế trái là **định nghĩa** — tiêu chuẩn của Legendre. Vế phải là **nghiệm** — chỉ tồn tại
khi các cột của $X$ độc lập tuyến tính. Công thức có nghịch đảo dùng để **chứng minh**; trên máy thì
không bao giờ tính nghịch đảo (mục 6).

### 4.3 Hàm mục tiêu là một cái bát, và vì sao chỉ có một đáy

$$S(\beta) = \beta^\top(X^\top X)\beta - 2y^\top X\beta + y^\top y$$

**Ý nghĩa.** $S$ là một **dạng toàn phương** với ma trận $X^\top X$ nửa xác định dương, nên:

- $S$ **lồi** luôn luôn — không có đáy giả.
- $S$ **lồi ngặt** khi và chỉ khi $X$ đủ hạng cột; khi đó đáy là **duy nhất**.
- $X$ không đủ hạng (hai cột trùng nhau, hoặc $p > n$) thì đáy là **một đường hay một mặt**: vô số
  $\beta$ cho cùng một $\hat y$. Đó là chỗ các biến thể có phạt như Ridge bước vào — cộng
  $\lambda\lVert\beta\rVert_2^2$ biến đáy phẳng thành đáy nhọn.

### 4.4 Sau khi có hệ số

$$\hat y = X\hat\beta = Hy,\qquad e = (I-H)y,\qquad \hat\sigma^2 = \frac{e^\top e}{n-p},\qquad H = X(X^\top X)^{-1}X^\top$$

**Ý nghĩa của $H$.** $H$ **chiếu** $y$ xuống không gian các dự đoán đạt được. Hai tính chất: $H^2 = H$
(chiếu hai lần như chiếu một lần), $\operatorname{tr}(H) = p$. Đường chéo $h_{ii}$ là **leverage** —
điểm $i$ tự kéo dự đoán của chính nó bao nhiêu; ở mục 0.5 hai máy ở hai đầu có 0,7, hai máy giữa có 0,3.

**Vì sao chia cho $n - p$.** Mô hình đã dùng $p$ bậc tự do để chọn $p$ hệ số, nên phần dư chỉ còn
$n - p$. Dưới giả định $\mathbb E[\varepsilon\mid X] = 0$ và $\operatorname{Var}(\varepsilon\mid X) =
\sigma^2 I$:

$$\mathbb E[e^\top e\mid X] = \sigma^2\operatorname{tr}(I - H) = \sigma^2(n-p).$$

Chia cho $n$ thì chệch xuống theo hệ số $(n-p)/n$ — ở ví dụ 25 điểm của mục 7.2 là 0,9200, tức **8,0 %**.

$$\operatorname{Var}(\hat\beta\mid X) = \sigma^2(X^\top X)^{-1},\qquad
\operatorname{SE}(\hat\beta_j) = \hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}$$

**Ý nghĩa.** Đây chính là **độ chính xác của Gauss** viết lại: Gauss tính $1/\sqrt{(N^{-1})_{jj}}$ với độ
chính xác của mỗi quan sát bằng 1 (mục 3.7). Sai số chuẩn hiện đại là nghịch đảo của nó nhân với
$\hat\sigma$. Hai trăm năm, cùng một ma trận.

$$R^2 = 1 - \frac{\lVert e\rVert^2}{\lVert y - \bar y\rVert^2}$$

**Ý nghĩa.** $R^2$ so mô hình với mô hình ngu nhất có thể — đoán $\bar y$ cho mọi điểm. $R^2 = 0{,}914286$
ở mục 0.5 nghĩa là tổng bình phương phần dư chỉ còn 1,2 trên 14 của tổng bình phương quanh trung bình.

### 4.5 Đi theo kích thước qua từng bước

Cột *n = 10 000, p = 100* lưu bằng `float64`, 8 byte mỗi số; nguồn `tools/v_linreg.py` mục 9.

| # | Bước | Phép toán | Ra | Kích thước | n = 10 000, p = 100 | Bỏ bước này thì hỏng gì |
|---|---|---|---|---|---|---|
| 0 | vào | — | $X$, $y$ | $(n,p)$, $(n,)$ | 8,00 MB · 80 KB | — |
| 1 | thêm cột chặn | ghép cột toàn 1 | $X$ | $(n,p)$ | +80 KB | đường bị ép qua gốc; $\sum e_i \ne 0$; $R^2$ có thể âm |
| 2 | lập $X^\top X$ | nhân chuyển vị | ma trận đối xứng | $(p,p)$ | **0,0800 MB** | không có phương trình chuẩn — QR và SVD **cố ý bỏ** bước này (mục 6) |
| 3 | lập $X^\top y$ | nhân chuyển vị | vectơ | $(p,)$ | 800 byte | — |
| 4 | giải | QR hoặc SVD trên $X$ | $\hat\beta$ | $(p,)$ | 800 byte | bước duy nhất có nhiều cách, và là chỗ cả mục 6 bàn |
| 5 | dự đoán | $X\hat\beta$ | $\hat y$ | $(n,)$ | 80 KB | — |
| 6 | phần dư | $y - \hat y$ | $e$ | $(n,)$ | 80 KB | không có $R^2$, $\hat\sigma^2$, không chẩn đoán được gì |
| 7 | phương sai | $e^\top e/(n-p)$ | một số | vô hướng | 8 byte | chia $n$ thì chệch xuống |
| 8 | sai số chuẩn | $\hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}$ | một số mỗi hệ số | $(p,)$ | 800 byte | hệ số trần, không biết tin được tới đâu |

**Chỗ đáng nhớ nhất là bước 2.** $X$ chiếm 8,00 MB, $X^\top X$ chỉ 0,0800 MB — **nhỏ hơn 100 lần**, và
tỉ số ấy tăng theo $n$: ở $n = 1\,000\,000$ là 10 000 lần, trong khi $X^\top X$ **vẫn** 0,0800 MB. Đó là
lý do phương trình chuẩn hấp dẫn, và cũng là lý do nó cộng dồn được qua nhiều lô dữ liệu. Cái giá là
số điều kiện bị bình phương (mục 6.1).

**Cái bẫy bộ nhớ:** $H$ cỡ $(n, n)$. Ở $n = 10\,000$ lập $H$ tường minh tốn **800,0 MB**; ở
$n = 100\,000$ tốn **80 000,0 MB**. Thứ thật sự cần từ $H$ chỉ là đường chéo, và lấy được từ phân rã
$X = QR$ mà không đụng tới $H$: $h_{ii} = \lVert Q_{i\cdot}\rVert^2$. Khối dưới kiểm hai cách cho cùng kết quả:

```python
import numpy as np

rng = np.random.default_rng(0)
n, p = 400, 5
X = np.c_[np.ones(n), rng.normal(size=(n, p - 1))]

H = X @ np.linalg.solve(X.T @ X, X.T)        # cách tường minh — (n, n)
Q, _ = np.linalg.qr(X)                       # cách nên dùng   — (n, p)
h_qr = (Q ** 2).sum(1)

print("lech toi da :", np.abs(np.diag(H) - h_qr).max())
print("tr(H) =", np.trace(H), " tong h_qr =", h_qr.sum(), " p =", p)
print("H@H = H ? lech", np.abs(H @ H - H).max())
```

Đầu ra thật trên máy dùng để viết bài (numpy 2.4.4):

```text
lech toi da : 1.734723475976807e-17
tr(H) = 5.0  tong h_qr = 5.000000000000002  p = 5
H@H = H ? lech 3.469446951953614e-17
```

Hai đường lệch nhau cỡ $10^{-17}$, dưới độ phân giải khoảng $2\times10^{-16}$ của `float64`, và cả hai xác
nhận $\operatorname{tr}(H) = p$.

---

## 5. Thuật toán từng bước

```
Thuật toán: bình phương tối thiểu thông thường
Đầu vào:  X cỡ (n, d) các đặc trưng; y cỡ (n,) đáp án
Đầu ra:   hệ số β̂ cỡ (p,), p = d + 1; sai số chuẩn SE cỡ (p,)

1. X ← ghép cột toàn 1 vào trước X                       (nếu cần hệ số chặn)
2. Phân rã X = Q R, Q cỡ (n, p) có cột trực chuẩn, R cỡ (p, p) tam giác trên
3. Nếu R có phần tử đường chéo ≈ 0: X không đủ hạng → dùng SVD, trả nghiệm chuẩn nhỏ nhất
4. Giải R β̂ = Qᵀ y bằng thế ngược                        (không lập nghịch đảo)
5. e ← y − X β̂
6. Kiểm: |Σ e| ≈ 0 (khi có chặn) và |Xᵀ e| ≈ 0
7. σ̂² ← eᵀ e / (n − p)
8. SE_j ← σ̂ · ‖ hàng j của R⁻¹ ‖                           (= σ̂ √[(XᵀX)⁻¹]_jj)
9. Trả về β̂, SE
```

Ba điều pseudocode không nói được:

* **Điều kiện dừng** — không có. Thuật toán trực tiếp chạy hết một lượt; khác hạ gradient ở mục 6 và
  khác Legendre, người lặp lại phép tính khi hàm phi tuyến (Gauss điều 180, mục 3.6).
* **Chỗ dễ sai khi cài.** Quên cột chặn thì bước 6 báo $\sum e \ne 0$. Dữ liệu mới phải được thêm cột
  chặn **giống hệt**. Ngưỡng "≈ 0" ở bước 3 phải **tương đối** theo phần tử lớn nhất của $R$, không tuyệt đối.
* **Độ nhạy.** Không có siêu tham số, nên không có gì để chỉnh — đó vừa là ưu điểm vừa là giới hạn:
  muốn đổi hành vi (chống ngoại lai, chống cột trùng) thì phải đổi **hàm mục tiêu**, không đổi được
  bằng một nút vặn (mục 9).

---

## 6. Bốn cách giải, và cách nào sai lúc nào

| Cách | Mạnh | Yếu | Khi chọn |
|---|---|---|---|
| phương trình chuẩn | dễ suy ra; nhanh khi $p$ nhỏ; cộng dồn được | **bình phương** số điều kiện | chỉ khi đã biết $X$ điều kiện tốt |
| QR | ổn định hơn nhiều; nhanh khi $n \gg p$ | bản đơn giản cần $X$ đủ hạng | mặc định cho bài toán đủ hạng |
| SVD / `lstsq` | nhận biết hạng; trả nghiệm chuẩn nhỏ nhất khi nghiệm không duy nhất | tốn hơn QR | **mặc định an toàn** trong numpy |
| hạ gradient / LSQR | đọc dữ liệu theo lô; khai thác ma trận thưa | nghiệm xấp xỉ; phải chọn tiêu chí dừng | dữ liệu rất lớn hoặc thưa |

Với ma trận dày và $n \gg p$, QR và SVD tốn cỡ $O(np^2)$ phép tính.

### 6.1 Vì sao phương trình chuẩn kém ổn định

Trong số học chính xác, $\kappa(X^\top X) = \kappa(X)^2$. Đo trên ma trận Vandermonde (các cột
$1, x, x^2, \ldots$ trên 60 điểm cách đều trong $[0, 1]$):

```python
import numpy as np

x = np.linspace(0, 1, 60)
for bac in (5, 9):
    X = np.vander(x, bac + 1, increasing=True)
    print(bac, np.linalg.cond(X), np.linalg.cond(X.T @ X))
```

| Bậc | $\kappa(X)$ | $\kappa(X^\top X)$ | $\kappa(X)^2$ |
|---|---:|---:|---:|
| 5 | $3{,}593\times10^{3}$ | $1{,}291\times10^{7}$ | $1{,}291\times10^{7}$ |
| 9 | $3{,}600\times10^{6}$ | $1{,}296\times10^{13}$ | $1{,}296\times10^{13}$ |
| 15 | $1{,}33\times10^{11}$ | $1{,}26\times10^{18}$ | $1{,}78\times10^{22}$ |

Nguồn: `tools/v_linreg.py` mục 8. Hai hàng đầu khớp định lý tới bốn chữ số. **Hàng thứ ba thì không**:
$\kappa(X)^2 = 1{,}78\times10^{22}$ nhưng `cond(X.T @ X)` tính ra $1{,}26\times10^{18}$ — vì $X^\top X$ **lập
bằng float64** đã mất hết chữ số, nên chính phép đo số điều kiện của nó cũng không còn đúng. Đó là triệu
chứng của đúng hiện tượng đang đo.

`float64` có khoảng 16 chữ số thập phân. $\kappa(X) \approx 10^{8}$ đã đẩy $\kappa(X^\top X)$ tới $10^{16}$ —
lập $X^\top X$ là **tự vứt đi một nửa số chữ số chính xác** trước khi bắt đầu giải.

### 6.2 Hệ quả trên một bài toán biết trước đáp án — và vì sao chữ số không tái lập được qua máy

Vandermonde bậc 15, **không nhiễu**, đáp án đúng là mọi hệ số bằng 1:

| Bộ giải | Sai số lớn nhất — máy dùng để viết bài | Sai số lớn nhất — bản cũ của bài, máy khác |
|---|---:|---:|
| phương trình chuẩn | **43,86** | 17,67 |
| QR | $6{,}759\times10^{-6}$ | $4{,}238\times10^{-6}$ |
| SVD | $9{,}718\times10^{-6}$ | $6{,}164\times10^{-6}$ |
| $\kappa(X^\top X)$ đo được | $1{,}26\times10^{18}$ | $1{,}38\times10^{18}$ |

Cột giữa: numpy 2.4.4 với OpenBLAS 0.3.31 trên Linux x86_64 (`tools/v_linreg.py` mục 8). Cột phải:
bản trước của bài này, kiểm ngày 2026-08-31 trên một máy Windows, và checklist của bản đó ghi *"khớp từng
chữ số"*.

**Kết quả không tái lập được qua máy, và đó là kết quả đúng.** Khi $\kappa(X^\top X)$ vượt $10^{16}$, phần
dư chữ số của phép giải là **nhiễu làm tròn**, và nhiễu đó phụ thuộc thứ tự cộng dồn của thư viện BLAS
— tức phụ thuộc máy và phiên bản. Hai máy cho hai con số 43,86 và 17,67, chênh hơn hai lần. Thứ **tái
lập được** là kết luận: trên cùng máy, phương trình chuẩn sai **6,490×10⁶ lần** nhiều hơn QR — sai hẳn
trên một bài toán mà mọi hệ số đúng đều bằng 1, trong khi QR sai ở chữ số thứ sáu.

> **Quy tắc thực hành.** Đừng viết `inv(X.T @ X) @ X.T @ y`. Dùng `np.linalg.lstsq(X, y, rcond=None)`:
> hàm giải thẳng bài toán, xử lý được cả hệ thiếu và thừa phương trình, trả nghiệm chuẩn nhỏ nhất khi có
> nhiều nghiệm. Nếu buộc dùng phương trình chuẩn, ít nhất dùng `solve(X.T @ X, X.T @ y)`.

### 6.3 Khi nào vẫn nên dùng phương trình chuẩn

Khi $p$ rất nhỏ so với $n$, các cột đã chuẩn hoá thang đo và $X$ đã được kiểm là điều kiện tốt — hoặc
khi cần cộng dồn $X^\top X$ và $X^\top y$ qua nhiều lô mà không giữ dữ liệu thô. Đó chính là tình thế của
Legendre và Gauss: hai, ba ẩn, hệ số đã được làm tròn tới ba chữ số, tính bằng tay — phương trình chuẩn là
lựa chọn đúng cho **người tính bằng tay**, và thành lựa chọn sai cho **máy tính float64** chỉ khi bài toán
có $\kappa$ lớn.

---

## 7. Cài đặt và số đo thật

### 7.1 Bản tối thiểu

Khối ngắn ở mục 0.1 là bản nên chạy đầu tiên. Trong dự án thật phải khớp trên tập huấn luyện và
đánh giá trên dữ liệu chưa dùng để học; ví dụ bốn máy dùng cả bốn điểm chỉ để minh hoạ phép tính.

### 7.2 Bốn cách giải trên cùng dữ liệu, đối chiếu nhau

Khối dưới giải một bộ 25 điểm sinh từ $y = 2 + 1{,}3x + \varepsilon$, $\sigma = 2$, bằng bốn cách; kiểm ba
tính chất hình học bằng `assert`; rồi dựng ca Vandermonde của mục 6.2.

```python
# -*- coding: utf-8 -*-
"""Hồi quy tuyến tính từ đầu — bốn cách giải cùng một bài toán, đối chiếu với nhau."""
import numpy as np


def fit_normal(X, y):
    """Phương trình chuẩn để minh họa; có thể kém ổn định số."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def fit_qr(X, y):
    """Phân rã QR rút gọn; giả sử X đủ hạng cột."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def fit_svd(X, y, rcond=1e-12):
    """SVD + giả nghịch đảo. Chạy được cả khi X không đủ hạng."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))


def fit_gd(X, y, lr=None, n_iter=2000):
    """Hạ gradient toàn batch, bản minh họa để đối chiếu nghiệm."""
    n, p = X.shape
    if lr is None:                       # 1/L với L là giá trị riêng lớn nhất của XᵀX/n
        lr = 1.0 / (np.linalg.eigvalsh(X.T @ X / n)[-1])
    b = np.zeros(p)
    for _ in range(n_iter):
        b -= lr * (X.T @ (X @ b - y)) / n
    return b


def summary(X, y, b):
    """Thống kê chuẩn: R², sai số chuẩn, thống kê t."""
    n, p = X.shape
    r = y - X @ b
    rss = r @ r
    tss = ((y - y.mean()) ** 2).sum()
    s2 = rss / (n - p)                                    # ước lượng σ² không chệch
    xtx_inv = np.linalg.solve(X.T @ X, np.eye(p))
    se = np.sqrt(np.diag(s2 * xtx_inv))
    return dict(r2=1 - rss / tss,
                r2_adj=1 - (rss / (n - p)) / (tss / (n - 1)),
                sigma=np.sqrt(s2), se=se, t=b / se, rss=rss)


def demo():
    rng = np.random.default_rng(3)
    n = 25
    x = np.sort(rng.uniform(0, 10, n))
    y = 2.0 + 1.3 * x + rng.normal(0, 2.0, n)
    X = np.c_[np.ones(n), x]

    print("Bốn cách giải, cùng một dữ liệu (n = %d, p = %d):" % X.shape)
    bs = {}
    for name, f in [("phương trình chuẩn", fit_normal), ("QR", fit_qr),
                    ("SVD", fit_svd), ("hạ gradient", fit_gd)]:
        bs[name] = f(X, y)
        print("   %-20s β = [%.6f, %.6f]" % (name, *bs[name]))
    ref = bs["QR"]
    print("   lệch lớn nhất so với QR: %.3e"
          % max(np.abs(v - ref).max() for v in bs.values()))

    s = summary(X, y, ref)
    print("\nThống kê:")
    print("   R²        = %.4f      R² hiệu chỉnh = %.4f" % (s["r2"], s["r2_adj"]))
    print("   σ̂        = %.4f      (giá trị thật khi sinh dữ liệu: 2.0)" % s["sigma"])
    print("   sai số chuẩn = [%.4f, %.4f]" % tuple(s["se"]))
    print("   thống kê t   = [%.3f, %.3f]" % tuple(s["t"]))
    print("   hệ số thật   = [2.0, 1.3]; khoảng beta ± 1.96 SE chứa hệ số thật:",
          np.all(np.abs(ref - [2.0, 1.3]) < 1.96 * s["se"]))

    # --- ba tính chất luôn đúng của nghiệm OLS, kiểm bằng assert ---
    r = y - X @ ref
    assert abs(r.sum()) < 1e-9, "phần dư phải tổng bằng 0 khi có hệ số chặn"
    assert abs(r @ x) < 1e-9, "phần dư phải trực giao với mọi cột của X"
    assert abs((X @ ref).mean() - y.mean()) < 1e-9, "đường hồi quy đi qua (x̄, ȳ)"
    print("\n3 tính chất hình học đều đúng: Σe = %.2e, eᵀx = %.2e, ŷ̄ − ȳ = %.2e"
          % (r.sum(), r @ x, (X @ ref).mean() - y.mean()))

    # --- khi nào phương trình chuẩn sập ---
    print("\nKhi X điều kiện xấu (Vandermonde bậc 15, KHÔNG nhiễu, đáp án đúng là toàn 1):")
    t = np.linspace(0, 1, 60)
    Xv = np.vander(t, 16, increasing=True)
    yv = Xv @ np.ones(16)
    for name, f in [("phương trình chuẩn", fit_normal), ("QR", fit_qr), ("SVD", fit_svd)]:
        e = np.abs(f(Xv, yv) - 1).max()
        print("   %-20s sai số lớn nhất = %.3e" % (name, e))
    print("   κ(X) = %.2e   κ(XᵀX) = %.2e" % (np.linalg.cond(Xv), np.linalg.cond(Xv.T @ Xv)))


if __name__ == "__main__":
    demo()
```

Đầu ra thật trên máy dùng để viết bài (numpy 2.4.4, OpenBLAS 0.3.31, Linux x86_64):

```text
Bốn cách giải, cùng một dữ liệu (n = 25, p = 2):
   phương trình chuẩn   β = [2.185812, 1.302401]
   QR                   β = [2.185812, 1.302401]
   SVD                  β = [2.185812, 1.302401]
   hạ gradient          β = [2.185812, 1.302401]
   lệch lớn nhất so với QR: 1.065e-07

Thống kê:
   R²        = 0.7614      R² hiệu chỉnh = 0.7510
   σ̂        = 2.1205      (giá trị thật khi sinh dữ liệu: 2.0)
   sai số chuẩn = [0.8304, 0.1520]
   thống kê t   = [2.632, 8.567]
   hệ số thật   = [2.0, 1.3]; khoảng beta ± 1.96 SE chứa hệ số thật: True

3 tính chất hình học đều đúng: Σe = 5.24e-14, eᵀx = 2.71e-13, ŷ̄ − ȳ = -3.55e-15

Khi X điều kiện xấu (Vandermonde bậc 15, KHÔNG nhiễu, đáp án đúng là toàn 1):
   phương trình chuẩn   sai số lớn nhất = 4.386e+01
   QR                   sai số lớn nhất = 6.759e-06
   SVD                  sai số lớn nhất = 9.718e-06
   κ(X) = 1.33e+11   κ(XᵀX) = 1.26e+18
```

Ba chữ số cuối của các dòng $\Sigma e$, $e^\top x$ và **bốn dòng Vandermonde** đổi theo máy (mục 6.2);
mọi dòng khác tái lập được.

Ba điều đáng đọc:

1. Trên bài toán dễ, **bốn cách trùng tới $1{,}065\times10^{-7}$** — phần lệch duy nhất là hạ gradient còn
   cách nghiệm sau 2000 vòng, không phải số học.
2. $\hat\sigma = 2{,}1205$ so với giá trị thật 2,0. Chia cho $n$ thay vì $n - p$ ra 2,0339 — *trông* gần
   hơn nhưng là ăn may: ước lượng đó chệch xuống một cách có hệ thống (mục 4.4).
3. Trên bài toán khó, phương trình chuẩn không "kém chính xác" mà **sai hẳn**.

Khoảng $\hat\beta \pm 1{,}96\,\text{SE}$ là xấp xỉ chuẩn để kiểm nhanh. Với mẫu nhỏ và nhiễu Gauss, khoảng
chính xác phải dùng mốc phân phối $t$ với $n - p$ bậc tự do.

### 7.3 Quy trình dùng thật

1. Xác định biến đích và đơn vị của từng đặc trưng.
2. Chia train / validation / test **trước** mọi bước học từ dữ liệu.
3. Mã hoá biến phân loại, xử lý giá trị thiếu bằng quy trình chỉ khớp trên tập train.
4. Thêm hệ số chặn nếu thư viện không tự thêm.
5. Khớp bằng `lstsq`, QR hoặc SVD; không dùng nghịch đảo tường minh.
6. So với baseline đoán trung bình; báo thước đo trên tập chưa thấy.
7. Vẽ phần dư (Hình 22). Nếu mục tiêu là diễn giải hệ số, kiểm thêm phương sai, tương quan sai số, cột
   trùng và cách lấy sai số chuẩn.
8. Chỉ ngoại suy ra ngoài miền của $X$ khi có lý do chuyên môn — đúng bài học Legendre tự rút ra ở mục
   3.4: khớp tốt trên số liệu có không có nghĩa tham số đúng với thế giới.

---

## 8. Vì sao bình phương lại ăn thua

Mục 1 cho thấy bình phương tối thiểu **thắng cách cũ** (tổng bình phương 7,0643 xuống 1,4329). Câu hỏi của
mục này sâu hơn: vì sao lại là **bình phương**, chứ không phải trị tuyệt đối hay luỹ thừa bậc bốn — cả
hai đều cũng "dùng hết dữ liệu". Có **ba lý do**, đến từ ba nguồn, và mỗi lý do có một phép đo đỡ.

| Lý do | Ai nêu | Cơ chế | Phép đo đỡ |
|---|---|---|---|
| **1. Tính được** | Legendre tr.73; Gauss điều 186 | đạo hàm của bình phương là **tuyến tính**, nên điều kiện cực tiểu là một hệ tuyến tính — giải được bằng tay năm 1805 | trị tuyệt đối không có đạo hàm tại 0; Gauss: *"ad calculos complicatissimos deferremur"*. Trên 25 điểm, L1 phải dò **300 cặp điểm** để tìm nghiệm (`v_linreg.py` mục 7) |
| **2. Hợp lý cực đại dưới sai số chuẩn** | Gauss điều 177–179 | $\Omega \propto e^{-h^2\sum v^2}$: tối đa hoá xác suất **là** tối thiểu hoá tổng bình phương | ước lượng đổi theo luật sai số: luật Gauss cho 5,04 (trung bình cộng), luật Laplace cho 5,21 (trung vị) trên cùng 11 quan sát (`v_linreg.py` mục 4) |
| **3. Cân bằng giữa các sai số** | Legendre tr.73 | mỗi điểm kéo nghiệm bằng lực tỉ lệ khoảng lệch, nên không nhóm nào lấn át — nghiệm là **trọng tâm** | L1 đi qua **đúng 2** điểm (bằng số ẩn) và bỏ mặc phần còn lại; bình phương tối thiểu có **0** phần dư bằng 0 — mọi điểm đều góp (`v_linreg.py` mục 7) |

**Cả ba lý do đều có mặt trái, và mặt trái nằm ở chính cơ chế.**

- Lý do 2 chỉ đúng **khi sai số thật sự chuẩn**. Gauss tự nói ông giả định điều đó như một tiên đề
  (mục 3.5). Sai số có đuôi dày thì lý do 2 biến mất, và cơ chế ở lý do 3 quay ra phản tác dụng.
- Lý do 3 — "lực tỉ lệ khoảng lệch" — nghĩa là **một điểm rất xa kéo rất mạnh**. Đo trên 60 điểm, cộng
  thêm 20, 50, 100 vào đúng một điểm:

| Dịch một điểm | Hệ số góc OLS | Lệch | Hệ số góc Huber | Lệch |
|---:|---:|---:|---:|---:|
| +0 | 1,4934 | 0,4 % | 1,4928 | 0,5 % |
| +20 | 1,5432 | 2,9 % | 1,4985 | 0,1 % |
| +50 | 1,6180 | 7,9 % | 1,4985 | 0,1 % |
| +100 | 1,7427 | **16,2 %** | 1,4985 | **0,1 %** |

Nguồn: `tools/v_linreg.py` mục 12; hệ số góc thật 1,5; một điểm trên 60 là 1,67 % dữ liệu. Huber ở đây cài
bằng bình phương tối thiểu có trọng số lặp (IRLS), thang đo MAD, $\delta = 1{,}345$ — **không** phải
`sklearn.HuberRegressor`, vốn ước lượng thang đo cùng lúc và có phạt L2 ẩn. Bản trước của bài này dùng
`sklearn` và ghi "Huber lệch 0 %"; trên máy viết bài không có `sklearn`, và bản IRLS cho 0,1 %. Hai con số
cùng nói một điều.

Cột OLS **tăng tuyến tính theo độ dịch** — đúng như lý do 3 dự đoán: điểm đó góp vào đạo hàm một lượng tỉ
lệ với khoảng lệch của nó. Cột Huber **đứng yên** từ +20 trở đi, vì Huber đổi bình phương thành tuyến tính
khi phần dư vượt ngưỡng, nên lực kéo **bị chặn trên**.

---

## 9. Giá phải trả — nó hỏng khi nào, và khi nào đừng dùng

### 9.1 Năm câu hỏi khác nhau, năm điều kiện khác nhau

| Câu hỏi | Cần gì |
|---|---|
| Có tính được một đường khớp không? | dữ liệu số và một bộ giải; $X$ không đủ hạng thì cần quy ước chọn nghiệm |
| Dự đoán dữ liệu mới có tốt không? | dạng mô hình đủ phù hợp; train và test cùng cơ chế sinh dữ liệu |
| Hệ số có đúng đại lượng thật không? | $\mathbb E[\varepsilon\mid X] = 0$ |
| Sai số chuẩn, khoảng tin cậy có đúng không? | thêm giả định phương sai đều và không tương quan, hoặc dùng sai số chuẩn bền |
| Hệ số có nghĩa nhân quả không? | thiết kế nghiên cứu; hồi quy một mình không cho điều đó |

### 9.2 Định lý Gauss–Markov, và nó KHÔNG nói gì

**Phát biểu.** Nếu $\mathbb E[\varepsilon\mid X] = 0$, $\operatorname{Var}(\varepsilon\mid X) = \sigma^2 I$ và $X$
đủ hạng cột, thì trong mọi ước lượng **tuyến tính theo $y$** và **không chệch**, OLS có hiệp phương sai
nhỏ nhất. Định lý **không cần** sai số chuẩn — khác lập luận của Gauss 1809 ở mục 3.6.

**Trực giác chứng minh.** Mọi ước lượng tuyến tính không chệch khác viết được thành $\tilde\beta = \hat\beta + Cy$
với $CX = 0$. Thành phần $Cy$ không đổi kỳ vọng nhưng cộng thêm $\sigma^2CC^\top$ — nửa xác định dương — vào
hiệp phương sai. Muốn giữ không chệch mà khác OLS thì chỉ **thêm** được nhiễu, không bớt được.

Ba chữ hay bị bỏ qua:

- **"tuyến tính"** — ước lượng phi tuyến như Huber thắng được OLS khi có ngoại lai (mục 8).
- **"không chệch"** — chấp nhận chệch một chút là thắng được. Hai cột tương quan 0,999939: OLS có sai số
  trung bình $\lVert\hat\beta - \beta\rVert$ bằng 16,0959, Ridge $\lambda = 1$ bằng 0,0974 — **nhỏ hơn
  165,32 lần** (`tools/v_linreg.py` mục 11).
- **"$\operatorname{Var}(\varepsilon\mid X) = \sigma^2 I$"** — vỡ giả định này thì OLS vẫn không chệch nhưng hết
  tối ưu, và sai số chuẩn cổ điển sai.

### 9.3 Nhìn phần dư

![Bốn giả định](<ols_assumptions.png>)

**Hình 22 — Biểu đồ phần dư khi từng giả định bị vỡ (hình tự vẽ).**

**Cách đọc hình.** Bốn bảng cùng trục: ngang là $\hat y$, dọc là phần dư $e$, đường xám là $e = 0$. Mẹo đọc
là **tìm hình dạng** — phần dư đúng chuẩn trông như một dải nhiễu đều.

| Bảng | Nhìn thấy | Nghĩa là |
|---|---|---|
| 1. Đúng giả định | dải đều quanh 0 | không có gì để sửa |
| 2. Bỏ sót độ cong | phần dư **cong chữ U** | đường thẳng bỏ sót cấu trúc dự đoán được |
| 3. Phương sai không đều | dải **loe hình phễu** | sai số chuẩn cổ điển sai |
| 4. Có ngoại lai | một điểm bay xa | điểm đó có thể kéo cả đường (mục 8) |

**Điều hình KHÔNG nói ra.** Biểu đồ phần dư chỉ thấy **dấu hiệu**. Giả định $\mathbb E[\varepsilon\mid X] = 0$
thường **không kiểm được** từ phần dư — đúng thứ đã đánh lừa phép đo kinh tuyến năm 1805: sai số vĩ độ
do lực hút cục bộ là sai số **có hệ thống**, và năm phần dư của Legendre (lớn nhất 1,8366 giây) trông hoàn
toàn vô hại.

<details>
<summary><b>Code sinh Hình 4</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({                          # cùng kiểu vẽ với Hình 1
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

rng = np.random.default_rng(11)
n = 120
xx = np.sort(rng.uniform(0, 10, n))
base_noise = rng.normal(0, 1, n)              # cùng nhiễu để so sánh công bằng
clean = 2 + 1.2 * xx + 1.5 * base_noise
yo = clean.copy()
yo[5] += 40                                   # chỉ thêm đúng 1 ngoại lai

sets = [
    ("Đúng giả định",            clean,
     "phần dư đều, không hình dạng"),
    ("Bỏ sót độ cong",           2 + 1.2 * xx + 0.35 * (xx - 5) ** 2 + 1.5 * base_noise,
     "đường thẳng bỏ sót cấu trúc hình chữ U"),
    ("Phương sai không đều",     2 + 1.2 * xx + (0.3 + 0.45 * xx) * base_noise,
     "SE cổ điển sai; dải nhiễu loe thành phễu"),
    ("Có điểm ngoại lai",        yo, "1 điểm có thể kéo cả đường"),
]

fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2))
for ax, (name, yv, note) in zip(axes, sets):
    Xv = np.c_[np.ones(n), xx]
    bb = np.linalg.lstsq(Xv, yv, rcond=None)[0]   # khớp OLS trên CHÍNH dữ liệu bị vỡ
    r = yv - Xv @ bb
    ax.scatter(Xv @ bb, r, s=9, color="#374151")
    ax.axhline(0, color="#6b7280", lw=1.3)
    ax.set_xlabel(r"$\hat y$"); ax.set_ylabel("phần dư" if ax is axes[0] else "")
    ax.set_title("%s\n%s" % (name, note), fontsize=8.4)

fig.savefig("ols_assumptions.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

Bốn bảng dùng **cùng $x$ và cùng vectơ nhiễu gốc**; mỗi bảng chỉ thêm đúng cấu trúc ghi ở tiêu đề.

### 9.4 Ba cái bẫy của $R^2$

1. $R^2$ trên tập huấn luyện **không giảm** khi thêm cột, kể cả cột ngẫu nhiên. Đủ cột để nội suy $n$ điểm
   thì $R^2 = 1$ mà vô dụng. Phải đánh giá trên dữ liệu chưa thấy.
2. $R^2$ cao không có nghĩa mô hình đúng: bảng 2 của Hình 22 là quan hệ cong khớp bằng đường thẳng.
3. $R^2$ **có thể âm** khi bỏ hệ số chặn hoặc khi tính trên tập kiểm tra — mô hình tệ hơn đoán $\bar y$.

### 9.5 Bảng giá: mỗi mặt được mua bằng cái gì

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
|---|---|---|---|
| **Dùng hết dữ liệu**, một tiêu chuẩn duy nhất để so hai nghiệm | tổng bình phương 7,0643 xuống 1,4329 trên ví dụ đầu tiên của Legendre | tiêu chuẩn "khớp tốt nhất" **không kiểm mô hình**: khớp kinh tuyến tốt hơn (6,48 so với 57,28 giây²) nhưng độ dẹt 1/148 bị chính Legendre bác | mục 1.2, 3.4; `v_linreg.py` mục 2–3 |
| **Nghiệm đóng**, không tốc độ học, không hạt giống | bốn bộ giải khác nhau trùng tới $1{,}065\times10^{-7}$ | chỉ có **vì** hàm mất mát là bình phương; đổi sang Huber là phải lặp | mục 7.2 |
| **Tối ưu dưới sai số chuẩn** (Gauss) | trung bình cộng 5,04 là hợp lý cực đại | luật sai số khác thì hết tối ưu: một điểm dịch +100 kéo hệ số góc lệch 16,2 %, Huber 0,1 % | mục 3.5, 8; `v_linreg.py` mục 4, 12 |
| **Hệ số diễn giải được**, có sai số chuẩn | $\text{SE} = 0{,}1520$, $t = 8{,}567$ ở ví dụ 25 điểm | **sụp khi cột tương quan**: tương quan 0,999939 cho độ lệch chuẩn của $\hat\beta_1$ bằng 14,27, **ngược dấu 48,2 %** số lần, trong khi tổng hai hệ số chỉ lệch chuẩn 0,1442 | `v_linreg.py` mục 11 |
| **Ít tham số**, chạy được với $n$ nhỏ | $p = 2$ mô tả 25 điểm | **chệch không chữa được bằng thêm dữ liệu**: trên dữ liệu cong, RMSE đường thẳng đứng ở 0,2938 khi $n$ tăng 4000 lần, gấp 2,9378 lần mức nhiễu | `v_linreg.py` mục 10 |
| **Cộng dồn được** $X^\top X$, $X^\top y$ | ở $n = 10^6$, $p = 100$: $X$ 800,00 MB, $X^\top X$ 0,0800 MB | **bình phương số điều kiện**: phương trình chuẩn sai 43,86 trên bài toán đáp án toàn 1; QR sai $6{,}759\times10^{-6}$; và chữ số cụ thể **đổi theo máy** | mục 6; `v_linreg.py` mục 8–9 |
| **Tối ưu trong lớp tuyến tính không chệch** (Gauss–Markov) | hiệp phương sai nhỏ nhất trong lớp đó | lớp đó **hẹp**: Ridge $\lambda = 1$ chệch mà sai số nhỏ hơn 165,32 lần | `v_linreg.py` mục 11 |

**Đọc dọc cột thứ ba** thấy một khuôn: bốn trong bảy cái giá đến từ **cùng một quyết định** — chọn bình
phương. Nó cho nghiệm đóng, tính lồi, lý do xác suất và công thức sai số chuẩn; nó cũng làm ngoại lai kéo
mạnh và làm hệ số nổ khi cột trùng. Các biến thể cùng giai đoạn — Ridge, Lasso, Elastic Net, Huber, hồi quy
phân vị, hồi quy Bayes — mỗi cái trả cái giá đó **rẻ hơn** bằng cách đổi đúng một chi tiết của hàm mục tiêu.

### 9.6 Khi nào đừng dùng — và dùng gì thay

| Tình huống | Dấu hiệu | Dùng gì thay |
|---|---|---|
| quan hệ cong rõ rệt | phần dư chữ U (Hình 22, bảng 2) | thêm cột đa thức hoặc spline — vẫn là hồi quy tuyến tính theo hệ số |
| ngoại lai thật, không xoá được | vài điểm xa hẳn; hệ số đổi mạnh khi bỏ một điểm | Huber, RANSAC, L1 |
| cột gần trùng | $\kappa(X)$ lớn; hệ số đổi dấu giữa các mẫu | Ridge, Elastic Net |
| cần **khoảng** chứ không chỉ trung bình | "90 % trường hợp dưới mức nào" | hồi quy phân vị |
| $p > n$ | nghiệm không duy nhất; $R^2$ huấn luyện bằng 1 | Lasso, Ridge, LARS |
| sai số có hệ thống, không ngẫu nhiên | phần dư trông đều nhưng tham số mâu thuẫn bằng chứng độc lập | **sửa mô hình**, không đổi thuật toán — bài học Legendre ở mục 3.4 |
| câu hỏi nhân quả | — | thiết kế nghiên cứu |

---

## 10. Ứng dụng thật

| Bài toán | Vai trò của hồi quy tuyến tính |
|---|---|
| **xác định hình dạng Trái Đất và độ dài của mét**, 1799–1805 | chính phép tính ở mục 3.3 và 3.9 — bốn cung Dunkerque–Barcelona, hai ẩn độ dẹt và độ dài một độ |
| **quỹ đạo sao chổi và tiểu hành tinh** | bài toán mà cả Legendre lẫn Gauss viết sách để giải |
| **hiệu chuẩn cảm biến** | trong một dải làm việc hẹp, quan hệ vật lý gần tuyến tính; phải kiểm độ cong trước khi ngoại suy |
| **kinh tế lượng, y sinh** | hệ số có đơn vị và sai số chuẩn — diễn giải được **khi** thiết kế nghiên cứu cho phép |
| **dữ liệu bảng cỡ vừa** | baseline bắt buộc trước mọi mô hình phức tạp. Grinsztajn và cộng sự 2022, tóm tắt tr.1: *"Results show that tree-based models remain state-of-the-art on medium-sized data (∼10K samples) even without accounting for their superior speed."* — tức mô hình cây, không phải tuyến tính, thắng ở chế độ đó; hồi quy tuyến tính là mốc để đo xem cây thắng bao nhiêu |
| **đầu ra của mạng nơ-ron** | tầng cuối của nhiều mạng là một phép biến đổi tuyến tính; *linear probing* học đúng một hồi quy tuyến tính trên biểu diễn đã học |

---

## 11. Đối chiếu với nguồn gốc — đã lấy gì, bỏ gì, vì sao

Quy tắc áp dụng: `Dàn ý mẫu` mục E.1d — phần có liên quan tới bình phương tối thiểu liệt kê **từng** đơn vị
đánh số; phần còn lại gộp theo đơn vị chia lớn nhất của chính nguồn, kèm trang và lý do.

### 11.1 Legendre 1805 — theo hình và bảng

Sách có **8 bảng đóng khung** và **1 bản hình gồm 5 hình**. Đếm bằng contact sheet toàn bộ 98 trang PDF
(`tools/pdfsheet.py`) rồi mở từng trang có khung; các cột số tính toán **không** đóng khung trong văn bản
không tính là bảng. Lớp text vỡ nên `tools/sopaper.py` không đếm được.

| Hình/Bảng | Trang in · PDF | Nội dung | Ở mục nào của bài | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| bảng quan sát sao chổi 1781 | tr.33 · PDF 45 | ba quan sát kinh độ, vĩ độ | — | dữ liệu của phương pháp tìm quỹ đạo (phần I), không dùng bình phương tối thiểu |
| bảng yếu tố quỹ đạo gần đúng / đã sửa, sao chổi 1781 | tr.41 · PDF 53 | so yếu tố của Legendre với Mémoires 1781 | — | kết quả của phương pháp quỹ đạo, phần I |
| bảng so vị trí tính và quan sát, sao chổi 1781 | tr.42 · PDF 54 | hai khối: yếu tố gần đúng và yếu tố đã sửa | — | như trên |
| bảng quan sát sao chổi 1769 (ví dụ thứ hai) | tr.43 · PDF 55 | ba quan sát tháng Chín | — | dữ liệu của phần I |
| bảng yếu tố gần đúng so với yếu tố thật, 1769 | tr.51 · PDF 63 | kiểm phương pháp phần I | — | như trên |
| bảng ba quan sát sao chổi 1769 lấy từ Pingré | tr.53 · PDF 65 | dữ liệu đầu vào của phần II | mục 3.1 nhắc tới | là **đầu vào** của ví dụ ở mục LVIII; bài dựng lại từ các hệ số $E', E'', E'''$ đã suy ra, không từ bảng thô |
| bảng vị trí tính so với quan sát theo yếu tố cuối | tr.66 · PDF 78 | sai số còn lại sau bình phương tối thiểu | — | kết quả thiên văn sau mục LIX; con số quyết định đã dẫn từ tr.65 |
| **bảng bốn cung kinh tuyến Delambre–Méchain** | **tr.76 · PDF 88** | **năm vĩ độ, bốn độ dài cung** | **mục 3.3, đã cắt (Hình 7), dựng lại toàn bộ** | — |
| bản hình Fig. 1–5 | sau tr.80 · PDF 93 | hình học quỹ đạo: mặt phẳng hoàng đạo, tam giác CTS | — | hình của phương pháp quỹ đạo phần I; phụ lục bình phương tối thiểu **không có hình nào** |

Tổng **9 dòng** (8 bảng + 1 bản hình). Đã cắt vào bài **9 trang** của phần ruột (Hình 2–10), gồm bảng
kinh tuyến.

### 11.2 Legendre 1805 — theo mục

Sách chia: một lời mở đầu không đánh số; **Première partie — Analyse du problème**; **Seconde partie —
Méthode pour corriger les élémens de l'orbite**; hai tiểu mục *Observations*; và **Appendice**. Mục đánh số
La Mã chạy liên tục từ I tới LXIII.

| Đơn vị của sách | Trang in · PDF | Nói về gì | Vào bài ở đâu | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| lời mở đầu | tr.[i]–[vi] · PDF 7–12 | lịch sử bài toán quỹ đạo (Newton, Lambert, Lagrange, Laplace); báo trước *« une appendice »* | mục 12 | không chứa phương pháp |
| Phần I, mục I–XLVII | tr.1–51 · PDF 13–63 | giải bài toán quỹ đạo sao chổi từ ba quan sát; hai ví dụ 1781 và 1769 | — | **bài toán khác**; không dùng bình phương tối thiểu |
| Phần II, mục XLVIII–LVI | tr.53–63 · PDF 65–75 | sửa yếu tố quỹ đạo gần đúng bằng hiệu chỉnh vi phân | mục 3.1 (bối cảnh) | dẫn tới hệ ba sai số của mục LVII, nhưng tự nó là thiên văn |
| **mục LVII** | **tr.63–64 · PDF 75–76** | ép $E''$, $E^{iv}$ bằng 0; câu *« cherchant le minimum de la somme des quarrés »* | **mục 1.2, 3.1** | — |
| **mục LVIII** | **tr.64–65 · PDF 76–77** | quy tắc nhân–cộng; hai phương trình chuẩn; nghiệm $z, u$ | **mục 3.1, dựng lại** | — |
| mục LIX | tr.65–66 · PDF 77–78 | yếu tố quỹ đạo cuối và bảng so sánh | mục 11.1 | kết quả thiên văn |
| mục LX | tr.66–67 · PDF 78–79 | so với yếu tố của Pingré; cảnh báo quỹ đạo 1769 có thể không phải parabol | — | thiên văn |
| **mục LXI** *Observations sur la méthode précédente* | tr.67–68 · PDF 79–80 | tóm bốn phần của phương pháp; phần tư: *« la somme des quarrés de ces trois quantités soit un minimum »* | mục 3.1 | nhắc lại, không có phép tính mới |
| **mục LXII** | tr.68–69 · PDF 80–81 | dùng bình phương tối thiểu với **hai hệ sai số** khi có hơn ba quan sát | mục 3.1 | không có ví dụ số; nội dung trùng quy tắc tổng quát của phụ lục |
| mục LXIII *Observations sur le calcul des corrections indéterminées* | tr.69–71 · PDF 81–83 | mẹo tính logarit của hiệu chỉnh nhỏ | — | kỹ thuật tính bảng logarit, không liên quan phương pháp |
| **Appendice**, tr.72–75 | **PDF 84–87** | phát biểu tổng quát; quy tắc tr.73; loại phương trình sai quá lớn; trung bình cộng; trọng tâm | **mục 1.1, 2, 3.2** | — |
| **Appendice, *Application à la mesure des degrés du méridien*** | **tr.76–80 · PDF 88–92** | hệ (a)–(d); độ dẹt 1/148 bị bác; giải lại với 1/320; lực hút cục bộ | **mục 3.3, 3.4, dựng lại** | — |

Tổng **12 dòng**. Mọi mục có dùng bình phương tối thiểu — LVII, LVIII, LXI, LXII và toàn bộ phụ lục — đều có
dòng riêng.

### 11.3 Gauss 1809 — theo hình và bảng

Quyển II mục III (PDF 221–240) **không có hình**; có **1 bảng** trong văn bản — bảng sáu loại trường hợp
$m, n, m', n', m'', n''$ ở điều 176 (tr.210, PDF 226) — và **1 sơ đồ** ba giả thuyết ở điều 188 (tr.223,
PDF 239). Phần còn lại của sách — ba bảng số cuối sách (*Tabula I–III*, PDF 261–266) và các bảng tính trong
quyển I — thuộc bài toán quỹ đạo.

| Hình/Bảng | Trang in · PDF | Nội dung | Ở mục nào của bài | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| bảng các trường hợp của điều 176 | tr.210 · PDF 226 | đếm trường hợp để chứng minh định lý Bayes rời rạc | — | là chứng minh phụ cho bước "xác suất hậu nghiệm tỉ lệ Ω"; bài giữ kết luận ở mục 3.5, không cần bảng đếm |
| sơ đồ ba giả thuyết khoảng cách, điều 188 | tr.223 · PDF 239 | phương pháp gần đúng rút gọn cho quỹ đạo | — | ngoài bình phương tối thiểu tổng quát |
| *Tabula I–III* và bảng tính quyển I | PDF 261–266 và rải rác | bảng thiên văn | — | bài toán quỹ đạo |

### 11.4 Gauss 1809 — theo mục

| Đơn vị của sách | Trang in · PDF | Nói về gì | Vào bài ở đâu | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| Lời tựa, quyển I (mục I–IV), quyển II mục I–II | PDF trước trang 221 | chuyển động theo Kepler; tìm quỹ đạo từ ba hoặc bốn quan sát | — | bài toán quỹ đạo |
| điều 172 | tr.205 · PDF 221 | mọi quan sát chỉ gần đúng; phải kết hợp **nhiều hơn** số quan sát tối thiểu | mục 1.1 (cùng ý Legendre) | trùng ý với câu đã trích của Legendre |
| điều 173 | tr.205–207 · PDF 221–223 | lấy trung bình có trọng số các vị trí gần nhau; độ chính xác $\sqrt{ee + e'e' + \cdots}$ | mục 3.6 | trọng số đã trình bày theo điều 179 |
| điều 174 | tr.207–208 · PDF 223–224 | chọn ba–bốn vị trí chuẩn cho chuỗi quan sát ngắn; mở đầu bài toán tổng quát | mục 3.5 | dẫn nhập |
| **điều 175** | **tr.208–210 · PDF 224–226** | quan sát độc lập → tích Ω | **mục 3.5, Hình 11** | — |
| điều 176 | tr.210–211 · PDF 226–227 | định lý: xác suất giả thuyết tỉ lệ xác suất biến cố | mục 3.5 | chứng minh phụ |
| **điều 177** | **tr.211–212 · PDF 227–228** | cực đại Ω; tiên đề trung bình cộng; suy ra $\varphi\Delta = \frac{h}{\sqrt\pi}e^{-hh\Delta\Delta}$ | **mục 3.5, Hình 12–13** | — |
| điều 178 | tr.212–213 · PDF 228–229 | luật tìm được không chặn sai số; $h$ là số đo độ chính xác | mục 0.3, 3.5 | — |
| **điều 179** | **tr.213–214 · PDF 229–230** | **nguyên lý bình phương tối thiểu**; quan sát không cùng độ chính xác | **mục 3.6, Hình 14–15** | — |
| **điều 180** | **tr.214–215 · PDF 230–231** | phương trình chuẩn $P = Q = R = S = 0$; khi nào nghiệm không xác định; hàm phi tuyến | **mục 3.6** | — |
| điều 181 | tr.215–216 · PDF 231–232 | một ẩn: $A = \frac{am + a'm' + \cdots}{aa + a'a' + \cdots}$ và độ chính xác của nó | mục 4.4 | trường hợp riêng của công thức tổng quát |
| **điều 182–183** | **tr.216–219 · PDF 232–235** | độ chính xác của từng ẩn qua phép khử | **mục 3.7, 4.4** | — |
| **điều 184** | **tr.219–220 · PDF 235–236** | ví dụ số bốn phương trình ba ẩn | **mục 3.7, Hình 16–17, dựng lại, tìm ra hệ số in sai** | — |
| điều 185 | tr.220 · PDF 236 | thay hệ số lẻ bằng số tròn cho dễ tính | — | mẹo tính tay; không đổi phương pháp |
| **điều 186** | **tr.220–221 · PDF 236–237** | luỹ thừa chẵn khác; câu về Legendre "Paris 1806"; nguyên lý Laplace | **mục 3.8, Hình 18, dựng lại** | — |
| điều 187–189 | tr.222–224 · PDF 238–240 | quay lại bài toán quỹ đạo; phương pháp gần đúng ba giả thuyết | — | bài toán quỹ đạo |
| mục IV, điều 190 trở đi | PDF 241 trở đi | quỹ đạo có nhiễu loạn | — | bài toán quỹ đạo |

Tổng **17 dòng**.

### 11.5 Stigler 1981 — theo hình, bảng và mục

| Hình/Bảng/Mục | Trang | Nội dung | Ở mục nào của bài | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| **Table 1** | 467 | bốn cung, số in sai 76545.74 | **mục 3.9, Hình 19, dựng lại** | — |
| **Figure 1** | 468 | bản đồ cung kinh tuyến | **mục 3.9, Hình 20** | — |
| **Figure 2** | 468 | hình học cung trên góc phần tư | **mục 3.9, Hình 20** | — |
| **Table 2** | 470 | các cách khớp bậc một so với Gauss | **mục 3.9, Hình 21** | — |
| mục 1 Introduction | 465 | tranh chấp Legendre – Adrain – Gauss | mục 3.9 | Adrain không được bàn thêm: không có PDF |
| mục 2 The extant evidence | 465–467 | bốn bằng chứng; lá thư 1799; Lindenau và von Zach 1806–1807 | mục 3.9 | — |
| mục 3 The meridian arc | 467–472 | dựng lại lá thư 1799; công thức (3.1)–(3.8); kết luận | mục 3.9, dựng lại (3.1) | công thức (3.2)–(3.8) là các cách viết khác; Bảng 2 đã cho kết quả của chúng |
| Appendix | 472–473 | khai triển bậc hai của Bessel (A.1)–(A.8) | — | Stigler tự nói không dựng lại được số của Gauss bằng nó |

Tổng **8 dòng** (2 hình, 2 bảng, 3 mục, 1 phụ lục).

---

## 12. Tài liệu tham khảo và điều kiện tái lập

```
[1] A.-M. Legendre, Nouvelles méthodes pour la détermination des orbites des
    comètes, Firmin Didot, Paris, 1805. Mục LVII–LVIII tr.63–65; phụ lục
    « Sur la méthode des moindres quarrés » tr.72–80, ký ngày 6 mars 1805.
    → NGUỒN GỐC thứ nhất. Hình 2–10 cắt từ file này.
    AI algorithms/_papers/1805 - Legendre - Nouvelles methodes pour la determination des orbites des cometes.pdf
    Bản số hoá ETH Zürich: https://doi.org/10.3931/e-rara-35115

[2] C. F. Gauss, Theoria motus corporum coelestium in sectionibus conicis solem
    ambientium, Perthes & Besser, Hamburg, 1809. Quyển II mục III, điều 172–189.
    → NGUỒN GỐC thứ hai. Hình 11–18 cắt từ file này.
    AI algorithms/_papers/1809 - Gauss - Theoria motus corporum coelestium.pdf

[3] S. M. Stigler, "Gauss and the Invention of Least Squares",
    The Annals of Statistics 9(3):465–474, 1981.
    → Trọng tài lịch sử ở mục 3.9. Hình 19–21 cắt từ file này.
    AI algorithms/_papers/1981 - Stigler - Gauss and the Invention of Least Squares.pdf

[4] L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still
    outperform deep learning on tabular data?", NeurIPS 2022. arXiv:2207.08815
    → Câu trích ở mục 10.
    AI algorithms/_papers/2022 - Grinsztajn et al - Why do tree-based models still outperform deep learning on tabular data.pdf

[5] G. James, D. Witten, T. Hastie, R. Tibshirani, An Introduction to Statistical
    Learning, Springer, 2013, chương 3. — trình bày nhập môn về giả định ở mục 9.
    AI algorithms/_papers/2013 - James et al - An Introduction to Statistical Learning.pdf

[6] C. M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006,
    chương 3. — cách nhìn Bayes của cùng mô hình.
    AI algorithms/_papers/2006 - Bishop - Pattern Recognition and Machine Learning.pdf

[7] NumPy Developers, numpy.linalg.lstsq, NumPy Reference.
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
```

**Mọi con số** của bài in ra từ `tools/v_linreg.py` (13 mục) hoặc từ khối mã in ngay trong bài, kèm đầu ra
đã dán. Script ban đầu `tools/v_gd1ss.py` mục 370/380/420 cũng đo năm con số của mục 8–9, nhưng gãy trên
máy không có `sklearn` ngay ở mục 350; ba phép đo đó đã chuyển sang `v_linreg.py` mục 10–12 bằng numpy thuần,
cho cùng con số trừ Huber (mục 8).

**Máy dùng để viết bài:** numpy 2.4.4, OpenBLAS 0.3.31, Linux x86_64. Các con số của ca Vandermonde bậc 15
(mục 6.2, 7.2) và ba chữ số cuối của $\Sigma e$, $e^\top x$ **không tái lập được qua máy**; mọi con số khác tái
lập tới chữ số in.

**Hình tự vẽ.** Hình 1 và Hình 22 do chính khối mã in dưới hình sinh ra; `tools/fig_linreg.py` là bản
trích từ bài bằng `tools/extract_fig.py`, và `tools/sohinh.py` so md5 giữa hai đường sinh.

---

## 13. Kiểm tra trước khi đóng bài

| Khối bắt buộc | Ở đâu | Bằng chứng |
|---|---|---|
| **IO** — nhận gì, trả gì | mục 0.1 | bảng kiểu và kích thước; khối mã chạy được với đầu ra thật |
| **KN** — khái niệm tại chỗ | mục 0.2, 0.3 | 30 dòng thuật ngữ; ba bảng ký hiệu (hiện đại, Legendre, Gauss) có cột "đọc là gì"; bốn chỗ trùng tên |
| **SS** — trước nó làm sao, tắc ở đâu | mục 1 | cách cũ đo bằng số: tổng bình phương 7,0643 so với 1,4329, sai số lớn nhất 2,248 so với 0,868 phút cung |
| **CT** — gồm những gì | mục 2, 3, 4, 5 | hình ba cách nhìn; 20 trang gốc đã cắt; bốn hệ đánh số (a)–(d) đủ công thức, ký hiệu, ý nghĩa; bảng đi theo kích thước; pseudocode |
| **VS** — vì sao ăn thua | mục 8 | ba lý do từ ba nguồn, mỗi lý do một phép đo; bảng dịch một điểm 0 → +100 |
| **KQ** — bằng chứng | mục 3 | định lượng: mọi ví dụ số của Legendre, Gauss, Stigler dựng lại cạnh số in; định tính: bốn chiếc máy, ví dụ sao chổi 1769 |
| **GIÁ** — mất gì, khi nào đừng dùng | mục 9 | bảng 4 cột bảy dòng, mỗi dòng có nguồn; bảy tình huống không nên dùng |

| Phép kiểm | Kết quả |
|---|---|
| mọi con số truy được về script, khối mã trong bài, hoặc trang in của nguồn | có — `v_linreg.py` 13 mục |
| mọi ví dụ số của nguồn gốc được dựng lại | có — Legendre LVIII, Legendre kinh tuyến (a)–(d) và (e), Gauss 184, Stigler (3.1) |
| mọi hình và bảng của phần liên quan được liệt kê | có — mục 11.1, 11.3, 11.5 |
| mọi mục của phần liên quan được liệt kê | có — mục 11.2, 11.4, 11.5 |
| mọi hệ phương trình đánh số có công thức, ký hiệu, ý nghĩa | có — (a), (b), (c), (d) ở mục 3.3 |
| kết quả âm được báo cáo | có — Legendre tự bác 1/148 (mục 3.4); Gauss in sai hệ số (mục 3.7); Stigler không dựng lại được 1/187 (mục 3.9); số Vandermonde không tái lập qua máy (mục 6.2) |
| đoạn trích tiếng Anh đối chiếu với lớp text PDF | chỉ Grinsztajn — `kiemtrich.py`; Stigler là bản quét |
| đoạn trích tiếng Pháp, Latinh | **chép từ ảnh trang**, không có phép kiểm tự động |
| hình tự vẽ khớp md5 giữa bài và `tools/` | `sohinh.py` |
| dẫn chiếu PDF trỏ tới file có thật | `kiempdf.py` |

**Bốn chỗ bài này cố ý để lại, ghi ra thay vì để người đọc tự phát hiện:**

1. **Không dựng lại được con số 1/187 của Gauss năm 1799.** Stigler cũng không (tr.471: *"none of my attempts
   was successful in simultaneously reproducing Gauss's values"*). Bài không thử khai triển bậc hai.
2. **Phân số trước $\alpha D$ ở dòng suy ra công thức (a)** không đọc chắc được trên ảnh; bài dựa vào hệ số
   270/π in rõ và vào việc chỉ ³⁄₂ khớp bảng hệ (a) (mục 3.3).
3. **Hằng số của $E^{v}$ trong hệ (e)** của Legendre ngầm chứa độ dẹt 1/241,4 thay vì 1/320; bài ghi nhận, không
   truy tới nguyên nhân (mục 3.4).
4. **Đoạn trích tiếng Pháp và Latinh không có phép kiểm tự động** — lớp text là OCR vỡ. Chúng được chép tay từ
   ảnh trang và có thể còn lỗi chép.
