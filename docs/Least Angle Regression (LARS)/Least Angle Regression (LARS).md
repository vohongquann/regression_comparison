# Least Angle Regression (LARS)

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 410
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Bradley Efron, Trevor Hastie, Iain Johnstone, Robert Tibshirani
Date: April 1, 2004
url: https://arxiv.org/abs/math/0406456

> **LARS — hồi quy góc nhỏ nhất.** Một **thuật toán dựng ĐƯỜNG NGHIỆM**, không phải một mô
> hình dự đoán mới. Nó đi từ mô hình rỗng tới nghiệm bình phương tối thiểu bằng một số hữu
> hạn bước thẳng, và tại mỗi bước nó đi theo hướng **tạo góc bằng nhau** với mọi biến đang
> hoạt động. Đổi một dòng trong nó thì ra đường **lasso**; đổi một dòng khác thì ra đường
> **forward stagewise**. Ba thuật toán trông khác hẳn nhau hoá ra là ba biến thể của cùng
> một phép dựng hình học — đó là nội dung chính của bài báo gốc.

---

## 0. Trước khi đọc

### 0.1 Nhận gì, trả ra gì

Chỗ khác biệt lớn nhất so với mọi bài khác trong nhóm hồi quy nằm ở **đầu ra**: LARS không
trả về một vectơ hệ số, nó trả về **cả một đường** hệ số — mọi mô hình từ rỗng tới đầy đủ,
trong một lần chạy.

| | Cách hiểu thông thường | Ký hiệu | Ví dụ ở mục 2 |
|---|---|---|---|
| **Đầu vào** | một bảng $n$ mẫu, $m$ cột, **đã tâm hoá và chuẩn hoá cột** | $X\in\mathbb{R}^{n\times m}$ | $442\times10$ |
| **Đáp án khi huấn luyện** | một số cho mỗi mẫu, **đã trừ trung bình** | $y\in\mathbb{R}^{n}$ | 442 số |
| **Siêu tham số** | **không có** | — | — |
| **Đầu ra 1** | dãy điểm gãy — nơi tập biến hoạt động đổi | $\hat C_0 > \hat C_1 > \dots$ | 11 giá trị |
| **Đầu ra 2** | hệ số tại **từng** điểm gãy | $B\in\mathbb{R}^{m\times(K+1)}$ | $10\times11$ |
| **Đầu ra 3** | thứ tự các biến lần lượt vào | danh sách | `[3, 9, 4, 7, …, 1]` |

```
X            shape (442, 10)     mỗi cột: Σxᵢⱼ = 0 và Σxᵢⱼ² = 1
y            shape (442,)        Σyᵢ = 0

B            shape (10, 11)      ← MỖI CỘT là một mô hình hoàn chỉnh
                                   cột  0 = toàn 0 (mô hình rỗng)
                                   cột 10 = nghiệm bình phương tối thiểu đầy đủ
C_hat        shape (11,)         ← mức tương quan chung, giảm dần về 0
```

Năm điều dễ hiểu sai, nói rõ ngay:

* **LARS không có siêu tham số.** Không có $\lambda$ nào để dò. Nó tính *toàn bộ* đường một
  lần; việc **chọn điểm nào trên đường** mới là chỗ dùng kiểm định chéo, và đó là một bước
  riêng, không rẻ đi chút nào nhờ LARS.
* **Giữa hai điểm gãy, hệ số đi THẲNG.** Nên 11 cột mô tả đủ một đường liên tục vô hạn
  điểm — muốn hệ số ở giữa thì nội suy tuyến tính. Đây chính là lý do đường nghiệm rẻ tới
  vậy, và là chỗ LARS khác hẳn mọi thuật toán lặp.
* **Chuẩn hoá cột là BẮT BUỘC, không phải tuỳ chọn.** Toàn bộ thuật toán so sánh
  $\lvert x_j^{\top}r\rvert$ giữa các cột; cột nào đo bằng đơn vị lớn hơn sẽ luôn thắng.
* **"Chuẩn hoá" ở đây là $\lVert x_j\rVert_2 = 1$, KHÔNG phải "độ lệch chuẩn bằng 1".** Hai
  quy ước lệch nhau đúng một hệ số $\sqrt{n-1} = 21{,}000000$ trên bộ dữ liệu của bài báo.
  Dùng nhầm thì thuật toán vẫn chạy, đường vẫn đúng hình dạng, và **mọi con số vẫn lệch**.
* **LARS gốc KHÔNG bằng đường lasso.** Phải thêm một quy tắc — "hệ số nào chạm 0 thì rời tập
  hoạt động" — mới ra đường lasso. Trên chính bộ dữ liệu của bài báo, quy tắc đó **thật sự
  được kích hoạt**: biến 7 rời tập ở bước 10 rồi quay lại ở bước 12.

### 0.2 Bảng ký hiệu — và các chỗ trùng tên

Bảng ở mục 0.3 giải nghĩa các **từ**; bảng này giải nghĩa các **ký hiệu**, kèm cách đọc thành
tiếng. Bài này dùng nhiều chỉ số và nhiều dấu mũ hơn mọi bài khác trong nhóm, vì thuật toán
phải nói liên tục về "tập biến đang hoạt động ở bước này".

| Ký hiệu | Đọc là gì | Nghĩa | Chiều / kiểu |
|---|---|---|---|
| $n$, $m$ | "en", "em" | số mẫu; **số cột của $X$** | vô hướng, nguyên |
| $X$, $y$ | "ích lớn", "y" | bảng dữ liệu vào (đã tâm hoá, đã chuẩn hoá cột); đáp án đã trừ trung bình | $\mathbb{R}^{n\times m}$, $\mathbb{R}^{n}$ |
| $x_j$ | "ích chỉ số j" | **một cột** của $X$, chuẩn Euclid bằng 1 | $\mathbb{R}^{n}$ |
| $\hat\mu$ | "muy mũ" | dự đoán hiện tại, $X\hat\beta$. Thuật toán được mô tả bằng $\hat\mu$ chứ không bằng $\hat\beta$, vì nó **đi trong không gian dự đoán** | $\mathbb{R}^{n}$ |
| $\hat\beta$ | "bê-ta mũ" | vectơ hệ số hiện tại | $\mathbb{R}^{m}$ |
| $\hat c$ | "xê mũ" | vectơ **tương quan hiện tại**, $\hat c = X^{\top}(y-\hat\mu)$ | $\mathbb{R}^{m}$ |
| $\hat C$ | **"xê mũ LỚN"** | giá trị lớn nhất trong $\lvert\hat c_j\rvert$ — mức tương quan **chung** của các biến hoạt động | vô hướng |
| $\mathcal{A}$ | **"tập a"** (*active set*) | tập chỉ số các biến đang được di chuyển hệ số | tập con của $\{1,\dots,m\}$ |
| $A_{\mathcal A}$ | **"A tập a"** | **một SỐ**, $\left(\mathbf 1^{\top}G_{\mathcal A}^{-1}\mathbf 1\right)^{-1/2}$. Nó là khoảng cách từ gốc tới đơn hình mở rộng, và là tốc độ giảm của $\hat C$ | vô hướng, $\in(0,1]$ |
| $s_j$ | "ét chỉ số j" | dấu của $\hat c_j$, dùng để đổi hướng cột cho mọi tương quan cùng dương | $\pm1$ |
| $X_{\mathcal A}$ | "ích tập a" | ma trận các cột hoạt động **đã đổi dấu**: cột thứ $j$ là $s_jx_j$ | $n\times\lvert\mathcal A\rvert$ |
| $G_{\mathcal A}$ | "gờ tập a", **ma trận Gram** | $X_{\mathcal A}^{\top}X_{\mathcal A}$ — bảng tích vô hướng giữa mọi cặp cột hoạt động | $\lvert\mathcal A\rvert\times\lvert\mathcal A\rvert$ |
| $w_{\mathcal A}$ | "w tập a" | hệ số dựng hướng đi: $A_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1$ | $\mathbb{R}^{\lvert\mathcal A\rvert}$ |
| $u_{\mathcal A}$ | "u tập a" | **hướng cách đều góc** — vectơ đơn vị tạo góc bằng nhau với mọi cột hoạt động | $\mathbb{R}^{n}$, dài 1 |
| $a$ | "a" | $X^{\top}u_{\mathcal A}$ — tương quan giữa **hướng đi** và từng cột | $\mathbb{R}^{m}$ |
| $\hat\gamma$ | "gam-ma mũ" | **độ dài bước**: đi bao xa thì một biến ngoài bắt kịp | vô hướng, $>0$ |
| $\bar\gamma$ | "gam-ma ngang" | quãng đường từ chỗ đang đứng tới nghiệm bình phương tối thiểu **trên tập hiện tại** | vô hướng, $>0$ |
| $\tilde\gamma$ | "gam-ma ngã" | quãng đường tới chỗ một hệ số **chạm 0** — chỉ có ở bản lasso | vô hướng, $>0$ |
| $\rho_k$ | "rô chỉ số k" | $\hat\gamma_k/\bar\gamma_k$ — LARS đi được bao nhiêu phần của quãng đường tới OLS | $\in(0,1]$ |
| $\bar y_k$ | "y ngang chỉ số k" | hình chiếu của $y$ xuống không gian sinh bởi $k$ cột đầu tiên được chọn | $\mathbb{R}^{n}$ |
| $t$ | "tê" | quãng đường đã đi, $\sum_j\lvert\hat\beta_j\rvert$ — trục ngang của Hình 5 | vô hướng, $\ge0$ |
| $T(\hat\beta)$, $S(\hat\beta)$ | "tê của", "ét của" | chuẩn 1 của hệ số; tổng bình phương phần dư | vô hướng |
| $\varepsilon$ | "ép-xi-lon" | **bước đi** của Forward Stagewise. **Không phải sai số** | vô hướng, $>0$ |
| $\mathcal{C}_{\mathcal A}$ | "nón C tập a" | **nón lồi** sinh bởi các cột đã đổi dấu — mọi tổ hợp hệ số KHÔNG ÂM | tập con của $\mathbb{R}^{n}$ |
| $\mathcal{S}_{\mathcal A}$ | "đơn hình tập a" | **đơn hình mở rộng**: tổ hợp có tổng hệ số bằng 1, hệ số được phép âm | tập con của $\mathbb{R}^{n}$ |
| $\operatorname{df}$ | "đê-ép" | bậc tự do, định nghĩa (4.4) | vô hướng |
| $C_p$ | "xê pê" | tiêu chí chọn mô hình (4.5) | vô hướng |
| $\eta(\cdot\,;t)$ | "ê-ta" | **ngưỡng mềm**: kéo mỗi giá trị về 0 một lượng $t$ rồi cắt tại 0 | $\mathbb{R}\to\mathbb{R}$ |

**Toán tử và ký hiệu nền:**

| Ký hiệu | Đọc là gì | Nghĩa |
|---|---|---|
| $x_j^{\top}r$ | "ích j chuyển vị nhân rờ" | tích vô hướng $\sum_i x_{ij}r_i$. Vì cả hai vectơ đã chuẩn hoá, nó tỉ lệ với cosin góc giữa chúng — nên **"tương quan lớn" và "góc nhỏ" là cùng một chuyện**, và đó là chỗ cái tên *least angle* ra đời |
| $\lVert x_j\rVert_2=1$ | "chuẩn hai của ích j bằng một" | mỗi cột dài đúng 1 đơn vị |
| $\mathbf 1$ | "vectơ một" | vectơ toàn số 1, độ dài $\lvert\mathcal A\rvert$ |
| $\varnothing$ | "tập rỗng" | trạng thái ban đầu của $\mathcal A$ |
| $\mathcal{L}(X)$ | "không gian sinh bởi X" | tập mọi tổ hợp tuyến tính của các cột |
| $\nabla\cdot g$ | "đi-véc-giăng của g" | $\sum_i \partial g_i/\partial y_i$ — tổng các đạo hàm riêng theo cùng chỉ số |
| $\min^{+}$ | "min cộng" | lấy nhỏ nhất **chỉ trên các giá trị dương** |

#### Bảy chỗ trùng tên — bốn chỗ nằm ngay trong CHÍNH bài báo

Đây là phần đắt nhất của bảng ký hiệu, vì nó gây lỗi thật khi viết mã chứ không chỉ gây khó
đọc. Bốn dòng đầu là trùng tên **bên trong một tài liệu**, tức người đọc không có cách nào
tránh bằng việc "chỉ đọc một nguồn".

| Chữ | Ở chỗ này nghĩa là | Ở chỗ kia nghĩa là |
|---|---|---|
| $A$ | $A_{\mathcal A}$ trong (2.5) là **một SỐ** | $\mathcal A$ là **một TẬP**; và trong Phụ lục A.2, $A$ lại là **một KHỐI của ma trận chia khối** $G_{\mathcal A^{+}}$ |
| $T$ | $T(\hat\beta) = \sum_j\lvert\hat\beta_j\rvert$ trong (1.4) — **một số** | trong Bổ đề 7 tr.28, $\mathcal T$ là **một KHOẢNG MỞ của trục $t$** |
| $\varepsilon$ | **bước đi** của Stagewise trong (1.7) | $\varepsilon = y - \mu$ là **vectơ phần dư** của mô phỏng trong (3.16), và $\varepsilon^{*}$ là mẫu lấy lại từ nó |
| $C$ | $\hat C$ là **tương quan lớn nhất** (2.9) | $\mathcal C_{\mathcal A}$ là **nón lồi** (3.12), còn $C_p$ là **tiêu chí chọn mô hình** (4.5) |
| $m$ | bài báo dùng $m$ cho **số cột** | mọi bài khác của database dùng $p$; còn $m$ ở bài `L-BFGS` là **số cặp nhớ** |
| $\gamma$ | $\hat\gamma$ là bước LARS, $\bar\gamma$ là quãng tới OLS, $\tilde\gamma$ là quãng tới chỗ đổi dấu | ở bài `XGBoost`, $\gamma$ là **phạt mỗi lá**; ở bài `SVM` nó là **độ rộng hạt nhân** |
| $\alpha$ | thư viện `scikit-learn` gọi mức phạt là `alpha` | bài báo này **không dùng** chữ $\alpha$ ở đâu cả; cột `alphas` mà `lars_path` trả về chính là $\hat C_k$ **chia cho $n$** |

> **Ba chỗ dễ nhầm, nhắc lại cho chắc:**
>
> 1. **$\hat c$ và $\hat C$ là hai đại lượng khác nhau.** $\hat c$ (thường) là cả vectơ tương
>    quan; $\hat C$ (hoa) là **giá trị lớn nhất** trong đó. Bất biến trung tâm của thuật toán
>    viết gọn bằng hai ký hiệu này: mọi $j\in\mathcal A$ đều có $\lvert\hat c_j\rvert=\hat C$.
> 2. **$A_{\mathcal A}$ là một SỐ, $\mathcal A$ là một TẬP.** Chữ $A$ trong $A_{\mathcal A}$
>    không phải chữ $\mathcal A$ viết lại; nó là ký hiệu của một đại lượng vô hướng có nghĩa
>    hình học riêng (Bổ đề 5 tr.27).
> 3. **$\varepsilon$ ở đây là bước đi, không phải sai số.** Trong bảng ở mục 1.2,
>    $\varepsilon = 0{,}01$ nghĩa là Forward Stagewise nhích $0{,}01$ mỗi lần — và cần
>    **356 310** lần mới đi được 90 % quãng đường.

### 0.3 Mười hai từ cần biết trước khi đọc tiếp

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Nghĩa |
|---|---|
| **Phần dư** *(residual)* | phần của $y$ mà mô hình hiện tại chưa giải thích được: $r = y - \hat\mu$ |
| **Tương quan hiện tại** *(current correlation)* | tích vô hướng $x_j^{\top}r$ giữa một cột và phần dư, **sau khi** cột đã được tâm hoá và chuẩn hoá. Không phải hệ số tương quan Pearson chưa chuẩn hoá |
| **Tập hoạt động** *(active set)* | các biến mà thuật toán đang di chuyển hệ số. Ở LARS gốc nó chỉ lớn dần; ở bản lasso nó có thể co lại |
| **Cách đều góc** *(equiangular)* | hướng đi tạo góc bằng nhau với mọi cột trong tập hoạt động — chính là cái tên của thuật toán |
| **Điểm gãy** *(breakpoint, knot)* | nơi một biến vào hoặc rời tập hoạt động. Giữa hai điểm gãy, đường hệ số là một đoạn thẳng |
| **Đường nghiệm** *(solution path)* | tập tất cả các nghiệm khi ràng buộc chạy từ chặt nhất tới lỏng nhất |
| **Ngưỡng mềm** *(soft threshold)* | phép biến đổi $\eta(y;t)$ kéo mỗi giá trị về phía 0 một lượng $t$ rồi cắt tại 0. Với thiết kế trực giao, LARS **chính là** phép này (Bổ đề 1 tr.22) |
| **Đơn hình mở rộng** *(extended simplex)* | tập mọi tổ hợp $\sum_j s_jx_jP_j$ có $\sum_jP_j = 1$, **cho phép $P_j$ âm** (5.7) |
| **Nón lồi** *(convex cone)* | tập mọi tổ hợp $\sum_j s_jx_jP_j$ có $P_j\ge0$, **không ràng buộc tổng** (3.12) |
| **Thống kê thứ tự** *(order statistic)* | dãy giá trị đã xếp giảm dần: $\lvert y\rvert_{(1)}\ge\lvert y\rvert_{(2)}\ge\dots$ |
| **Bậc tự do** *(degrees of freedom)* | ở đây **không** phải "số tham số" mà là $\sum_i\operatorname{cov}(\hat\mu_i,y_i)/\sigma^2$ — đo xem dự đoán chạy theo dữ liệu bao nhiêu (4.4) |
| **Khả vi hầu khắp** *(almost differentiable)* | tính chất yếu hơn khả vi, vừa đủ để công thức Stein (4.12) dùng được. Bài báo định nghĩa lại nó ở Ghi chú A.1 tr.46 |

### 0.4 Chạy tay một ví dụ, in ra TỪNG bước

Ví dụ nhỏ nhất còn thấy được hình học: bốn quan sát, hai cột **vuông góc**. Biến 1 mạnh gấp
ba biến 2. Mọi con số dưới đây do [`tools/v_lars.py`](<../../../tools/v_lars.py>) mục 24 in ra —
không con số nào được gõ tay.

```
   X =
        [-0.707107, +0.000000]
        [+0.000000, -0.707107]
        [+0.707107, +0.000000]
        [+0.000000, +0.707107]
   y = [-2.121320, -0.707107, +2.121320, +0.707107]
   kiểm quy ước (1.1): Σy = 0.0e+00 · Σx_{i1} = 0.0e+00 · Σx_{i1}² = 1.000000 · Σx_{i2}² = 1.000000
   hai cột có TRỰC GIAO không? x₁ᵀx₂ = 0.0e+00

   ── BƯỚC 1 ──
   (2.8)  ĉ = Xᵀ(y − μ̂) = [+3.000000, +1.000000]
   (2.9)  Ĉ = max|ĉ_j| = 3.000000  ->  A = {1}
   (2.10) s_j = dấu(ĉ_j) = [+1]
   (2.5)  G_A = [[1.]]
          A_A = (1ᵀG_A⁻¹1)^(−1/2) = 1.000000
   (2.6)  w_A = A_A·G_A⁻¹·1 = [1.000000]
          u_A = X_A·w_A = [-0.707107, +0.000000, +0.707107, +0.000000]
   (2.7)  kiểm: X_Aᵀu_A = [1.000000] phải bằng A_A·1 = 1.000000 ; ‖u_A‖ = 1.000000
   (2.11) a = Xᵀu_A = [+1.000000, +0.000000]
   (2.13) j = 2, Ĉ−ĉ / A−a = (2.000000)/(1.000000) = 2.000000  <- hợp lệ
   (2.13) j = 2, Ĉ+ĉ / A+a = (4.000000)/(1.000000) = 4.000000  <- hợp lệ
   (2.12) γ̂ = 2.000000 ; μ̂ ← μ̂ + γ̂·u_A
          β̂ = [2.000000, 0.000000] ; Σ|β̂| = 2.000000
   (2.16) tương quan chung mới: Ĉ − γ̂·A_A = 1.000000 ; đo lại = 1.000000

   ── BƯỚC 2 ──
   (2.8)  ĉ = Xᵀ(y − μ̂) = [+1.000000, +1.000000]
   (2.9)  Ĉ = max|ĉ_j| = 1.000000  ->  A = {1, 2}
   (2.10) s_j = dấu(ĉ_j) = [+1, +1]
   (2.5)  G_A = [[1. 0.]  [0. 1.]]
          A_A = (1ᵀG_A⁻¹1)^(−1/2) = 0.707107
   (2.6)  w_A = A_A·G_A⁻¹·1 = [0.707107, 0.707107]
          u_A = X_A·w_A = [-0.500000, -0.500000, +0.500000, +0.500000]
   (2.7)  kiểm: X_Aᵀu_A = [0.707107, 0.707107] phải bằng A_A·1 = 0.707107 ; ‖u_A‖ = 1.000000
   (2.11) a = Xᵀu_A = [+0.707107, +0.707107]
   (2.13) không còn biến ngoài -> quy ước tr.10: γ̂ = Ĉ/A_A = 1.414214
   (2.12) γ̂ = 1.414214 ; μ̂ ← μ̂ + γ̂·u_A
          β̂ = [3.000000, 1.000000] ; Σ|β̂| = 4.000000
   (2.16) tương quan chung mới: Ĉ − γ̂·A_A = 0.000000 ; đo lại = 0.000000

   Nghiệm OLS của bài toán này: [3.000000, 1.000000]
   LARS dừng ở      : [3.000000, 1.000000] -> lệch 4.44e-16
```

Đọc lại hai bước đó bằng lời, vì đây là toàn bộ thuật toán:

1. **Hỏi cột nào ăn khớp với phần dư nhất.** Lúc đầu phần dư chính là $y$, và
   $\hat c = (3{,}000000;\ 1{,}000000)$ — biến 1 thắng.
2. **Đi theo biến đó, nhưng KHÔNG đi hết cỡ.** Forward Selection sẽ đi thẳng tới
   $\hat\beta_1 = 3$. LARS chỉ đi tới $\hat\beta_1 = 2{,}000000$ — đúng chỗ mà tương quan
   của biến 1 tụt xuống bằng tương quan của biến 2, cả hai bằng $1{,}000000$.
3. **Từ đó đi theo đường phân giác.** Hướng $u_{\mathcal A}$ tạo góc bằng nhau với cả hai
   cột: $X_{\mathcal A}^{\top}u_{\mathcal A} = (0{,}707107;\ 0{,}707107)$, hai thành phần
   **bằng nhau**, đúng như (2.7) đòi.
4. **Không còn biến nào ở ngoài để bắt kịp**, nên bước cuối dùng quy ước ở tr.10: đi trọn
   $\hat\gamma = \hat C/A_{\mathcal A} = 1{,}414214$, và đúng lúc đó tương quan chung về 0 —
   tức đã tới nghiệm bình phương tối thiểu $(3;\ 1)$.

Con số đáng nhớ nhất ở đây là **2,000000**. Nó không phải một tham số ai đặt ra, cũng không
phải kết quả của một vòng lặp dò dẫm: nó là nghiệm của một phương trình bậc nhất mà (2.13)
viết sẵn. Toàn bộ cái hay của LARS nằm ở chỗ **"đi tới khi có biến khác bắt kịp" là một câu
hỏi giải được bằng công thức**, chứ không phải một câu hỏi phải bò từng bước nhỏ để trả lời.

### 0.5 Nguồn của bài này, và những chỗ nguồn tự bất đồng

**Nguồn ruột — lấy đủ 100 %:**

| Nguồn | Có gì | Bài này lấy |
|---|---|---|
| **Efron, Hastie, Johnstone & Tibshirani (2004), *Least Angle Regression*, The Annals of Statistics 32(2):407–451** — `_papers/2004 - Efron et al - Least Angle Regression.pdf`, bản arXiv math/0406456v2, 48 trang | 9 hình · 1 bảng · **145 công thức đánh số** · 8 mục đánh số + phụ lục · 4 định lý · 17 bổ đề | **9/9 hình · 1/1 bảng · 145/145 công thức · 8/8 mục + phụ lục** |

Mọi con số trong bài sinh ra từ [`tools/v_lars.py`](<../../../tools/v_lars.py>) — 25 mục, chạy
trong **khoảng 25 giây**, đầu ra đầy đủ ở [`tools/_out_lars.txt`](<../../../tools/_out_lars.txt>).
Tám hình tự vẽ có mã in ngay trong bài; bản trích ra nằm ở
[`tools/fig_lars.py`](<../../../tools/fig_lars.py>).

> **Khai báo về mọi con số thời gian trong bài.** Chúng là **một lần chạy**, trên một máy
> cụ thể, với **số luồng BLAS đặt cứng bằng 1** (biến môi trường đặt trước khi nạp `numpy`).
> Đổi số luồng là đổi thứ tự cộng dồn dấu phẩy động, nên nó đổi cả chữ số cuối của các con
> số khác. Mọi con số **không** dính đồng hồ thì dựng lại được từng chữ số.

#### Mười bảy chỗ đã đối chiếu — sáu chỗ nguồn in sai hoặc tự bất đồng

Bài báo này dựng lại được rất tốt: phần lớn các khẳng định khớp **tuyệt đối**, kể cả những
khẳng định không ai thường kiểm. Danh sách dưới đây chia làm hai nửa.

**Nửa thứ nhất — khớp, và khớp tới mức đáng ngạc nhiên:**

| # | Khẳng định của bài báo | Đo lại |
|---|---|---|
| 1 | LARS đi hết đường trong $m$ bước; lasso tốn 12 bước "thay vì 10" (tr.12) | **10 và 12** — khớp |
| 2 | Stagewise cần **13** bước LARS có sửa đổi (tr.14) | **13** — khớp |
| 3 | Chỗ Stagewise thu nhỏ tập: $\mathcal A = \{3,9,4,7,2,10,5,8\}$ co về $\hat{\mathcal B} = \mathcal A - \{3,7\}$ (tr.14) | **khớp cả tập $\mathcal A$ lẫn hai phần tử bị bỏ** |
| 4 | Thứ tự biến vào là 3, 9, 4, 7, …, 1 (tr.4 và tr.9) | `[3, 9, 4, 7, 2, 10, 5, 8, 6, 1, 7]` — khớp |
| 5 | Tại $t = 1000$ chỉ các biến 3, 9, 4, 7 có mặt (tr.4) | **khớp** |
| 6 | Mô hình bậc hai (3.15) có "true $R^2$" bằng **0,416** (tr.15) | **0,415630** |
| 7 | Trên mô hình đó, LARS tốn **64** bước (tr.12) | **64** — khớp |
| 8 | $C_p$ nhỏ nhất tại $k = 7$ với $m = 10$ (chú thích Hình 7) | **$k = 7$** — khớp |
| 9 | Thiết kế trực giao cho ngưỡng mềm tại thống kê thứ tự (Bổ đề 1 tr.22) | lệch $1{,}78\times10^{-15}$ |
| 10 | $A_{\mathcal A}$ là khoảng cách từ gốc tới đơn hình mở rộng (Bổ đề 5 tr.27) | lệch $\le 4{,}44\times10^{-16}$, và **0/5110** cặp vi phạm tính đơn điệu |
| 11 | $\operatorname{trace}(M_k) = k$, và $M_k$ là đạo hàm cục bộ (tr.21) | **đúng 10 chữ số**, và khớp sai phân hữu hạn tới $10^{-7}$ |

**Nửa thứ hai — sáu chỗ in sai, và bảy chỗ đo ra khác con số được in:**

| # | Chỗ | Bài báo in | Đo lại |
|---|---|---|---|
| 12 | **(3.20) tr.16** | tử số $1-\rho_k^2$ | phải là $(1-\rho_k)^2$; bản in **sai tới 6,30 lần** |
| 13 | **(4.13) tr.21** | $\hat\mu_k = \bar y_k - \cot_k\cdot\lVert\bar y_{k+1}-\bar y_k\rVert$ | vế trái là VECTƠ, vế phải là vectơ trừ một SỐ — thiếu thừa số $u_k$ |
| 14 | **tr.3, ngay sau (1.1)** | *"numerical results are expressed in the original units"* | $t$ cuối in ở Hình 1 là **3460,00**, tức con số của quy ước (1.1); theo đơn vị gốc nó là **107,1213** — lệch **32,30 lần** |
| 15 | **tr.31** | *"Using Lemmas 3 and 4 it can be shown that the (T, S) curve is always convex"* | Bổ đề 3 nói về tính liên tục, Bổ đề 4 nói về dấu; hai bổ đề thật sự cho tính lồi là **7 và 9** |
| 16 | **tr.13** | *"there is no contradiction between (3.12) and (3.13)"* | (3.13) nằm ở **trang sau**; hai công thức đang so là (3.10) và (3.11) |
| 17 | **tr.14** | *"(3.13) $\hat\mu_{\mathcal A^{+}}-\hat\mu_{\mathcal A}=\hat\gamma u_{\hat{\mathcal B}}$ … **as in (3.13)***" | công thức tự dẫn chính nó |
| 18 | **tr.17** | *"As in (2.3), (3.5) the calculation of $\tilde\gamma$ and $\hat\gamma$ is easy"* | $\hat\gamma$ tính bằng **(2.13)**, không phải (2.3) |
| 19 | **tr.40, §7 điểm 4** | *"the simple approximation formula (4.12)"* | "xấp xỉ đơn giản" là **(4.9)**; (4.12) là công thức Stein |
| 20 | **Bảng 1 tr.3** | in $x_8$ của bệnh nhân 441 là **5** | dữ liệu có **4,79**; bảng là bản ĐÃ LÀM TRÒN mà chú thích không nói |
| 21 | **tr.12** | Lasso tốn **103** bước trên mô hình $m = 64$ | **104** |
| 22 | **tr.14** | Stagewise tốn **255** bước trên mô hình đó | **251** |
| 23 | **chú thích Hình 7** | $C_p$ nhỏ nhất tại $k = 16$ với $m = 64$ | **$k = 15$**; tại $k=16$ giá trị cao hơn cực tiểu 1,6341 |
| 24 | **chú thích Hình 1** | hai bảng *"differ slightly for large t as shown in the track of covariate 8"* | đo ba cách, **không cách nào** chỉ vào biến 8 |
| 25 | **tr.15, Hình 5** | đỉnh **0,963** tại $k=10$; Forward Selection đỉnh **0,950** tại $k=3$ | **0,966422** tại $k=11$; **0,956009** tại $k=3$ |
| 26 | **tr.15** | $\hat\mu^{(40)*}$ trung bình có **33,23** hệ số khác 0 với Stagewise | **35,59** (Lasso 35,52 so với 35,83 — khớp; LARS 40 — khớp) |

Và hai điều không phải lỗi của bài báo nhưng người đọc hôm nay cần biết:

* **Xấp xỉ $\operatorname{df} \doteq k$ hụt một cách có hệ thống trên chính bộ diabetes.** Đo
  bằng đúng thủ tục bootstrap (4.6)–(4.8) với $B = 500$: $\operatorname{df}$ thấp hơn $k$ một
  cách **có ý nghĩa thống kê** ở $k = 6, 7, 8$. Đó không mâu thuẫn với Định lý 4, vì Định lý
  4 đòi **điều kiện nón dương (4.11)** — mà bộ diabetes **vi phạm điều kiện đó ở 418 trên
  1023 tập con**, tức 40,86 %.
* **Chế độ `method="lar"` của `scikit-learn` PHÂN KỲ trên chính mô hình bậc hai của bài
  báo.** Nó chạy 96 bước thay vì 64, $\sum_j\lvert\hat\beta_j\rvert$ ở bước cuối lên tới
  $8{,}3768\times10^{14}$, và **không phát một cảnh báo nào**. Bản viết từ đầu theo đúng
  (2.4)–(2.13) cho 64 bước và về đúng nghiệm bình phương tối thiểu. Mục 3.4 của bài này nói
  kỹ chỗ này.

---

## 1. Trước LARS người ta làm thế nào, và tắc ở đâu

### 1.1 Hai thuật toán cũ, hai kiểu hỏng ngược nhau

Bài toán đặt ra ở tr.2: có nhiều biến, muốn chọn một tập con nhỏ mà dự đoán vẫn tốt. Hai
thuật toán tự động đã có sẵn, và chúng hỏng theo hai cách đối nghịch.

| Thuật toán | Mỗi bước làm gì | Hỏng ở đâu |
|---|---|---|
| **Forward Selection** | chọn biến tương quan nhất rồi khớp **HẾT CỠ** vào nó, chiếu mọi biến còn lại vuông góc với nó, lặp lại | quá tham. Khớp hết cỡ vào $x_{j_1}$ có thể **xoá sạch** phần đóng góp của một biến tốt nhưng tương quan với $x_{j_1}$. Bài báo gọi thẳng là *"a dangerously greedy algorithm"* (tr.16) |
| **Forward Stagewise** | chọn biến tương quan nhất rồi chỉ nhích **một bước $\varepsilon$ tí xíu**, lặp lại | quá chậm. Số bước tỉ lệ **nghịch** với $\varepsilon$, và muốn đường mượt gấp mười thì trả giá gấp mười, mãi mãi |

> *"Forward Selection is an aggressive fitting technique that can be overly greedy, perhaps
> eliminating at the second step useful predictors that happen to be correlated with
> $x_{j_1}$."* — tr.2
>
> Dịch: *Forward Selection là một kỹ thuật khớp hung hăng, có thể quá tham, và có khi ngay ở
> bước thứ hai đã loại mất những biến dự báo có ích chỉ vì chúng tương quan với $x_{j_1}$.*

> *"Forward Stagewise, as described below, is a much more cautious version of Forward
> Selection, which may take thousands of tiny steps as it moves toward a final model."* — tr.2
>
> Dịch: *Forward Stagewise, mô tả dưới đây, là một bản thận trọng hơn hẳn của Forward
> Selection, và nó có thể tốn hàng nghìn bước tí hon trên đường tới mô hình cuối.*

### 1.2 "Hàng nghìn bước" là một tính từ trá hình — đo nó bằng số

Câu *"thousands of tiny steps"* nghe như một lời than. Đo thật trên chính bộ dữ liệu của bài
báo (442 bệnh nhân, 10 biến), cho Forward Stagewise chạy tới **90 % quãng đường** tới nghiệm
bình phương tối thiểu:

| Thuật toán | Số bước để đi 90 % đường |
|---|---:|
| Forward Stagewise, $\varepsilon = 10$ | **không tới được** sau 500 000 bước |
| Forward Stagewise, $\varepsilon = 1$ | 3 566 |
| Forward Stagewise, $\varepsilon = 0{,}1$ | 35 634 |
| Forward Stagewise, $\varepsilon = 0{,}01$ | **356 310** |
| **LARS** | **10 — và đó là CẢ đường, không phải 90 %** |

Đọc bảng này theo chiều dọc. Ba dòng giữa cho thấy số bước **tỉ lệ nghịch đúng với
$\varepsilon$**: chia $\varepsilon$ cho 10 thì nhân số bước lên 10. Không có điểm dừng nào
trong cái đánh đổi đó — nó cứ thế mãi.

Dòng đầu là dòng dạy được nhiều nhất, và nó không phải chuyện $\varepsilon$ "quá lớn nên
kém chính xác". Với $\varepsilon = 10$, thuật toán **không bao giờ tới đích**, kể cả sau
500 000 bước. Lý do: gần nghiệm bình phương tối thiểu, mọi $\lvert\hat c_j\rvert$ đều nhỏ hơn
$\varepsilon$, nên hệ số nhảy $+\varepsilon$ rồi $-\varepsilon$ rồi lại $+\varepsilon$, dao
động quanh đích mà $\sum_j\lvert\hat\beta_j\rvert$ không tăng thêm. Bước quá thô không làm
đường xấu đi một chút — nó làm thuật toán **không hội tụ**.

Điều đó dẫn thẳng tới một phép đo về chính Hình 1 của bài báo, mục 3.1 sẽ nói kỹ: bài báo
viết bảng phải của Hình 1 được dựng bằng **6000 bước Stagewise**, và trục ngang của nó chạy
tới $t = 3459{,}98$ như bảng trái. Quét $\varepsilon$ trên lưới $0{,}4$ tới $3{,}0$, giá trị
đi xa nhất trong đúng 6000 bước là $\varepsilon = 0{,}80$, và nó chỉ tới **97,8 %** quãng
đường.

Mã sinh mọi con số trong mục này: [`tools/v_lars.py`](<../../../tools/v_lars.py>) mục 7.

### 1.3 Ý tưởng của LARS, nói bằng một câu

Thay vì bò từng bước $\varepsilon$ rồi mỗi lần hỏi *"đã tới chưa"*, hãy **giải xem còn bao xa
thì có biến khác bắt kịp**, rồi nhảy thẳng tới đó.

Câu đó nghe hiển nhiên. Chỗ khó — và chỗ đóng góp thật của bài báo — là **"đi theo hướng
nào"**. Forward Stagewise với $\varepsilon\to0$ không đi theo một biến nào cả: khi hai biến
có tương quan bằng nhau, nó nhảy qua nhảy lại giữa chúng, và quỹ đạo trung bình là một hướng
**ở giữa**. LARS viết thẳng ra cái hướng ở giữa đó — hướng **cách đều góc** — nên nó tính
được độ dài bước bằng một công thức đóng.

Ba hệ quả, mỗi hệ quả là một mục của bài báo:

1. Hướng cách đều góc giữ cho mọi biến hoạt động **cùng một mức tương quan**, và mức đó giảm
   tuyến tính. Đó là bất biến của thuật toán (mục 2.2 của bài này).
2. Thêm một quy tắc "hệ số chạm 0 thì rời tập" thì đường LARS **trở thành** đường lasso
   (mục 2.3) — và đó là cách tính lasso rẻ hơn quy hoạch toàn phương một bậc độ lớn.
3. Thay hướng cách đều góc bằng **hình chiếu của nó vào nón lồi** thì đường LARS **trở
   thành** đường forward stagewise lý tưởng (mục 2.4).

### 1.4 Bảng được — mất

> | Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
> |---|---|---|---|
> | Đi hết đường thay vì xấp xỉ | 356 310 bước → **10 bước** | phải giải một hệ tuyến tính $k\times k$ ở **mỗi** điểm gãy; một bước LARS đắt hơn một bước Stagewise rất nhiều | mục 1.2 · mục 2.2 |
> | Không còn tham số $\varepsilon$ | nghiệm xác định, không phụ thuộc lưới | đổi lại **nhạy với suy biến**: hai cột trùng nhau làm $G_{\mathcal A}$ không nghịch đảo được | mục 5.2 |
> | Chi phí cùng bậc với một lần khớp OLS | tỉ số phép tính **1,98 lần**, không tăng theo $m$ | chỉ đúng khi $m < n$ và khi cài đặt bằng cập nhật Cholesky; bản ngây thơ chậm hơn **3,3 lần** | mục 2.11 |
> | Cho luôn cả đường lasso và đường stagewise | thêm đúng một quy tắc mỗi đường | ba đường **không** trùng nhau vô điều kiện; chúng chỉ trùng dưới điều kiện nón dương (4.11) | mục 2.3 · 2.4 · 2.7 |
> | Ước lượng bậc tự do rẻ | $\operatorname{df}\approx k$, không tốn thêm phép tính nào | là **xấp xỉ**; trên chính bộ diabetes nó hụt có ý nghĩa ở $k = 6,7,8$ | mục 2.7 |
> | Thay được kiểm định chéo | **KHÔNG** — vẫn phải chọn một điểm trên đường | LARS chỉ làm rẻ việc *dựng* đường; việc *chọn* điểm tốn đúng như cũ | mục 5.1 |
> | Là lựa chọn mặc định ngày nay | **không còn**. Hạ theo toạ độ với khởi động ấm rẻ hơn cho một lưới $\lambda$ dày | LARS mạnh nhất khi câu hỏi là *"các điểm gãy nằm chính xác ở đâu"* | mục 6 |

Hai dòng cuối là hai dòng thua, và chúng nên được đọc cùng nhau. LARS thắng rõ khi câu hỏi
là *"toàn bộ đường nghiệm, chính xác tới từng điểm gãy"*; nó không thắng khi câu hỏi chỉ là
*"một mô hình tốt, càng nhanh càng hay"*.

---

## 2. Phần ruột — đi theo đúng thứ tự bài báo

> **Cách đọc mục này.** Bài báo có **145 công thức đánh số**, trong đó khoảng một nửa là các
> bước đại số bên trong chứng minh của §5, §6 và Phụ lục. Mục này viết ra **mọi** công thức
> đánh số, nhưng theo hai mức: công thức nào **phát biểu phương pháp hoặc một kết quả** thì
> có đủ ba phần — công thức · mọi ký hiệu kèm chiều · **ý nghĩa trừu tượng**; công thức nào
> chỉ là một bước biến đổi bên trong chứng minh thì nằm trong bảng gom của chính chứng minh
> đó, kèm cột *"nó nói gì"* và cột *"dùng ở đâu"*. Lý do: viết ba phần cho một dòng khai
> triển trung gian là làm loãng đúng những chỗ đáng đọc.

### 2.1 Bài toán và hai đối thủ — §1 của bài báo, tr.2–5

#### Bảng 1 và bộ dữ liệu

![Bảng 1 của Efron 2004, trang 3](<paper_lars_efron2004_tr3_bang1_du_lieu_diabetes.png>)

**Hình 1 — Bảng 1 của Efron, Hastie, Johnstone & Tibshirani (2004), trang 3.** Nguồn:
*Least Angle Regression*, The Annals of Statistics 32(2):407–451 —
[`_papers/2004 - Efron et al - Least Angle Regression.pdf`](<../../_papers/2004 - Efron et al - Least Angle Regression.pdf>).

**Chú thích gốc, nguyên văn:** *"Diabetes study: 442 diabetes patients were measured on 10
baseline variables; a prediction model was desired for the response variable, a measure of
disease progression one year after baseline."*

*Dịch: Nghiên cứu tiểu đường: 442 bệnh nhân tiểu đường được đo trên 10 biến nền; người ta
cần một mô hình dự đoán cho biến đáp ứng, là một thước đo mức tiến triển bệnh sau một năm.*

**Nhận xét của chính tác giả, nguyên văn, tr.2:** *"Ten baseline variables, age, sex, body
mass index, average blood pressure and six blood serum measurements, were obtained for each
of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of
disease progression one year after baseline. The statisticians were asked to construct a
model that predicted response y from covariates x1, x2, . . ., x10. Two hopes were evident
here, that the model would produce accurate baseline predictions of response for future
patients and that the form of the model would suggest which covariates were important
factors in disease progression."*

*Dịch: Mười biến nền — tuổi, giới tính, chỉ số khối cơ thể, huyết áp trung bình và sáu phép
đo huyết thanh — được thu thập cho mỗi bệnh nhân trong $n = 442$ bệnh nhân tiểu đường, cùng
với biến đáp ứng cần quan tâm là một thước đo định lượng mức tiến triển bệnh sau một năm.
Các nhà thống kê được yêu cầu dựng một mô hình dự đoán $y$ từ các biến $x_1,\dots,x_{10}$.
Hai kỳ vọng hiện rõ ở đây: mô hình cho dự đoán chính xác cho bệnh nhân tương lai, và dạng
của mô hình gợi ra biến nào là yếu tố quan trọng trong tiến triển bệnh.*

**Cách đọc bảng.** Tám dòng in ra là sáu bệnh nhân đầu và hai bệnh nhân cuối; 434 dòng giữa
bị thay bằng dấu chấm. Dựng lại từ dữ liệu gốc thì tám dòng đó khớp — **nhưng chỉ sau khi
làm tròn**:

```
      BN  |  x1  x2   x3     x4    x5    x6     x7   x8     x9     x10 |   y
        1 |     59      2   32.1    101    157   93.2     38      4 4.8598     87 |   151
        2 |     48      1   21.6     87    183  103.2     70      3 3.8918     69 |    75
        3 |     72      2   30.5     93    156   93.6     41      4 4.6728     85 |   141
        4 |     24      1   25.3     84    198  131.4     40      5 4.8903     89 |   206
        5 |     50      1     23    101    192  125.4     52      4 4.2905     80 |   135
        6 |     23      1   22.6     89    139   64.8     61      2 4.1897     68 |    97
      441 |     36      1     30     95    201  125.2     42   4.79 5.1299     85 |   220
      442 |     36      1   19.6     71    250  133.2     97      3 4.5951     92 |    57

   Bảng 1 in x8 của bệnh nhân 441 là 5; dữ liệu có 4.79.
   Bảng 1 in x9 của bệnh nhân 1 là 4.9; dữ liệu có 4.8598.
   -> Bảng 1 là bản ĐÃ LÀM TRÒN, và chú thích bảng không nói điều đó.
```

**Điều bảng KHÔNG nói ra.** Cột $x_8$ trông như một số nguyên ở cả tám dòng in ra, nên người
đọc dễ kết luận nó là một biến rời rạc. Nó không phải: bệnh nhân 441 có $x_8 = 4{,}79$. Chỉ
có $x_2$ (giới tính) mới thật sự là biến hai giá trị — và đó chính là biến mà §3.3 loại khỏi
danh sách bình phương khi dựng mô hình bậc hai.

#### Công thức (1.1) — quy ước chuẩn hoá, và một câu về đơn vị

![Công thức (1.1) của Efron 2004, trang 3](<paper_lars_efron2004_tr3_ct1_1_chuan_hoa.png>)

**Hình 2 — công thức (1.1), trang 3.** Nguồn: cùng bài báo.

$$\sum_{i=1}^{n}y_i = 0,\qquad \sum_{i=1}^{n}x_{ij} = 0,\qquad
\sum_{i=1}^{n}x_{ij}^{2} = 1 \quad\text{với } j = 1,2,\dots,m. \tag{1.1}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $y_i$ | vô hướng | đáp ứng của mẫu thứ $i$, đã trừ trung bình |
| $x_{ij}$ | vô hướng | giá trị biến $j$ ở mẫu $i$, đã tâm hoá và chia cho chuẩn Euclid của cột |
| $n$, $m$ | nguyên | số mẫu (442), số biến (10) |

**Ý nghĩa trừu tượng.** Ba đẳng thức này **không** phải thủ tục làm sạch dữ liệu — chúng là
điều kiện để mọi phép so sánh trong thuật toán có nghĩa. Đẳng thức thứ nhất bỏ hệ số chặn ra
khỏi bài toán. Hai đẳng thức sau đặt **mọi cột lên cùng một thước đo**, nên câu hỏi *"cột nào
tương quan mạnh nhất với phần dư"* mới là một câu hỏi về **dữ liệu** chứ không phải về **đơn
vị đo**. Đổi $x_j$ từ mét sang xentimét mà không chuẩn hoá lại thì thứ tự các biến vào đổi
theo.

Kiểm lại trên dữ liệu thật:

```
   Chuẩn hoá theo (1.1): Σyᵢ = -3.411e-12 · Σxᵢⱼ lớn nhất = 9.999e-14 · Σxᵢⱼ² = 1.0000000000
   `load_diabetes()` của sklearn ĐÃ theo đúng quy ước này: lệch 2.776e-17
   Nếu dùng nhầm quy ước 'độ lệch chuẩn = 1' thì mỗi cột dài √(n−1) = 21.000000 lần
```

> **Chỗ nguồn tự bất đồng — số 14 trong bảng ở mục 0.5.** Ngay sau (1.1), bài báo viết:
>
> *"This is assumed to be the case in the theory which follows, except that numerical results
> are expressed in the original units of the diabetes example."*
>
> *Dịch: Điều này được giả định trong phần lý thuyết sau đây, chỉ trừ việc các kết quả bằng
> số được biểu diễn theo đơn vị gốc của ví dụ tiểu đường.*
>
> Nhưng con số $t = 3460{,}00$ in ở tr.4 và vẽ trên trục ngang Hình 1 **là con số của quy ước
> (1.1)**, không phải của đơn vị gốc:
>
> ```
>    Σ|β̂| của nghiệm OLS, theo quy ước (1.1)      : 3459.9776
>    Σ|β̂| của nghiệm OLS, theo ĐƠN VỊ GỐC         : 107.1213
>    ... hai giá trị cách nhau 32.30 lần.
> ```
>
> Đây là chỗ đã được ghi lại một lần ở bài [Lasso Regression](<../Lasso Regression/Lasso Regression.md>) khi bài
> đó dựng lại Bảng 1 của Tibshirani 1996; nay nó hiện ra lần thứ hai, ở chính Hình 1 của bài
> báo này. **Một câu tuyên bố về đơn vị là một dữ kiện phải kiểm, y như một con số.**

#### Công thức (1.2)–(1.5) — bài toán lasso

$$\hat\mu = \sum_{j=1}^{m}x_j\hat\beta_j = X\hat\beta,
\qquad [\,X_{n\times m} = (x_1,x_2,\dots,x_m)\,] \tag{1.2}$$

$$S(\hat\beta) = \lVert y - \hat\mu\rVert^{2} = \sum_{i=1}^{n}(y_i - \hat\mu_i)^{2} \tag{1.3}$$

$$T(\hat\beta) = \sum_{j=1}^{m}\lvert\hat\beta_j\rvert \tag{1.4}$$

Hai đại lượng (1.3) và (1.4) là hai thứ đối nghịch mà cả bài toán phải cân: $S$ đo **mô hình
khớp dữ liệu tới đâu**, $T$ đo **mô hình tốn bao nhiêu độ lớn hệ số**. Mọi thuật toán trong
bài báo này đều là một cách đi trên mặt phẳng $(T, S)$, và Hình 43 vẽ chính mặt phẳng ấy.

![Công thức (1.5) của Efron 2004, trang 4](<paper_lars_efron2004_tr4_ct1_5_lasso.png>)

**Hình 3 — công thức (1.5), trang 4.** Nguồn: cùng bài báo.

$$\textbf{Lasso: } \text{cực tiểu hoá } S(\hat\beta)
\text{ với ràng buộc } T(\hat\beta)\le t \tag{1.5}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\hat\mu$ | $\mathbb{R}^{n}$ | vectơ dự đoán |
| $S(\hat\beta)$ | vô hướng $\ge0$ | tổng bình phương phần dư |
| $T(\hat\beta)$ | vô hướng $\ge0$ | chuẩn 1 của vectơ hệ số — "tổng độ lớn hệ số" |
| $t$ | vô hướng $\ge0$ | trần đặt lên $T$; $t = 0$ cho mô hình rỗng, $t$ đủ lớn cho nghiệm OLS |

**Ý nghĩa trừu tượng.** (1.5) nói rằng lasso **không** là một hàm mục tiêu mới mà là bài toán
bình phương tối thiểu cũ **cộng một ngân sách**. Ngân sách đo bằng chuẩn 1, và chính chuẩn 1
— chứ không phải chuẩn 2 — là thứ đẩy một số hệ số về **đúng 0**. Đọc theo hình học: tập
$\{\beta : \sum_j\lvert\beta_j\rvert\le t\}$ là một khối kim cương có **đỉnh nhọn nằm trên
các trục toạ độ**, nên mặt đẳng trị của $S$ khi phình ra rất hay chạm nó ở một đỉnh — mà ở
đỉnh thì phần lớn toạ độ bằng 0.

Chỗ đáng nhớ: LARS **không giải** (1.5) bằng cách tối ưu. Nó đi dọc đường nghiệm và dùng
điều kiện dừng để tính ra các điểm gãy — và mục 2.9 chứng minh hai cách cho cùng một kết
quả.

#### Công thức (1.6)–(1.7) — Forward Stagewise

![Công thức (1.6)–(1.7) của Efron 2004, trang 5](<paper_lars_efron2004_tr5_ct1_6_1_7_stagewise.png>)

**Hình 4 — công thức (1.6) và (1.7), trang 5.** Nguồn: cùng bài báo.

$$\hat c = c(\hat\mu) = X^{\top}(y-\hat\mu) \tag{1.6}$$

$$\hat j = \arg\max_j\lvert\hat c_j\rvert,
\qquad \hat\mu \to \hat\mu + \varepsilon\cdot\operatorname{sign}(\hat c_{\hat j})\cdot x_{\hat j} \tag{1.7}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\hat c$ | $\mathbb{R}^{m}$ | **vectơ tương quan hiện tại** — một số cho mỗi cột |
| $\hat j$ | chỉ số | cột đang ăn khớp nhất với phần dư |
| $\varepsilon$ | vô hướng $>0$ | độ dài mỗi bước nhích. **Không phải sai số** |

**Ý nghĩa trừu tượng.** (1.7) là toàn bộ Forward Stagewise, và nó ngắn tới mức dễ bị coi
thường. Điều đáng chú ý là **chữ "small" trong câu của tác giả mang toàn bộ sức nặng**:

> *"'Small' is important here: the 'big' choice $\varepsilon = \lvert\hat c_{\hat j}\rvert$
> leads to the classic Forward Selection technique, which can be overly greedy, impulsively
> eliminating covariates which are correlated with $x_{\hat j}$."* — tr.5
>
> *Dịch: Chữ "nhỏ" ở đây là quan trọng: lựa chọn "lớn"
> $\varepsilon = \lvert\hat c_{\hat j}\rvert$ dẫn thẳng tới kỹ thuật Forward Selection cổ
> điển, vốn có thể quá tham, bốc đồng loại bỏ những biến có tương quan với $x_{\hat j}$.*

Nghĩa là **Forward Selection và Forward Stagewise là cùng một dòng mã, khác nhau đúng một
tham số**. Một đầu của thang $\varepsilon$ cho thuật toán tham nhất, đầu kia cho thuật toán
thận trọng nhất, và LARS sẽ nằm ở giữa — nhưng nằm giữa theo một cách **tính được**, không
phải bằng cách chọn một $\varepsilon$ trung dung.

#### Hình 1 của bài báo — hai đường nghiệm gần như trùng nhau

![Hình 1 của Efron 2004, trang 4](<paper_lars_efron2004_tr4_hinh1_lasso_va_stagewise.png>)

**Hình 5 — Hình 1 của Efron 2004, trang 4.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"Estimates of regression coefficients $\hat\beta_j$,
$j = 1, 2, \dots, 10$, for the diabetes study. (Left panel) Lasso estimates, as a function of
$t = \sum_j\lvert\hat\beta_j\rvert$. The covariates enter the regression equation sequentially
as $t$ increases, in order $j = 3, 9, 4, 7, \dots, 1$. (Right panel) The same plot for Forward
Stagewise Linear Regression. The two plots are nearly identical, but differ slightly for large
$t$ as shown in the track of covariate 8."*

*Dịch: Ước lượng hệ số hồi quy $\hat\beta_j$, $j = 1,\dots,10$, cho nghiên cứu tiểu đường.
(Bảng trái) Ước lượng lasso, vẽ theo $t = \sum_j\lvert\hat\beta_j\rvert$. Các biến lần lượt
đi vào phương trình hồi quy khi $t$ tăng, theo thứ tự $j = 3, 9, 4, 7, \dots, 1$. (Bảng phải)
Cùng đồ thị ấy cho hồi quy tuyến tính Forward Stagewise. Hai đồ thị gần như y hệt nhau, chỉ
khác chút ít ở $t$ lớn, thấy được qua vết của biến số 8.*

**Nhận xét của chính tác giả, nguyên văn, tr.5:** *"The right panel of Figure 1 shows the
coefficient plot for Stagewise applied to the diabetes data. The estimates were built up in
6000 Stagewise steps [making $\varepsilon$ in (1.7) small enough to conceal the
'Etch-a-Sketch' staircase seen in Figure 2, Section 2]. The striking fact is the similarity
between the Lasso and Stagewise estimates. Although their definitions look completely
different, the results are nearly, but not exactly, identical."*

*Dịch: Bảng phải của Hình 1 vẽ hệ số của Stagewise áp cho dữ liệu tiểu đường. Các ước lượng
được dựng qua 6000 bước Stagewise [chọn $\varepsilon$ trong (1.7) đủ nhỏ để giấu đi cái bậc
thang kiểu "Etch-a-Sketch" thấy ở Hình 2, mục 2]. Điều nổi bật là sự giống nhau giữa ước
lượng Lasso và Stagewise. Dù định nghĩa của chúng trông hoàn toàn khác nhau, kết quả gần
như — nhưng không hoàn toàn — trùng nhau.*

**Cách đọc hình.** Trục ngang là $t = \sum_j\lvert\hat\beta_j\rvert$, trục dọc là hệ số, số
ở mép phải là số hiệu biến. Ba điều nhìn thấy được, và cả ba dựng lại được bằng số:

* **Thứ tự các đường tách khỏi trục 0** chính là thứ tự biến vào. Dựng lại cho
  `[3, 9, 4, 7, 2, 10, 5, 8, 6, 1, 7]` — bốn biến đầu và biến thứ mười khớp đúng chú thích.
* **Mỗi đường gồm các đoạn thẳng nối nhau**, không cong. Đó là hệ quả của Bổ đề 7 tr.28.
* **Ở bảng trái có một mũi tên** đánh dấu chỗ một đường chạm lại 0 rồi đi tiếp — đó là biến
  7, và đó là chỗ duy nhất sửa đổi lasso (3.6) được kích hoạt trên bộ này.

**Điều hình KHÔNG nói ra, và đây là hai chỗ đo được:**

1. **Chú thích chỉ vào biến 8, phép đo thì không.** Đo lệch giữa hai đường theo từng biến,
   bằng ba cách:

   ```
      trên CẢ đường  — biến 7 : lệch lớn nhất  52.3827 tại t = 2062.1
      trên CẢ đường  — biến 8 : lệch lớn nhất  43.0596 tại t = 2062.1
      ...
      t lớn        — biến 5 : lệch lớn nhất  31.0181
      t lớn        — biến 7 : lệch lớn nhất  30.3483
      t lớn        — biến 8 : lệch lớn nhất  24.2453
      ...
      sau 6000 bước, Σ|β̂| = 3012.4872 (87.1% đường); lệch với đường lasso theo biến:
         biến 3 :   2.8382
         biến 6 :   1.7287
   ```

   Ba cách đo cho ba câu trả lời khác nhau, và **không cách nào chỉ vào biến 8**. Cách thứ
   ba là cách công bằng nhất với bài báo, vì nó dùng đúng bản $\varepsilon$ hữu hạn 6000
   bước; ở đó hai đường sát nhau tới mức lệch lớn nhất chỉ **2,8382** trên thang hệ số cỡ
   800 — tức nhận xét *"nearly identical"* đúng hơn hẳn nhận xét về biến 8.
2. **6000 bước không phủ hết trục ngang.** Quét $\varepsilon$ từ 0,4 tới 3,0, giá trị đi xa
   nhất trong đúng 6000 bước là $\varepsilon = 0{,}80$ và nó chỉ tới **97,8 %** quãng đường.
   Nguyên nhân là chỗ đã nói ở mục 1.2: gần nghiệm OLS, bước $\varepsilon$ dao động thay vì
   tiến.

### 2.2 Thuật toán LARS — §2 của bài báo, tr.5–10

#### Hình 2 — hình học với hai biến

![Hình 2 của Efron 2004, trang 6](<paper_lars_efron2004_tr6_hinh2_hinh_hoc_hai_bien.png>)

**Hình 6 — Hình 2 của Efron 2004, trang 6.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"The LARS algorithm in the case of $m = 2$ covariates;
$\bar y_2$ is the projection of $y$ into $\mathcal L(x_1, x_2)$. Beginning at $\hat\mu_0 = 0$,
the residual vector $\bar y_2 - \hat\mu_0$ has greater correlation with $x_1$ than $x_2$; the
next LARS estimate is $\hat\mu_1 = \hat\mu_0 + \hat\gamma_1 x_1$, where $\hat\gamma_1$ is
chosen such that $\bar y_2 - \hat\mu_1$ bisects the angle between $x_1$ and $x_2$; then
$\hat\mu_2 = \hat\mu_1 + \hat\gamma_2 u_2$, where $u_2$ is the unit bisector;
$\hat\mu_2 = \bar y_2$ in the case $m = 2$, but not for the case $m > 2$; see Figure 4. The
staircase indicates a typical Stagewise path. Here LARS gives the Stagewise track as
$\varepsilon\to0$, but a modification is necessary to guarantee agreement in higher
dimensions; see Section 3.2."*

*Dịch: Thuật toán LARS trong trường hợp $m = 2$ biến; $\bar y_2$ là hình chiếu của $y$ xuống
$\mathcal L(x_1,x_2)$. Xuất phát ở $\hat\mu_0 = 0$, vectơ phần dư $\bar y_2 - \hat\mu_0$ có
tương quan với $x_1$ lớn hơn với $x_2$; ước lượng LARS tiếp theo là
$\hat\mu_1 = \hat\mu_0 + \hat\gamma_1 x_1$, trong đó $\hat\gamma_1$ được chọn sao cho
$\bar y_2 - \hat\mu_1$ chia đôi góc giữa $x_1$ và $x_2$; rồi
$\hat\mu_2 = \hat\mu_1 + \hat\gamma_2 u_2$, với $u_2$ là vectơ đơn vị phân giác;
$\hat\mu_2 = \bar y_2$ khi $m = 2$, nhưng không đúng khi $m > 2$; xem Hình 4. Bậc thang biểu
thị một quỹ đạo Stagewise điển hình. Ở đây LARS cho đúng vết Stagewise khi
$\varepsilon\to0$, nhưng cần một sửa đổi để bảo đảm điều đó ở số chiều cao hơn; xem mục 3.2.*

**Nhận xét của chính tác giả, nguyên văn, tr.6:** *"The LARS procedure works roughly as
follows. As with classic Forward Selection, we start with all coefficients equal to zero, and
find the predictor most correlated with the response, say $x_{j_1}$. We take the largest step
possible in the direction of this predictor until some other predictor, say $x_{j_2}$, has as
much correlation with the current residual. At this point LARS parts company with Forward
Selection. Instead of continuing along $x_{j_1}$, LARS proceeds in a direction equiangular
between the two predictors until a third variable $x_{j_3}$ earns its way into the 'most
correlated' set."*

*Dịch: Thủ tục LARS đại khái chạy thế này. Cũng như Forward Selection cổ điển, ta bắt đầu
với mọi hệ số bằng 0 và tìm biến dự báo tương quan nhất với đáp ứng, gọi là $x_{j_1}$. Ta đi
bước dài nhất có thể theo hướng biến đó cho tới khi một biến khác, gọi là $x_{j_2}$, có tương
quan với phần dư hiện tại lớn bằng vậy. Đúng chỗ này LARS chia tay Forward Selection. Thay vì
đi tiếp dọc $x_{j_1}$, LARS đi theo một hướng cách đều góc giữa hai biến cho tới khi biến thứ
ba $x_{j_3}$ giành được chỗ trong tập "tương quan nhất".*

**Cách đọc hình.** Toàn bộ hình nằm trong mặt phẳng sinh bởi hai cột, $\mathcal L(x_1,x_2)$.
Theo thứ tự câu chuyện:

1. $\bar y_2$ là **hình chiếu của $y$** xuống mặt phẳng đó — tức nghiệm bình phương tối thiểu
   trên hai biến. Đó là đích cuối.
2. Xuất phát $\hat\mu_0 = 0$ ở gốc. Phần dư lúc này tạo **góc nhỏ hơn** với $x_1$ so với
   $x_2$, nên $x_1$ vào trước.
3. LARS đi dọc $x_1$ **không phải hết cỡ** mà đúng tới $\hat\mu_1$, chỗ phần dư mới **chia
   đôi góc** giữa $x_1$ và $x_2$. Chia đôi góc nghĩa là hai tương quan bằng nhau.
4. Từ $\hat\mu_1$, LARS đi theo $u_2$ — **vectơ phân giác đơn vị** — tới đích.
5. **Đường bậc thang** răng cưa là quỹ đạo Forward Stagewise với $\varepsilon$ nhỏ. Nó bò
   zigzag quanh đúng đoạn thẳng mà LARS đi một nhát.

**Điều hình KHÔNG nói ra.** Hình này vẽ trường hợp $m = 2$, nơi bước thứ hai đi thẳng tới
đích. **Với $m > 2$ thì không**, và chính chú thích đã cảnh báo bằng câu *"but not for the
case $m > 2$; see Figure 4"*. Hình 12 của bài này là chỗ nói rõ điều đó. Ví dụ chạy tay ở
mục 0.4 là đúng tình huống của Hình 2, với $\hat\gamma_1 = 2{,}000000$ và
$\hat\gamma_2 = 1{,}414214$.

#### Công thức (2.1)–(2.3) — dựng bài toán hai biến

$$c(\hat\mu) = X^{\top}(y-\hat\mu) = X^{\top}(\bar y_2 - \hat\mu) \tag{2.1}$$

$$\hat\mu_1 = \hat\mu_0 + \hat\gamma_1 x_1 \tag{2.2}$$

$$\hat\mu_2 = \hat\mu_1 + \hat\gamma_2 u_2 \tag{2.3}$$

**Ý nghĩa trừu tượng của (2.1).** Đây là lý do cả thuật toán vẽ được trong một mặt phẳng:
tương quan hiện tại **chỉ phụ thuộc hình chiếu của $y$** xuống không gian cột, không phụ
thuộc phần của $y$ vuông góc với mọi cột. Phần vuông góc ấy là phần không thuật toán nào
chạm tới được, nên có thể quên nó đi khi vẽ hình.

#### Công thức (2.4)–(2.7) — vectơ cách đều góc

![Công thức (2.4)–(2.7) của Efron 2004, trang 7](<paper_lars_efron2004_tr7_ct2_4_2_7_cach_deu_goc.png>)

**Hình 7 — công thức (2.4)–(2.7), trang 7.** Nguồn: cùng bài báo.

$$X_{\mathcal A} = (\cdots s_jx_j\cdots)_{j\in\mathcal A} \tag{2.4}$$

$$G_{\mathcal A} = X_{\mathcal A}^{\top}X_{\mathcal A},
\qquad A_{\mathcal A} = \left(\mathbf 1_{\mathcal A}^{\top}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A}\right)^{-1/2} \tag{2.5}$$

$$u_{\mathcal A} = X_{\mathcal A}w_{\mathcal A},
\qquad w_{\mathcal A} = A_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} \tag{2.6}$$

$$X_{\mathcal A}^{\top}u_{\mathcal A} = A_{\mathcal A}\mathbf 1_{\mathcal A},
\qquad \lVert u_{\mathcal A}\rVert^{2} = 1 \tag{2.7}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $s_j$ | $\pm1$ | dấu của tương quan hiện tại; nhân vào cột để **mọi tương quan cùng dương** |
| $X_{\mathcal A}$ | $n\times k$ | các cột hoạt động đã đổi dấu, $k = \lvert\mathcal A\rvert$ |
| $G_{\mathcal A}$ | $k\times k$ | ma trận Gram — đối xứng, xác định dương khi các cột độc lập tuyến tính |
| $\mathbf 1_{\mathcal A}$ | $\mathbb{R}^{k}$ | vectơ toàn số 1 |
| $A_{\mathcal A}$ | vô hướng $\in(0,1]$ | hệ số chuẩn hoá — và, theo Bổ đề 5, là **khoảng cách từ gốc tới đơn hình mở rộng** |
| $w_{\mathcal A}$ | $\mathbb{R}^{k}$ | hệ số tổ hợp dựng ra hướng đi |
| $u_{\mathcal A}$ | $\mathbb{R}^{n}$, dài 1 | **hướng cách đều góc** |

**Ý nghĩa trừu tượng.** (2.7) là chỗ cả khối công thức này tồn tại để phục vụ. Nó nói hai
điều cùng lúc, và cả hai đều cần thiết:

* $X_{\mathcal A}^{\top}u_{\mathcal A} = A_{\mathcal A}\mathbf 1$ — hướng đi tạo **cùng một
  tích vô hướng** với mọi cột hoạt động. Vì các cột đều dài 1, tích vô hướng bằng nhau nghĩa
  là **góc bằng nhau**. Đó là cái tên của thuật toán.
* $\lVert u_{\mathcal A}\rVert = 1$ — hướng đi có độ dài chuẩn hoá, nên $\hat\gamma$ đo đúng
  quãng đường đi được trong không gian dự đoán.

Còn $A_{\mathcal A}$ thì không phải một hằng số kỹ thuật: nó là **giá trị chung của tích vô
hướng đó**, và mục 2.9 sẽ cho nó một nghĩa hình học (Bổ đề 5), còn (2.16) cho nó nghĩa động
học — nó chính là **tốc độ giảm của $\hat C$**.

Kiểm (2.7) ở mọi bước của đường LARS trên bộ diabetes:

```
      lệch lớn nhất của X_A'u_A − A_A·1 : 3.775e-15
      lệch lớn nhất của ‖u_A‖ − 1       : 2.998e-15
```

#### Công thức (2.8)–(2.13) — trọn vẹn thuật toán

![Công thức (2.8)–(2.13) của Efron 2004, trang 8](<paper_lars_efron2004_tr8_ct2_8_2_13_thuat_toan.png>)

**Hình 8 — công thức (2.8)–(2.13), trang 8.** Nguồn: cùng bài báo.

$$\hat c = X^{\top}(y - \hat\mu_{\mathcal A}) \tag{2.8}$$

$$\hat C = \max_j\{\lvert\hat c_j\rvert\},
\qquad \mathcal A = \{j : \lvert\hat c_j\rvert = \hat C\} \tag{2.9}$$

$$s_j = \operatorname{sign}\{\hat c_j\}\quad\text{với } j\in\mathcal A \tag{2.10}$$

$$a \equiv X^{\top}u_{\mathcal A} \tag{2.11}$$

$$\hat\mu_{\mathcal A^{+}} = \hat\mu_{\mathcal A} + \hat\gamma\,u_{\mathcal A} \tag{2.12}$$

$$\hat\gamma = \min_{j\in\mathcal A^{c}}{}^{+}
\left\{\frac{\hat C - \hat c_j}{A_{\mathcal A} - a_j},\ \frac{\hat C + \hat c_j}{A_{\mathcal A} + a_j}\right\} \tag{2.13}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\mathcal A^{c}$ | tập | các biến **ngoài** tập hoạt động |
| $a_j$ | vô hướng | tương quan giữa hướng đi và cột $j$ |
| $\min^{+}$ | — | lấy nhỏ nhất **chỉ trên các giá trị dương** |
| $\hat\gamma$ | vô hướng $>0$ | độ dài bước |

**Ý nghĩa trừu tượng — và đây là chỗ đáng đọc nhất của cả bài báo.** (2.13) trông như một
công thức rối rắm, nhưng nó chỉ là nghiệm của một phương trình bậc nhất. Đi theo
$u_{\mathcal A}$ một quãng $\gamma$ thì:

* tương quan của một biến **hoạt động** tụt theo $\hat C - \gamma A_{\mathcal A}$ — mọi biến
  hoạt động tụt **cùng tốc độ**, đó là (2.16);
* tương quan của một biến **ngoài** tụt theo $\hat c_j - \gamma a_j$ — mỗi biến một tốc độ.

Cho hai biểu thức bằng nhau rồi giải ra $\gamma$ thì được đúng hai phân số trong (2.13): một
cho trường hợp biến ngoài bắt kịp từ phía dương, một từ phía âm. Lấy nghiệm dương nhỏ nhất
tức là **lấy sự kiện xảy ra SỚM NHẤT**.

Nói cách khác: **"đi tới khi có biến khác bắt kịp" không phải một điều kiện dừng phải dò tìm,
mà là một phương trình giải được.** Đó là toàn bộ chỗ LARS nhanh hơn Forward Stagewise —
không phải nhờ bước lớn hơn, mà nhờ **biết trước bước phải dài bao nhiêu**.

#### Công thức (2.14)–(2.18) — vì sao (2.13) đúng

$$\mu(\gamma) = \hat\mu_{\mathcal A} + \gamma u_{\mathcal A} \tag{2.14}$$

$$c_j(\gamma) = x_j^{\top}(y - \mu(\gamma)) = \hat c_j - \gamma a_j \tag{2.15}$$

$$\lvert c_j(\gamma)\rvert = \hat C - \gamma A_{\mathcal A}
\qquad\text{với } j\in\mathcal A \tag{2.16}$$

$$\lvert\hat c_{kj}\rvert = \lvert x_j^{\top}(y-\hat\mu_{k-1})\rvert \tag{2.17}$$

$$\hat C_k = \max\{\lvert\hat c_{kj}\rvert\} = \hat C_{k-1} - \hat\gamma_{k-1}A_{k-1} \tag{2.18}$$

**Ý nghĩa trừu tượng của (2.16).** Đây là **bất biến** của LARS, và nó là thứ làm mọi thứ
khác chạy được. Nó nói: trong suốt một bước, mọi biến hoạt động giữ **đúng cùng một mức**
tương quan tuyệt đối, và mức chung ấy giảm **tuyến tính** với hệ số $A_{\mathcal A}$. Một
biến ngoài vào tập hoạt động đúng lúc nó **chạm** mức chung đó — không sớm hơn, không muộn
hơn.

Kiểm bằng số, đi nửa bước rồi so với công thức:

```
      γ =  237.359 : đo được   712.076445 · công thức   712.076445 · lệch 2.27e-13
      γ =  474.718 : đo được   474.717630 · công thức   474.717630 · lệch 2.27e-13
      γ =  712.076 : đo được   237.358815 · công thức   237.358815 · lệch 2.84e-14
```

Và dãy $\hat C_k$ theo (2.18) trên cả đường:

```
      949.4353 889.3138 452.8957 316.0734 130.1295 88.7843 68.9648 19.9812 5.4775 5.0882 0.0000
      dãy có giảm đơn điệu không? CÓ
```

Con số cuối bằng 0 không phải trùng hợp: ở nghiệm bình phương tối thiểu, phần dư **trực
giao** với mọi cột, nên mọi tương quan bằng 0 cùng lúc.

#### Vẽ chính bất biến đó ra

![Bất biến cách đều góc, đo trên bộ diabetes](<lars_bat_bien.png>)

**Hình 9 — bất biến "cách đều góc", đo trên dữ liệu thật (hình tự vẽ).** Ba hình gốc ở trên
giải thích bất biến bằng **hình học** (Hình 6) và cho thấy **hệ quả** của nó (Hình 5). Hình
này bù đúng chỗ còn thiếu: nó vẽ **chính đại lượng mà bất biến nói tới**, đo trên chính bộ
dữ liệu của bài báo.

**Cách đọc hình.** Trục ngang là bước của đường lasso, trục dọc là
$\lvert x_j^{\top}(y - X\hat\beta)\rvert$ — độ lớn tương quan giữa biến $j$ và phần dư hiện
tại. Mười đường, mỗi đường một biến, trục dọc theo thang log.

* **Chấm đặc** = biến đang trong tập hoạt động; **chấm rỗng** = đang ở ngoài.
* Điều cần nhìn: **mọi chấm đặc ở cùng một bước nằm chồng khít lên nhau**. Đó là (2.16), và
  độ trải lớn nhất đo được trên cả đường là $1{,}49\times10^{-12}$ — bằng nhau trong dung sai
  số học, không phải bằng nhau tuyệt đối.
* **Đường đen đứt là trần** $\hat C_k$. Biến ngoài không bao giờ vượt trần, và vào tập đúng
  lúc **chạm** trần.
* **Vạch cam ở bước 10** đánh dấu chỗ sửa đổi lasso (3.6) bỏ biến 7 ra: chấm của nó chuyển
  từ đặc sang rỗng, rồi đặc lại ở bước 12.

**Điều hình KHÔNG nói ra.** Bước cuối có giá trị thật bằng **0**, và trên thang log thì
không vẽ được số 0; hình đặt một sàn ở 0,3 để chấm ấy còn hiện ra. Người đọc nhìn vào cột
cuối sẽ thấy mọi chấm nằm trên sàn chứ không phải nằm ở 0 — đó là quy ước vẽ, không phải
phép đo.

<details>
<summary><b>Mã sinh Hình 9</b> — trong đó có bản cài đặt LARS viết từ đầu theo (2.4)–(2.13)</summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# chuẩn hoá đúng quy ước (1.1) tr.3: cột trừ trung bình rồi chia cho CHUẨN EUCLID
d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()
def huong(X, A, c):                              # (2.4)–(2.6), (2.11)
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    wA = AA * iG
    uA = XA @ wA
    return s, AA, wA, uA, X.T @ uA


def lars(X, y, kieu="lar", eps=1e-12):
    """LARS của Efron 2004 theo đúng (2.4)–(2.13); kieu = lar | lasso | stage."""
    n, m = X.shape
    beta, mu, A, B, bo_ky = np.zeros(m), np.zeros(n), [], [np.zeros(m)], []
    vua_bo = -1
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        if kieu == "stage":                      # chiếu u_A vào nón lồi (3.12)
            P = nnls(X[:, A] * s, uA)[0]
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                bo_ky.append((len(B), [A[i] + 1 for i in range(len(A)) if P[i] <= 1e-10]))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A = Bh
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]     # (2.13)
        gam = min(ung) if ung else C / AA
        d_ = np.zeros(m); d_[A] = s * wA         # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A:
                if abs(d_[j]) > eps:
                    g = -beta[j] / d_[j]         # (3.4)–(3.5)
                    if eps < g < gt:
                        gt, jt = g, j
            if gt < gam:
                gam, bo = gt, jt                 # (3.6)
        beta = beta + gam * d_
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            bo_ky.append((len(B), [bo + 1]))
            vua_bo = bo
        B.append(beta.copy())
        if not con and bo is None:
            break
    return np.array(B).T, bo_ky


B = lars(X, y, "lasso")[0]
K = B.shape[1]

C = np.array([np.abs(X.T @ (y - X @ B[:, k])) for k in range(K)])     # K × 10
trong = np.array([np.abs(B[:, k]) > 1e-12 for k in range(K)])         # K × 10
SAN = 0.3                                        # sàn để vẽ được giá trị 0 trên trục log

fig, ax = plt.subplots(figsize=(10.6, 4.8))
for j in range(10):
    v = np.maximum(C[:, j], SAN)
    mau = plt.cm.tab10(j % 10)
    ax.plot(range(K), v, lw=1.0, alpha=.45, color="#9ca3af", zorder=1)
    on, off = trong[:, j], ~trong[:, j]
    ax.plot(np.arange(K)[on], v[on], "o", ms=6.2, zorder=3, color=mau)
    ax.plot(np.arange(K)[off], v[off], "o", ms=6.2, zorder=2,
            mfc="white", mec=mau, mew=1.1)
    kv = int(np.argmax(on))
    ax.annotate(str(j + 1), (kv, v[kv]), fontsize=7.5, fontweight="bold", color=mau,
                xytext=(0, 9), textcoords="offset points", ha="center", zorder=5)

do_trai = max(np.ptp(C[k, trong[k]]) for k in range(1, K) if trong[k].any())
tran = np.array([C[k, trong[k]].max() if trong[k].any() else C[k].max() for k in range(K)])
ax.plot(range(K), np.maximum(tran, SAN), "k--", lw=1.3, zorder=4,
        label=r"trần $\hat C_k$ — giá trị chung của mọi biến hoạt động")
ax.axvline(10, color="#ea580c", lw=1.4, ls=":", zorder=4)
ax.annotate("bước 10: (3.6) bỏ biến 7\nra khỏi tập hoạt động", (10, 3.0),
            xytext=(-14, 34), textcoords="offset points", fontsize=8,
            color="#ea580c", ha="right",
            arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.0))
ax.plot([], [], "o", ms=6.2, color="#374151", label="chấm ĐẶC = biến đang hoạt động")
ax.plot([], [], "o", ms=6.2, mfc="white", mec="#374151", mew=1.1,
        label="chấm RỖNG = biến ở ngoài")
ax.set_yscale("log")
ax.set_ylim(SAN * 0.8, C.max() * 2.6)
ax.set_xticks(range(K))
ax.set_xlabel("bước của đường lasso (LARS có sửa đổi (3.6))")
ax.set_ylabel(r"$|x_j^{\top}(y - X\hat\beta)|$")
ax.legend(fontsize=8, loc="lower left", ncol=1)
ax.set_title("Bất biến cách đều góc, đo trên chính bộ diabetes của bài báo\n"
             "mọi chấm đặc ở cùng một bước CHỒNG KHÍT lên nhau — độ trải lớn nhất "
             "%.2e\n" % do_trai +
             "biến ngoài không bao giờ vượt trần, và vào đúng lúc nó CHẠM trần",
             fontsize=9.5)
fig.savefig("lars_bat_bien.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

for k in range(1, K):
    c = C[k, trong[k]]
    ngoai = C[k, ~trong[k]]
    assert c.max() - c.min() < 1e-9, "bất biến vỡ ở bước %d" % k
    assert len(ngoai) == 0 or ngoai.max() <= c.max() + 1e-9, "biến ngoài vượt trần ở bước %d" % k
print("bất biến đúng ở cả %d bước; độ trải lớn nhất %.2e"
      % (K - 1, max(np.ptp(C[k, trong[k]]) for k in range(1, K) if trong[k].any())))
```

</details>

#### Công thức (2.19)–(2.22) — LARS đứng ở đâu so với OLS

![Công thức (2.19)–(2.22) của Efron 2004, trang 10](<paper_lars_efron2004_tr10_ct2_19_2_22_quan_he_ols.png>)

**Hình 10 — công thức (2.19)–(2.22), trang 10.** Nguồn: cùng bài báo.

$$\bar y_k = \hat\mu_{k-1} + X_kG_k^{-1}X_k^{\top}(y-\hat\mu_{k-1})
= \hat\mu_{k-1} + \frac{\hat C_k}{A_k}u_k \tag{2.19}$$

$$X_k^{\top}(y-\hat\mu_{k-1}) = \hat C_k\mathbf 1_{\mathcal A} \tag{2.20}$$

$$\bar\gamma_k \equiv \frac{\hat C_k}{A_k} \tag{2.21}$$

$$\hat\mu_k - \hat\mu_{k-1} = \frac{\hat\gamma_k}{\bar\gamma_k}(\bar y_k - \hat\mu_{k-1}) \tag{2.22}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\bar y_k$ | $\mathbb{R}^{n}$ | hình chiếu của $y$ xuống $\mathcal L(X_k)$ — **nghiệm OLS trên $k$ biến đang có** |
| $\bar\gamma_k$ | vô hướng | quãng đường từ chỗ đang đứng tới $\bar y_k$ |
| $\hat\gamma_k/\bar\gamma_k$ | $\in(0,1]$ | tỉ lệ quãng đường LARS thật sự đi — ký hiệu $\rho_k$ |

**Ý nghĩa trừu tượng.** (2.22) là câu trả lời gọn nhất cho *"LARS khác Forward Selection ở
đâu"*. Cả hai đi theo **cùng một đường thẳng**, từ $\hat\mu_{k-1}$ về phía $\bar y_k$.
Forward Selection đi trọn: $\rho_k = 1$. LARS chỉ đi một phần: $\rho_k < 1$. Không có gì bí
ẩn trong "hướng cách đều góc" — nó chính là hướng về nghiệm OLS của tập hiện tại.

Đo $\rho_k$ trên cả đường:

```
   k | Ĉ_k       | A_k      | γ̄_k = Ĉ_k/A_k | γ̂_k        | ρ_k = γ̂/γ̄ | lệch (2.19)
    1 |  949.4353 | 1.000000 |      949.4353 |    60.1215 |   0.063323 | 2.84e-14
    2 |  889.3138 | 0.850340 |     1045.8331 |   513.2277 |   0.490736 | 1.56e-13
    3 |  452.8957 | 0.779404 |      581.0799 |   175.5475 |   0.302106 | 8.53e-14
    4 |  316.0734 | 0.716910 |      440.8829 |   259.3684 |   0.588293 | 8.53e-14
    5 |  130.1295 | 0.466376 |      279.0230 |    88.6522 |   0.317724 | 1.56e-13
    6 |   88.7843 | 0.453713 |      195.6840 |    43.6830 |   0.223232 | 1.35e-13
    7 |   68.9648 | 0.360219 |      191.4524 |   135.9829 |   0.710270 | 1.99e-13
    8 |   19.9812 | 0.268511 |       74.4146 |    54.0150 |   0.725865 | 3.30e-13
    9 |    5.4775 | 0.069747 |       78.5342 |     5.5816 |   0.071072 | 1.52e-12
   10 |    5.0882 | 0.069214 |       73.5146 |    73.5146 |   1.000000 | 1.69e-12
   -> ρ_k < 1 ở mọi bước trừ bước cuối? ĐÚNG ; ρ lớn nhất trước bước cuối = 0.725865
```

Cột $\rho_k$ đọc được nhiều hơn vẻ ngoài của nó. Ở bước 1, LARS chỉ đi **6,33 %** quãng
đường mà Forward Selection sẽ đi — tức nó thận trọng tới mức gần như không nhúc nhích. Ở
bước 8 nó đi **72,59 %**. Và ở bước cuối $\rho = 1$ **theo quy ước**, vì lúc đó không còn
biến ngoài nào để bắt kịp nên bài báo định nghĩa $\hat\gamma_m = \bar\gamma_m$ (tr.10).

#### Hình 3 và Hình 4 của bài báo

![Hình 3 của Efron 2004, trang 9](<paper_lars_efron2004_tr9_hinh3_lars_goc.png>)

**Hình 11 — Hình 3 của Efron 2004, trang 9.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"LARS analysis of the diabetes study: (left) estimates of
regression coefficients $\hat\beta_j$, $j = 1, 2, \dots, 10$; plotted versus
$\sum\lvert\hat\beta_j\rvert$; plot is slightly different than either Lasso or Stagewise,
Figure 1; (right) absolute current correlations as function of LARS step; variables enter
active set (2.9) in order $3, 9, 4, 7, \dots, 1$; heavy curve shows maximum current
correlation $\hat C_k$ declining with $k$."*

*Dịch: Phân tích LARS cho nghiên cứu tiểu đường: (trái) ước lượng hệ số hồi quy
$\hat\beta_j$, vẽ theo $\sum\lvert\hat\beta_j\rvert$; đồ thị khác chút ít so với cả Lasso lẫn
Stagewise ở Hình 1; (phải) trị tuyệt đối của tương quan hiện tại theo bước LARS; các biến đi
vào tập hoạt động (2.9) theo thứ tự $3, 9, 4, 7, \dots, 1$; đường đậm cho thấy tương quan
lớn nhất $\hat C_k$ giảm dần theo $k$.*

**Nhận xét của chính tác giả, nguyên văn, tr.9:** *"Figure 3 concerns the LARS analysis of
the diabetes data. The complete algorithm required only $m = 10$ steps of procedure
(2.8)–(2.13), with the variables joining the active set $\mathcal A$ in the same order as for
the Lasso: 3, 9, 4, 7, . . . , 1. Tracks of the regression coefficients $\hat\beta_j$ are
nearly but not exactly the same as either the Lasso or Stagewise tracks of Figure 1."*

*Dịch: Hình 3 nói về phân tích LARS trên dữ liệu tiểu đường. Toàn bộ thuật toán chỉ cần
$m = 10$ bước của thủ tục (2.8)–(2.13), với các biến gia nhập tập hoạt động $\mathcal A$
theo đúng thứ tự như ở Lasso: 3, 9, 4, 7, …, 1. Vết của các hệ số hồi quy $\hat\beta_j$ gần
giống nhưng không hoàn toàn giống vết Lasso hay Stagewise ở Hình 1.*

**Cách đọc hình.** Bảng phải chính là Hình 9 của bài này vẽ bằng thang tuyến tính: mỗi bước
một cột chấm, đường đậm là trần $\hat C_k$. Bảng trái là đường LARS gốc — so với Hình 5 thì
nó **không** có chỗ nào chạm lại 0.

**Điều hình KHÔNG nói ra, và đây là chỗ quan trọng.** Câu *"nearly but not exactly the same"*
đo được: so hai đường LARS và lasso tại cùng $t$, **lệch hệ số lớn nhất là 50,6412** trên
thang hệ số cỡ 792. Không nhỏ. Cái khác nhau không nằm ở hình dạng chung mà nằm ở **một
khoảng của đường**, đúng chỗ biến 7 đổi dấu.

![Hình 4 của Efron 2004, trang 10](<paper_lars_efron2004_tr10_hinh4_lars_va_ols.png>)

**Hình 12 — Hình 4 của Efron 2004, trang 10.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"At each stage the LARS estimate $\hat\mu_k$ approaches, but
does not reach, the corresponding OLS estimate $\bar y_k$."*

*Dịch: Ở mỗi chặng, ước lượng LARS $\hat\mu_k$ tiến lại gần, nhưng không chạm tới, ước lượng
OLS tương ứng $\bar y_k$.*

**Nhận xét của chính tác giả, nguyên văn, tr.10:** *"It is easy to see that $\hat\gamma_k$,
(2.12), is always less than $\bar\gamma_k$, so that $\hat\mu_k$ lies closer than $\bar y_k$
to $\hat\mu_{k-1}$. … The exception is at the last stage: since $\mathcal A_m$ contains all
covariates, (2.13) is not defined. By convention the algorithm takes
$\hat\gamma_m = \bar\gamma_m = \hat C_m/A_m$, making $\hat\mu_m = \bar y_m$ and
$\hat\beta_m$ equal the OLS estimate for the full set of $m$ covariates."*

*Dịch: Dễ thấy $\hat\gamma_k$ ở (2.12) luôn nhỏ hơn $\bar\gamma_k$, nên $\hat\mu_k$ nằm gần
$\hat\mu_{k-1}$ hơn là $\bar y_k$. … Ngoại lệ là chặng cuối: vì $\mathcal A_m$ chứa mọi
biến, (2.13) không xác định. Theo quy ước, thuật toán lấy
$\hat\gamma_m = \bar\gamma_m = \hat C_m/A_m$, khiến $\hat\mu_m = \bar y_m$ và
$\hat\beta_m$ bằng đúng ước lượng OLS trên toàn bộ $m$ biến.*

**Cách đọc hình.** Hình vẽ ba chặng liên tiếp trong không gian dự đoán $\mathbb{R}^{n}$, chứ
không phải trong không gian hệ số. Mỗi chặng có hai điểm: $\bar y_k$ là đích OLS của tập
hiện tại, còn $\hat\mu_k$ là chỗ LARS dừng lại — **luôn nằm trên đoạn nối, luôn chưa tới
đích**. Rồi tập hoạt động lớn thêm một biến, đích mới $\bar y_{k+1}$ xuất hiện ở chỗ khác,
và câu chuyện lặp lại.

**Điều hình KHÔNG nói ra.** Hình vẽ khoảng cách $\hat\mu_k$ tới $\bar y_k$ như một đoạn
ngắn đều đặn, nhưng cột $\rho_k$ đo ở trên cho thấy nó **rất không đều**: từ 6,33 % ở bước 1
tới 72,59 % ở bước 8. Và chính đại lượng $\rho_k$ đó sẽ quay lại ở công thức (3.20) — chỗ
bài báo in sai, mục 2.6 của bài này.

### 2.3 Sửa đổi thứ nhất: LARS thành LASSO — §3.1 của bài báo, tr.11–12

#### Công thức (3.1)–(3.5) — chỗ LARS gốc có thể sai luật

![Công thức (3.1)–(3.5) của Efron 2004, trang 11](<paper_lars_efron2004_tr11_ct3_1_3_5_sua_doi_lasso.png>)

**Hình 13 — công thức (3.1)–(3.5), trang 11.** Nguồn: cùng bài báo.

$$\operatorname{sign}(\hat\beta_j) = \operatorname{sign}(\hat c_j) = s_j \tag{3.1}$$

$$w_{\mathcal A} = A_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} \tag{3.2}$$

$$\mu(\gamma) = X\beta(\gamma),\qquad \beta_j(\gamma) = \hat\beta_j + \gamma\hat d_j
\quad\text{với } j\in\mathcal A \tag{3.3}$$

$$\gamma_j = -\hat\beta_j/\hat d_j \tag{3.4}$$

$$\tilde\gamma = \min_{\gamma_j>0}\{\gamma_j\} \tag{3.5}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\hat d$ | $\mathbb{R}^{m}$ | hướng đi **trong không gian hệ số**: bằng $s_jw_{\mathcal A j}$ với $j\in\mathcal A$, bằng 0 ở ngoài |
| $\gamma_j$ | vô hướng | quãng đường tới chỗ hệ số $j$ **đi qua 0** |
| $\tilde\gamma$ | vô hướng $>0$ | quãng đường tới lần đổi dấu **sớm nhất**; bằng $\infty$ nếu không có |

**Ý nghĩa trừu tượng.** (3.1) là một **điều kiện bắt buộc của nghiệm lasso**, không phải một
lựa chọn thiết kế: nó suy ra từ điều kiện dừng của bài toán (1.5), và mục 2.9 chứng minh nó
(Bổ đề 8). Nội dung của nó: ở một nghiệm lasso, **hệ số và tương quan phải cùng dấu**. Trực
giác: nếu hệ số dương mà tương quan âm thì giảm hệ số vừa làm phần dư nhỏ đi vừa làm ngân
sách $T$ nhỏ đi — tức điểm đang xét không thể là nghiệm.

LARS gốc **không ép** điều kiện đó. Nên câu hỏi tự nhiên là: nó có vi phạm thật không? Đo
trên bộ diabetes:

```
   (3.1) dấu(β̂_j) = dấu(ĉ_j) cho mọi j hoạt động — kiểm trên CẢ hai đường.
   Bỏ qua bước cuối cùng, nơi Ĉ = 0 nên dấu của ĉ chỉ còn là nhiễu số học.
      LARS gốc :  0/ 45 cặp (bước, biến) VI PHẠM điều kiện dấu
      lasso    :  0/ 63 cặp (bước, biến) VI PHẠM điều kiện dấu
   -> Tại các ĐIỂM GÃY, cả hai đường đều không vi phạm. Nhưng chỗ vi phạm của LARS
      gốc nằm GIỮA hai điểm gãy, nên phải quét trên một lưới dày mới thấy:
      LARS gốc :  626/4001 điểm trên lưới VI PHẠM — chỉ ở biến [7], trong khoảng t = 2917.9 .. 3458.2
      lasso    :    0/4001 điểm trên lưới VI PHẠM
```

Bốn dòng này là toàn bộ lý do mục 3.1 của bài báo tồn tại — **và cách đo được chúng cũng là
một bài học.** Kiểm tại các **điểm gãy** thì cả hai đường đều sạch: 0/45 và 0/63. Kết luận
"LARS gốc không vi phạm (3.1)" khi ấy là **sai**, vì chỗ vi phạm không nằm ở điểm gãy mà nằm
**bên trong một đoạn**: hệ số của biến 7 đi qua 0 ở giữa bước cuối, chứ không đi qua 0 tại một
nút. Quét trên lưới 4001 điểm thì nó hiện ra ngay: **626 điểm vi phạm, chỉ ở biến 7, trong
khoảng $t$ từ 2917,9 tới 3458,2** — tức 15,6 % cuối của đường.

Đường lasso không vi phạm **điểm nào**, đúng như Bổ đề 8 tr.29 đòi hỏi. Và khoảng vi phạm của
LARS gốc chính là khoảng mà (3.6) cắt ra bằng cách bỏ biến 7 khỏi tập hoạt động.

#### Công thức (3.6) và Định lý 1

![Công thức (3.6) và Định lý 1 của Efron 2004, trang 12](<paper_lars_efron2004_tr12_ct3_6_dinh_ly1.png>)

**Hình 14 — sửa đổi lasso (3.6) và Định lý 1, trang 12.** Nguồn: cùng bài báo.

**Phát biểu gốc, nguyên văn:** *"**Lasso modification.** If $\tilde\gamma < \hat\gamma$, stop
the ongoing LARS step at $\gamma = \tilde\gamma$ and remove $\tilde j$ from the calculation of
the next equiangular direction."*

*Dịch: **Sửa đổi lasso.** Nếu $\tilde\gamma < \hat\gamma$, dừng bước LARS đang chạy tại
$\gamma = \tilde\gamma$ và bỏ $\tilde j$ ra khỏi phép tính hướng cách đều góc kế tiếp.*

$$\hat\mu_{\mathcal A^{+}} = \hat\mu_{\mathcal A} + \tilde\gamma\,u_{\mathcal A},
\qquad \mathcal A^{+} = \mathcal A - \{\tilde j\} \tag{3.6}$$

**Định lý 1, nguyên văn:** *"Under the Lasso modification, and assuming the 'one at a time'
condition discussed below, the LARS algorithm yields all Lasso solutions."*

*Dịch: Dưới sửa đổi lasso, và với giả thiết điều kiện "mỗi lần một biến" bàn ở dưới, thuật
toán LARS cho ra mọi nghiệm lasso.*

**Ý nghĩa trừu tượng.** Cả sửa đổi gói trong một câu: **thêm một ứng viên nữa vào danh sách
"chuyện gì xảy ra trước"**. LARS gốc chỉ hỏi *"biến ngoài nào bắt kịp sớm nhất"*; bản lasso
hỏi thêm *"hệ số trong nào chạm 0 sớm nhất"*, rồi lấy cái nào tới trước. Đó là lý do tập
hoạt động của LARS gốc chỉ lớn dần, còn tập của lasso có thể **co lại**.

**Nhận xét của chính tác giả về chỗ nó kích hoạt, nguyên văn, tr.12:** *"The Lasso diagram in
Figure 1 was actually calculated using the modified LARS algorithm. Modification (3.6) came
into play only once, at the arrowed point in the left panel. There $\mathcal A$ contained all
10 indices while $\mathcal A^{+} = \mathcal A - \{7\}$. Variable 7 was restored to the active
set one LARS step later, the next and last step then taking $\hat\beta$ all the way to the
full OLS solution. The brief absence of variable 7 had an effect on the tracks of the others,
noticeably $\hat\beta_8$. The price of using Lasso instead of unmodified LARS comes in the
form of added steps, 12 instead of 10 in this example."*

*Dịch: Đồ thị Lasso ở Hình 1 thật ra được tính bằng thuật toán LARS đã sửa đổi. Sửa đổi (3.6)
chỉ can thiệp đúng một lần, tại điểm có mũi tên ở bảng trái. Ở đó $\mathcal A$ chứa cả 10 chỉ
số trong khi $\mathcal A^{+} = \mathcal A - \{7\}$. Biến 7 được khôi phục vào tập hoạt động
một bước LARS sau đó, và bước kế tiếp, cũng là bước cuối, đưa $\hat\beta$ đi thẳng tới nghiệm
OLS đầy đủ. Việc biến 7 vắng mặt trong chốc lát có ảnh hưởng tới vết của các biến khác, rõ
nhất là $\hat\beta_8$. Cái giá của việc dùng Lasso thay cho LARS không sửa đổi nằm ở số bước
thêm ra, 12 thay vì 10 trong ví dụ này.*

Dựng lại **từng chi tiết** của đoạn đó:

```
   Sửa đổi (3.6) kích hoạt 1 lần: biến 7 tại bước 10
   Thứ tự biến vào trên đường lasso (đếm cả lần vào lại): [3, 9, 4, 7, 2, 10, 5, 8, 6, 1, 7]
      trong bước 10, A có 10 biến (1,2,3,4,5,6,7,8,9,10); sau (3.6), A+ còn 9 biến (1,2,3,4,5,6,8,9,10)
      biến 7 vắng mặt ở đúng các bước [10, 11], rồi trở lại ở bước 12
```

Bốn khẳng định, bốn lần khớp: kích hoạt **đúng một lần** · $\mathcal A$ có **đủ 10** chỉ số ·
$\mathcal A^{+}$ bỏ **đúng biến 7** · biến 7 trở lại sau **đúng một bước**. Và số bước là
**12 so với 10**.

#### Hình 15 — ba đường cạnh nhau

![Ba đường nghiệm LARS, Lasso và Stagewise trên bộ diabetes](<lars_ba_duong.png>)

**Hình 15 — ba đường nghiệm dựng lại từ đầu, đặt cạnh nhau (hình tự vẽ).** Hình 5 và Hình 11
của bài báo vẽ hai trong ba đường, ở hai trang cách nhau năm trang, với hai thang trục khác
nhau. Hình này bù chỗ đó: **cùng một thang, cùng một lần chạy, ba bảng cạnh nhau**, và đánh
dấu mọi chỗ tập hoạt động bị thu nhỏ.

**Cách đọc hình.** Ba bảng dùng chung trục dọc. Trục ngang là
$t = \sum_j\lvert\hat\beta_j\rvert$, số ở mép phải là số hiệu biến.

* **Bảng trái, LARS gốc, 10 bước.** Không có chỗ nào tập hoạt động co lại. Đường của biến 7
  đi xuống, **cắt qua 0 và tiếp tục xuống** — đó chính là chỗ nó vi phạm (3.1).
* **Bảng giữa, lasso, 12 bước.** Ở đúng chỗ ấy, đường của biến 7 **dừng lại ở 0**, nằm yên
  hai bước, rồi mới đi tiếp. Nhãn cam ghi *"bỏ 7"*.
* **Bảng phải, Stagewise, 13 bước.** Có **hai** lần thu nhỏ: bỏ $\{3,7\}$ cùng lúc, rồi sau
  đó bỏ $\{3\}$.

Ba đường **cùng xuất phát từ gốc và cùng kết thúc ở nghiệm bình phương tối thiểu** — cả ba
đều được kiểm bằng `assert` trong mã sinh hình.

**Điều hình KHÔNG nói ra.** Nhìn bằng mắt thì ba bảng "gần như giống nhau", đúng như bài báo
nói. Nhưng lệch lớn nhất giữa đường LARS và đường lasso, đo tại cùng $t$, là **50,6412** trên
thang hệ số cỡ 792 — tức khoảng 6 %. "Gần như giống nhau" là một nhận xét về **hình dạng**,
không phải về **giá trị**.

<details>
<summary><b>Mã sinh Hình 15</b> — chứa cả ba biến thể của thuật toán</summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()


def huong(X, A, c):                              # (2.4)–(2.6), (2.11)
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    wA = AA * iG
    uA = XA @ wA
    return s, AA, wA, uA, X.T @ uA


def lars(X, y, kieu="lar", eps=1e-12):
    """LARS của Efron 2004 theo đúng (2.4)–(2.13); kieu = lar | lasso | stage."""
    n, m = X.shape
    beta, mu, A, B, bo_ky = np.zeros(m), np.zeros(n), [], [np.zeros(m)], []
    vua_bo = -1
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        if kieu == "stage":                      # chiếu u_A vào nón lồi (3.12)
            P = nnls(X[:, A] * s, uA)[0]
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                bo_ky.append((len(B), [A[i] + 1 for i in range(len(A)) if P[i] <= 1e-10]))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A = Bh
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]     # (2.13)
        gam = min(ung) if ung else C / AA
        d_ = np.zeros(m); d_[A] = s * wA         # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A:
                if abs(d_[j]) > eps:
                    g = -beta[j] / d_[j]         # (3.4)–(3.5)
                    if eps < g < gt:
                        gt, jt = g, j
            if gt < gam:
                gam, bo = gt, jt                 # (3.6)
        beta = beta + gam * d_
        mu = mu + gam * uA
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            bo_ky.append((len(B), [bo + 1]))
            vua_bo = bo
        B.append(beta.copy())
        if not con and bo is None:
            break
    return np.array(B).T, bo_ky


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), sharey=True)
for ax, kieu, ten in zip(axes, ("lar", "lasso", "stage"),
                         ("LARS gốc — mục 2\n(Hình 3 bảng trái)",
                          "Lasso — sửa đổi (3.6)\n(Hình 1 bảng trái)",
                          "Stagewise — chiếu vào nón lồi\n(Hình 1 bảng phải)")):
    B, bo_ky = lars(X, y, kieu)
    t = np.abs(B).sum(0)
    for j in range(10):
        ax.plot(t, B[j], lw=1.5, color=plt.cm.tab10(j % 10))
        ax.annotate(str(j + 1), (t[-1], B[j, -1]), fontsize=7.5, fontweight="bold",
                    color=plt.cm.tab10(j % 10), xytext=(4, -3), textcoords="offset points")
    for k, js in bo_ky:
        ax.annotate("bỏ %s" % ",".join(str(v) for v in js), (t[k], 0),
                    xytext=(0, -72), textcoords="offset points", fontsize=8,
                    color="#ea580c", ha="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#ea580c", lw=0.7),
                    arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.0))
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xlim(-60, t[-1] * 1.09)
    ax.set_title("%s\n%d bước" % (ten, B.shape[1] - 1), fontsize=9.5)
    ax.set_xlabel(r"$t = \sum_j |\hat\beta_j|$")
axes[0].set_ylabel(r"hệ số $\hat\beta_j$")
fig.suptitle("Ba đường nghiệm trên bộ diabetes — cùng một điểm đầu, cùng một điểm cuối, "
             "khác nhau ở giữa\nsố bước 10 · 12 · 13 khớp đúng ba con số bài báo nêu "
             "ở tr.9, tr.12 và tr.14", fontsize=10.5, y=1.10)
fig.savefig("lars_ba_duong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

for kieu, doi in (("lar", 10), ("lasso", 12), ("stage", 13)):
    B, _ = lars(X, y, kieu)
    assert B.shape[1] - 1 == doi, (kieu, B.shape[1] - 1)
    assert np.abs(B[:, -1] - np.linalg.lstsq(X, y, rcond=None)[0]).max() < 1e-8
print("ba đường: 10, 12, 13 bước; cả ba kết thúc đúng nghiệm OLS")
```

</details>

### 2.4 Sửa đổi thứ hai: LARS thành STAGEWISE — §3.2 của bài báo, tr.12–14

#### Công thức (3.7)–(3.12)

![Công thức (3.8)–(3.12) của Efron 2004, trang 13](<paper_lars_efron2004_tr13_ct3_8_3_12_stagewise.png>)

**Hình 16 — công thức (3.8)–(3.12), trang 13.** Nguồn: cùng bài báo.

$$x_2^{\top}(y - \hat\mu_1 - \varepsilon x_1) > x_1^{\top}(y - \hat\mu_1 - \varepsilon x_1) \tag{3.7}$$

$$N_j \equiv \#\{\text{số bước chọn chỉ số } j\},\qquad j = 1,2,\dots,m \tag{3.8}$$

$$P \equiv (N_1, N_2, \dots, N_m)/N \tag{3.9}$$

$$\mu = \hat\mu + N\varepsilon X_{\mathcal A}P_{\mathcal A} \tag{3.10}$$

$$\mu_{\mathcal A} + \gamma X_{\mathcal A}w_{\mathcal A},
\qquad w_{\mathcal A} = A_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} \tag{3.11}$$

$$\mathcal C_{\mathcal A} = \left\{v = \sum_{j\in\mathcal A}s_jx_jP_j,\ P_j\ge0\right\} \tag{3.12}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $N$ | nguyên | tổng số bước $\varepsilon$ đã đi |
| $N_j$ | nguyên $\ge0$ | trong đó bao nhiêu bước chọn biến $j$ |
| $P$ | $\mathbb{R}^{m}$, $\ge0$, tổng 1 | **tỉ lệ** các bước dành cho từng biến |
| $\mathcal C_{\mathcal A}$ | tập con của $\mathbb{R}^{n}$ | **nón lồi** sinh bởi các cột đã đổi dấu |

**Ý nghĩa trừu tượng — và đây là chỗ then chốt của cả mục.** So (3.10) với (3.11):

* Stagewise đi theo $X_{\mathcal A}P_{\mathcal A}$ với **mọi $P_j\ge0$**, vì $P_j$ đếm số
  bước và số bước không âm được.
* LARS đi theo $X_{\mathcal A}w_{\mathcal A}$, và **$w_{\mathcal A}$ CÓ THỂ ÂM**.

Nếu $w_{\mathcal A}$ có thành phần âm thì hai thuật toán **không thể trùng nhau** — Stagewise
không có cách nào đi theo hướng đó. Đó là toàn bộ nội dung của (3.12): nón lồi
$\mathcal C_{\mathcal A}$ là **tập mọi hướng Stagewise đi được**, và câu hỏi thành ra
*"$u_{\mathcal A}$ có nằm trong nón đó không"*.

> **Chỗ nguồn tự bất đồng — số 16 trong bảng ở mục 0.5.** Ngay sau (3.12), bài báo viết
> *"If $u_{\mathcal A}\in\mathcal C_{\mathcal A}$ then there is no contradiction between
> (3.12) and (3.13)."* Nhưng ở trang đó **chưa có (3.13)** — công thức ấy nằm ở trang sau, và
> nó nói về chuyện khác. Hai công thức đang được so là **(3.10) và (3.11)**, đúng như câu
> ngay trên đó đã viết: *"Comparing (3.10) with (3.11) shows that LARS cannot agree with
> Stagewise if $w_{\mathcal A}$ has negative components."*

#### Định lý 2 và sửa đổi Stagewise

![Định lý 2 của Efron 2004, trang 13](<paper_lars_efron2004_tr13_dinh_ly2.png>)

**Hình 17 — sửa đổi Stagewise và Định lý 2, trang 13.** Nguồn: cùng bài báo.

**Phát biểu gốc, nguyên văn:** *"**Stagewise modification.** Proceed as in (2.8)–(2.13),
except with $u_{\mathcal A}$ replaced by $u_{\hat{\mathcal B}}$, the unit vector lying along
the projection of $u_{\mathcal A}$ into $\mathcal C_{\mathcal A}$."* … *"**Theorem 2.** Under
the Stagewise modification, the LARS algorithm yields all Stagewise solutions."*

*Dịch: **Sửa đổi Stagewise.** Làm như (2.8)–(2.13), chỉ thay $u_{\mathcal A}$ bằng
$u_{\hat{\mathcal B}}$, vectơ đơn vị nằm dọc hình chiếu của $u_{\mathcal A}$ vào
$\mathcal C_{\mathcal A}$. … **Định lý 2.** Dưới sửa đổi Stagewise, thuật toán LARS cho ra
mọi nghiệm Stagewise.*

**Ý nghĩa trừu tượng.** "Chiếu vào nón lồi" nghe trừu tượng, nhưng nó là một bài toán quen:
tìm $P\ge0$ cực tiểu $\lVert u_{\mathcal A} - X_{\mathcal A}P\rVert$ — tức **bình phương tối
thiểu không âm** (NNLS). Những biến có $P_j = 0$ ở nghiệm chính là những biến bị **đẩy ra
khỏi tập hoạt động**. Nên sửa đổi Stagewise cũng là một cách để tập hoạt động co lại, nhưng
theo một lý do khác hẳn lasso: lasso bỏ biến vì **hệ số chạm 0**, Stagewise bỏ biến vì
**hướng đi đòi nó chạy ngược chiều**.

$$\hat\mu_{\mathcal A^{+}} - \hat\mu_{\mathcal A} = \hat\gamma u_{\hat{\mathcal B}}
= \hat\gamma X_{\hat{\mathcal B}}w_{\hat{\mathcal B}} \tag{3.13}$$

$$\operatorname{sign}(\hat\beta_j^{+} - \hat\beta_j) = s_j \tag{3.14}$$

Đẳng thức này nói **hệ số Stagewise chỉ đi ra xa 0, không bao giờ lùi lại**, và đó là ràng
buộc chặt nhất trong ba thuật toán.

> **Chỗ nguồn tự bất đồng — số 17.** Câu giới thiệu (3.13) ở tr.14 kết thúc bằng *"as in
> (3.13)"* — **công thức tự dẫn chính nó**. Theo mạch văn, chỗ ấy phải dẫn tới (3.11).

**Ý nghĩa trừu tượng của (3.14).** Vì $w_{\hat{\mathcal B}}$ nằm trong nón lồi nên mọi thành
phần của nó không âm, nên **hiệu hệ số giữa hai bước liên tiếp luôn cùng dấu với tương quan**.
Nói bằng lời: **hệ số Stagewise không bao giờ lùi ngược chiều**, nó chỉ đi ra xa 0 hoặc đứng
yên. Kiểm trên cả đường:

```
      0/77 hiệu VI PHẠM -> Stagewise thật sự không bao giờ lùi ngược chiều.
```

Đặt ba thuật toán cạnh nhau như bài báo làm ở tr.14:

| Thuật toán | Ràng buộc về dấu |
|---|---|
| **Stagewise** | **hiệu** $\hat\beta_j$ giữa hai bước liên tiếp cùng dấu với $\hat c_j$ — chặt nhất |
| **Lasso** | **bản thân** $\hat\beta_j$ cùng dấu với $\hat c_j$ — chặt vừa |
| **LARS** | **không ràng buộc dấu** (trừ điều Bổ đề 4 nói về biến vừa vào) — lỏng nhất |

> *"From this point of view, Lasso is intermediate between the LARS and Stagewise methods."*
> — tr.14
>
> *Dịch: Nhìn từ góc này, Lasso nằm ở giữa LARS và Stagewise.*

Và chỗ nó xảy ra thật, dựng lại đúng câu của tác giả:

**Nhận xét của chính tác giả, nguyên văn, tr.14:** *"This happened at the arrowed point in the
right panel of Figure 1: there the set $\mathcal A = \{3, 9, 4, 7, 2, 10, 5, 8\}$ was decreased
to $\hat{\mathcal B} = \mathcal A - \{3, 7\}$. It took a total of 13 modified LARS steps to
reach the full OLS solution $\bar\beta_m = (X'X)^{-1}X'y$."*

*Dịch: Chuyện đó xảy ra tại điểm có mũi tên ở bảng phải của Hình 1: ở đó tập
$\mathcal A = \{3, 9, 4, 7, 2, 10, 5, 8\}$ bị thu về $\hat{\mathcal B} = \mathcal A - \{3, 7\}$.
Tổng cộng phải 13 bước LARS đã sửa đổi mới tới nghiệm OLS đầy đủ.*

```
   Đường LARS–Stagewise tự dựng: 13 bước; tới OLS chưa? lệch 7.958e-11
   Các lần tập hoạt động BỊ THU NHỎ: [(7, [3, 7], [2, 3, 4, 5, 7, 8, 9, 10]), (11, [3], [...])]
      dựng lại: tại bước 7, A = [2, 3, 4, 5, 7, 8, 9, 10] bị thu về B̂ = A − [3, 7]
      -> KHỚP cả tập A lẫn tập bị bỏ
```

Đây là chỗ đáng nhớ nhất của cả mục 2 này. Một câu trong thân bài báo, không kèm bảng, không
kèm hình, nêu ra **một tập tám phần tử và hai phần tử bị bỏ** — và phép dựng lại cho đúng cả
tám lẫn cả hai. Không phép kiểm nào khác chạm tới được một câu như thế; chỉ có cách cài đặt
lại thuật toán mới kiểm được.

#### Hình 18 — ε → 0 nghĩa là gì, vẽ ra

![Stagewise với bước epsilon hội tụ về đường lý tưởng](<lars_eps.png>)

**Hình 18 — Stagewise bước $\varepsilon$ hội tụ về đường lý tưởng (hình tự vẽ).** Hình 6 của
bài báo vẽ cái bậc thang bằng tay, trong mặt phẳng hai chiều, để minh hoạ trực giác. Hình này
bù chỗ đó bằng **phép đo**: bậc thang thật, trên dữ liệu thật 10 biến, ở bốn giá trị
$\varepsilon$.

**Cách đọc hình.** Bảng trái vẽ vết của hai biến (3 và 9) theo $t$, với ba giá trị
$\varepsilon$ cộng đường lý tưởng (nét đứt đen). Bảng phải vẽ lệch lớn nhất giữa đường
$\varepsilon$ và đường lý tưởng, **đo tại cùng một $t$**, theo thang log hai chiều.

* $\varepsilon$ càng nhỏ, bậc thang càng phẳng ra và càng bám sát nét đứt.
* Trên bảng phải, các điểm với $\varepsilon\le2$ nằm gần một đường thẳng có độ dốc 1 trong
  thang log-log — tức **sai số tỉ lệ THUẬN với $\varepsilon$**. Đó chính là nghĩa của cụm
  *"idealized Stagewise procedure in which the step size $\varepsilon$ goes to zero"* ở tr.13.
* Số bước ghi cạnh mỗi điểm cho thấy vế còn lại của cái đánh đổi: 221 bước ở
  $\varepsilon = 8$ lên **28 234** bước ở $\varepsilon = 0{,}0625$.

**Điều hình KHÔNG nói ra, và nó được đánh dấu bằng hai chữ thập đỏ.** Với
$\varepsilon\ge4$, phép so mất nghĩa: bước đi thô hơn cả một đoạn của đường lý tưởng, nên
"lệch tại cùng $t$" không còn đo cái nó định đo. Dãy sai số ở hai điểm đó **không đơn điệu**
(3,74 rồi 9,12), và đó là lỗi của **thước đo**, không phải của thuật toán. Giữ chúng lại
trong hình thay vì bỏ đi, vì một dãy không đơn điệu là dấu hiệu phải đọc, không phải thứ để
giấu.

<details>
<summary><b>Mã sinh Hình 18</b></summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()


def stagewise_eps(X, y, eps, t_dich, tran=400000):
    """(1.7) — cập nhật ĉ bằng ma trận Gram nên mỗi bước chỉ tốn O(m), không O(nm)."""
    G = X.T @ X
    c = X.T @ y
    b = np.zeros(X.shape[1])
    duong = [b.copy()]
    for i in range(tran):
        j = int(np.argmax(np.abs(c)))
        s = np.sign(c[j])
        b[j] += eps * s
        c -= eps * s * G[:, j]
        duong.append(b.copy())
        if np.abs(b).sum() >= t_dich:
            return np.array(duong).T, i + 1
    return np.array(duong).T, tran


def huong(X, A, c):
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)
    return s, AA, AA * iG, XA @ (AA * iG), X.T @ (XA @ (AA * iG))


def lars_stage(X, y, eps=1e-12):
    """Sửa đổi Stagewise: thay u_A bằng hình chiếu của nó vào nón lồi C_A, (3.12)."""
    n, m = X.shape
    beta, mu, A, B = np.zeros(m), np.zeros(n), [], [np.zeros(m)]
    while True:
        c = X.T @ (y - mu)
        if A and max(abs(c[j]) for j in A) < 1e-9 * np.abs(X.T @ y).max():
            break
        ngoai = [j for j in range(m) if j not in A]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        P = nnls(X[:, A] * s, uA)[0]
        A = [A[i] for i in range(len(A)) if P[i] > 1e-10]
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)
        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]
        gam = min(ung) if ung else C / AA
        dd = np.zeros(m); dd[A] = s * wA
        beta = beta + gam * dd
        mu = mu + gam * uA
        B.append(beta.copy())
        if not con:
            break
    return np.array(B).T


b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
t_het = np.abs(b_ols).sum()
B_id = lars_stage(X, y)
t_id = np.abs(B_id).sum(0)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
mau = {4.0: "#fca5a5", 1.0: "#f97316", 0.25: "#2563eb"}
for eps in (4.0, 1.0, 0.25):
    D, _ = stagewise_eps(X, y, eps, 0.62 * t_het)
    tt = np.abs(D).sum(0)
    ax.plot(tt, D[2], lw=1.2, color=mau[eps], label=r"$\varepsilon = %g$" % eps)
    ax.plot(tt, D[8], lw=1.2, color=mau[eps])
ax.plot(t_id, B_id[2], "k--", lw=1.8, label="đường lý tưởng (Định lý 2)")
ax.plot(t_id, B_id[8], "k--", lw=1.8)
ax.set_xlim(0, 0.62 * t_het)
ax.set_ylim(-20, 620)
ax.annotate("biến 3", (0.60 * t_het, B_id[2][np.argmin(np.abs(t_id - 0.60 * t_het))]),
            xytext=(-42, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.annotate("biến 9", (0.60 * t_het, B_id[8][np.argmin(np.abs(t_id - 0.60 * t_het))]),
            xytext=(-42, -16), textcoords="offset points", fontsize=8.5, fontweight="bold")
ax.set_xlabel(r"$t = \sum_j |\hat\beta_j|$"); ax.set_ylabel(r"hệ số $\hat\beta_j$")
ax.set_title("Bước ε càng nhỏ, bậc thang càng phẳng ra thành đường lý tưởng\n"
             "(đây là thứ 'staircase' trong Hình 2 tr.6 vẽ bằng tay)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

# So tại CÙNG MỘT t cho mọi ε: điểm dừng của mỗi lần chạy rơi vào một t khác nhau,
# và so ở những t khác nhau thì dãy sai số không còn đơn điệu theo ε — lỗi của THƯỚC ĐO,
# không phải của thuật toán.
T_SO = 0.5 * t_het
eps_l, le_l, buoc_l = [], [], []
for eps in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625):
    bb, kk = stagewise_eps(X, y, eps, T_SO * 1.02)
    t_bb = np.abs(bb).sum(0)
    tai_T = np.array([np.interp(T_SO, t_bb, bb[j]) for j in range(10)])
    ref = np.array([np.interp(T_SO, t_id, B_id[j]) for j in range(10)])
    eps_l.append(eps); le_l.append(np.abs(tai_T - ref).max()); buoc_l.append(kk)
eps_l = np.array(eps_l); le_l = np.array(le_l)
tho = eps_l >= 4.0                               # hai giá trị ε quá thô, tách riêng
ax2.loglog(eps_l[~tho], le_l[~tho], "o-", ms=6, lw=1.6, color="#7c3aed",
           label=r"lệch đo được, $\varepsilon \leq 2$")
ax2.loglog(eps_l[tho], le_l[tho], "x", ms=9, mew=2.0, color="#dc2626",
           label=r"$\varepsilon \geq 4$: bước THÔ hơn cả một đoạn, phép so mất nghĩa")
r_ = eps_l[~tho]
ax2.loglog(r_, r_ * le_l[~tho][0] / r_[0], "--", lw=1.4, color="#6b7280",
           label=r"đường tham chiếu tỉ lệ THUẬN với $\varepsilon$")
for e, l, k in zip(eps_l, le_l, buoc_l):
    ax2.annotate("%d bước" % k, (e, l), xytext=(6, -11), textcoords="offset points",
                 fontsize=7.5, color="#4b5563")
ax2.set_xlabel(r"bước $\varepsilon$"); ax2.set_ylabel(r"lệch với đường lý tưởng tại $t$ cố định")
ax2.set_title("Với ε ≤ 2 sai số giảm TỈ LỆ THUẬN với ε, còn số bước tăng tỉ lệ NGHỊCH\n"
              "muốn mượt gấp đôi thì trả giá gấp đôi, mãi mãi", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper left")
fig.savefig("lars_eps.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("số bước ứng với ε = %s : %s" % (eps_l, buoc_l))
print("lệch ứng với ε = %s" % ["%.4f" % v for v in le_l])
print("tỉ số lệch/ε : %s" % ["%.4f" % (l / e) for e, l in zip(eps_l, le_l)])
```

</details>

### 2.5 Mô phỏng so ba thuật toán — §3.3 của bài báo, tr.14–16

#### Công thức (3.15)–(3.17)

![Công thức (3.15) của Efron 2004, trang 14](<paper_lars_efron2004_tr14_ct3_15_mo_hinh_bac_hai.png>)

**Hình 19 — công thức (3.15), trang 14.** Nguồn: cùng bài báo.

$$\textbf{Mô hình bậc hai: } 10 \text{ hiệu ứng chính},\ 45 \text{ tương tác},\
9 \text{ bình phương} \tag{3.15}$$

$$y^{*} = \mu + \varepsilon^{*} \tag{3.16}$$

$$\operatorname{pe}(\hat\mu) = 1 - \lVert\hat\mu-\mu\rVert^{2}/\lVert\mu\rVert^{2} \tag{3.17}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\mu$ | $\mathbb{R}^{n}$ | "sự thật" của mô phỏng: $X\beta$ với $\beta$ lấy từ 10 bước LARS trên dữ liệu gốc |
| $\varepsilon$ | $\mathbb{R}^{n}$ | phần dư $y - \mu$. **Chữ này ở đây KHÔNG phải bước đi** |
| $\varepsilon^{*}$ | $\mathbb{R}^{n}$ | mẫu lấy lại **có hoàn lại** từ các thành phần của $\varepsilon$ |
| $\operatorname{pe}$ | $\in(-\infty,1]$ | tỉ lệ giải thích được; $\operatorname{pe}(0)=0$, $\operatorname{pe}(\mu)=1$ |

**Ý nghĩa trừu tượng của (3.17).** Đây **không** phải $R^2$ và cũng không phải sai số dự
đoán. Nó đo khoảng cách từ ước lượng tới **giá trị thật $\mu$**, chuẩn hoá theo độ lớn của
$\mu$ — một đại lượng chỉ tính được trong mô phỏng, nơi ta biết $\mu$. Chính vì thế nó là
thước đo đúng để so ba thuật toán: nó bỏ qua nhiễu và chỉ hỏi *"ước lượng có gần sự thật
không"*.

**Cách dựng mô hình bậc hai — và một chỗ bài báo không nói.** (3.15) nói "10 + 45 + 9 = 64",
nhưng **không nói tích được lấy từ cột nào**: từ cột thô hay từ cột đã chuẩn hoá theo (1.1)?
Hai cách cho hai ma trận thiết kế khác nhau. Để chính con số 0,416 của bài báo phân xử:

```
   Dựng tích từ cột NÀO? Bài báo không nói. Thử cả hai, để con số 0,416 phân xử:
      từ cột đã chuẩn hoá (1.1)    -> true R² = 0.415630
      từ cột theo đơn vị gốc       -> true R² = 0.503001
```

Chênh lệch giữa hai cách là **0,087**, không thể nhầm với sai số làm tròn. Cách đầu cho
0,415630 so với **0,416** in trong bài báo — khớp tới ba chữ số thập phân. Vậy tích được lấy
từ cột **đã chuẩn hoá**, và từ đây mọi con số của §3.3 và §4 dựng lại được.

Đây là khuôn đã gặp ở nhiều bài khác: **khi bài báo không nói rõ một lựa chọn, hãy thử mọi
cách hợp lý rồi để một con số của chính bài báo chọn giùm.**

#### Hình 5 của bài báo và bản dựng lại

![Hình 5 của Efron 2004, trang 15](<paper_lars_efron2004_tr15_hinh5_mo_phong.png>)

**Hình 20 — Hình 5 của Efron 2004, trang 15.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"Simulation study comparing LARS, Lasso and Stagewise
algorithms; 100 replications of model (3.15)–(3.16). Solid curve shows average proportion
explained, (3.17), for LARS estimates as function of number of steps
$k = 1, 2, \dots, 40$; Lasso and Stagewise give nearly identical results; small dots indicate
plus or minus one standard deviation over the 100 simulations. Classic Forward Selection
(heavy dashed curve) rises and falls more abruptly."*

*Dịch: Nghiên cứu mô phỏng so các thuật toán LARS, Lasso và Stagewise; 100 lần lặp mô hình
(3.15)–(3.16). Đường liền vẽ tỉ lệ giải thích trung bình, (3.17), cho ước lượng LARS theo số
bước $k = 1,\dots,40$; Lasso và Stagewise cho kết quả gần như y hệt; các chấm nhỏ chỉ cộng
trừ một độ lệch chuẩn qua 100 lần mô phỏng. Forward Selection cổ điển (đường đứt đậm) lên và
xuống đột ngột hơn.*

**Nhận xét của chính tác giả, nguyên văn, tr.15–16:** *"Figure 5's most striking message is
that the three algorithms performed almost identically, and rather well. The average
proportion explained rises quickly, reaching a maximum of 0.963 at $k = 10$, and then declines
slowly as $k$ grows to 40. … The dashed curve in Figure 5 tracks the average proportion
explained by classic Forward Selection. It rises very quickly, to a maximum of 0.950 after
$k = 3$ steps, and then falls back more abruptly than the LARS–Lasso–Stagewise curves. This
behavior agrees with the characterization of Forward Selection as a dangerously greedy
algorithm."*

*Dịch: Thông điệp nổi bật nhất của Hình 5 là ba thuật toán chạy gần như y hệt nhau, và chạy
khá tốt. Tỉ lệ giải thích trung bình lên nhanh, đạt cực đại 0,963 tại $k = 10$, rồi giảm chậm
khi $k$ tăng tới 40. … Đường đứt trong Hình 5 vẽ tỉ lệ giải thích trung bình của Forward
Selection cổ điển. Nó lên rất nhanh, tới cực đại 0,950 sau $k = 3$ bước, rồi tụt lại đột ngột
hơn các đường LARS–Lasso–Stagewise. Hành vi đó phù hợp với việc mô tả Forward Selection là
một thuật toán tham một cách nguy hiểm.*

![Dựng lại Hình 5 — bốn thuật toán trên 100 lần lặp](<lars_mo_phong.png>)

**Hình 21 — dựng lại Hình 5 bằng chính thủ tục §3.3 (hình tự vẽ).**

**Cách đọc hình.** Trục ngang là số bước $k$, trục dọc là $\operatorname{pe}$ trung bình trên
100 lần lặp. Bốn đường; chấm nhạt là $\pm1$ độ lệch chuẩn của LARS.

**Bốn con số của chú thích, đối chiếu:**

| Bài báo nói | Đo lại |
|---|---|
| ba đường LARS/Lasso/Stagewise *"nearly identical"* | tại $k = 40$, ba đường cách nhau **0,015193** |
| cực đại **0,963** tại $k = 10$ | **0,966422** tại $k = 11$ |
| độ lệch chuẩn *"roughly $\pm0{,}02$"* | trung bình **0,022894** |
| Forward Selection cực đại **0,950** tại $k = 3$ | **0,956009** tại $k = 3$ |

Ba trong bốn dòng khớp về **chất** và lệch nhẹ về **lượng** — đúng mức chênh mà 100 lần lặp
với hạt giống khác nhau tạo ra: với độ lệch chuẩn 0,0229 trên 100 lần, sai số chuẩn của
trung bình là khoảng 0,0023, và chênh 0,0034 nằm trong khoảng một tới hai lần sai số đó.

**Điều hình KHÔNG nói ra, và nó là điều đáng nhớ nhất của cả mục.** Forward Selection lên
**nhanh hơn** — sau 3 bước nó đã đạt 0,9560 trong khi LARS mới ở 0,8867. Nếu chỉ nhìn ba
bước đầu thì Forward Selection thắng. Cái làm nó thua là **chỗ nó tụt xuống**: tại $k = 40$
nó còn 0,8299 trong khi LARS còn 0,8999. Nên nhận xét *"dangerously greedy"* không nói về tốc
độ lên mà nói về **độ dốc đi xuống sau đỉnh** — và người đọc chỉ thấy điều đó khi nhìn cả
đường, không phải khi nhìn con số cực đại.

Một chỗ nữa đo được và bài báo có nói:

```
   Bài báo tr.15: *μ̂^(40)* averaged 33.23 nonzero terms with Stagewise, compared to
                  35.83 for Lasso and 40 for LARS.*
      Stagewise : trung bình số hệ số khác 0 ở bước 40 = 35.5900  (bài báo 33.23)
      Lasso     : trung bình số hệ số khác 0 ở bước 40 = 35.5200  (bài báo 35.83)
      LARS      : trung bình số hệ số khác 0 ở bước 40 = 40.0000  (bài báo 40.00)
```

Hai dòng dưới khớp, dòng Stagewise lệch **2,36**. Đó là chỗ lệch lớn nhất của cả mục 2 này,
và nó nhất quán với chỗ lệch khác cùng họ: số bước Stagewise trên mô hình $m = 64$ đo được
**251** so với **255** bài báo nêu. Cả hai đều dính tới bản cài đặt sửa đổi Stagewise, nơi
quy tắc phá hoà và ngưỡng coi $P_j$ là 0 có ảnh hưởng thật.

<details>
<summary><b>Mã sinh Hình 21</b></summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Trung bình trên 100 lần lặp của mô hình (3.15)–(3.16); `tools/v_lars.py` mục 23 in ra.
pe_lar = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
          0.9499846647152862, 0.9576921960697135, 0.961458250035188, 0.9636565167276252,
          0.9655683013856813, 0.9663587533276058, 0.9664215946576782, 0.9661503758712067,
          0.9654225353654488, 0.9640054497548314, 0.9626560941512722, 0.9613381404251197,
          0.9596821500093218, 0.9578209438065511, 0.9555312256565532, 0.953829856691638,
          0.9516619196783792, 0.9493813851697237, 0.9473947158962684, 0.944808642708031,
          0.942532379020172, 0.9399211919494506, 0.9376799909928732, 0.9347056063648586,
          0.9322551935841894, 0.92888660295163, 0.9262251093004045, 0.9234794153767786,
          0.9200382777157533, 0.917162824402744, 0.9142059979443059, 0.9113782309499264,
          0.9090176699056954, 0.90635004834155, 0.9029951292396702, 0.8998684502813599]
pe_lasso = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
            0.9499846647152862, 0.9576921960697135, 0.9614400803090214, 0.9636134646553534,
            0.9655355499950666, 0.9662947328926659, 0.9664044544782391, 0.9661602846741324,
            0.9654263280071708, 0.9640803682703529, 0.9627765528151508, 0.9614611230281151,
            0.9599259226388019, 0.95816009017857, 0.9561762196016724, 0.9544014720077297,
            0.9528017238420513, 0.950522618801289, 0.9484176370486099, 0.9463630446934185,
            0.9442225225666627, 0.9420607607927226, 0.9396949756693962, 0.9375758627701591,
            0.935133210173877, 0.9328533035033751, 0.9305242608093822, 0.9280701859746268,
            0.9256971715638999, 0.9232161351710918, 0.9206081881532522, 0.9182239371560565,
            0.9161123746440438, 0.913886500003819, 0.9113732428732785, 0.9093210906935684]
pe_stage = [0.0, 0.16137326080068815, 0.7410025520685758, 0.8867110557927856, 0.9350087550592083,
            0.9499840971226687, 0.9576912805244342, 0.9614686599091726, 0.9635251843438235,
            0.9655947769729758, 0.9663695466621641, 0.9665137898860617, 0.9663243113102569,
            0.9657666868765616, 0.9645333747845557, 0.9635824702787835, 0.9621141891420484,
            0.9607368508067836, 0.9594665228566375, 0.9578473015979266, 0.9561974147949506,
            0.9546738175677503, 0.9528911186828534, 0.9511996861095068, 0.9493604565502447,
            0.9474698940539311, 0.9454835037929022, 0.9433090955182284, 0.9414471887306022,
            0.9391338237569948, 0.937429632502935, 0.9353875743740392, 0.9334003400713742,
            0.9310364538337607, 0.9285904200248525, 0.9264561485738453, 0.9240994742513574,
            0.921777964408827, 0.9198064748549973, 0.9172320116141034, 0.9150617103413135]
pe_fs = [0.0, 0.6972223593296433, 0.9425632100025563, 0.9560087865983492, 0.950345383351585,
         0.9446973008912662, 0.9357902713259486, 0.9299300087723165, 0.9222093416690079,
         0.915919906000439, 0.9101383405292719, 0.9048849248621356, 0.8991327651703804,
         0.8939620571742349, 0.8898345391015936, 0.8860798767522493, 0.8823185757173988,
         0.8788436064135609, 0.8754819561475565, 0.8725808590290292, 0.8698701757653601,
         0.8670366043797304, 0.8642586531316633, 0.8618924310041476, 0.8601317550832228,
         0.8580451195277611, 0.8558467400971653, 0.8538709460558905, 0.8513724244952986,
         0.8493424429577768, 0.8476798551333508, 0.8457922216722076, 0.8441174907694274,
         0.8424221616512232, 0.8402995083679639, 0.838385814882017, 0.8368641685799239,
         0.8351872297567746, 0.8335009276490701, 0.8315077280578658, 0.8298674871053177]
sd_lar = [0.0, 0.10073285876923689, 0.10789147513603316, 0.054496970677128796,
          0.03538153852684438, 0.026091836456359953, 0.022508801690708034,
          0.021275547435047885, 0.01998331144348188, 0.017926525845322124,
          0.016769326684056857, 0.016119133769422176, 0.015906472970491848,
          0.015545471239661608, 0.01555847665084218, 0.015152927440251375,
          0.015058925907182295, 0.015307847666781218, 0.015328259659380443,
          0.015031126999257466, 0.015086454536766417, 0.015391092129780754,
          0.01604804399043123, 0.016081781838553355, 0.01615365841910121,
          0.016399082484252963, 0.016263111791931687, 0.016316186523168314,
          0.01679884702322542, 0.01724328117076876, 0.017398788320719942,
          0.017661327999642453, 0.018414299873111765, 0.018812274656518267,
          0.01922733707726031, 0.019758383389068847, 0.01986770799186052,
          0.02052066078224579, 0.02021624304918996, 0.021006755132725888,
          0.021922781236303087]

k = np.arange(41)
fig, ax = plt.subplots(figsize=(10.2, 5.2))
ax.plot(k, pe_lar, "-", lw=2.2, color="#1d4ed8", label="LARS")
ax.plot(k, pe_lasso, "--", lw=1.6, color="#059669", label="Lasso")
ax.plot(k, pe_stage, ":", lw=2.0, color="#ea580c", label="Stagewise")
ax.plot(k, pe_fs, "-", lw=2.2, color="#b91c1c", dashes=(6, 3),
        label="Forward Selection cổ điển")
ax.plot(k, np.array(pe_lar) + np.array(sd_lar), ".", ms=3.0, color="#93c5fd")
ax.plot(k, np.array(pe_lar) - np.array(sd_lar), ".", ms=3.0, color="#93c5fd")
ax.plot([], [], ".", ms=6, color="#93c5fd", label="± 1 độ lệch chuẩn của LARS")

kb = int(np.argmax(pe_lar))
ax.plot([kb], [pe_lar[kb]], "o", ms=10, mfc="none", mec="#1d4ed8", mew=2.0)
ax.annotate("LARS cao nhất %.4f tại k = %d\n(bài báo: 0,963 tại k = 10)"
            % (pe_lar[kb], kb), (kb, pe_lar[kb]), xytext=(34, -28),
            textcoords="offset points", fontsize=8.5, color="#1d4ed8",
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.0))
kf = int(np.argmax(pe_fs))
ax.plot([kf], [pe_fs[kf]], "o", ms=10, mfc="none", mec="#b91c1c", mew=2.0)
ax.annotate("Forward Selection cao nhất %.4f tại k = %d\nrồi TỤT nhanh hơn hẳn "
            "(bài báo: 0,950 tại k = 3)" % (pe_fs[kf], kf), (kf, pe_fs[kf]),
            xytext=(46, 24), textcoords="offset points", fontsize=8.5, color="#b91c1c",
            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0))

ax.set_xlabel("số bước k")
ax.set_ylabel(r"tỉ lệ giải thích được  $\mathrm{pe}(\hat\mu) = 1 - \|\hat\mu-\mu\|^2/\|\mu\|^2$  (3.17)")
ax.set_ylim(0.60, 1.005)
ax.set_title("Dựng lại Hình 5 tr.15 — 100 lần lặp mô hình (3.15)–(3.16), m = 64\n"
             "ba thuật toán LARS · Lasso · Stagewise gần như KHÔNG phân biệt được nhau",
             fontsize=10)
ax.legend(fontsize=8.5, loc="lower left", ncol=2)
fig.savefig("lars_mo_phong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("LARS đỉnh %.6f tại k = %d ; Forward Selection đỉnh %.6f tại k = %d"
      % (max(pe_lar), int(np.argmax(pe_lar)), max(pe_fs), int(np.argmax(pe_fs))))
print("khoảng cách ba đường LARS/Lasso/Stagewise tại k = 40: %.6f"
      % (max(pe_lar[40], pe_lasso[40], pe_stage[40])
         - min(pe_lar[40], pe_lasso[40], pe_stage[40])))
```

</details>

### 2.6 Bốn biến thể khác — §3.4 của bài báo, tr.16–17

#### Positive Lasso: (3.18)–(3.19)

![Công thức (3.18)–(3.19) của Efron 2004, trang 16](<paper_lars_efron2004_tr16_ct3_18_3_19_positive_lasso.png>)

**Hình 22 — công thức (3.18)–(3.19), trang 16.** Nguồn: cùng bài báo.

$$\text{cực tiểu hoá } S(\hat\beta) \text{ với ràng buộc } T(\hat\beta)\le t
\text{ và mọi } \hat\beta_j\ge0 \tag{3.18}$$

$$\hat\gamma = \min_{j\in\mathcal A^{c}}{}^{+}
\left\{\frac{\hat C-\hat c_j}{A_{\mathcal A}-a_j}\right\} \tag{3.19}$$

**Ý nghĩa trừu tượng.** Bài báo nói cả sửa đổi bằng ba dòng: đổi $\lvert\hat c_j\rvert$ thành
$\hat c_j$ ở **cả hai chỗ** trong (2.9); đặt $s_j = 1$ thay cho (2.10); và bỏ nhánh thứ hai
của (2.13). Cách hiểu hình học được chính tác giả nêu ra:

> *"The changes above amount to considering the $x_j$ as generating half-lines rather than
> full one-dimensional spaces."* — tr.16
>
> *Dịch: Những thay đổi trên tương đương với việc coi mỗi $x_j$ sinh ra một NỬA đường thẳng
> chứ không phải cả một không gian một chiều.*

Mỗi biến chỉ được đi **một chiều**. Hệ quả tức thì: nghiệm cuối không còn là nghiệm OLS nữa
mà là nghiệm bình phương tối thiểu **có ràng buộc không âm**.

**Nhận xét của chính tác giả, nguyên văn, tr.16:** *"The positive Lasso usually does not
converge to the full OLS solution $\bar\beta_m$, even for very large choices of $t$."*

*Dịch: Positive Lasso thường không hội tụ về nghiệm OLS đầy đủ $\bar\beta_m$, ngay cả với
những giá trị $t$ rất lớn.*

Dựng lại, và kiểm bằng một đường tính hoàn toàn khác — hàm `nnls` của `scipy`:

```
   Đường positive lasso: 5 bước; hệ số cuối có âm không? nhỏ nhất = 0.000000
   Nghiệm bình phương tối thiểu KHÔNG ÂM (scipy.optimize.nnls): lệch 1.023e-12
      Σ|β̂| của positive lasso ở cuối : 1439.7988
      Σ|β̂| của OLS đầy đủ            : 3459.9776
      OLS có mấy hệ số ÂM? 3/10 -> nên ràng buộc β ≥ 0 CẮT MẤT nghiệm OLS
      RSS: positive lasso 1358786.98 · OLS 1263985.79 · tệ hơn 7.5002%
   Số biến vào được: 5/10 (các biến [3, 4, 8, 9, 10])
```

Hai đường tính độc lập trùng nhau tới $10^{-12}$: sửa đổi LARS ba dòng cho **đúng** nghiệm
NNLS. Và câu *"usually does not converge to the full OLS"* có lý do đếm được: nghiệm OLS có
**3 trên 10** hệ số âm, nên ràng buộc $\hat\beta\ge0$ cắt mất nó. Cái giá là RSS tệ hơn
**7,5002 %**, và chỉ **5 trên 10** biến vào được mô hình.

#### LARS–OLS hybrid: (3.20) — CHỖ IN SAI THỨ NHẤT

![Công thức (3.20) của Efron 2004, trang 16](<paper_lars_efron2004_tr16_ct3_20_lars_ols_hybrid.png>)

**Hình 23 — công thức (3.20), trang 16, đúng như bài báo in.** Nguồn: cùng bài báo.

Bài báo in:

$$R^{2}(\bar\beta_k) - R^{2}(\hat\beta_k)
= \frac{1-\rho_k^{2}}{\rho_k(2-\rho_k)}\left[R^{2}(\hat\beta_k) - R^{2}(\hat\beta_{k-1})\right],
\qquad \rho_k = \hat\gamma_k/\bar\gamma_k \tag{3.20 — bản in}$$

**Ý nghĩa trừu tượng của cái mà (3.20) định nói.** Sau $k$ bước, LARS đã chọn ra một tập biến
$\mathcal A_k$ nhưng **chưa khớp hết cỡ** vào chúng. "LARS–OLS hybrid" là ý tưởng dùng LARS
để **chọn** biến rồi dùng OLS để **ước lượng** hệ số trên đúng tập đó. (3.20) trả lời: làm
thế thì $R^2$ tăng thêm bao nhiêu — và câu trả lời được viết **hoàn toàn bằng $\rho_k$**, tức
bằng tỉ lệ quãng đường mà bước vừa rồi đã đi.

**Suy lại công thức từ hình học của Hình 12 — Hình 4 của bài báo.** Đặt $S(\cdot)$ là tổng bình phương phần dư.
Ba dữ kiện, tất cả đều đã có:

* $\hat\mu_k = \hat\mu_{k-1} + \hat\gamma_ku_k$ và $\bar y_k = \hat\mu_{k-1}+\bar\gamma_ku_k$
  — hai điểm trên **cùng một đường thẳng**, (2.19) và (2.22);
* $y - \bar y_k \perp \mathcal L(X_k)$ và $u_k\in\mathcal L(X_k)$ — hình chiếu thì vuông góc;
* $\lVert u_k\rVert = 1$ — (2.7).

Từ đó, theo định lý Pythagore:

$$S(\hat\mu_k) - S(\bar y_k) = (\bar\gamma_k-\hat\gamma_k)^{2} = \bar\gamma_k^{2}(1-\rho_k)^{2}$$

$$S(\hat\mu_{k-1}) - S(\hat\mu_k) = \bar\gamma_k^{2} - \bar\gamma_k^{2}(1-\rho_k)^{2}
= \bar\gamma_k^{2}\,\rho_k(2-\rho_k)$$

Chia hai đẳng thức cho nhau, và nhớ $R^2 = 1 - S/\lVert y\rVert^{2}$:

$$\boxed{\;R^{2}(\bar\beta_k) - R^{2}(\hat\beta_k)
= \frac{(1-\rho_k)^{2}}{\rho_k(2-\rho_k)}\left[R^{2}(\hat\beta_k) - R^{2}(\hat\beta_{k-1})\right]\;}
\tag{3.20 — bản đúng}$$

Tử số là $(1-\rho_k)^{2}$, **không** phải $1-\rho_k^{2}$. Đo cả mười bước:

```
    k |   ρ_k    | VẾ TRÁI đo được | (1−ρ)²/(ρ(2−ρ))·Δ | 1−ρ²/(ρ(2−ρ))·Δ | sai số bản in
    1 | 0.063323 |   0.30174600    |    0.30174600     |   0.34254468  |    1.14 lần
    2 | 0.490736 |   0.10822873    |    0.10822873     |   0.31681085  |    2.93 lần
    3 | 0.302106 |   0.06274550    |    0.06274550     |   0.11706823  |    1.87 lần
    4 | 0.588293 |   0.01257054    |    0.01257054     |   0.04849497  |    3.86 lần
    5 | 0.317724 |   0.01382713    |    0.01382713     |   0.02670521  |    1.93 lần
    6 | 0.223232 |   0.00881504    |    0.00881504     |   0.01388168  |    1.57 lần
    7 | 0.710270 |   0.00117392    |    0.00117392     |   0.00692965  |    5.90 lần
    8 | 0.725865 |   0.00015877    |    0.00015877     |   0.00099958  |    6.30 lần
    9 | 0.071072 |   0.00203055    |    0.00203055     |   0.00234126  |    1.15 lần
   10 | 1.000000 |   -0.00000000   |    0.00000000     |   0.00000000  |    0.00 lần

   Công thức (1 − ρ)² khớp 10/10 dòng trong sai số 1.39e-15;
   bản in 1 − ρ² lệch tới 6.30 lần ở dòng tệ nhất.
   Hệ số giữa hai bản là (1+ρ)/(1−ρ), bằng 1 chỉ khi ρ = 0 và ra ∞ khi ρ → 1.
```

![Đối chiếu công thức (3.20) bản in với bản đúng](<lars_ct320.png>)

**Hình 24 — công thức (3.20): bản in so với bản suy lại (hình tự vẽ).**

**Cách đọc hình.** Bảng trái: ba cột cho mỗi bước $k$ — vế trái đo trực tiếp (đen), bản
$(1-\rho)^2$ (xanh), bản in $1-\rho^2$ (đỏ). Trục dọc là log đối xứng vì các giá trị trải qua
bốn bậc độ lớn. **Cột xanh chồng khít cột đen ở cả mười bước; cột đỏ thì không.** Bảng phải:
tỉ số giữa hai bản là $(1+\rho)/(1-\rho)$, vẽ thành đường liền, với mười điểm đo được nằm
đúng trên đó.

**Vì sao lỗi này sống được.** Ba lý do, và cả ba đáng ghi:

1. **Hai bản trùng nhau tại $\rho = 0$**, và lệch chỉ 1,14 lần tại $\rho = 0{,}063$ — tức ở
   bước đầu tiên, chỗ người ta hay kiểm nhất, nó gần như đúng.
2. **Hai bản cùng cho 0 tại $\rho = 1$**, tức ở bước cuối. Cả hai đầu của dãy đều "trông
   được".
3. **Bài báo không in một con số nào cho (3.20).** Nó chỉ nói kết quả *"on the order of 0.01
   for $k\ge4$"*, và điều đó **đúng với cả hai bản** ở phần lớn các bước. Không có bảng số
   nào để đối chiếu, nên không có gì báo động.

Và đo lại chính câu định tính ấy:

```
   Bài báo tr.17: *The increases in R² were small … on the order of 0.01 for k ≥ 4
                  compared with R² ≐ 0.50*
      R² của mô hình đầy đủ = 0.517748
      mức tăng khi thay β̂_k bằng OLS trên cùng tập biến, k = 4..9: 0.0126 0.0138 0.0088 0.0012 0.0002 0.0020
```

$R^2 = 0{,}517748$ so với *"$\doteq 0{,}50$"* — khớp. Sáu mức tăng nằm trong khoảng
0,0002–0,0138, tức *"on the order of 0.01"* cũng khớp. **Câu định tính đúng; chỉ dòng công
thức là sai.**

<details>
<summary><b>Mã sinh Hình 24</b></summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# `tools/v_lars.py` mục 12 và 23 in ra ba mảng này
rho = [0.06332340658933504, 0.4907357695538847, 0.3021055872663274, 0.5882932706030122,
       0.3177236980040751, 0.22323214020969445, 0.7102700478790898, 0.7258650199952871,
       0.0710721109996991, 0.9999999999999966]
ve_trai = [0.30174599602085383, 0.10822872766909852, 0.06274549583049804,
           0.0125705398921373, 0.01382712633977834, 0.008815043969504055,
           0.00117392379690473, 0.00015877310695899816, 0.0020305496913188836,
           -7.771561172376096e-16]
delta_R2 = [0.04217776420452635, 0.30907878776564135, 0.0660803826640356,
            0.061590873697595305, 0.015876628878104482, 0.005794679233685929,
            0.012810775005050457, 0.0019539765120778485, 0.00032260034371911495,
            0.0020619539159140166]

rho = np.array(rho); ve_trai = np.array(ve_trai); delta_R2 = np.array(delta_R2)
dung = (1 - rho) ** 2 / (rho * (2 - rho)) * delta_R2      # (1 − ρ)²  — suy từ Hình 4
nhu_in = (1 - rho ** 2) / (rho * (2 - rho)) * delta_R2    # 1 − ρ²    — bản in tr.16
k = np.arange(1, 11)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7))
w = 0.27
ax.bar(k - w, ve_trai, w, color="#111827", label="VẾ TRÁI, đo trực tiếp")
ax.bar(k, dung, w, color="#059669", label=r"$(1-\rho_k)^2 / [\rho_k(2-\rho_k)]$ — suy lại")
ax.bar(k + w, nhu_in, w, color="#dc2626",
       label=r"$(1-\rho_k^2) / [\rho_k(2-\rho_k)]$ — BẢN IN tr.16")
ax.set_yscale("symlog", linthresh=1e-4)
ax.set_xticks(k)
ax.set_xlabel("bước k"); ax.set_ylabel(r"$R^2(\bar\beta_k) - R^2(\hat\beta_k)$")
ax.set_title("Công thức (3.20): cột xanh CHỒNG KHÍT cột đen, cột đỏ thì không\n"
             "(trục dọc log đối xứng vì các giá trị trải qua bốn bậc)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

ti = (1 + rho) / (1 - rho)
r = np.linspace(0.0, 0.95, 300)
ax2.plot(r, (1 + r) / (1 - r), lw=2.0, color="#dc2626",
         label=r"hệ số sai của bản in: $(1+\rho)/(1-\rho)$")
ax2.plot(rho[:-1], ti[:-1], "o", ms=7, color="#111827", zorder=3,
         label="10 bước đo được trên bộ diabetes")
for i, dx, dy, ha in ((1, -10, 16, "right"), (6, -16, -34, "right"), (7, 16, 14, "left")):
    ax2.annotate("k = %d\nsai %.2f lần" % (i + 1, ti[i]), (rho[i], ti[i]),
                 xytext=(dx, dy), textcoords="offset points", fontsize=8,
                 ha=ha, color="#111827",
                 arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8))
ax2.axhline(1, color="#059669", lw=1.4, ls="--", label="bằng nhau (chỉ khi ρ = 0)")
ax2.set_xlabel(r"$\rho_k = \hat\gamma_k / \bar\gamma_k$")
ax2.set_ylabel("bản in / bản đúng")
ax2.set_ylim(0, 12)
ax2.set_title("Hai bản chỉ trùng nhau tại ρ = 0 và phân kỳ khi ρ → 1\n"
              "bước nào LARS đi gần hết tới OLS thì chỗ sai càng lớn", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper left")
fig.savefig("lars_ct320.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

assert np.abs(dung - ve_trai).max() < 1e-12, "bản (1−ρ)² KHÔNG khớp"
print("bản (1−ρ)² khớp vế trái tới %.2e trên cả 10 bước" % np.abs(dung - ve_trai).max())
print("bản in 1−ρ² sai lớn nhất %.2f lần (tại k = %d)"
      % (ti[:-1].max(), int(np.argmax(ti[:-1])) + 1))
```

</details>

#### Hai biến thể còn lại

**Main effects first (tr.17).** Chạy LARS trên bốn biến chính trước, lấy $\mathcal A_4$, rồi
chạy lại LARS với $y$ thay bằng $y - \hat\mu_4$ và $X$ thay bằng ma trận $n\times6$ của các
tương tác $x_{3:9}, x_{3:4}, \dots, x_{4:7}$. Ý nghĩa: **áp một thứ tự ưu tiên lên thuật
toán** — không cho tương tác cạnh tranh với hiệu ứng chính. Đây là một ràng buộc từ kiến thức
chuyên ngành, không phải từ dữ liệu.

**Backward Lasso (tr.17).** Chạy ngược đường lasso, xuất phát **từ nghiệm OLS đầy đủ**. Lý do
làm được nằm ở (3.1): vì mọi hệ số của $\bar\beta_m$ khác 0 và dấu của chúng phải trùng với
$s_j$ của bước LARS cuối, nên hướng cách đều góc cuối cùng **tính ngược ra được**.

> *"The crucial property of the Lasso that makes backward navigation possible is (3.1), which
> permits calculation of the correct equiangular direction $u_{\mathcal A}$ at each step. In
> this sense Lasso can be just as well thought of as a backward-moving algorithm. This is not
> the case for LARS or Stagewise, both of which are inherently forward-moving algorithms."*
> — tr.17
>
> *Dịch: Tính chất then chốt của Lasso làm cho việc đi ngược khả thi là (3.1), vốn cho phép
> tính đúng hướng cách đều góc $u_{\mathcal A}$ ở mỗi bước. Theo nghĩa ấy, Lasso hoàn toàn có
> thể được coi là một thuật toán đi lùi. LARS và Stagewise thì không: cả hai vốn dĩ là thuật
> toán chỉ đi tới.*

**Ý nghĩa đáng nhớ.** Lasso được định nghĩa bằng một bài toán tối ưu **không có chiều thời
gian**, nên đường nghiệm của nó đi xuôi hay ngược đều được. LARS và Stagewise được định nghĩa
bằng một **thủ tục**, nên chúng chỉ có một chiều. Ba thuật toán cho gần cùng một đường, nhưng
chỉ một trong ba có tính chất đó.

> **Chỗ nguồn in sai — số 18 trong bảng ở mục 0.5.** Câu cuối của đoạn Backward Lasso viết
> *"As in (2.3), (3.5) the calculation of $\tilde\gamma$ and $\hat\gamma$ is easy."* Nhưng
> $\hat\gamma$ được tính bằng **(2.13)**, không phải (2.3) — (2.3) là
> $\hat\mu_2 = \hat\mu_1 + \hat\gamma_2u_2$, một đẳng thức cập nhật chứ không phải một công
> thức tính bước.

### 2.7 Bậc tự do và $C_p$ — §4 của bài báo, tr.17–21

Đây là mục trả lời câu hỏi mà mọi mục trước đó đã đẩy sang: **LARS cho cả một đường, vậy chọn
điểm nào?** Bài báo không trả lời bằng kiểm định chéo mà bằng một tiêu chí giải tích — và
điều đó chỉ làm được vì cấu trúc của LARS đơn giản một cách khác thường.

#### Công thức (4.1)–(4.5) — định nghĩa bậc tự do

$$y \sim (\mu, \sigma^{2}I) \tag{4.1}$$

$$(\hat\mu_i-\mu_i)^{2} = (y_i-\hat\mu_i)^{2} - (y_i-\mu_i)^{2} + 2(\hat\mu_i-\mu_i)(y_i-\mu_i) \tag{4.2}$$

$$E\left\{\frac{\lVert\hat\mu-\mu\rVert^{2}}{\sigma^{2}}\right\}
= E\left\{\frac{\lVert y-\hat\mu\rVert^{2}}{\sigma^{2}} - n\right\}
+ 2\sum_{i=1}^{n}\frac{\operatorname{cov}(\hat\mu_i, y_i)}{\sigma^{2}} \tag{4.3}$$

Đây là chỗ cả mục 4 đứng lên: **sai số thật không đo được, nhưng nó bằng một đại lượng đo
được cộng một số hạng hiệu chỉnh** — và số hạng hiệu chỉnh ấy chính là thứ sẽ được đặt tên là
bậc tự do.

![Công thức (4.4)–(4.5) của Efron 2004, trang 18](<paper_lars_efron2004_tr18_ct4_4_4_5_df_cp.png>)

**Hình 25 — công thức (4.4)–(4.5), trang 18.** Nguồn: cùng bài báo.

$$\operatorname{df}_{\mu,\sigma^{2}} = \sum_{i=1}^{n}\operatorname{cov}(\hat\mu_i, y_i)/\sigma^{2} \tag{4.4}$$

$$C_p(\hat\mu) = \frac{\lVert y-\hat\mu\rVert^{2}}{\sigma^{2}} - n + 2\operatorname{df}_{\mu,\sigma^{2}} \tag{4.5}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\mu$ | $\mathbb{R}^{n}$ | giá trị kỳ vọng thật của $y$ — không biết được |
| $\sigma^{2}$ | vô hướng $>0$ | phương sai nhiễu, giả thiết bằng nhau ở mọi mẫu |
| $\operatorname{cov}(\hat\mu_i,y_i)$ | vô hướng | **dự đoán ở mẫu $i$ chạy theo đáp án ở mẫu $i$ bao nhiêu** |
| $\operatorname{df}$ | vô hướng | tổng các hiệp phương sai đó, chuẩn hoá theo $\sigma^2$ |

**Ý nghĩa trừu tượng — và đây là chỗ hay nhất của cả mục 4.** (4.3) là một đẳng thức **hoàn
toàn tổng quát**: nó không giả thiết gì về $\hat\mu$ ngoài việc nó là một hàm của $y$. Đọc
theo lời: **sai số thật = sai số quan sát được − $n$ + hai lần một đại lượng đo mức mô hình
"nghe theo" dữ liệu.** Đại lượng cuối được đặt tên là bậc tự do, và (4.5) chỉ là (4.3) viết
lại thành một tiêu chí ước lượng được.

Chỗ đáng nhớ: định nghĩa (4.4) **không nhắc tới số tham số**. Với một mô hình tuyến tính
$\hat\mu = My$ thì nó rút về $\operatorname{trace}(M)$, và với OLS trên $k$ biến thì nó ra
đúng $k$ — nhưng đó là **hệ quả**, không phải định nghĩa. Với một thủ tục **thích nghi** như
LARS, nơi tập biến phụ thuộc chính $y$, "số tham số" không còn là một khái niệm rõ ràng, còn
(4.4) thì vẫn rõ ràng.

#### Công thức (4.6)–(4.10) — đo df bằng bootstrap, và xấp xỉ đơn giản

$$y^{*}\sim N(\bar\mu, \bar\sigma^{2}),\qquad \hat\mu^{*} = g(y^{*}) \tag{4.6}$$

$$\widehat{\operatorname{cov}}_i = \frac{\sum_{b=1}^{B}\hat\mu_i^{*}(b)\left[y_i^{*}(b)-y_i^{*}(\cdot)\right]}{B-1},
\qquad y^{*}(\cdot) = \frac{\sum_{b=1}^{B}y^{*}(b)}{B} \tag{4.7}$$

$$\widehat{\operatorname{df}} = \sum_{i=1}^{n}\widehat{\operatorname{cov}}_i/\bar\sigma^{2} \tag{4.8}$$

Ba công thức (4.6)–(4.8) là một thủ tục **đo bậc tự do bằng thực nghiệm**: sinh dữ liệu giả
quanh một mô hình đã biết, chạy lại đúng thuật toán, rồi xem dự đoán chạy theo dữ liệu bao
nhiêu. Nó không cần biết thuật toán bên trong làm gì — chỉ cần chạy lại được.

![Công thức (4.9) của Efron 2004, trang 18](<paper_lars_efron2004_tr18_ct4_9_xap_xi_don_gian.png>)

**Hình 26 — công thức (4.9), trang 18 — "xấp xỉ đơn giản".** Nguồn: cùng bài báo.

$$\operatorname{df}(\hat\mu_k) \doteq k \tag{4.9}$$

Một dòng, và nó là kết quả được nhắc lại nhiều nhất của cả bài báo: **số bước LARS đã đi
chính là bậc tự do đã tiêu.**

![Công thức (4.10) của Efron 2004, trang 19](<paper_lars_efron2004_tr19_ct4_10_cp.png>)

**Hình 27 — công thức (4.10), trang 19.** Nguồn: cùng bài báo.

$$C_p(\hat\mu_k) \doteq \lVert y-\hat\mu_k\rVert^{2}/\bar\sigma^{2} - n + 2k \tag{4.10}$$

**Ý nghĩa trừu tượng của (4.9).** Dấu $\doteq$ ở đây là một lời tuyên bố mạnh gói trong một
ký hiệu nhỏ. Nó nói: **sau $k$ bước LARS, mô hình "nghe theo" dữ liệu đúng bằng mức một mô
hình OLS trên $k$ biến ĐƯỢC CHỌN TRƯỚC.** Điều đó phản trực giác, vì LARS đã **dùng $y$** để
chọn ra $k$ biến ấy, và phép chọn đó lẽ ra phải tốn thêm bậc tự do. (4.9) nói phần tốn thêm
đó **được bù lại** bởi việc LARS co hệ số — và hai thứ triệt tiêu nhau vừa khéo.

Hệ quả thực hành của (4.10), chính tác giả nêu:

> *"The formula, which is the same as the $C_p$ estimate of risk for an OLS estimator based on
> a subset of $k$ preselected predictor vectors, has the great advantage of not requiring any
> further calculations beyond those for the original LARS estimates. The formula applies only
> to LARS, and not to Lasso or Stagewise."* — tr.19
>
> *Dịch: Công thức này, giống đúng ước lượng $C_p$ của rủi ro cho một ước lượng OLS dựa trên
> một tập con gồm $k$ vectơ dự báo được chọn TRƯỚC, có cái lợi rất lớn là không đòi thêm bất
> kỳ phép tính nào ngoài những phép tính đã làm cho ước lượng LARS gốc. Công thức chỉ áp cho
> LARS, không áp cho Lasso hay Stagewise.*

Câu cuối là câu người đọc hay bỏ qua: **(4.10) KHÔNG dùng được cho lasso.** Lý do được nói ở
tr.21: với lasso, tổng số bước có thể vượt $m$, nên $k$ không còn là một đại lượng đếm được
theo cùng nghĩa.

#### Hình 6 và Hình 7 của bài báo

![Hình 6 của Efron 2004, trang 19](<paper_lars_efron2004_tr19_hinh6_bac_tu_do.png>)

**Hình 28 — Hình 6 của Efron 2004, trang 19.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"Degrees of freedom for LARS estimates $\hat\mu_k$: (left)
diabetes study, Table 1, $k = 1, 2, \dots, m = 10$; (right) quadratic model (3.15) for the
diabetes data, $m = 64$. Solid line is simple approximation $\mathrm{df}_k = k$. Dashed lines
are approximate 95% confidence intervals for the bootstrap estimates. Each panel based on
$B = 500$ bootstrap replications."*

*Dịch: Bậc tự do của các ước lượng LARS $\hat\mu_k$: (trái) nghiên cứu tiểu đường, Bảng 1,
$k = 1,\dots,m = 10$; (phải) mô hình bậc hai (3.15) cho dữ liệu tiểu đường, $m = 64$. Đường
liền là xấp xỉ đơn giản $\mathrm{df}_k = k$. Các đường đứt là khoảng tin cậy 95 % xấp xỉ cho
các ước lượng bootstrap. Mỗi bảng dựa trên $B = 500$ lần lặp bootstrap.*

**Nhận xét của chính tác giả, nguyên văn, tr.18–19:** *"The left panel of Figure 6 shows
$\widehat{\mathrm{df}}_k$ for the diabetes data LARS estimates $\hat\mu_k$,
$k = 1, 2, \dots, m = 10$. It portrays a startlingly simple situation that we will call the
'simple approximation' … The right panel also applies to the diabetes data, but this time with
the quadratic model (3.15), having $m = 64$ predictors. We see that the simple approximation
(4.9) is again accurate within the limits of the bootstrap computation (4.8), where $B = 500$
replications were divided into 10 groups of 50 each in order to calculate Student-$t$
confidence intervals."*

*Dịch: Bảng trái của Hình 6 cho thấy $\widehat{\mathrm{df}}_k$ cho các ước lượng LARS trên dữ
liệu tiểu đường, $k = 1,\dots,m = 10$. Nó phác ra một tình huống đơn giản đến mức gây sững
sờ, mà các tác giả gọi là "xấp xỉ đơn giản" … Bảng phải cũng áp cho dữ liệu tiểu đường,
nhưng lần này với mô hình bậc hai (3.15) có $m = 64$ biến dự báo. Ta thấy xấp xỉ đơn giản
(4.9) lại chính xác trong giới hạn của phép tính bootstrap (4.8), nơi $B = 500$ lần lặp được
chia thành 10 nhóm 50 để tính khoảng tin cậy Student-$t$.*

![Hình 7 của Efron 2004, trang 20](<paper_lars_efron2004_tr20_hinh7_cp.png>)

**Hình 29 — Hình 7 của Efron 2004, trang 20.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"$C_p$ estimates of risk (4.10) for the two situations of
Figure 6: (left) $m = 10$ model has smallest $C_p$ at $k = 7$; (right) $m = 64$ model has
smallest $C_p$ at $k = 16$."*

*Dịch: Ước lượng $C_p$ của rủi ro (4.10) cho hai tình huống của Hình 6: (trái) mô hình
$m = 10$ có $C_p$ nhỏ nhất tại $k = 7$; (phải) mô hình $m = 64$ có $C_p$ nhỏ nhất tại
$k = 16$.*

**Nhận xét của chính tác giả, nguyên văn, tr.19:** *"Figure 7 displays $C_p(\hat\mu_k)$ as a
function of $k$ for the two situations of Figure 6. Minimum $C_p$ was achieved at steps
$k = 7$ and $k = 16$, respectively. Both of the minimum $C_p$ models looked sensible, their
first several selections of 'important' covariates agreeing with an earlier model based on a
detailed inspection of the data assisted by medical expertise."*

*Dịch: Hình 7 vẽ $C_p(\hat\mu_k)$ theo $k$ cho hai tình huống của Hình 6. $C_p$ nhỏ nhất đạt
được lần lượt ở bước $k = 7$ và $k = 16$. Cả hai mô hình cực tiểu $C_p$ đều trông hợp lý, và
mấy lựa chọn đầu tiên về biến "quan trọng" của chúng trùng với một mô hình trước đó dựng bằng
cách xem xét kỹ dữ liệu có sự trợ giúp của chuyên môn y khoa.*

#### Hình 30 — dựng lại cả hai hình bằng đúng thủ tục bootstrap

![Dựng lại Hình 6 và Hình 7 — bậc tự do và Cp](<lars_df_cp.png>)

**Hình 30 — dựng lại Hình 6 và Hình 7 (hình tự vẽ).** Hai hình của bài báo nằm ở hai trang
khác nhau và không đánh dấu chỗ nào xấp xỉ (4.9) hụt. Hình này đặt bốn bảng vào một chỗ và
**khoanh đỏ mọi điểm mà $\operatorname{df} = k$ nằm ngoài khoảng tin cậy**.

**Cách đọc hình.** Hàng trên là bậc tự do: chấm xanh kèm thanh sai số là ước lượng bootstrap
với $B = 500$, chia 10 nhóm 50 và dùng $t$ với 9 bậc tự do; đường xanh lá là xấp xỉ
$\operatorname{df} = k$. Hàng dưới là $C_p$: vòng xanh là cực tiểu đo được, ô vuông đỏ là chỗ
bài báo nói.

**Bốn con số, đối chiếu:**

```
   Hình 6 bảng TRÁI — diabetes m = 10, B = 500 lần bootstrap, σ̄ = 54.1542:
      k        :      1      2      3      4      5      6      7      8      9     10
      df ước   :   0.96   2.08   2.90   3.97   4.84   5.76   6.61   7.59   8.70  10.08
      sai số   :   0.07   0.13   0.12   0.10   0.10   0.07   0.12   0.15   0.16   0.15
      |df − k|  lớn nhất = 0.4133 ; trung bình = 0.1846
      số k mà df = k nằm trong khoảng tin cậy 95% (t với 9 bậc tự do, ±2.262·se): 7/10
      các k mà df ĐO ĐƯỢC THẤP HƠN k một cách có ý nghĩa: [6, 7, 8]

   Hình 6 bảng PHẢI — mô hình bậc hai m = 64, B = 500, σ̄ = 53.2303:
      |df − k| lớn nhất = 2.8434 tại k = 52 ; trung bình = 1.2346

   Hình 7 — C_p(μ̂_k) = ‖y − μ̂_k‖²/σ̄² − n + 2k, công thức (4.10):
      m = 10: C_p nhỏ nhất tại k = 7 (bài báo: 7) · C_p = 6.8775
         dãy C_p: 451.72 416.03 141.80 84.74 31.69 19.51 16.33 6.88 7.13 8.84 9.00
         mô hình được chọn gồm các biến [2, 3, 4, 5, 7, 9, 10]
      m = 64: C_p nhỏ nhất tại k = 15 (bài báo: 16) · C_p = 16.2003
         C_p quanh cực tiểu: k=13:19.861 k=14:18.529 k=15:16.200 k=16:17.834 k=17:19.091
         C_p tại k = 16 là 17.8344, cao hơn cực tiểu 1.6341
```

Ba điều đáng nói:

1. **$C_p$ cực tiểu tại $k = 7$ với $m = 10$ — khớp tuyệt đối**, và mô hình được chọn gồm bảy
   biến `[2, 3, 4, 5, 7, 9, 10]`. Đây là một phép dựng lại sạch, và nó không cần bootstrap:
   (4.10) chỉ cần RSS của đường LARS cộng một ước lượng $\bar\sigma^{2}$.
2. **Với $m = 64$ thì cực tiểu ở $k = 15$, không phải 16.** Chênh một bước, và $C_p$ ở hai chỗ
   cách nhau 1,6341 — tức không phải chuyện làm tròn. Đây là chỗ nhạy với cách ước lượng
   $\bar\sigma^{2}$: bài này dùng RSS của mô hình đầy đủ chia $n - m - 1$, và bài báo không
   nói nó dùng mẫu số nào.
3. **Xấp xỉ (4.9) hụt một cách có hệ thống**, và đó là phần đáng đọc nhất. Không phải nó lệch
   ngẫu nhiên hai phía: ở $k = 6, 7, 8$ thì $\operatorname{df}$ đo được **thấp hơn $k$ một
   cách có ý nghĩa thống kê**. Với $m = 64$, lệch lớn nhất là **2,8434** tại $k = 52$ và lệch
   trung bình là 1,2346.

   Điều này **không** mâu thuẫn với Định lý 4, và lý do nằm ngay ở mục sau.

<details>
<summary><b>Mã sinh Hình 30</b></summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Mọi mảng dưới đây do `tools/v_lars.py` mục 23 in ra ĐỦ CHỮ SỐ — dán thẳng vào đây
# để hình dựng lại được trong vài giây thay vì chạy lại 500 lần bootstrap.
df10 = [0.9626394119323122, 2.0838681560558876, 2.8964655750846005, 3.9745861473434734,
        4.8370680925234915, 5.756363075530417, 6.609884404884125, 7.586668098713742,
        8.697189067358309, 10.08328952369671]
se10 = [0.07267620788047585, 0.1303442562425645, 0.11687749484606942, 0.09950925774330967,
        0.09948264527166381, 0.07455407498332303, 0.12344261727610725, 0.14544018423521277,
        0.16488380072704717, 0.14552895413275152]
K64 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64]
df64 = [0.9184276721576267, 4.123928132236446, 7.025050709352413, 10.130360982340857,
        12.94850254008435, 15.800020455549076, 18.709049540931012, 21.5429314261086,
        24.32115577239908, 26.964381637815688, 29.78909610102815, 32.57381709645876,
        35.27402767107544, 38.04927227526457, 40.63874798438391, 43.39121718170785,
        46.21065414381041, 49.15659331116208, 52.35200449334537, 55.73711384050036,
        59.20610967788954, 63.53479562130559]
se64 = [0.06241106851223841, 0.15529491040040283, 0.15441617200028973, 0.1761963272825003,
        0.1881440319456216, 0.154697142741991, 0.1936298925761718, 0.2399923571000519,
        0.2456561825993646, 0.2943351984918918, 0.29981191920806727, 0.28462659101162546,
        0.32112996526608145, 0.30160150385999035, 0.34027278726413, 0.39114219835225034,
        0.38359475694981965, 0.3868975161583985, 0.37631916388591385, 0.42079963147802435,
        0.42030423362016994, 0.3843495497036518]
Cp10 = [451.72439585243205, 416.0290990203355, 141.79784615368544, 84.74019607957268,
        31.69492969416592, 19.505599141908647, 16.326752944624047, 6.87745079283394,
        7.131134315067243, 8.842818517775129, 8.999999999999318]
Cp64 = [483.01779975642194, 446.00261711330563, 162.09923690295034, 102.97370672400143,
        62.25868206827761, 56.13264296495191, 55.89647101419581, 47.76315559871239,
        48.40136756892156, 46.83347339528689, 44.61053546415087, 24.83953936665324,
        23.063090355561314, 19.86085846009979, 18.52852042215892, 16.200264094219392,
        17.83438734594864, 19.090842719861712, 19.627720616513102, 21.134815060609924,
        23.081667027472406, 24.306476103023726, 26.083256325759976, 26.96446268605831,
        28.063302423480934, 29.820392723963266, 30.876810621534787, 31.15739715056145,
        32.75546874163234, 33.532437685059676, 35.519280866496956, 36.40322952040407,
        34.05995057686505, 33.15599172137149, 30.21386211337517, 27.310184697722605,
        29.24229864711873, 29.615836408942698, 29.359056995840604, 31.146134150827436,
        29.849755044675362, 31.708895805707584, 32.183116942471486, 34.04255693211883,
        35.984634279864906, 37.174051663381476, 38.21492279066115, 40.109036910672444,
        41.744035512056826, 42.18977320082627, 43.444470422921995, 45.29818487078222,
        46.346092826292136, 46.931674174243994, 48.529010395402224, 48.99854396574949,
        50.695394804752766, 52.48290882074565, 53.85712289809453, 55.730280801809386,
        57.06821731543931, 58.27349242405313, 60.16648726557992, 61.00352338952126,
        62.9999999999996]

T9 = 2.262                                       # t Student 9 bậc tự do, hai phía 95 %
fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.0))

for a, K, df, se, ten in ((ax[0, 0], list(range(1, 11)), df10, se10, "m = 10 — bộ diabetes"),
                          (ax[0, 1], K64, df64, se64, "m = 64 — mô hình bậc hai (3.15)")):
    K = np.array(K, float); df = np.array(df); se = np.array(se)
    a.plot([0, K.max()], [0, K.max()], "-", lw=1.6, color="#059669",
           label=r"xấp xỉ đơn giản (4.9): $\mathrm{df} = k$")
    a.errorbar(K, df, yerr=T9 * se, fmt="o", ms=5, lw=1.1, capsize=3,
               color="#1d4ed8", label="bootstrap (4.6)–(4.8), B = 500")
    ngoai = np.abs(df - K) > T9 * se
    if ngoai.any():
        a.plot(K[ngoai], df[ngoai], "o", ms=11, mfc="none", mec="#dc2626", mew=1.8,
               label="k NẰM NGOÀI khoảng tin cậy 95 %")
    a.set_xlabel("số bước k"); a.set_ylabel(r"$\mathrm{df}(\hat\mu_k)$")
    a.set_title("%s — dựng lại Hình 6 tr.19\nlệch lớn nhất %.4f"
                % (ten, np.abs(df - K).max()), fontsize=9.5)
    a.legend(fontsize=8, loc="upper left")

for a, Cp, doi, ten in ((ax[1, 0], Cp10, 7, "m = 10"), (ax[1, 1], Cp64, 16, "m = 64")):
    Cp = np.array(Cp)
    k = int(np.argmin(Cp))
    a.plot(range(len(Cp)), Cp, "-o", ms=4, lw=1.4, color="#7c3aed")
    a.plot([k], [Cp[k]], "o", ms=11, mfc="none", mec="#059669", mew=2.0,
           label="cực tiểu đo được: k = %d" % k)
    a.plot([doi], [Cp[doi]], "s", ms=9, mfc="none", mec="#dc2626", mew=1.8,
           label="bài báo nói: k = %d" % doi)
    a.set_xlabel("số bước k"); a.set_ylabel(r"$C_p(\hat\mu_k)$")
    a.set_ylim(Cp.min() - 6, min(Cp.max(), Cp[max(0, k - 6)] + 60))
    a.set_title("%s — dựng lại Hình 7 tr.20, công thức (4.10)" % ten, fontsize=9.5)
    a.legend(fontsize=8, loc="upper right")

fig.suptitle("Bậc tự do và $C_p$: xấp xỉ df ≈ k đúng tới đâu, và nó chọn ra mô hình nào",
             fontsize=11, y=0.995)
fig.tight_layout()
fig.savefig("lars_df_cp.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("m=10: C_p nhỏ nhất tại k = %d ; m=64: tại k = %d"
      % (int(np.argmin(Cp10)), int(np.argmin(Cp64))))
print("m=10: %d/10 điểm có df = k nằm NGOÀI khoảng tin cậy 95%%"
      % int((np.abs(np.array(df10) - np.arange(1, 11)) > T9 * np.array(se10)).sum()))
```

</details>

#### Định lý 3 — trường hợp trực giao

![Định lý 3 của Efron 2004, trang 19](<paper_lars_efron2004_tr19_dinh_ly3.png>)

**Hình 31 — Định lý 3, trang 19.** Nguồn: cùng bài báo.

**Phát biểu gốc, nguyên văn:** *"If the covariate vectors $x_1, x_2, \dots, x_m$ are mutually
orthogonal, then the $k$-step LARS estimate $\hat\mu_k$ has $\mathrm{df}(\hat\mu_k) = k$."*

*Dịch: Nếu các vectơ biến $x_1,\dots,x_m$ đôi một trực giao thì ước lượng LARS sau $k$ bước
$\hat\mu_k$ có $\mathrm{df}(\hat\mu_k) = k$.*

Chú ý chữ "=" chứ không phải "$\doteq$": đây là **đẳng thức**, không phải xấp xỉ. Kiểm bằng
bootstrap với $B = 2000$ trên một thiết kế trực giao $8\times8$:

```
      μ = [ 6. -4.  3.  0.  0.  0.  0.  0.], σ = 1
      k  :       1       2       3       4       5       6       7
      df :   1.029   2.066   3.124   4.080   5.087   6.081   7.111
      |df − k| lớn nhất = 0.1238 (B = 2000 lần bootstrap)
```

Lệch lớn nhất 0,1238 với 2000 lần lặp — cỡ đúng nhiễu Monte Carlo, không phải cỡ một chệch có
hệ thống. So với bộ diabetes, nơi lệch lên tới 0,4133 với $B = 500$ và **lệch cùng một
chiều** ở ba giá trị $k$ liên tiếp, khác biệt rất rõ.

#### Điều kiện nón dương (4.11), công thức Stein (4.12), và Định lý 4

![Điều kiện nón dương và công thức Stein, trang 20](<paper_lars_efron2004_tr20_ct4_11_non_duong_stein.png>)

**Hình 32 — điều kiện nón dương (4.11) và công thức Stein (4.12), trang 20.** Nguồn: cùng bài báo.

**Phát biểu gốc, nguyên văn:** *"**Positive cone condition.** For all possible subsets
$X_{\mathcal A}$ of the full design matrix $X$, $G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} > 0$,
where the inequality is taken element-wise."*

*Dịch: **Điều kiện nón dương.** Với mọi tập con $X_{\mathcal A}$ có thể có của ma trận thiết
kế đầy đủ $X$, ta có $G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} > 0$, bất đẳng thức hiểu theo
từng phần tử.*

$$G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} > 0 \tag{4.11}$$

$$\sum_{i=1}^{n}\operatorname{cov}(g_i, y_i)/\sigma^{2} = E[\nabla\cdot g(y)] \tag{4.12}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $G_{\mathcal A}^{-1}\mathbf 1$ | $\mathbb{R}^{k}$ | chính là $w_{\mathcal A}/A_{\mathcal A}$, tức **hướng đi trong không gian hệ số**, chưa chuẩn hoá |
| $\nabla\cdot g$ | vô hướng | divergence: $\sum_i\partial g_i/\partial y_i$ |

**Ý nghĩa trừu tượng của (4.11).** Nhìn qua thì nó là một điều kiện đại số khô khan về ma
trận Gram. Nhìn kỹ thì nó nói một điều rất cụ thể: **hướng cách đều góc luôn nằm trong nón
lồi**, tức $u_{\mathcal A}\in\mathcal C_{\mathcal A}$ — vì $w_{\mathcal A}$ tỉ lệ thuận với
$G_{\mathcal A}^{-1}\mathbf 1$ và $A_{\mathcal A}>0$. Nối lại với mục 2.4: nếu (4.11) đúng cho
mọi tập con thì **sửa đổi Stagewise không bao giờ phải làm gì**, nên LARS, lasso và Stagewise
**trùng nhau**. Chính tác giả nói thế:

> *"It is also easy to show that LARS, Lasso and Stagewise all coincide under the positive cone
> condition, so the degrees-of-freedom formula applies to them too in this case."* — tr.20
>
> *Dịch: Cũng dễ chứng minh rằng LARS, Lasso và Stagewise trùng nhau dưới điều kiện nón dương,
> nên công thức bậc tự do cũng áp được cho chúng trong trường hợp đó.*

**Ý nghĩa trừu tượng của (4.12) — công thức Stein.** Nó đổi một đại lượng **thống kê** (tổng
hiệp phương sai, phải lấy kỳ vọng trên mọi $y$ có thể) thành một đại lượng **giải tích** (tổng
đạo hàm riêng, tính được tại một $y$ cụ thể). Đó là chỗ toàn bộ chứng minh Định lý 4 đứng
trên. Cái giá: (4.12) đòi $g$ **khả vi hầu khắp** và đòi $y$ phân phối chuẩn — và đúng chỗ
giả thiết khả vi ấy là chỗ Phụ lục A phải làm việc.

![Định lý 4 của Efron 2004, trang 20](<paper_lars_efron2004_tr20_dinh_ly4.png>)

**Hình 33 — Định lý 4, trang 20.** Nguồn: cùng bài báo.

**Phát biểu gốc, nguyên văn:** *"Under the positive cone condition,
$\mathrm{df}(\hat\mu_k) = k$."*

*Dịch: Dưới điều kiện nón dương, $\mathrm{df}(\hat\mu_k) = k$.*

**Nhận xét của chính tác giả, nguyên văn, tr.20:** *"The positive cone condition holds if $X$
is orthogonal. It is strictly more general than orthogonality, but counterexamples (such as
the diabetes data) show that not all design matrices $X$ satisfy it."*

*Dịch: Điều kiện nón dương đúng khi $X$ trực giao. Nó thực sự tổng quát hơn tính trực giao,
nhưng các phản ví dụ (chẳng hạn chính dữ liệu tiểu đường) cho thấy không phải mọi ma trận
thiết kế $X$ đều thoả nó.*

Câu *"such as the diabetes data"* là một khẳng định **đếm được**, và bài báo không đếm. Đếm:

```
   (4.11): với MỌI tập con A, G_A⁻¹·1_A > 0 theo từng phần tử.
      đếm được: 418 / 1023 tập con VI PHẠM, tức 40.86%
      tập con nhỏ nhất vi phạm có 3 phần tử; vài ví dụ:
         A = (2, 5, 6)      -> phần tử nhỏ nhất của G_A⁻¹·1 = -0.021161
         A = (5, 6, 7)      -> phần tử nhỏ nhất của G_A⁻¹·1 = -1.266891
         A = (5, 6, 8)      -> phần tử nhỏ nhất của G_A⁻¹·1 = -0.060373
         A = (5, 6, 9)      -> phần tử nhỏ nhất của G_A⁻¹·1 = -0.518984

   Thiết kế TRỰC GIAO có thoả không? (bài báo nói có)
      0/255 tập con vi phạm trên ma trận đơn vị 8×8.

   Dưới điều kiện nón dương, ba thuật toán có TRÙNG nhau không? (bài báo tr.20 nói có)
      6/6 thiết kế ngẫu nhiên THOẢ (4.11): ba đường trùng nhau tới 1e-8.
      Còn trên diabetes — thiết kế KHÔNG thoả — ba đường có 10, 12 và 13 bước.
```

Ba khẳng định của đoạn đó, ba lần kiểm, ba lần khớp — và **một con số mới**: 40,86 % tập con
vi phạm. Đó không phải "vi phạm ở một vài chỗ hiếm"; đó là gần một nửa.

#### Hình 34 — điều kiện nón dương, đếm ra

![Điều kiện nón dương trên bộ diabetes, đếm theo cỡ tập con](<lars_non_duong.png>)

**Hình 34 — (4.11) vi phạm bao nhiêu, và vi phạm mạnh cỡ nào (hình tự vẽ).** Bài báo nêu điều
kiện nón dương bằng một dòng đại số rồi nói dữ liệu tiểu đường là phản ví dụ, không kèm con
số nào. Hình này bù đúng chỗ đó.

**Cách đọc hình.** Bảng trái: với từng cỡ tập con $k = 1..10$, cột xám là tổng số tập con cỡ
đó, cột đỏ là số tập vi phạm (4.11). Bảng phải: **phần tử nhỏ nhất** của
$G_{\mathcal A}^{-1}\mathbf 1$ trong mọi tập cỡ $k$, thang log đối xứng, vùng hồng là vùng âm.

Ba điều đọc ra:

* **Vi phạm bắt đầu từ cỡ 3.** Mọi tập một hoặc hai biến đều thoả — với $k\le2$ thì
  $G_{\mathcal A}^{-1}\mathbf 1 > 0$ tự động khi tương quan nhỏ hơn 1.
* **Tỉ lệ vi phạm đạt đỉnh ở giữa** rồi lên tới 100 % ở các cỡ lớn: cỡ 9 vi phạm 10/10, cỡ 10
  vi phạm 1/1. Tức **chính tập đầy đủ 10 biến cũng vi phạm**.
* **Vi phạm không sát ngưỡng.** Giá trị âm nhất là **−42,70**, không phải −0,001. Đây không
  phải chuyện sai số số học đẩy một giá trị qua 0.

**Nối lại với Hình 30, và đây là kết luận của cả mục 2.7.** Định lý 4 nói
$\operatorname{df} = k$ **chính xác** dưới điều kiện (4.11). Bộ diabetes vi phạm (4.11) ở 418
trên 1023 tập con. Nên việc $\operatorname{df}$ đo được **thấp hơn $k$ có ý nghĩa ở $k = 6,7,8$**
không phải một phản bác Định lý 4 — nó là đúng thứ lý thuyết cho phép xảy ra khi giả thiết
không thoả. Hai phép đo ở hai mục khác nhau, ghép lại thành một câu chuyện nhất quán.

<details>
<summary><b>Mã sinh Hình 34</b></summary>

```python
import itertools
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
G = X.T @ X

vp, tong, nho_nhat = [], [], []
for k in range(1, 11):
    v_, t_, m_ = 0, 0, []
    for A in itertools.combinations(range(10), k):
        t_ += 1
        z = np.linalg.solve(G[np.ix_(A, A)], np.ones(k))          # (4.11)
        m_.append(z.min())
        if (z <= 0).any():
            v_ += 1
    vp.append(v_); tong.append(t_); nho_nhat.append(min(m_))

k = np.arange(1, 11)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax.bar(k, tong, color="#e5e7eb", label="tổng số tập con cỡ k")
ax.bar(k, vp, color="#dc2626", label=r"số tập con VI PHẠM $G_{\mathcal{A}}^{-1}\mathbf{1} > 0$")
for i in range(10):
    if vp[i]:
        ax.annotate("%d/%d" % (vp[i], tong[i]), (k[i], max(vp[i], 6)),
                    xytext=(0, 5), textcoords="offset points", fontsize=7.5,
                    ha="center", color="#7f1d1d")
ax.set_xticks(k)
ax.set_xlabel("cỡ của tập con A"); ax.set_ylabel("số tập con")
ax.set_title("Điều kiện nón dương (4.11) trên CHÍNH bộ diabetes\n"
             "%d trên %d tập con vi phạm — %.2f %%"
             % (sum(vp), sum(tong), 100 * sum(vp) / sum(tong)), fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

ax2.plot(k, nho_nhat, "o-", ms=6, lw=1.6, color="#1d4ed8")
ax2.axhline(0, color="#dc2626", lw=1.4, ls="--",
            label="ngưỡng của (4.11): mọi phần tử phải DƯƠNG")
ax2.fill_between(k, nho_nhat, 0, where=np.array(nho_nhat) < 0, color="#fecaca", alpha=.7)
ax2.set_yscale("symlog", linthresh=1e-2)
ax2.set_xticks(k)
ax2.set_xlabel("cỡ của tập con A")
ax2.set_ylabel(r"phần tử NHỎ NHẤT của $G_{\mathcal{A}}^{-1}\mathbf{1}$ trong mọi tập cỡ k")
ax2.set_title("Vi phạm không phải sát ngưỡng: giá trị âm nhất là %.2f\n"
              "Định lý 4 tr.20 đòi (4.11), nên nó KHÔNG áp được cho bộ này"
              % min(nho_nhat), fontsize=9.5)
ax2.set_ylim(min(nho_nhat) * 9.0, 4.0)
ax2.legend(fontsize=8, loc="lower center")
fig.savefig("lars_non_duong.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("vi phạm theo cỡ 1..10: %s" % vp)
print("tổng: %d/%d tập con vi phạm (%.2f%%); phần tử âm nhất %.6f"
      % (sum(vp), sum(tong), 100 * sum(vp) / sum(tong), min(nho_nhat)))
```

</details>

#### Công thức (4.13)–(4.15) — CHỖ IN SAI THỨ HAI

![Công thức (4.13)–(4.15) của Efron 2004, trang 21](<paper_lars_efron2004_tr21_ct4_13_4_15_delta.png>)

**Hình 35 — công thức (4.13)–(4.15), trang 21, đúng như bài báo in.** Nguồn: cùng bài báo.

Bài báo in:

$$\hat\mu_k = \bar y_k - \cot_k\cdot\lVert\bar y_{k+1}-\bar y_k\rVert \tag{4.13 — bản in}$$

$$\cot_k = \frac{u_k^{\top}u_{k+1}}{\left[1-(u_k^{\top}u_{k+1})^{2}\right]^{1/2}} \tag{4.14}$$

$$\hat\mu_k^{*} = \hat\mu_k + M_k(y^{*}-y),
\qquad M_k = P_k - \cot_k\cdot u_kv_k^{\top} \tag{4.15}$$

**Vế trái của (4.13) là một vectơ trong $\mathbb{R}^{442}$; vế phải là một vectơ trừ đi một
SỐ.** Thiếu thừa số $u_k$. Công thức đúng:

$$\hat\mu_k = \bar y_k - \cot_k\cdot\lVert\bar y_{k+1}-\bar y_k\rVert\cdot u_k
\tag{4.13 — bản đúng}$$

**Suy lại.** Ba dữ kiện, tất cả lấy từ hình học của Hình 12 (Hình 4 của bài báo):

* $\bar y_{k+1} - \hat\mu_k = \bar\gamma_{k+1}u_{k+1}$ theo (2.19) áp cho bước $k+1$;
* $\bar y_k - \hat\mu_k = (\bar\gamma_k - \hat\gamma_k)u_k$, nằm trong $\mathcal L(X_k)$;
* $\bar y_{k+1} - \bar y_k \perp \mathcal L(X_k)$, vì $\bar y_k$ là hình chiếu.

Tách $\bar y_{k+1}-\hat\mu_k$ thành hai thành phần đó: một thành phần dọc $u_k$ dài
$\bar\gamma_k-\hat\gamma_k$, một thành phần vuông góc dài
$\lVert\bar y_{k+1}-\bar y_k\rVert$. Vậy $\tan$ của góc giữa $u_{k+1}$ và $u_k$ bằng tỉ số hai
độ dài đó, tức

$$\bar\gamma_k - \hat\gamma_k = \cot_k\cdot\lVert\bar y_{k+1}-\bar y_k\rVert,$$

và nhân hai vế với vectơ đơn vị $u_k$ thì ra (4.13) bản đúng. Đo cả chín bước:

```
    k | trace(M_k) | (4.13) có u_k: lệch | (4.13) như in: lệch | M_k có là Jacobian?
    1 | 1.0000000000 |      2.49e-13       |       89.6640      |  lệch 8.24e-10
    2 | 2.0000000000 |      4.26e-13       |       53.3675      |  lệch 1.23e-07
    3 | 3.0000000000 |      9.95e-14       |       29.2149      |  lệch 1.44e-07
    4 | 4.0000000000 |      1.42e-13       |       18.1660      |  lệch 5.27e-08
    5 | 5.0000000000 |      1.54e-12       |        6.2019      |  lệch 5.06e-08
    6 | 6.0000000000 |      2.84e-13       |        4.9743      |  lệch 8.71e-08
    7 | 7.0000000000 |      4.01e-13       |        6.7962      |  lệch 1.03e-07
    8 | 8.0000000000 |      3.55e-13       |        2.5637      |  lệch 3.87e-08
    9 | 9.0000000000 |      1.94e-12       |        0.2989      |  lệch 1.20e-07
```

Ba cột, ba kết luận:

1. **$\operatorname{trace}(M_k) = k$ tới 10 chữ số.** Lý do gọn: $u_k\in\mathcal L(X_k)$ còn
   $v_k\perp\mathcal L(X_k)$, nên $u_k^{\top}v_k = 0$ và số hạng thứ hai của (4.15) **không
   đóng góp gì** vào vết. Vết còn lại là $\operatorname{trace}(P_k) = k$. Đó là toàn bộ lập
   luận "delta method" mà tr.21 nêu ra như một cách hiểu thứ hai cho (4.9), bên cạnh chứng
   minh bằng công thức Stein.
2. **Bản đúng khớp tới $10^{-12}$; bản in lệch tới 89,66.** Con số lệch giảm dần theo $k$ chỉ
   vì $\lVert\bar y_{k+1}-\bar y_k\rVert$ giảm dần — bản in không "gần đúng ở cuối", nó chỉ
   nhỏ dần cùng với đại lượng nó viết sai.
3. **$M_k$ đúng là đạo hàm cục bộ.** Nhân $M_k$ với một hướng $\Delta y$ bất kỳ cho cùng kết
   quả với sai phân hữu hạn của **chính thuật toán LARS**, sai số cỡ $10^{-7}$ — đúng mức
   nhiễu của sai phân hai phía với bước $10^{-6}$ trong `float64`.

Phép kiểm thứ ba là phép kiểm đáng giá nhất, vì nó kiểm một tuyên bố mà bài báo chỉ nói bằng
lời: *"(4.15) holds within a neighborhood of $y$ such that the LARS choices
$\mathcal L(X_k)$ and $v_k$ remain the same"*. Không ai buộc phải tin câu đó — chạy lại thuật
toán ở $y\pm h\Delta y$ rồi chia là kiểm được.

### 2.8 Ba mục con của §4 — tr.21–25

#### §4.1 Thiết kế trực giao: (4.16)–(4.18) và Bổ đề 1

![Bổ đề 1 và công thức (4.17)–(4.18), trang 22](<paper_lars_efron2004_tr22_ct4_17_bo_de1.png>)

**Hình 36 — Bổ đề 1 và công thức (4.17)–(4.18), trang 22.** Nguồn: cùng bài báo.

$$\lvert y\rvert_{(1)}\ge\lvert y\rvert_{(2)}\ge\cdots\ge\lvert y\rvert_{(n)}
\ge\lvert y\rvert_{(n+1)} := 0 \tag{4.16}$$

$$\hat\mu_{k,i}(y) = \begin{cases}
y_i - \lvert y\rvert_{(k+1)}, & y_i > \lvert y\rvert_{(k+1)},\\
0, & \lvert y_i\rvert \le \lvert y\rvert_{(k+1)},\\
y_i + \lvert y\rvert_{(k+1)}, & y_i < -\lvert y\rvert_{(k+1)},
\end{cases} \tag{4.17}$$

$$= \eta\left(y_i;\ \lvert y\rvert_{(k+1)}\right) \tag{4.18}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $\lvert y\rvert_{(k)}$ | vô hướng | giá trị tuyệt đối **lớn thứ $k$** trong $y$ |
| $\eta(y_1;t)$ | $\mathbb{R}$ | **ngưỡng mềm**: $y_1-t$, $0$, hay $y_1+t$ tuỳ dấu và độ lớn |

**Ý nghĩa trừu tượng — và đây là kết quả đẹp nhất của §4.** Với thiết kế trực giao,
$x_j = e_j$, toàn bộ LARS **rút về một công thức đóng không cần vòng lặp nào**: bước thứ $k$
chính là **ngưỡng mềm tại thống kê thứ tự thứ $k+1$**. Không có ma trận Gram, không có hệ
tuyến tính, không có $\hat\gamma$ phải giải.

Nói cách khác: **ở trường hợp trực giao, LARS chính là lasso và chính là phép ngưỡng mềm** —
ba thứ trùng nhau hoàn toàn. Mọi sự phức tạp của (2.4)–(2.13) sinh ra từ **tương quan giữa
các cột**, không phải từ bản chất của bài toán.

Kiểm bằng số:

```
   y = +4.0818 -5.1113 +0.8362 -1.1355 -0.9053 -0.4312 -4.0400 -0.4639
   |y|_(k) xếp giảm dần: 5.1113 4.0818 4.0400 1.1355 0.9053 0.8362 0.4639 0.4312

   Bổ đề 1 tr.22: μ̂_k,i = η(y_i ; |y|_(k+1)) — ngưỡng mềm tại thống kê thứ tự thứ k+1
      lệch lớn nhất giữa đường LARS và công thức ngưỡng mềm: 1.776e-15
      ngưỡng tại từng bước: k=0:5.1113 k=1:4.0818 k=2:4.0400 k=3:1.1355 ...

   Trong chứng minh: G_A = I, A_A = |A|^(−1/2), u_A = |A|^(−1/2)·1, a_{k,j} = 0
      |A| = 1: A_A đo 1.00000000 · công thức 1.00000000 | max|a_j| ngoài A = 0.00e+00
      |A| = 3: A_A đo 0.57735027 · công thức 0.57735027 | max|a_j| ngoài A = 0.00e+00
      |A| = 5: A_A đo 0.44721360 · công thức 0.44721360 | max|a_j| ngoài A = 0.00e+00
```

Bốn đại lượng mà chứng minh dùng — $G_{\mathcal A} = I$, $A_{\mathcal A} = \lvert\mathcal A\rvert^{-1/2}$,
$u_{\mathcal A} = \lvert\mathcal A\rvert^{-1/2}\mathbf 1$, $a_{k,j} = 0$ với $j\notin\mathcal A_k$ —
đều kiểm được, và cả bốn khớp.

Bài báo cũng xử lý trường hợp có **hoà** (nhiều giá trị tuyệt đối bằng nhau):

> *"The argument clearly extends to the case in which there are ties in the order statistics
> (4.16): if $\lvert y\rvert_{(k+1)} = \cdots = \lvert y\rvert_{(k+r)}$, then
> $\mathcal A_k(y)$ expands by $r$ variables at step $k+1$ and $\hat\mu_{k+\nu}(y)$,
> $\nu = 1, \dots, r$, are all determined at the same time and are equal to
> $\hat\mu_{k+1}(y)$."* — tr.23
>
> *Dịch: Lập luận rõ ràng mở rộng được cho trường hợp có hoà trong các thống kê thứ tự (4.16):
> nếu $\lvert y\rvert_{(k+1)} = \cdots = \lvert y\rvert_{(k+r)}$ thì $\mathcal A_k(y)$ mở rộng
> thêm $r$ biến ở bước $k+1$, và các $\hat\mu_{k+\nu}(y)$ với $\nu = 1,\dots,r$ được xác định
> cùng lúc và bằng nhau, bằng $\hat\mu_{k+1}(y)$.*

Và chứng minh Định lý 4 trong trường hợp trực giao chỉ gọn một dòng, vì (4.17) cho ngay
divergence:

$$\nabla\cdot\hat\mu_k = \sum_i\frac{\partial\hat\mu_{k,i}}{\partial y_i}(y)
= \sum_i I\{\lvert y_i\rvert > \lvert y\rvert_{(k+1)}\} = k \quad\text{hầu chắc chắn}$$

Đọc theo lời: đạo hàm của ngưỡng mềm bằng **1 ở những toạ độ vượt ngưỡng và 0 ở những toạ độ
bị cắt**, nên tổng của chúng đúng bằng **số toạ độ sống sót** — tức $k$.

```
   Chứng minh của bài báo: ∇·μ̂_k = Σᵢ 1{|y_i| > |y|_(k+1)} = k hầu chắc chắn.
      đếm trực tiếp trên y ở trên: k=0:0 k=1:1 k=2:2 k=3:3 k=4:4 k=5:5 k=6:6 k=7:7
```

#### §4.2 Công thức divergence: (4.19)–(4.24) và Bổ đề 2

![Công thức (4.19) của Efron 2004, trang 23](<paper_lars_efron2004_tr23_ct4_19_divergence.png>)

**Hình 37 — công thức (4.19), trang 23.** Nguồn: cùng bài báo.

$$\nabla\cdot\hat\mu_k(y) = k \tag{4.19}$$

**Bổ đề 2, nguyên văn:** *"There is an open set $\mathcal G_k$ of full measure such that, at
all $y\in\mathcal G_k$, $\hat\mu_k(y)$ is locally linear and $\nabla\cdot\hat\mu_k(y) = k$."*

*Dịch: Tồn tại một tập mở $\mathcal G_k$ có độ đo đầy đủ sao cho tại mọi $y\in\mathcal G_k$,
$\hat\mu_k(y)$ là tuyến tính cục bộ và $\nabla\cdot\hat\mu_k(y) = k$.*

**Ý nghĩa trừu tượng.** (4.19) là **phần dễ** của Định lý 4, và bài báo nói thẳng điều đó:
divergence bằng $k$ **hầu khắp nơi**, không cần điều kiện nón dương. Điều kiện nón dương chỉ
cần cho **phần khó**: bảo đảm $\hat\mu_k$ đủ trơn để công thức Stein (4.12) dùng được. Nói
cách khác, Định lý 4 gồm hai nửa, và nửa hình học thì luôn đúng còn nửa giải tích thì không.

Năm công thức của chứng minh, gom lại:

| Số hiệu | Công thức | Nó nói gì | Dùng ở đâu |
|---|---|---|---|
| (4.20) | $\hat\mu_k(y) = \sum_{l=1}^{k}\gamma_l(y)u_l$ | ước lượng LARS là **tổng các bước**, mỗi bước một hướng | cơ sở để lấy đạo hàm từng số hạng |
| (4.21) | $\nabla\cdot\hat\mu_k(y) = \sum_{l=1}^{k}\langle\nabla\gamma_l, u_l\rangle$ | divergence tách thành $k$ số hạng độc lập | đổi bài toán thành tính $k$ tích vô hướng |
| (4.22) | $\gamma_k(y) = b_{k+1}^{-1}\langle\delta_{k+1}, y-\hat\mu_{k-1}\rangle$ | (2.13) viết lại thành một **hàm tuyến tính của $y$** | cho phép lấy $\nabla\gamma_k$ |
| (4.23) | $u_k\in\mathcal L_k$, $\mathcal L_{k+1}\subset\mathcal L_k$ | các không gian "cách đều góc" **lồng nhau** và co dần | điều kiện để phép truy hồi ở (4.24) đóng lại |
| (4.24) | $\langle\nabla\gamma_k, u_k\rangle = 1$ và $\langle\nabla\gamma_k, u\rangle = 0$ với $u\in\mathcal L_{k+1}$ | mỗi bước đóng góp **đúng 1** vào divergence, và không đóng góp gì cho các bước sau | cộng $k$ số 1 lại, ra (4.19) |

**Ý nghĩa trừu tượng của cả khối.** Mỗi bước LARS "tiêu" **đúng một** bậc tự do, và các bước
**không chồng chéo nhau** — đó là nội dung của (4.24), và nó là lý do sâu xa của (4.9). Cái
làm chuyện này chạy được là tính lồng nhau (4.23): không gian các hướng cách đều góc co lại
một chiều mỗi khi thêm một biến.

> **Một chỗ in thiếu trong chứng minh.** Ở tr.25, dòng dẫn tới (4.24) viết
> $\langle\delta_{k+1},u\rangle = \langle x_l - x_{k+1}\rangle$ — dấu ngoặc trong chỉ có **một**
> đối số. Theo mạch, nó phải là $\langle x_l - x_{k+1},\,u\rangle$. Đây là lỗi in, không ảnh
> hưởng lập luận.

#### §4.3 Bổ đề 3 và chỗ Định lý 4 đóng lại

**Bổ đề 3, nguyên văn, tr.25:** *"Under the positive cone condition, $\hat\mu_k(y)$ is
continuous and almost differentiable."*

*Dịch: Dưới điều kiện nón dương, $\hat\mu_k(y)$ liên tục và khả vi hầu khắp.*

**Ý nghĩa trừu tượng.** Đây là mảnh ghép cuối: Bổ đề 2 cho divergence bằng $k$ hầu khắp, Bổ
đề 3 cho phép dùng (4.12), và ghép hai thứ lại ra Định lý 4. Chứng minh Bổ đề 3 nằm ở Phụ lục,
và nó là phần dài nhất của cả bài báo — mục 2.13 của bài này nói kỹ.

Điều đáng ghi: bài báo **không** chứng minh Định lý 4 khi (4.11) không thoả, và nói rõ vì sao:

> *"Focusing specifically on LARS, it will turn out that $\nabla\cdot\hat\mu_k(y) = k$ in all
> situations with probability 1, but that the continuity assumptions underlying (4.12) and
> SURE can fail in certain nonorthogonal cases where the positive cone condition does not
> hold."* — tr.20
>
> *Dịch: Tập trung riêng vào LARS, sẽ hoá ra rằng $\nabla\cdot\hat\mu_k(y) = k$ trong mọi tình
> huống với xác suất 1, nhưng các giả thiết liên tục làm nền cho (4.12) và SURE có thể không
> đúng trong một số trường hợp không trực giao khi điều kiện nón dương không thoả.*

Câu đó **dự đoán đúng** điều mục 2.7 đo được: trên bộ diabetes — không thoả (4.11) —
$\operatorname{df}$ lệch khỏi $k$ một cách có hệ thống. Và bài báo cũng nêu mức tin của mình
vào xấp xỉ:

> *"A range of simulations suggested that the simple approximation is quite accurate even when
> the $x_j$'s are highly correlated and that it requires concerted effort at pathology to make
> $\mathrm{df}(\hat\mu_k)$ much different than $k$."* — tr.21
>
> *Dịch: Một loạt mô phỏng cho thấy xấp xỉ đơn giản khá chính xác ngay cả khi các $x_j$ tương
> quan cao, và phải cố ý dựng ca bệnh lý mới làm $\mathrm{df}(\hat\mu_k)$ khác $k$ nhiều.*

Phép đo ở mục 2.7 đồng ý với câu này về **chất** — lệch lớn nhất 0,4133 trên thang 1..10 là
nhỏ — nhưng thêm một chi tiết câu này không nói: **lệch nhỏ mà CÓ HƯỚNG**. Ba giá trị $k$
liên tiếp cùng lệch xuống, và cả ba vượt khoảng tin cậy. Một lệch có hướng khác hẳn một lệch
ngẫu nhiên cùng độ lớn, vì nó không co lại khi tăng $B$.

Cuối §4 còn một khẳng định về lasso, và bài báo tự nói nó chưa có chứng minh:

> *"However, we have found empirically that an intuitively plausible result holds: the degrees
> of freedom is well approximated by the number of nonzero predictors in the model.
> Specifically, starting at step 0, let $\ell(k)$ be the index of the last model in the Lasso
> sequence containing $k$ predictors. Then $\mathrm{df}(\hat\mu_{\ell(k)}) \doteq k$. We do not
> yet have any mathematical support for this claim."* — tr.21
>
> *Dịch: Tuy nhiên, các tác giả thấy bằng thực nghiệm rằng một kết quả hợp trực giác vẫn đúng:
> bậc tự do được xấp xỉ tốt bởi số biến khác 0 trong mô hình. Cụ thể, bắt đầu từ bước 0, gọi
> $\ell(k)$ là chỉ số của mô hình cuối cùng trong dãy Lasso chứa $k$ biến. Khi đó
> $\mathrm{df}(\hat\mu_{\ell(k)}) \doteq k$. Các tác giả chưa có bất kỳ chỗ dựa toán học nào
> cho khẳng định này.*

Câu cuối đáng ghi vì nó là một lời tự khai về giới hạn, và vì nó **đã được trả lời sau đó**
bởi đúng ba trong bốn tác giả cùng một người nữa, trong một bài báo riêng về bậc tự do của
lasso. Bài này không có PDF đó trong repo nên không trích số; chỉ ghi rằng câu *"we do not yet
have any mathematical support"* là một món nợ mà chính nhóm tác giả về sau trả.

### 2.9 Vì sao sửa đổi lasso ĐÚNG — §5 của bài báo, tr.25–34

Mục này chứng minh Định lý 1. Nó dài mười trang và gồm bảy bổ đề, nhưng cấu trúc thì đơn
giản: **liệt kê mọi ràng buộc mà một nghiệm lasso bắt buộc phải thoả, rồi chỉ ra chỉ có đúng
một tập hoạt động thoả hết.** Bổ đề 4–6 chuẩn bị công cụ; Bổ đề 7–10 dựng bốn ràng buộc; đoạn
cuối ghép chúng lại.

#### Bổ đề 4 — biến mới vào theo chiều "đúng": (5.1)–(5.6)

**Phát biểu gốc, nguyên văn, tr.25:** *"If $x_k$ is the only addition to the active set at the
end of step $k-1$, then the coefficient vector $w_k = A_kG_k^{-1}\mathbf 1_k$ for the
equiangular vector $u_k = X_kw_k$, (2.6), has its $k$th component $w_{kk}$ agreeing in sign
with the current correlation $c_{kk} = x_k^{\top}(y-\hat\mu_{k-1})$. Moreover, the regression
vector $\hat\beta_k$ for $\hat\mu_k = X\hat\beta_k$ has its $k$th component $\hat\beta_{kk}$
agreeing in sign with $c_{kk}$."*

*Dịch: Nếu $x_k$ là bổ sung duy nhất vào tập hoạt động ở cuối bước $k-1$ thì vectơ hệ số
$w_k = A_kG_k^{-1}\mathbf 1_k$ của vectơ cách đều góc $u_k = X_kw_k$, (2.6), có thành phần thứ
$k$ là $w_{kk}$ cùng dấu với tương quan hiện tại
$c_{kk} = x_k^{\top}(y-\hat\mu_{k-1})$. Hơn nữa, vectơ hồi quy $\hat\beta_k$ của
$\hat\mu_k = X\hat\beta_k$ có thành phần thứ $k$ là $\hat\beta_{kk}$ cùng dấu với $c_{kk}$.*

**Nhận xét của chính tác giả, nguyên văn, tr.26:** *"Lemma 4 says that new variables enter the
LARS active set in the 'correct' direction, a weakened version of the Lasso requirement (3.1).
This will turn out to be a crucial connection for the LARS–Lasso relationship."*

*Dịch: Bổ đề 4 nói rằng các biến mới đi vào tập hoạt động LARS theo chiều "đúng", một phiên
bản yếu hơn của yêu cầu (3.1) của Lasso. Điều này hoá ra là một mối nối then chốt cho quan hệ
LARS–Lasso.*

| Số hiệu | Công thức | Nó nói gì | Dùng ở đâu |
|---|---|---|---|
| (5.1) | $w_k = A_k\hat C_k^{-1}\left[(X_k^{\top}X_k)^{-1}X_k^{\top}(y-\hat\mu_{k-1})\right] := A_k\hat C_k^{-1}w_k^{*}$ | $w_k$ **tỉ lệ THUẬN, hệ số dương**, với hệ số hồi quy của phần dư lên $X_k$ | đổi câu hỏi về dấu của $w_k$ thành câu hỏi về dấu của một hệ số OLS |
| (5.2) | $X_k^{\top}(y-\bar y_{k-1}) = (0,\delta)^{\top}$ với $\delta > 0$ | phần dư sau bước $k-1$ **trực giao với $k-1$ cột cũ**, và tương quan với cột MỚI thì dương | đây là chỗ $\delta > 0$ ra đời |
| (5.3) | $c_k(\gamma) < / = / > c_j(\gamma)$ tuỳ $\gamma$ so với $\hat\gamma_{k-1}$ | tương quan của biến mới **giảm chậm hơn** của các biến cũ | giải thích vì sao nó bắt kịp rồi vượt |
| (5.4)–(5.5) | $w_k^{*} = (X_k^{\top}X_k)^{-1}(0,\delta)^{\top} + (X_k^{\top}X_k)^{-1}X_k^{\top}\left[(\bar\gamma_{k-1}-\hat\gamma_{k-1})u_{k-1}\right]$ | tách $w_k^{*}$ làm hai phần | phần đầu dương vì $X_k^{\top}X_k$ xác định dương; phần sau **bằng 0** vì $u_{k-1}\in\mathcal L(X_{k-1})$ |
| (5.6) | $\hat\beta_{kk} = \hat\beta_{k-1,k} + \hat\gamma_kw_{kk}$ | hệ số của biến mới xuất phát từ **0** | nên dấu của nó bằng dấu của $w_{kk}$ |

**Ý nghĩa trừu tượng của cả bổ đề.** Một biến vừa vào tập hoạt động thì hệ số của nó **bắt
đầu đi đúng chiều mà tương quan chỉ ra**. Nghe hiển nhiên, nhưng nó không hiển nhiên: hướng
cách đều góc là một tổ hợp của **mọi** cột hoạt động, và không có gì bảo đảm sẵn rằng thành
phần ứng với biến mới lại cùng dấu với tương quan của nó. Bổ đề 4 đúng là chỗ bảo đảm ấy, và
nó là mảnh ghép làm Ràng buộc 3 ở cuối mục dùng được.

Kiểm cả mười bước:

```
    k | biến mới | dấu ĉ_kk | dấu w_kk | dấu β̂_kk | khớp?
    1 |     3    |   +1     |   +1     |    +1     | có
    2 |     9    |   +1     |   +1     |    +1     | có
    3 |     4    |   +1     |   +1     |    +1     | có
    4 |     7    |   -1     |   -1     |    -1     | có
    5 |     2    |   -1     |   -1     |    -1     | có
    6 |    10    |   +1     |   +1     |    +1     | có
    7 |     5    |   -1     |   -1     |    -1     | có
    8 |     8    |   +1     |   +1     |    +1     | có
    9 |     6    |   +1     |   +1     |    +1     | có
   10 |     1    |   -1     |   -1     |    -1     | có
   -> Bổ đề 4 đúng ở mọi bước? ĐÚNG
```

#### Bổ đề 5 — $A_{\mathcal A}$ là một khoảng cách: (5.7)–(5.11)

![Bổ đề 5 của Efron 2004, trang 27](<paper_lars_efron2004_tr27_bo_de5.png>)

**Hình 38 — Bổ đề 5, trang 27.** Nguồn: cùng bài báo.

$$\mathcal S_{\mathcal A} = \left\{v = \sum_{j\in\mathcal A}s_jx_jP_j : \sum_{j\in\mathcal A}P_j = 1\right\} \tag{5.7}$$

**Phát biểu gốc, nguyên văn:** *"The point in $\mathcal S_{\mathcal A}$ nearest the origin is
$v_{\mathcal A} = A_{\mathcal A}u_{\mathcal A} = A_{\mathcal A}X_{\mathcal A}w_{\mathcal A}$
where $w_{\mathcal A} = A_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A}$, with length
$\lVert v_{\mathcal A}\rVert = A_{\mathcal A}$. If $\mathcal A\subseteq\mathcal B$, then
$A_{\mathcal A}\ge A_{\mathcal B}$, the largest possible value being $A_{\mathcal A} = 1$ for
$\mathcal A$ a singleton."*

*Dịch: Điểm trong $\mathcal S_{\mathcal A}$ gần gốc nhất là
$v_{\mathcal A} = A_{\mathcal A}u_{\mathcal A}$, với độ dài
$\lVert v_{\mathcal A}\rVert = A_{\mathcal A}$. Nếu $\mathcal A\subseteq\mathcal B$ thì
$A_{\mathcal A}\ge A_{\mathcal B}$, giá trị lớn nhất có thể là $A_{\mathcal A} = 1$ khi
$\mathcal A$ chỉ có một phần tử.*

| Số hiệu | Công thức | Nó nói gì |
|---|---|---|
| (5.8) | $v_{\mathcal A} = A_{\mathcal A}u_{\mathcal A} = A_{\mathcal A}X_{\mathcal A}w_{\mathcal A}$ | điểm gần gốc nhất nằm **dọc đúng hướng cách đều góc** |
| (5.9) | $P^{\top}G_{\mathcal A}P - \lambda(\mathbf 1_{\mathcal A}^{\top}P - 1)$ | hàm Lagrange của bài toán cực tiểu có ràng buộc |
| (5.10) | $P_{\mathcal A} = A_{\mathcal A}^{2}G_{\mathcal A}^{-1}\mathbf 1_{\mathcal A} = A_{\mathcal A}w_{\mathcal A}$ | nghiệm của bài toán đó |
| (5.11) | $\lVert v_{\mathcal A}\rVert^{2} = A_{\mathcal A}^{4}\mathbf 1^{\top}G_{\mathcal A}^{-1}\mathbf 1 = A_{\mathcal A}^{2}$ | độ dài bình phương rút gọn về $A_{\mathcal A}^{2}$ |

**Ý nghĩa trừu tượng — và đây là chỗ $A_{\mathcal A}$ thôi là một hằng số kỹ thuật.** Bổ đề 5
cho $A_{\mathcal A}$ **ba nghĩa cùng lúc**, và ba nghĩa ấy sẽ được dùng ở ba chỗ khác nhau:

1. **Hình học:** khoảng cách từ gốc tới đơn hình mở rộng sinh bởi các cột hoạt động.
2. **Động học:** tốc độ giảm của $\hat C$ theo (2.16).
3. **Thứ tự:** thêm biến vào thì $A_{\mathcal A}$ **giảm**. Nên càng nhiều biến hoạt động,
   $\hat C$ giảm càng chậm — tức bước LARS càng dài ra.

Điểm thứ ba là chỗ Ràng buộc 4 ở cuối mục dựa vào: trong số các tập ứng viên, tập nào có
$A_{\mathcal A}$ **nhỏ nhất** thì làm $S$ giảm nhanh nhất, và lasso chọn đúng tập đó.

Kiểm cả ba phần:

```
   tập A                            | A_A theo (2.5) | khoảng cách tối tiểu đo trực tiếp | lệch
   [3]                              | 1.00000000     |     1.00000000           | 1.11e-16
   [3, 9]                           | 0.85034009     |     0.85034009           | 0.00e+00
   [3, 4, 9]                        | 0.77940354     |     0.77940354           | 1.11e-16
   [2, 3, 4, 7, 9]                  | 0.65912639     |     0.65912639           | 4.44e-16
   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  | 0.43474097     |     0.43474097           | 2.78e-16
      số cặp (A, A∪{j}) VI PHẠM tính đơn điệu: 0 / 5110
      A_A của 10 tập một phần tử: 1.000000 ... 1.000000
      A_A của tập đầy đủ 10 biến: 0.153899 (nhỏ nhất trong 1023 tập)
```

Ba khẳng định, ba lần khớp: khoảng cách đúng bằng $A_{\mathcal A}$ · tính đơn điệu đúng ở
**cả 5110 cặp** · và $A_{\mathcal A} = 1$ ở đúng mười tập một phần tử.

Một chi tiết đáng ghi: $A_{\mathcal A}$ của tập đầy đủ là **0,153899**, không phải
$0{,}43474097$ như dòng thứ năm của bảng trên. Hai con số ấy khác nhau vì **dấu**: bảng trên
dùng dấu $s_j$ lấy từ $\hat c$ tại $\hat\mu = 0$, còn dòng cuối quét mọi tập con với dấu
dương. Cùng một tập chỉ số, hai cách định hướng cột, hai giá trị $A_{\mathcal A}$.

#### Bổ đề 6 — $S$ là một tam thức bậc hai: (5.12)–(5.14)

$$\beta(\gamma) = \hat\beta + \gamma d,\qquad S(\gamma) = \lVert y - X\beta(\gamma)\rVert^{2} \tag{5.12}$$

$$S(\gamma) - S(0) = -2\hat c^{\top}d\,\gamma + d^{\top}X^{\top}Xd\,\gamma^{2} \tag{5.13}$$

$$\dot S(0) = -2\hat c^{\top}d,\qquad \ddot S(0) = 2d^{\top}X^{\top}Xd \tag{5.14}$$

**Ý nghĩa trừu tượng.** Đi theo bất kỳ hướng thẳng nào trong không gian hệ số thì $S$ là một
**parabol**, và hai hệ số của parabol ấy đọc thẳng ra được: hệ số bậc nhất là tương quan hiện
tại chiếu lên hướng đi, hệ số bậc hai là độ dài bình phương của bước trong không gian dự
đoán. Toàn bộ §5 sau đó chỉ là chọn $d$ sao cho parabol đi xuống nhanh nhất mà vẫn giữ ngân
sách $T$.

```
      γ = 0.5 : đo     233.513077 · công thức     233.513077 · lệch 2.63e-10
      γ = 2.0 : đo     948.227558 · công thức     948.227558 · lệch 1.78e-10
      γ = 0.5 : đo     -41.754813 · công thức     -41.754813 · lệch 1.40e-10
```

#### Bổ đề 7 — đường lasso là các đoạn thẳng: (5.15)–(5.27)

![Bổ đề 7 của Efron 2004, trang 28](<paper_lars_efron2004_tr28_bo_de7.png>)

**Hình 39 — Bổ đề 7, trang 28.** Nguồn: cùng bài báo.

$$\mathcal A = \{j : \hat\beta_j(t)\ne0\} \tag{5.15}$$

$$\hat\mu(t) = \hat\mu(t_0) + A_{\mathcal A}(t-t_0)u_{\mathcal A} \tag{5.16}$$

$$\hat\beta_{\mathcal A}(t) = \hat\beta_{\mathcal A}(t_0) + S_{\mathcal A}A_{\mathcal A}(t-t_0)w_{\mathcal A} \tag{5.17}$$

**Ý nghĩa trừu tượng — đây là bổ đề quan trọng nhất của §5.** Nó nói: **trên một khoảng mà
tập biến khác 0 không đổi, nghiệm lasso đi THẲNG, và đi đúng theo hướng cách đều góc.** Hai
vế của Định lý 1 gặp nhau ở đây — lasso được định nghĩa bằng tối ưu, LARS bằng thủ tục, và
bổ đề này nói chúng đi cùng một hướng.

Chín công thức của chứng minh:

| Số hiệu | Công thức | Nó nói gì | Dùng ở đâu |
|---|---|---|---|
| (5.18) | $S(\hat\beta_{\mathcal A}) = \lVert y - X_{\mathcal A}S_{\mathcal A}\hat\beta_{\mathcal A}\rVert^{2}$ | viết lại $S$ chỉ theo các toạ độ hoạt động | rút bài toán về $\lvert\mathcal A\rvert$ chiều |
| (5.19) | $\sum_{\mathcal A}s_j\hat\beta_j = t$ và $\operatorname{sign}(\hat\beta_j) = s_j$ | hai ràng buộc: **đúng ngân sách** và **đúng dấu** | dạng đầy đủ của bài toán con |
| (5.20) | cực tiểu $S$ với ràng buộc $\sum s_j\hat\beta_j = t$ | **bỏ được ràng buộc dấu** vì nghiệm nằm hẳn bên trong | biến bài toán bất đẳng thức thành đẳng thức |
| (5.21) | $\tfrac12\lVert y-X_{\mathcal A}S_{\mathcal A}\hat\beta_{\mathcal A}\rVert^{2} + \lambda\sum_{\mathcal A}s_j\hat\beta_j$ | dạng Lagrange | chuyển sang điều kiện dừng |
| (5.22) | $-S_{\mathcal A}X_{\mathcal A}^{\top}(y-X_{\mathcal A}S_{\mathcal A}\hat\beta_{\mathcal A}) + \lambda S_{\mathcal A}\mathbf 1 = 0$ | **điều kiện dừng** | chỗ Bổ đề 8 rút ra trực tiếp |
| (5.23)–(5.24) | $\hat\beta_{\mathcal A}(t_2)-\hat\beta_{\mathcal A}(t_1) = (\lambda_1-\lambda_2)S_{\mathcal A}G_{\mathcal A}^{-1}\mathbf 1$ | lấy hiệu hai điều kiện dừng ở hai giá trị $t$ | cho thấy hiệu **tỉ lệ với $G^{-1}\mathbf 1$** |
| (5.25) | $t_2-t_1 = (\lambda_1-\lambda_2)A_{\mathcal A}^{-2}$ | quy đổi giữa hai tham số hoá: **$t$ và $\lambda$** | đây là chỗ $A_{\mathcal A}^{2}$ thành tốc độ |
| (5.26) | $\hat\beta_{\mathcal A}(t_2)-\hat\beta_{\mathcal A}(t_1) = S_{\mathcal A}A_{\mathcal A}(t_2-t_1)w_{\mathcal A}$ | ghép (5.24) với (5.25) | ra đúng (5.17) |
| (5.27) | $D(t) = \{\beta : \sum\lvert\beta_j\rvert\le t\}$ | khối kim cương ràng buộc | nói rằng nghiệm chạy dọc **một cạnh** của khối |

**Ý nghĩa trừu tượng của (5.25), vì nó hay bị lướt qua.** Bài toán lasso có hai cách tham số
hoá: bằng ngân sách $t$ hoặc bằng nhân tử $\lambda$. (5.25) nói tỉ lệ quy đổi giữa hai cách
là $A_{\mathcal A}^{-2}$ — **một hằng số trên mỗi đoạn, nhưng đổi từ đoạn này sang đoạn
khác**. Hệ quả tức thì, chính bài báo nêu ra ở câu cuối chứng minh: $\hat C(t)$ là một **hàm
tuyến tính từng khúc, giảm**, với độ dốc $-A_{\mathcal A}^{2}$.

Kiểm cả hai:

```
   Bổ đề 7 (5.16): trong một đoạn có A không đổi, μ̂(t) = μ̂(t₀) + A_A·(t − t₀)·u_A
      đoạn 1, t =    18.04 : lệch 8.882e-16
      đoạn 2, t =   241.19 : lệch 1.066e-14
      đoạn 4, t =  1178.34 : lệch 2.842e-14

   (5.25): t₂ − t₁ = (λ₁ − λ₂)·A_A⁻², tức Ĉ(t) giảm TUYẾN TÍNH với hệ số A_A²
   đoạn | A_A²      | −dĈ/dt đo được | lệch
      1 | 1.000000 |    1.000000    | 4.66e-15
      2 | 0.723078 |    0.723078    | 3.33e-16
      3 | 0.607470 |    0.607470    | 1.11e-16
      4 | 0.513960 |    0.513960    | 1.11e-16
      5 | 0.217506 |    0.217506    | 1.53e-15
```

#### Bổ đề 8 và Bổ đề 9 — điều kiện dấu và điều kiện cực trị: (5.28)–(5.35)

![Bổ đề 8 của Efron 2004, trang 29](<paper_lars_efron2004_tr29_bo_de8.png>)

**Hình 40 — Bổ đề 8, trang 29.** Nguồn: cùng bài báo.

$$\hat c_j = \hat C\cdot\operatorname{sign}(\hat\beta_j)\quad\text{với } j\in\mathcal A \tag{5.28}$$

$$\operatorname{sign}(\hat\beta_j) = \operatorname{sign}(\hat c_j)\quad\text{với } j\in\mathcal A \tag{5.29}$$

(5.28) nói **mọi tương quan hoạt động có cùng độ lớn $\hat C$ và dấu khớp với dấu hệ số**;
(5.29) là hệ quả trực tiếp, và nó chính là điều kiện (3.1) mà mục 2.3 dùng — nay đã có chứng
minh thay vì chỉ là một quan sát.

![Bổ đề 9 của Efron 2004, trang 30](<paper_lars_efron2004_tr30_bo_de9.png>)

**Hình 41 — Bổ đề 9, trang 30.** Nguồn: cùng bài báo.

$$\lvert c_j(t)\rvert = \hat C(t)\ \text{với } j\in\mathcal A,
\qquad \lvert c_j(t)\rvert \le \hat C(t)\ \text{với } j\notin\mathcal A \tag{5.30}$$

**Ý nghĩa trừu tượng.** Hai bổ đề này là **bất biến (2.16) của LARS, phát biểu lại cho
lasso** — nhưng suy ra từ điều kiện tối ưu chứ không từ cách dựng thuật toán. Đó chính là
chỗ hai thế giới gặp nhau: LARS có tính chất "mọi tương quan hoạt động bằng nhau" **theo xây
dựng**; lasso có tính chất đó **theo bắt buộc**. Và (5.29) là (3.1), nay đã được chứng minh.

Bốn công thức phụ trợ:

| Số hiệu | Công thức | Nó nói gì |
|---|---|---|
| (5.31) | $R_t(d) = -\dot S(0)/\dot T(0)$ | **tỉ lệ đổi chác**: đi theo $d$ thì mỗi đơn vị ngân sách mua được bao nhiêu phần giảm của $S$ |
| (5.32) | $R_t(d) = 2\left[\hat C_t\sum_{\mathcal A}d_j + \sum_{\mathcal A^{c}}c_j(t)d_j\right]\big/\left[\sum_{\mathcal A}d_j + \sum_{\mathcal A^{c}}\lvert d_j\rvert\right]$ | viết $R_t$ ra tường minh |
| (5.33) | $R_t(d) = 2\hat C_t$ khi $d$ chỉ có toạ độ trong $\mathcal A$ | mọi hướng "chỉ dùng biến đang hoạt động" đều cho cùng tỉ lệ đổi chác |
| (5.34) | $R_t(d) = 2\lvert c_j(t)\rvert$ khi $d$ chỉ có một toạ độ $j$ | dùng một biến ngoài thì tỉ lệ bằng tương quan của nó |
| (5.35) | $R_t \equiv R_t(w_{\mathcal A})$ | tỉ lệ dọc đúng hướng lasso đang đi, tức **độ dốc của đường $(T,S)$** |

**Ý nghĩa trừu tượng của (5.33)–(5.34) đặt cạnh nhau.** Nếu tồn tại một biến ngoài có
$\lvert c_j\rvert > \hat C$ thì (5.34) cho tỉ lệ đổi chác **lớn hơn** (5.33) — tức có một
hướng tốt hơn hướng lasso đang đi, mâu thuẫn với việc nó là nghiệm. Đó là toàn bộ chứng minh
vế thứ hai của (5.30), và nó là một lập luận **kinh tế** chứ không phải giải tích: *"nếu có
món hời hơn thì anh đã không chọn món này"*.

#### Bổ đề 10 và bốn ràng buộc: (5.36)–(5.44)

![Bổ đề 10 của Efron 2004, trang 31](<paper_lars_efron2004_tr31_bo_de10.png>)

**Hình 42 — Bổ đề 10, trang 31.** Nguồn: cùng bài báo.

$$\mathcal A_1 = \{j : \hat\beta_j\ne0\},\qquad
\mathcal A_0 = \{j : \hat\beta_j = 0 \text{ và } \lvert\hat c_j\rvert = \hat C\} \tag{5.36}$$

$$R(d) = -\dot S(0)/\dot T(0) \le 2\hat C \tag{5.37}$$

$$\Delta S = -2\hat C\Delta T + L(d)^{2}(\Delta T)^{2} \tag{5.38}$$

$$L(d) = \lVert Xd/d_{+}\rVert \tag{5.39}$$

| Số hiệu | Công thức | Nó nói gì |
|---|---|---|
| (5.40)–(5.41) | khai triển $R(d)$ theo ba nhóm $\mathcal A_1$, $\mathcal A_0$, $\mathcal A_2$ | cho thấy $R(d) < 2\hat C$ **trừ khi** $d_j = 0$ với mọi $j\in\mathcal A_2$ |
| (5.42) | $T(\gamma) = T(0) + \gamma\sum_{\mathcal A_{10}}d_j$ | khi đó ngân sách tăng tuyến tính, hệ số đơn giản |
| (5.43) | $\Delta S \ge -2\hat C\Delta T + A_{\mathcal A}^{2}(\Delta T)^{2}$ | **chặn dưới** cho $\Delta S$, đạt được đúng khi $d$ là hướng cách đều góc |
| (5.44) | $\hat\beta_{\mathcal A}(\gamma) = \hat\beta_{\mathcal A} + \gamma S_{\mathcal A}w_{\mathcal A}$ | bước đi tiếp theo, dùng cho Ràng buộc 1 |

**Ý nghĩa trừu tượng của (5.38) và (5.43) đặt cạnh nhau — đây là chỗ cả §5 dồn về.** (5.38)
nói: đi thêm một lượng ngân sách $\Delta T$ thì $S$ giảm một lượng có **hai số hạng** — số
hạng bậc nhất $-2\hat C\Delta T$ **giống nhau cho mọi hướng hợp lệ**, còn số hạng bậc hai
$L(d)^{2}(\Delta T)^{2}$ thì khác nhau. Muốn $S$ giảm nhiều nhất thì phải làm $L(d)$ **nhỏ
nhất**, và Bổ đề 5 nói giá trị nhỏ nhất ấy chính là $A_{\mathcal A}$, đạt được đúng tại hướng
cách đều góc.

Nói bằng lời: **mọi hướng hợp lệ đều giảm $S$ với cùng một tốc độ tức thời; hướng cách đều
góc thắng vì nó giữ được tốc độ ấy LÂU NHẤT.** Đó là một câu về đạo hàm cấp hai, không phải
về đạo hàm cấp một — và nó là lý do sâu xa vì sao "cách đều góc" là hướng đúng.

Kiểm (5.37) bằng 20 000 hướng ngẫu nhiên tại $t = 1000$:

```
      R(d) lớn nhất trong 20000 hướng: 436.347825 ; 2Ĉ = 517.955512 ; vượt trần? KHÔNG
      R(d) theo đúng hướng cách đều góc u_A: 517.955589 -> đạt ĐÚNG trần 2Ĉ = 517.955512.
```

Hai mươi nghìn hướng ngẫu nhiên, **không hướng nào vượt trần**, và hướng cách đều góc thì
chạm đúng trần. Đó là (5.37) và (5.35) cùng lúc.

**Bốn ràng buộc, tr.33.** Đoạn cuối §5 gom mọi thứ lại thành bốn điều kiện mà tập hoạt động
$\mathcal A$ của lasso bắt buộc phải thoả:

| Ràng buộc | Nội dung | Suy từ đâu |
|---|---|---|
| 1 | $\mathcal A_1\subseteq\mathcal A$ | Bổ đề 7 qua (5.44): các hệ số đang khác 0 vẫn khác 0 ngay sau đó |
| 2 | $\mathcal A\subseteq\mathcal A_{10}$ | Bổ đề 10 qua (5.37): hướng đi không được dùng biến ngoài $\mathcal A_{10}$ |
| 3 | $w_{\mathcal A}$ không được ngược dấu $\hat c_j$ ở bất kỳ $j\in\mathcal A_0$ | Bổ đề 8: nếu ngược thì dấu hệ số và dấu tương quan sẽ lệch ngay sau đó |
| 4 | trong số các tập thoả 1–3, $\mathcal A$ phải làm $A_{\mathcal A}$ **nhỏ nhất** | Bổ đề 10 qua (5.43): để $S$ giảm nhanh nhất |

**Ý nghĩa trừu tượng.** Bốn ràng buộc này **xác định duy nhất** tập hoạt động, và đó là toàn
bộ chứng minh Định lý 1. Chỗ hay là nó không phải một phép quy nạp phức tạp: ba ràng buộc đầu
thu hẹp danh sách ứng viên xuống còn đúng hai khả năng — $\mathcal A_1$ hoặc $\mathcal A_{10}$
— và ràng buộc thứ tư chọn giữa hai. Khi một biến vừa **vào** thì Bổ đề 4 bảo đảm Ràng buộc 3
không bị vi phạm nên chọn $\mathcal A_{10}$; khi một biến vừa **bị bỏ** thì chính Ràng buộc 3
loại $\mathcal A_{10}$ nên chọn $\mathcal A_1$. Đúng hai nhánh của sửa đổi (3.6).

Bài báo cũng nói rõ giới hạn của Định lý 1 khi điều kiện "mỗi lần một biến" không thoả:

> *"Since one-at-a-time computations, perhaps with some added $y$ jitter, apply to all
> practical situations, the LARS algorithm described in Section 7 is not equipped to handle
> many-at-a-time problems."* — tr.34
>
> *Dịch: Vì các phép tính mỗi-lần-một-biến, có thể kèm thêm chút nhiễu cho $y$, áp dụng được
> cho mọi tình huống thực tế, thuật toán LARS mô tả ở mục 7 không được trang bị để xử lý các
> bài toán nhiều-biến-một-lúc.*

Đây là một lời tự khai về **cài đặt**, không phải về lý thuyết: bài báo nói cách xử lý trường
hợp hoà ở tr.33–34, nhưng chương trình đi kèm thì không cài.

#### Hình 8 của bài báo và bản dựng lại

![Hình 8 của Efron 2004, trang 30](<paper_lars_efron2004_tr30_hinh8_duong_ts.png>)

**Hình 43 — Hình 8 của Efron 2004, trang 30.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"Plot of $S$ versus $T$ for Lasso applied to diabetes data;
points indicate the 12 modified LARS steps of Figure 1; triangle is $(T, S)$ boundary point at
$t = 1000$; dashed arrow is tangent at $t = 1000$, negative slope $R_t$, (5.31). The $(T, S)$
curve is a decreasing, convex, quadratic spline."*

*Dịch: Đồ thị $S$ theo $T$ cho Lasso áp lên dữ liệu tiểu đường; các điểm đánh dấu 12 bước LARS
đã sửa đổi của Hình 1; hình tam giác là điểm biên $(T, S)$ tại $t = 1000$; mũi tên đứt là
tiếp tuyến tại $t = 1000$, độ dốc âm $R_t$, (5.31). Đường $(T, S)$ là một spline bậc hai,
giảm và lồi.*

**Nhận xét của chính tác giả, nguyên văn, tr.31:** *"The argument above relies on the fact
that $R_t(d)$ cannot be greater than $R_t$, or else there would be $(T, S)$ values lying below
the optimal curve. Using Lemmas 3 and 4 it can be shown that the $(T, S)$ curve is always
convex, as in Figure 8, being a quadratic spline with $\dot S(T) = -2\hat C(T)$ and
$\ddot S(T) = 2A_{\mathcal A}^{2}$."*

*Dịch: Lập luận ở trên dựa vào việc $R_t(d)$ không thể lớn hơn $R_t$, nếu không sẽ có những
giá trị $(T, S)$ nằm dưới đường tối ưu. Dùng Bổ đề 3 và Bổ đề 4, có thể chứng minh rằng đường
$(T, S)$ luôn lồi, như ở Hình 8, và là một spline bậc hai với $\dot S(T) = -2\hat C(T)$ và
$\ddot S(T) = 2A_{\mathcal A}^{2}$.*

> **Chỗ nguồn dẫn sai — số 15 trong bảng ở mục 0.5.** Câu trên dẫn **Bổ đề 3 và Bổ đề 4**.
> Bổ đề 3 nói $\hat\mu_k(y)$ liên tục và khả vi hầu khắp **dưới điều kiện nón dương** — một
> phát biểu về tính trơn theo $y$, không phải về đường $(T,S)$. Bổ đề 4 nói về **dấu** của
> biến mới vào. Hai bổ đề thật sự cho hai vế của câu ấy là:
>
> * $\dot S(T) = -2\hat C(T)$ — từ **Bổ đề 9** qua (5.33) và (5.35);
> * $\hat C(T)$ giảm, nên $\dot S$ tăng, nên đường lồi — từ **Bổ đề 7**, câu cuối chứng minh,
>   chỗ nói $\hat C(t)$ là hàm tuyến tính từng khúc giảm.
>
> Cả hai vế đều đo được, và cả hai đều đúng; chỉ dòng dẫn chiếu là sai địa chỉ.

![Dựng lại Hình 8 — đường (T, S) của lasso](<lars_ts_loi.png>)

**Hình 44 — dựng lại Hình 8, và kiểm cả hai vế của câu vừa trích (hình tự vẽ).**

**Cách đọc hình.** Bảng trái là đường $(T, S)$ với 12 điểm gãy, tam giác đỏ ở $t = 1000$ và
tiếp tuyến tại đó. Bảng phải vẽ **đạo hàm cấp hai**: đường tím là $\ddot S$ đo bằng sai phân
trên lưới 1200 điểm, các đoạn xanh lá là công thức $2A_{\mathcal A}^{2}$ tính riêng cho từng
đoạn.

Ba con số:

```
      S có giảm đơn điệu không? CÓ ; T có tăng đơn điệu không? CÓ
      đạo hàm cấp hai theo sai phân: nhỏ nhất 3.2782e-03 -> LỒI trên cả đường
      Ṡ(1000) đo -517.955512 · công thức −2Ĉ(1000) = −2×258.977756 = -517.955512 · lệch 5.68e-13
      S tại t = 1000 : 1463282.9944 ; T = 1000 ; Ĉ(1000) = 258.977756
```

**Điều hình KHÔNG nói ra.** $\ddot S$ là hằng số **trên mỗi đoạn** nhưng **nhảy bậc** ở mỗi
điểm gãy — các đoạn xanh lá có độ cao khác nhau. Đó chính là nghĩa của chữ *"quadratic
spline"*: không phải một parabol, mà là nhiều mảnh parabol nối nhau, và mỗi mảnh có độ cong
riêng bằng $2A_{\mathcal A}^{2}$ của tập hoạt động trên đoạn đó. Vì $A_{\mathcal A}$ giảm khi
tập lớn ra (Bổ đề 5), **đường càng về sau càng ít cong** — tức lợi ích biên của việc nới ngân
sách giảm dần.

<details>
<summary><b>Mã sinh Hình 44</b></summary>

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

d = load_diabetes(scaled=False)
X = d.data - d.data.mean(0)
X = X / np.sqrt((X ** 2).sum(0))
y = d.target - d.target.mean()
B = lars_path(X, y, method="lasso")[2]
T = np.abs(B).sum(0)
S = ((y[:, None] - X @ B) ** 2).sum(0)


def he_so(t):                                    # đường nghiệm TUYẾN TÍNH giữa hai nút
    return np.array([np.interp(t, T, B[j]) for j in range(10)])


luoi = np.linspace(0, T[-1], 1200)
S_luoi = np.array([((y - X @ he_so(t)) ** 2).sum() for t in luoi])
b1k = he_so(1000.0)
S1k = ((y - X @ b1k) ** 2).sum()
C1k = np.abs(X.T @ (y - X @ b1k)).max()          # Ĉ(1000)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
ax.plot(luoi, S_luoi, lw=2.0, color="#2563eb", zorder=2)
ax.plot(T, S, "o", ms=6.5, color="#1e3a8a", zorder=3,
        label="12 điểm gãy của đường lasso")
ax.plot([1000.0], [S1k], "^", ms=12, color="#dc2626", zorder=4,
        label=r"điểm $t = 1000$")
dx = np.array([-520.0, 760.0])
ax.plot(1000 + dx, S1k - 2 * C1k * dx, "--", lw=1.6, color="#dc2626", zorder=4,
        label=r"tiếp tuyến, độ dốc $\dot S = -2\hat C(t)$")
ax.annotate(r"$\dot S(1000) = -2 \times %.4f = %.4f$" % (C1k, -2 * C1k),
            (1000 + dx[1], S1k - 2 * C1k * dx[1]),
            xytext=(-6, 26), textcoords="offset points", fontsize=8.5, color="#dc2626",
            ha="right")
ax.set_xlabel(r"$T = \sum_j |\hat\beta_j|$")
ax.set_ylabel(r"$S = \|y - X\hat\beta\|^2$")
ax.set_title("Dựng lại Hình 8 tr.30 — đường (T, S) của lasso\n"
             "giảm, LỒI, và là một spline bậc hai", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# đạo hàm cấp hai đo bằng sai phân, so với công thức 2·A_A² của từng đoạn
dd = np.diff(S_luoi, 2) / (luoi[1] - luoi[0]) ** 2
ax2.plot(luoi[1:-1], dd, lw=1.4, color="#7c3aed", label=r"$\ddot S$ đo bằng sai phân")
for k in range(1, len(T)):
    A = list(np.flatnonzero(np.abs(B[:, k]) > 1e-12))
    if not A:
        continue
    c = X.T @ (y - X @ B[:, k - 1])
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s
    iG = np.linalg.solve(XA.T @ XA, np.ones(len(A)))
    AA2 = 1.0 / (np.ones(len(A)) @ iG)
    ax2.plot([T[k - 1], T[k]], [2 * AA2, 2 * AA2], lw=3.0, color="#059669",
             solid_capstyle="butt")
ax2.plot([], [], lw=3.0, color="#059669", label=r"công thức $2A_{\mathcal{A}}^2$ của từng đoạn")
ax2.axhline(0, color="#6b7280", lw=0.8)
ax2.set_xlabel(r"$T = \sum_j |\hat\beta_j|$")
ax2.set_ylabel(r"$\ddot S(T)$")
ax2.set_ylim(-0.25, 2.5)
ax2.set_title(r"$\ddot S$ là HẰNG SỐ trên mỗi đoạn và LUÔN DƯƠNG"
              "\nđó là toàn bộ lý do đường (T, S) lồi", fontsize=9.5)
ax2.legend(fontsize=8, loc="upper right")
fig.savefig("lars_ts_loi.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

assert dd.min() > -1e-6, "đường (T, S) có chỗ LÕM"
# S là đa thức bậc hai TRÊN MỖI ĐOẠN, nên sai phân hai phía với bước nằm gọn trong đoạn
# cho đạo hàm CHÍNH XÁC; dùng lưới chung ở trên thì bước vắt qua điểm gãy và sai số lớn.
h = 20.0
doc = (((y - X @ he_so(1000 + h)) ** 2).sum() - ((y - X @ he_so(1000 - h)) ** 2).sum()) / (2 * h)
print("Ṡ(1000) đo %.6f · công thức −2Ĉ(1000) = −2×%.6f = %.6f · lệch %.2e"
      % (doc, C1k, -2 * C1k, abs(doc + 2 * C1k)))
print("đạo hàm cấp hai nhỏ nhất trên cả đường: %.4e -> LỒI" % dd.min())
```

</details>

### 2.10 Vì sao sửa đổi Stagewise ĐÚNG — §6 của bài báo, tr.34–39

Mục này chứng minh Định lý 2, và nó ngắn hơn §5 vì dùng lại nhiều công cụ. Cấu trúc: ba ràng
buộc mà một hướng Stagewise hợp lệ phải thoả, rồi chứng minh **chỉ có đúng một hướng** thoả cả
ba.

#### Bổ đề 11 và Ràng buộc I: (6.1)–(6.3)

![Ràng buộc I của Efron 2004, trang 34](<paper_lars_efron2004_tr34_ct6_3_rang_buoc1.png>)

**Hình 45 — Ràng buộc I và công thức (6.3), trang 34.** Nguồn: cùng bài báo.

$$\lvert\hat c_j(N)-\hat c_j\rvert = \lvert x_j^{\top}(\hat\mu(N)-\hat\mu)\rvert
\le \lVert x_j\rVert\cdot\lVert\hat\mu(N)-\hat\mu\rVert \le \gamma \tag{6.1}$$

$$\hat\mu(\gamma) = \hat\mu + \gamma v,\qquad v = X_{\mathcal A}P_{\mathcal A} \tag{6.2}$$

$$\mathcal S_{\mathcal A}^{+} = \left\{v : v = \sum_{j\in\mathcal A}x_jP_j,\ P_j\ge0,\
\sum_{j\in\mathcal A}P_j = 1\right\} \tag{6.3}$$

**Bổ đề 11, nguyên văn:** *"For sufficiently small $\varepsilon$, only $j\in\mathcal A$ can
have $P_j = N_j/N > 0$."*

*Dịch: Với $\varepsilon$ đủ nhỏ, chỉ những $j\in\mathcal A$ mới có thể có $P_j = N_j/N > 0$.*

**Ý nghĩa trừu tượng.** (6.1) là một bất đẳng thức Cauchy–Schwarz đơn giản, nhưng nó gánh
toàn bộ Bổ đề 11: nếu đi một quãng ngắn thì tương quan không đổi được nhiều, nên một biến
đang kém xa mức $\hat C$ **không kịp** trở thành biến tương quan nhất. Nói cách khác, Stagewise
với bước nhỏ **không thể nhảy ra ngoài tập hoạt động**.

#### Ràng buộc II và III: (6.4)–(6.7)

![Ràng buộc II và III của Efron 2004, trang 35](<paper_lars_efron2004_tr35_ct6_6_6_7_rang_buoc23.png>)

**Hình 46 — Ràng buộc II, III và công thức (6.6)–(6.7), trang 35.** Nguồn: cùng bài báo.

$$v_{\mathcal B} = A_{\mathcal B}^{2}X_{\mathcal B}G_{\mathcal B}^{-1}\mathbf 1_{\mathcal B}
= A_{\mathcal B}u_{\mathcal B} \tag{6.4}$$

$$\hat c_j(\gamma) = x_j^{\top}(y-\hat\mu-\gamma v) = \hat c_j - \gamma x_j^{\top}v \tag{6.5}$$

$$x_j^{\top}v_{\mathcal B} = A_{\mathcal B}^{2}\quad\text{với } j\in\mathcal B \tag{6.6}$$

$$x_j^{\top}v_{\mathcal B} \ge A_{\mathcal B}^{2}\quad\text{với } j\in\mathcal A-\mathcal B \tag{6.7}$$

| Ràng buộc | Nội dung | Vì sao bắt buộc |
|---|---|---|
| **I** | $v\in\mathcal S_{\mathcal A}^{+}$ | $P_j$ đếm số bước, không âm được |
| **II** | $v$ phải tỉ lệ với $u_{\mathcal B}$ của một tập con $\mathcal B$ | nếu không, các tương quan trong $\mathcal B$ **giảm không đều**, và thuật toán (1.7) sẽ không đi theo $v$ nữa |
| **III** | $x_j^{\top}v_{\mathcal B}\ge A_{\mathcal B}^{2}$ với $j\in\mathcal A-\mathcal B$ | các biến bị bỏ ra phải **tụt nhanh ít nhất bằng** các biến ở lại; nếu không chúng lập tức quay vào |

**Ý nghĩa trừu tượng của bộ ba.** Ba ràng buộc này mô tả **chính xác tập các hướng mà một
thủ tục kiểu (1.7) có thể đi theo**, mà không cần nói gì về $\varepsilon$. Đó là cách bài báo
biến một thủ tục rời rạc thành một bài toán hình học liên tục: thay vì hỏi *"dãy bước
$\varepsilon$ hội tụ về đâu"*, hỏi *"hướng nào thoả ba điều kiện này"*.

#### Bổ đề 12 và hình học nón lồi: (6.8)–(6.18)

![Hình 9 của Efron 2004, trang 37](<paper_lars_efron2004_tr37_hinh9_hinh_hoc_non_loi.png>)

**Hình 47 — Hình 9 của Efron 2004, trang 37.** Nguồn: cùng bài báo.

**Chú thích gốc, nguyên văn:** *"The geometry of the LARS–Stagewise modification."*

*Dịch: Hình học của sửa đổi LARS–Stagewise.*

**Nhận xét của chính tác giả, nguyên văn, tr.36:** *"The nearest point to $u_{\mathcal A}$ in
$\mathcal C_{\mathcal A}$, say $\hat u_{\mathcal A}$, is of the form $\sum_{\mathcal A}x_j\hat P_j$
with $\hat P_j\ge0$. Therefore $\hat u_{\mathcal A}$ exists strictly within face
$\hat{\mathcal B}$, where $\hat{\mathcal B} = \{j : \hat P_j > 0\}$, and must equal
$\mathrm{Proj}_{\hat{\mathcal B}}(u_{\mathcal A})$."*

*Dịch: Điểm gần $u_{\mathcal A}$ nhất trong $\mathcal C_{\mathcal A}$, gọi là
$\hat u_{\mathcal A}$, có dạng $\sum_{\mathcal A}x_j\hat P_j$ với $\hat P_j\ge0$. Do đó
$\hat u_{\mathcal A}$ nằm hẳn bên trong mặt $\hat{\mathcal B}$, với
$\hat{\mathcal B} = \{j : \hat P_j > 0\}$, và phải bằng
$\mathrm{Proj}_{\hat{\mathcal B}}(u_{\mathcal A})$.*

**Cách đọc hình.** Hình vẽ nón lồi $\mathcal C_{\mathcal A}$ như một hình chóp mở, và vectơ
$u_{\mathcal A}$ nằm **ngoài** nón đó. Hình chiếu của nó vào nón rơi lên một **mặt** của
chóp — mặt ứng với tập con $\hat{\mathcal B}$ — và chính hình chiếu ấy là hướng Stagewise đi.
Siêu phẳng $H$ trong hình là mặt phẳng tách $u_{\mathcal A}$ khỏi nón, và nó là công cụ của
chứng minh Bổ đề 12.

**Điều hình KHÔNG nói ra.** Hình vẽ trường hợp hình chiếu rơi lên một mặt **hai chiều** của
chóp ba chiều, tức $\lvert\hat{\mathcal B}\rvert = \lvert\mathcal A\rvert - 1$. Trên bộ
diabetes, lần thu nhỏ đầu tiên bỏ **hai** biến cùng lúc: $\mathcal A$ tám phần tử co về
$\hat{\mathcal B}$ sáu phần tử. Hình chiếu rơi lên một mặt **thấp hơn hai chiều**, và hình
không vẽ trường hợp đó.

Mười một công thức của chứng minh Bổ đề 12, gom lại:

| Số hiệu | Công thức | Nó nói gì |
|---|---|---|
| (6.8) | $\hat v = d\hat\beta(z)/dz$ | đạo hàm phải của đường Stagewise theo một tham số hoá đơn điệu |
| (6.9) | $\mathrm{Proj}_{\mathcal B}(u_{\mathcal A}) = (A_{\mathcal A}/A_{\mathcal B})u_{\mathcal B}$ | **hình chiếu của hướng cách đều góc vào một mặt lại là hướng cách đều góc của mặt đó** |
| (6.10) | $\mathrm{Proj}_{\mathcal B}(v_{\mathcal A}) = (A_{\mathcal A}/A_{\mathcal B})^{2}v_{\mathcal B}$ | cùng điều đó, viết cho $v$ |
| (6.11) | $u_{\mathcal A} = \cos\cdot u_{\mathcal B} + \sin\cdot z_{\mathcal B}$ | tách $u_{\mathcal A}$ thành phần trong mặt và phần vuông góc |
| (6.12) | $u_{\mathcal B} = (1,0,0,0)$, $u_{\mathcal A} = (\cos,0,\sin,0)$ sau một phép xoay | đưa hình học về dạng đọc được bằng toạ độ |
| (6.13) | $x_j = (A_{\mathcal B}, x_{j2}, 0, 0)$ với $j\in\mathcal B$ | toạ độ đầu **buộc phải** bằng $A_{\mathcal B}$ vì (2.7) |
| (6.14) | $x_\ell = (x_{\ell1}, x_{\ell2}, x_{\ell3}, x_{\ell4})$ với $\ell\in\mathcal A-\mathcal B$ | đặt tên toạ độ cho các biến bị bỏ ra |
| (6.15) | $A_{\mathcal A} = \cos\cdot x_{\ell1} + \sin\cdot x_{\ell3}$ | viết ràng buộc (2.7) trong hệ toạ độ ấy |
| (6.16) | $x_\ell^{\top}v_{\hat{\mathcal B}} = A_{\hat{\mathcal B}}x_{\ell1} \ge A_{\hat{\mathcal B}}^{2}$ | **chiều thuận**: hình chiếu thoả Ràng buộc III |
| (6.17) | $x_\ell^{\top}v_{\mathcal B} = A_{\mathcal B}x_{\ell1} < A_{\mathcal B}^{2}$ | **chiều đảo**: mọi mặt khác VI PHẠM Ràng buộc III |
| (6.18) | $d\hat\beta(z)/dz = \phi(\hat{\mathcal A})$ | hướng đi là **hàm chỉ của tập tương quan cực đại**, không của lịch sử |

**Ý nghĩa trừu tượng của (6.9) — công thức đẹp nhất của §6.** Chiếu một vectơ cách đều góc
vào không gian con sinh bởi một tập con thì **được đúng vectơ cách đều góc của tập con đó**,
nhân một hệ số. Tính chất ấy không hiển nhiên chút nào, và nó là lý do cả phép chiếu vào nón
lồi vẫn cho ra một hướng "cách đều góc" — chỉ là của một tập nhỏ hơn. Nhờ vậy, sửa đổi
Stagewise **không phá vỡ cấu trúc** của thuật toán: nó chỉ đổi tập.

Kiểm (6.9) và hai ràng buộc tại đúng chỗ chúng kích hoạt trên bộ diabetes:

```
   Tại bước 7 của đường LARS–Stagewise, u_A nằm NGOÀI nón lồi C_A.
      A  = [2, 3, 4, 5, 7, 8, 9, 10] -> A_A = 0.26851128 ; w_A có 2 thành phần ÂM
      B̂  = [2, 4, 5, 8, 9, 10] -> A_B̂ = 0.31359291
      Bổ đề 5 đòi A ⊆ B thì A_A ≥ A_B; ở đây B̂ ⊆ A nên A_B̂ ≥ A_A: 0.31359291 ≥ 0.26851128 -> ĐÚNG

   (6.9): hình chiếu của u_A vào L(X_B̂) bằng (A_A/A_B̂)·u_B̂
      lệch = 8.049e-16 ; Cos = A_A/A_B̂ = 0.85624153 ; u_A'u_B̂ đo được = 0.85624153

   Ràng buộc II (6.6): x_j'v_B̂ = A_B̂² với j ∈ B̂ ; Ràng buộc III (6.7): ≥ A_B̂² với j ∈ A − B̂
      biến 2  ∈ B̂ : x_j'v_B̂ = 0.09834052  =  A_B̂² = 0.09834052
      biến 3  ∈ A−B̂: x_j'v_B̂ = 0.10293343  ≥  A_B̂² = 0.09834052
      biến 4  ∈ B̂ : x_j'v_B̂ = 0.09834052  =  A_B̂² = 0.09834052
      biến 5  ∈ B̂ : x_j'v_B̂ = 0.09834052  =  A_B̂² = 0.09834052
      biến 7  ∈ A−B̂: x_j'v_B̂ = 0.17252342  ≥  A_B̂² = 0.09834052
      biến 8  ∈ B̂ : x_j'v_B̂ = 0.09834052  =  A_B̂² = 0.09834052
```

Sáu biến ở lại cho **đúng cùng một giá trị 0,09834052**; hai biến bị bỏ cho giá trị **lớn
hơn**. Đó là Ràng buộc II và Ràng buộc III, đúng như phát biểu, tại đúng chỗ bài báo nói.

#### (6.19)–(6.22): LARS và Stagewise tối ưu hai thứ KHÁC NHAU

![Công thức (6.19) của Efron 2004, trang 38](<paper_lars_efron2004_tr38_ct6_19_toc_do_giam_S.png>)

**Hình 48 — công thức (6.19), trang 38.** Nguồn: cùng bài báo.

$$-\left.\frac{\partial S(\gamma)}{\partial\gamma}\right|_{0}
= 2\hat C\cdot\frac{u_{\mathcal A}^{\top}u}{A_{\hat{\mathcal A}}} \tag{6.19}$$

Tốc độ giảm tức thời của $S$ theo một hướng $u$ **chỉ phụ thuộc hình chiếu của $u$ lên hướng
cách đều góc**. Hệ quả: $u_{\mathcal A}$ làm $S$ giảm nhanh nhất, và mọi hướng khác chậm hơn
đúng theo cosin của góc lệch.

![Công thức (6.20)–(6.22) của Efron 2004, trang 39](<paper_lars_efron2004_tr39_ct6_20_6_22_ti_so.png>)

**Hình 49 — công thức (6.20)–(6.22), trang 39.** Nguồn: cùng bài báo.

$$\left.\frac{\partial S_{\text{Stage}}(\gamma)}{\partial\gamma}\right|_{0}
\Big/ \left.\frac{\partial S_{\text{LARS}}(\gamma)}{\partial\gamma}\right|_{0}
= \frac{A_{\hat{\mathcal A}}}{A_{\hat{\mathcal B}}} \tag{6.20}$$

$$-\left.\frac{\partial\hat C(\gamma)}{\partial\gamma}\right|_{0}
= \min_{\hat{\mathcal A}}\{\lvert x_j^{\top}u\rvert\} \tag{6.21}$$

$$\left.\frac{\partial\hat C_{\text{LARS}}(\gamma)}{\partial\gamma}\right|_{0}
\Big/ \left.\frac{\partial\hat C_{\text{Stage}}(\gamma)}{\partial\gamma}\right|_{0}
= \frac{A_{\hat{\mathcal A}}}{A_{\hat{\mathcal B}}} \tag{6.22}$$

**Ý nghĩa trừu tượng — đây là kết luận của cả §6, và nó đổi cách hiểu về Stagewise.** Hai tỉ
số (6.20) và (6.22) **bằng nhau**, nhưng chúng nói về hai đại lượng khác nhau, và chúng nghịch
chiều nhau:

* Theo (6.20): Stagewise giảm $S$ **chậm hơn** LARS, đúng hệ số $A_{\hat{\mathcal A}}/A_{\hat{\mathcal B}} < 1$.
* Theo (6.22): Stagewise giảm $\hat C$ **nhanh hơn** LARS, đúng cùng hệ số.

Nghĩa là hai thuật toán không phải "một cái tốt hơn cái kia" — chúng **tối ưu hai tiêu chí
khác nhau**. Chính tác giả nói ra:

> *"The original motivation for the Stagewise procedure was to minimize residual squared error
> within a framework of parsimonious forward search. However, (6.20) shows that Stagewise is
> less greedy than LARS in this regard, it being more accurate to describe Stagewise as
> striving to minimize the maximum absolute residual correlation."* — tr.39
>
> *Dịch: Động cơ ban đầu của thủ tục Stagewise là cực tiểu hoá sai số bình phương phần dư
> trong khuôn khổ một phép tìm kiếm tới tiết kiệm. Tuy nhiên, (6.20) cho thấy Stagewise kém
> tham hơn LARS ở khía cạnh đó, và mô tả chính xác hơn là Stagewise đang cố cực tiểu hoá
> tương quan phần dư tuyệt đối lớn nhất.*

Đây là một chỗ **đính chính chính mình** rất thẳng thắn: thủ tục được nghĩ ra để làm một
việc, và hoá ra nó làm một việc khác. Đo lại tại đúng chỗ (6.20) và (6.22) kích hoạt:

```
      −∂S/∂γ theo u_A  (LARS)      : 513.682634
      −∂S/∂γ theo u_B̂  (Stagewise) : 439.836405
      (6.20) tỉ số Stage/LARS = 0.85624153 ; A_A/A_B̂ = 0.85624153 ; lệch 2.11e-15
      (6.21) −∂Ĉ/∂γ theo u_A  = 0.26851128 (= A_A  = 0.26851128)
             −∂Ĉ/∂γ theo u_B̂ = 0.31359291 (= A_B̂ = 0.31359291)
      (6.22) tỉ số LARS/Stage = 0.85624153 = A_A/A_B̂ ; lệch 4.33e-15
      -> LARS giảm S nhanh hơn Stagewise 1.1679 lần, nhưng giảm Ĉ CHẬM hơn đúng bấy nhiêu.
```

Hai tỉ số ra **đúng cùng một số 0,85624153**, lệch cỡ $10^{-15}$. Và con số ấy chính là
$\cos$ của góc giữa hai hướng, theo (6.11) — tức ba đại lượng tưởng không liên quan hoá ra
là một.

### 2.11 Chi phí tính toán — §7 của bài báo, tr.39–40

**Khẳng định gốc, nguyên văn, tr.39:** *"The entire sequence of steps in the LARS algorithm
with $m < n$ variables requires $O(m^{3} + nm^{2})$ computations — the cost of a least squares
fit on $m$ variables."*

*Dịch: Toàn bộ dãy bước của thuật toán LARS với $m < n$ biến cần $O(m^{3} + nm^{2})$ phép
tính — đúng bằng chi phí của một lần khớp bình phương tối thiểu trên $m$ biến.*

**Nhận xét của chính tác giả về cách đạt được, nguyên văn, tr.39:** *"In detail, at the $k$th
of $m$ steps, we compute $m - k$ inner products $c_{jk}$ of the nonactive $x_j$ with the
current residuals to identify the next active variable, and then invert the $k\times k$ matrix
$G_k = X_k^{\top}X_k$ to find the next LARS direction. We do this by updating the Cholesky
factorization $R_{k-1}$ of $G_{k-1}$ found at the previous step … Hence the LARS sequence can
be seen as a Cholesky factorization with a guided ordering of the variables."*

*Dịch: Chi tiết hơn, ở bước thứ $k$ trong $m$ bước, ta tính $m-k$ tích vô hướng $c_{jk}$ giữa
các $x_j$ chưa hoạt động với phần dư hiện tại để xác định biến hoạt động kế tiếp, rồi nghịch
đảo ma trận $k\times k$ là $G_k = X_k^{\top}X_k$ để tìm hướng LARS kế tiếp. Ta làm việc đó
bằng cách cập nhật phân tích Cholesky $R_{k-1}$ của $G_{k-1}$ tìm được ở bước trước … Vì thế
dãy LARS có thể được nhìn như một phân tích Cholesky với một thứ tự biến có định hướng.*

**Câu cuối là câu đáng nhớ nhất của cả mục 7.** Phân tích Cholesky của $X^{\top}X$ là thứ mà
một lần khớp bình phương tối thiểu dày đặc phải làm dù sao. LARS làm **đúng phép phân tích
ấy**, chỉ khác ở chỗ nó **chọn thứ tự các cột theo dữ liệu** thay vì theo thứ tự có sẵn. Nhìn
thế thì "LARS rẻ bằng một lần OLS" thôi là một phép màu — nó là một cách phát biểu lại cùng
một phép tính.

**Cách kiểm cho đúng, và đây là một bài học về phép đo.** Bấm đồng hồ trên một vòng lặp Python
thì **không kiểm được** khẳng định này: chi phí thông dịch át mất phần số học. Cách đúng là
**đếm phép tính**:

```
   Cách kiểm ĐÚNG là ĐẾM PHÉP TÍNH, không phải bấm đồng hồ:
      · mỗi bước k: (m − k) tích vô hướng độ dài n  -> (m − k)·n
      · cập nhật Cholesky hạng 1                     -> k²/2 + O(k)
      · hai phép giải tam giác để lấy G_A⁻¹·1       -> k²
      · dựng u_A = X_A w_A và a = X'u_A              -> n·k + n·m

      n     | m    | LARS (phép)  | OLS (phép)   | LARS/OLS | m gấp đôi -> LARS ×
      2000 |   20 | 1.6043e+06 | 8.0267e+05 |    2.00  | —
      2000 |   40 | 6.4332e+06 | 3.2213e+06 |    2.00  | 4.01
      2000 |   80 | 2.5861e+07 | 1.2971e+07 |    1.99  | 4.02
      2000 |  160 | 1.0447e+08 | 5.2565e+07 |    1.99  | 4.04
      2000 |  320 | 4.2606e+08 | 2.1572e+08 |    1.98  | 4.08
      -> tỉ số LARS/OLS dừng lại quanh 1.98 chứ không tăng theo m
```

**Tỉ số dừng ở 1,98 và KHÔNG tăng theo $m$** — đó chính xác là nghĩa của *"the same order of
magnitude as ordinary least squares"*. Không phải "bằng nhau", mà là "cùng bậc": LARS tốn
khoảng **hai lần** một lần khớp OLS.

Đo đồng hồ để đối chiếu, với $n$ đủ lớn cho phần số học át chi phí thông dịch:

```
      n      | m    | LARS Cholesky | LARS giải lại  | OLS đầy đủ | LARS/OLS
      20000 |   40 |     0.0591 s  |     0.1024 s  |   0.0107 s |    5.5 lần
      20000 |   80 |     0.2268 s  |     0.5357 s  |   0.0395 s |    5.7 lần
      20000 |  160 |     0.8180 s  |     2.9585 s  |   0.1305 s |    6.3 lần
      (hai bản LARS cho cùng đường: lệch 2.93e-14)
```

Hai điều đọc ra, và cả hai nên nói thẳng:

1. **Cập nhật Cholesky thắng rõ bản ngây thơ** — khoảng **3,6 lần** ở $m = 160$ trong lần
   chạy in ở trên. Đó là phần mà §7 nói tới, và nó đo được. (Cột giây **không tái lập
   được**: chạy lại cho con số khác ở chữ số thứ ba; xem khai báo ở mục 0.5.)
2. **Tỉ số so với OLS thì KHÔNG dừng ở 2** mà chạy 5,5 → 5,7 → 6,3. Lý do không nằm ở thuật
   toán mà ở cài đặt: `numpy.linalg.lstsq` gọi một thủ tục LAPACK duy nhất, còn bản LARS này
   gọi $m$ lần các phép BLAS nhỏ từ Python. **Bảng đếm phép tính là bảng kiểm khẳng định của
   bài báo; bảng đồng hồ là bảng mô tả cài đặt cụ thể này.** Hai bảng trả lời hai câu hỏi
   khác nhau, và gộp chúng lại là cách chắc chắn để đọc sai cả hai.

#### Trường hợp $m \gg n$

![Nhận xét về m ≫ n của Efron 2004, trang 40](<paper_lars_efron2004_tr40_m_lon_hon_n.png>)

**Hình 50 — nhận xét về trường hợp $m\gg n$, trang 40.** Nguồn: cùng bài báo.

**Nguyên văn:** *"In this case LARS terminates at the saturated least squares fit after
$n - 1$ variables have entered the active set [at a cost of $O(n^{3})$ operations]. (This
number is $n - 1$ rather than $n$, because the columns of $X$ have been mean centered, and
hence it has row-rank $n - 1$.)"* … *"Although the model involves no more than $n - 1$
variables at any time, the number of different variables ever to have entered the model during
the entire sequence can be—and typically is—greater than $n - 1$."*

*Dịch: Trong trường hợp này LARS kết thúc ở phép khớp bình phương tối thiểu bão hoà sau khi
$n-1$ biến đã vào tập hoạt động [với chi phí $O(n^{3})$ phép tính]. (Con số là $n-1$ chứ không
phải $n$, vì các cột của $X$ đã được tâm hoá nên nó có hạng theo hàng là $n-1$.)* … *Dù mô
hình không bao giờ có quá $n-1$ biến tại bất kỳ thời điểm nào, số biến KHÁC NHAU từng đi vào
mô hình trong cả dãy có thể — và thường là — lớn hơn $n-1$.*

Hai khẳng định, hai phép đếm:

```
      n = 40  m = 300  :  49 bước · số biến khác 0 NHIỀU NHẤT = 39 (n − 1 = 39)
                          số biến TỪNG vào mô hình = 42 (> n − 1: có)
      n = 60  m = 1000 : 101 bước · số biến khác 0 NHIỀU NHẤT = 59 (n − 1 = 59)
                          số biến TỪNG vào mô hình = 75 (> n − 1: có)
```

**Cả hai khớp tuyệt đối.** Số hệ số khác 0 lớn nhất bằng **đúng $n-1$** ở cả hai cỡ, và số
biến từng vào thì **vượt** $n-1$ — 42 so với 39, và 75 so với 59. Câu thứ hai là câu người
đọc hay bỏ qua, và nó có hệ quả thực hành: **danh sách biến "được lasso chọn" phụ thuộc vào
điểm dừng, và tổng số biến từng xuất hiện có thể lớn hơn nhiều so với số biến trong bất kỳ mô
hình nào trên đường.**

Bài báo cũng tự nêu một giới hạn ở đúng mục này:

> *"The model sequence, particularly near the saturated end, tends to be quite variable with
> respect to small changes in $y$."* — tr.40
>
> *Dịch: Dãy mô hình, nhất là gần đầu bão hoà, có xu hướng khá biến động trước những thay đổi
> nhỏ của $y$.*

### 2.12 Boosting — §8 của bài báo, tr.40–41

![Đoạn về least squares boosting, trang 41](<paper_lars_efron2004_tr41_boosting.png>)

**Hình 51 — đoạn mô tả least squares boosting, trang 41.** Nguồn: cùng bài báo.

**Nguyên văn:** *"Specifically we start with the residual $r = y$ and the fit $\hat y = 0$. We
fit a tree in $x_1, x_2, \dots, x_m$ to the response $y$ giving a fitted tree $t_1$ (an
$n$-vector of fitted values). Then we update $\hat y$ to $\hat y + \varepsilon\cdot t_1$, $r$
to $y - \hat y$ and continue for many iterations. Here $\varepsilon$ is a small positive
constant. Empirical studies show that small values of $\varepsilon$ work better than
$\varepsilon = 1$: in fact, for prediction accuracy 'the smaller the better.'"*

*Dịch: Cụ thể, ta bắt đầu với phần dư $r = y$ và phép khớp $\hat y = 0$. Ta khớp một cây trên
$x_1,\dots,x_m$ với đáp ứng $y$, được cây $t_1$ (một vectơ $n$ chiều các giá trị khớp). Rồi ta
cập nhật $\hat y$ thành $\hat y + \varepsilon\cdot t_1$, $r$ thành $y - \hat y$, và tiếp tục
nhiều vòng. Ở đây $\varepsilon$ là một hằng số dương nhỏ. Các nghiên cứu thực nghiệm cho thấy
$\varepsilon$ nhỏ chạy tốt hơn $\varepsilon = 1$: thật ra, về độ chính xác dự đoán thì "càng
nhỏ càng tốt".*

**Ý nghĩa trừu tượng — và chỗ nối.** Đặt cạnh (1.7) thì thấy ngay: **least squares boosting
chính là Forward Stagewise, chỉ khác là "biến dự báo" bây giờ là TẬP MỌI CÂY có thể khớp
được, thay vì $m$ cột.** Khớp một cây vào phần dư là một cách tìm "biến tương quan nhất" bằng
tìm kiếm số. Nên câu hỏi *"vì sao $\varepsilon$ nhỏ lại tốt hơn"* trở thành câu hỏi *"vì sao
đường lasso tốt hơn nghiệm OLS"* — và câu sau thì đã có câu trả lời.

**Nhưng bài báo dừng lại ở PHỎNG ĐOÁN, và nói rõ như vậy:**

> *"Hastie, Tibshirani and Friedman (2001) noted the striking similarity between Forward
> Stagewise regression and the Lasso, and conjectured that this may help explain the success of
> the Forward Stagewise process used in least squares boosting. That is, in some sense least
> squares boosting may be carrying out a Lasso fit on the infinite set of tree predictors."*
> — tr.41
>
> *Dịch: Hastie, Tibshirani và Friedman (2001) đã ghi nhận sự giống nhau nổi bật giữa hồi quy
> Forward Stagewise và Lasso, và PHỎNG ĐOÁN rằng điều này có thể giúp giải thích thành công
> của quá trình Forward Stagewise dùng trong least squares boosting. Tức là, theo một nghĩa
> nào đó, least squares boosting có thể đang thực hiện một phép khớp Lasso trên tập vô hạn các
> cây dự báo.*

**§8 là mục DUY NHẤT của bài báo không có một con số nào.** Không bảng, không hình, không phép
đo. Hai chữ *"conjectured"* và *"may be"* là lời tự khai đúng mức. Dựng lại ý đó ở quy mô nhỏ,
với bộ học cơ sở là các **gốc cây** một chiều:

```
      n = 120 điểm, m = 60 gốc cây (mỗi gốc là một ngưỡng chia)
      hạng của ma trận thiết kế: 60 ; số điều kiện 4.208e+01
      ε = 0.2000  (    127 bước): lệch lớn nhất với đường lasso 0.719902 (thang hệ số 2.099)
      ε = 0.0500  (    507 bước): lệch lớn nhất với đường lasso 0.177114 (thang hệ số 2.099)
      ε = 0.0125  (   2026 bước): lệch lớn nhất với đường lasso 0.031938 (thang hệ số 2.099)
```

$\varepsilon$ giảm 16 lần thì lệch giảm **22,5 lần** — tức **ít nhất tỉ lệ thuận**, cùng
chiều với phép đo ở mục 2.4 trên dữ liệu 10 biến (ở đó tỉ số lệch/$\varepsilon$ đứng quanh
0,5–0,8 chứ không trôi). Đó là nội dung phỏng đoán của §8, ở một quy mô kiểm được. Nó **không**
chứng minh phỏng đoán cho trường hợp cây thật, nơi tập cơ sở vô hạn và không trực giao; nó chỉ
cho thấy cơ chế đứng vững ở chỗ kiểm được.

Bài báo kết thúc bằng một đề xuất, và đề xuất ấy cũng không được đo:

> *"One such idea is a modified form of Forward Stagewise: we find the best tree as usual, but
> rather than taking a small step in only that tree, we take a small least squares step in all
> trees currently in our model. One can show that for small step sizes this procedure
> approximates LARS; its advantage is that it can be carried out on an infinite set of
> predictors such as trees."* — tr.41
>
> *Dịch: Một ý như thế là một dạng sửa đổi của Forward Stagewise: ta tìm cây tốt nhất như
> thường lệ, nhưng thay vì đi một bước nhỏ chỉ trên cây đó, ta đi một bước bình phương tối
> thiểu nhỏ trên MỌI cây hiện có trong mô hình. Có thể chứng minh rằng với bước nhỏ, thủ tục
> này xấp xỉ LARS; cái lợi của nó là làm được trên một tập vô hạn các biến dự báo như tập các
> cây.*

Câu *"One can show"* không kèm chứng minh và không kèm phép đo. Đó là câu cuối cùng của phần
nội dung bài báo.

### 2.13 Phụ lục — tr.41–47

Phụ lục dài sáu trang và có một việc duy nhất: **chứng minh Bổ đề 3**, tức chứng minh
$\hat\mu_k(y)$ đủ trơn theo $y$ để công thức Stein (4.12) dùng được. Đây là phần dài nhất của
bài báo, và nó tồn tại vì một lý do dễ bỏ qua: **LARS là một hàm của $y$ có thể NHẢY**, khi
một thay đổi nhỏ của $y$ làm đổi biến nào vào trước.

| Phần | Nội dung | Công thức |
|---|---|---|
| **A.1** | Bổ đề 13: tồn tại tập mở $\mathcal G_k$ có độ đo đầy đủ mà trên đó tập hoạt động **cố định cục bộ** và $\hat\mu_k$ **tuyến tính cục bộ**. Bổ đề 14: gần một điểm thường, chỉ một biến vào, và $\hat\gamma_k$ liên tục, Lipschitz | (A.1)–(A.4) |
| **A.2** | Bổ đề 15: công thức cho thành phần "+" của $G_{\mathcal A^{+}}^{-1}\mathbf 1$, và hệ quả $x_{+}^{\top}u_{\mathcal A} < A_{\mathcal A}$ **dưới điều kiện nón dương** | (A.5)–(A.6) |
| **A.3** | Bổ đề 16 và 17: xử lý **điểm bội** — nơi hai biến vào cùng lúc. Ghi chú A.1 định nghĩa lại "khả vi hầu khắp". Rồi chứng minh Bổ đề 3 | — |

Sáu công thức, gom lại:

| Số hiệu | Công thức | Nó nói gì |
|---|---|---|
| (A.1) | $\lVert\Delta g\rVert \le L\lVert\Delta y\rVert$ | định nghĩa Lipschitz cục bộ |
| (A.2) | $w^{\top}(Ry-\gamma u_k) = x_{j_1}^{\top}(Ry-\gamma u_k) = x_{j_2}^{\top}(Ry-\gamma u_k)$ | điều kiện để **hai** biến cùng bắt kịp — và tập các $y$ như thế nằm trong một **hợp hữu hạn siêu phẳng**, tức có độ đo 0 |
| (A.3) | $\hat C_k(y_0)-\hat\gamma_kA_k = \hat c_{k,j}(y_0)-\hat\gamma_ka_{k,j}$ với $j = k+1$, và $>$ với $j > k+1$ | biến $k+1$ bắt kịp **đúng trước** mọi biến khác |
| (A.4) | $\hat\gamma_k(y) = \dfrac{(x_\ell-x_{k+1})^{\top}(y-\hat\mu_{k-1}(y))}{(x_\ell-x_{k+1})^{\top}u_k}$ | (2.13) viết lại thành một **thương của hai hàm tuyến tính** — đây là chỗ tính trơn ra đời |
| (A.5) | $(G_{\mathcal A^{+}}^{-1}\mathbf 1_{\mathcal A^{+}})_{+} = (1-a)^{-1}\left(1-\dfrac{x_{+}^{\top}u_{\mathcal A}}{A_{\mathcal A}}\right)$ | công thức nghịch đảo ma trận chia khối |
| (A.6) | $x_{+}^{\top}u_{\mathcal A} < A_{\mathcal A}$ | **hệ quả của điều kiện nón dương**: biến mới luôn có tương quan với hướng đi NHỎ HƠN mức chung |

**Ý nghĩa trừu tượng của (A.6) — và đây là chỗ điều kiện nón dương thật sự làm việc.** Nó bảo
đảm mẫu số $A_{\mathcal A} - a_j$ trong (2.13) **luôn dương** cho biến sắp vào. Không có nó,
mẫu số có thể đổi dấu, và khi mẫu số đổi dấu thì $\hat\gamma_k$ — theo (A.4) là một thương —
có thể **nhảy**. Một hàm nhảy thì không khả vi hầu khắp, và công thức Stein (4.12) sập. Đó
đúng là chỗ Định lý 4 cần (4.11), và cũng đúng là chỗ nó có thể hỏng khi (4.11) không thoả —
như bộ diabetes.

**Ghi chú A.1 tr.46** là một chỗ nhỏ mà đáng đọc, vì nó là tác giả tự đính chính một định
nghĩa:

> *"We say that a function $g:\mathbb R^{n}\to\mathbb R$ is almost differentiable if it is
> absolutely continuous on almost all line segments parallel to the coordinate axes, and its
> partial derivatives (which consequently exist a.e.) are locally integrable. This definition
> of almost differentiability appears superficially to be weaker than that given by Stein, but
> it is in fact precisely the property used in his proof."* — tr.46
>
> *Dịch: Ta nói một hàm $g:\mathbb R^{n}\to\mathbb R$ là khả vi hầu khắp nếu nó liên tục tuyệt
> đối trên hầu hết các đoạn thẳng song song với các trục toạ độ, và các đạo hàm riêng của nó
> (do đó tồn tại hầu khắp nơi) khả tích địa phương. Định nghĩa khả vi hầu khắp này thoạt nhìn
> có vẻ yếu hơn định nghĩa của Stein, nhưng thật ra nó đúng là tính chất được dùng trong chứng
> minh của ông.*

**Hai chỗ ký hiệu không nhất quán trong Phụ lục**, ghi lại cho người đọc khỏi mất thời gian:
(A.2) dùng chữ $w$ ở chỗ mà phần Quy ước tr.41–42 gọi là $x_\ell$, và cùng chữ $w$ ấy xuất
hiện lại ở đoạn chứng minh Lipschitz của Bổ đề 14 (*"we set $\delta = w - x_{k+1}$"*). Đây là
lỗi in, không ảnh hưởng lập luận — nhưng nó **trùng tên với $w_{\mathcal A}$ của (2.6)**, một
đại lượng hoàn toàn khác.

---

## 3. Thuật toán, viết đủ để dựng lại

### 3.1 Sáu bước, không bỏ chi tiết nào

Giả định: $y$ đã trừ trung bình; mỗi cột $x_j$ đã trừ trung bình và chia cho chuẩn Euclid của
chính nó, tức $\lVert x_j\rVert_2 = 1$ — quy ước (1.1). "Độ lệch chuẩn bằng 1" là một quy ước
**khác**, lệch một hệ số $\sqrt{n-1}$.

| # | Làm gì | Công thức của bài báo | Chi phí |
|---|---|---|---|
| 1 | Khởi tạo: $\hat\mu = 0$, $\mathcal A = \varnothing$, $\hat\beta = 0$ | — | $O(1)$ |
| 2 | Tính tương quan hiện tại $\hat c = X^{\top}(y-\hat\mu)$ | (2.8) | $O(nm)$ |
| 3 | Lấy $\hat C = \max_j\lvert\hat c_j\rvert$; đưa mọi $j$ đạt cực đại vào $\mathcal A$; đặt $s_j = \operatorname{sign}(\hat c_j)$ | (2.9), (2.10) | $O(m)$ |
| 4 | Dựng hướng cách đều góc: đổi dấu cột thành $X_{\mathcal A}$, lấy $G_{\mathcal A} = X_{\mathcal A}^{\top}X_{\mathcal A}$, rồi $A_{\mathcal A}$, $w_{\mathcal A}$, $u_{\mathcal A}$, và $a = X^{\top}u_{\mathcal A}$ | (2.4)–(2.7), (2.11) | $O(k^{2} + nk + nm)$ |
| 5 | Tính bước $\hat\gamma$ = nghiệm dương nhỏ nhất trong $2(m-k)$ ứng viên; nếu $\mathcal A$ đã đầy thì lấy $\hat\gamma = \hat C/A_{\mathcal A}$ | (2.13) + quy ước tr.10 | $O(m)$ |
| 6 | Cập nhật $\hat\mu \leftarrow \hat\mu + \hat\gamma u_{\mathcal A}$ và $\hat\beta_{\mathcal A} \leftarrow \hat\beta_{\mathcal A} + \hat\gamma s\odot w_{\mathcal A}$; quay lại bước 2 | (2.12), (3.3) | $O(n + k)$ |

**Sửa đổi lasso** — thêm đúng một khối giữa bước 5 và 6:

> Tính $\hat d_j = s_jw_{\mathcal A j}$ với $j\in\mathcal A$, rồi
> $\gamma_j = -\hat\beta_j/\hat d_j$ và $\tilde\gamma = \min_{\gamma_j>0}\gamma_j$. Nếu
> $\tilde\gamma < \hat\gamma$: lấy bước $\tilde\gamma$ thay cho $\hat\gamma$, đặt
> $\hat\beta_{\tilde j} = 0$ **chính xác**, và bỏ $\tilde j$ khỏi $\mathcal A$. Bước kế tiếp
> **không** cho $\tilde j$ vào lại ngay.

**Sửa đổi Stagewise** — thay bước 4:

> Sau khi có $u_{\mathcal A}$, giải bài toán bình phương tối thiểu **không âm**
> $\min_{P\ge0}\lVert u_{\mathcal A} - X_{\mathcal A}P\rVert$. Đặt
> $\hat{\mathcal B} = \{j : P_j > 0\}$, rồi dựng lại $A_{\hat{\mathcal B}}$,
> $w_{\hat{\mathcal B}}$, $u_{\hat{\mathcal B}}$ trên $\hat{\mathcal B}$ và dùng chúng cho các
> bước sau. $\mathcal A$ co về $\hat{\mathcal B}$.

Bốn chi tiết mà một bản cài đặt thật buộc phải xử lý, và bài báo có nói tới cả bốn:

1. **Chặn mẫu số.** Trong (2.13), chỉ nhận ứng viên có mẫu số dương và thương dương. Một mẫu
   số âm nghĩa là biến đó đang **rời xa** mức chung, không bao giờ bắt kịp.
2. **Bước cuối.** Khi $\mathcal A$ chứa mọi biến thì (2.13) không còn ứng viên nào; quy ước
   tr.10 lấy $\hat\gamma = \hat C/A_{\mathcal A}$, đúng bằng quãng tới nghiệm OLS.
3. **Cấm vào lại ngay sau khi bị bỏ.** Không có ràng buộc này thì biến vừa bị bỏ vẫn có
   $\lvert\hat c_j\rvert = \hat C$ nên (2.9) sẽ nhận nó lại tức thì, và thuật toán quay vòng.
4. **Dừng khi bão hoà.** Với $m > n$, phần dư về 0 sau $n-1$ biến; khi ấy $\hat C \approx 0$
   và phải dừng, nếu không $G_{\mathcal A}$ suy biến.

### 3.2 Mã chạy được, và đầu ra thật

<details>
<summary><b>Bản cài đặt đầy đủ ba biến thể</b> — <code>tools/v_lars.py</code>, khoảng 80 dòng</summary>

```python
import numpy as np
from scipy.optimize import nnls


def chuan(M):
    """Quy ước (1.1): mỗi cột trừ trung bình rồi chia cho CHUẨN EUCLID."""
    M = np.asarray(M, dtype=float)
    M = M - M.mean(0)
    return M / np.sqrt((M ** 2).sum(0))


def huong(X, A, c):
    """(2.4)–(2.6) + (2.11): trả về (s, A_A, w_A, u_A, a) cho tập hoạt động A."""
    s = np.sign(c[A]); s[s == 0] = 1.0
    XA = X[:, A] * s                                   # (2.4)
    GA = XA.T @ XA                                     # (2.5)
    iG = np.linalg.solve(GA, np.ones(len(A)))
    AA = 1.0 / np.sqrt(np.ones(len(A)) @ iG)           # (2.5)
    wA = AA * iG                                       # (2.6)
    uA = XA @ wA                                       # (2.6)
    return s, AA, wA, uA, X.T @ uA                     # (2.11)


def lars(X, y, kieu="lar", eps=1e-12):
    """LARS của Efron 2004; kieu = lar | lasso | stage. Trả về (B, nhật ký)."""
    n, m = X.shape
    beta, mu, A, B, sk = np.zeros(m), np.zeros(n), [], [np.zeros(m)], []
    vua_bo = -1
    nguong_het = 1e-9 * max(1.0, np.abs(X.T @ y).max())
    while True:
        c = X.T @ (y - mu)                             # (2.8)
        if A and max(abs(c[j]) for j in A) < nguong_het:
            break                                      # bão hoà: phần dư đã trực giao
        ngoai = [j for j in range(m) if j not in A and j != vua_bo]
        if ngoai:
            jb = max(ngoai, key=lambda j: abs(c[j]))
            if not A or abs(c[jb]) >= max(abs(c[j]) for j in A) - 1e-9:
                A = A + [jb]                           # (2.9)
                sk.append(("vao", jb, len(B) - 1))
        vua_bo = -1
        if not A:
            break
        s, AA, wA, uA, a = huong(X, A, c)
        C = max(abs(c[j]) for j in A)

        if kieu == "stage":                            # chiếu u_A vào nón lồi (3.12)
            P = nnls(X[:, A] * s, uA)[0]
            Bh = [A[i] for i in range(len(A)) if P[i] > 1e-10]
            if len(Bh) < len(A):
                sk.append(("bo", [A[i] for i in range(len(A)) if P[i] <= 1e-10],
                           len(B) - 1, sorted(A)))
            s, AA, wA, uA, a = huong(X, Bh, c)
            A = Bh

        con = [j for j in range(m) if j not in A]
        ung = [t / mm for j in con
               for t, mm in ((C - c[j], AA - a[j]), (C + c[j], AA + a[j]))
               if mm > eps and t / mm > eps]           # (2.13)
        gam = min(ung) if ung else C / AA              # quy ước bước cuối, tr.10

        d = np.zeros(m); d[A] = s * wA                 # (3.3)
        bo = None
        if kieu == "lasso":
            gt, jt = np.inf, None
            for j in A:
                if abs(d[j]) > eps:
                    g = -beta[j] / d[j]                # (3.4)
                    if eps < g < gt:
                        gt, jt = g, j                  # (3.5)
            if gt < gam:
                gam, bo = gt, jt                       # (3.6)

        beta = beta + gam * d
        mu = mu + gam * uA                             # (2.12)
        if bo is not None:
            beta[bo] = 0.0
            A = [j for j in A if j != bo]
            sk.append(("bo", [bo], len(B)))
            vua_bo = bo
        B.append(beta.copy())
        if not con and bo is None:
            break
    return np.array(B).T, sk
```

</details>

Chạy nó trên bộ dữ liệu của bài báo, rồi đối chiếu với `scikit-learn`:

```
   LARS   : tự viết 10 bước · sklearn 10 bước · lệch hệ số lớn nhất 7.885e-11
   Lasso  : tự viết 12 bước · sklearn 12 bước · lệch hệ số lớn nhất 1.708e-10
   Bài báo tr.12: *12 instead of 10 in this example* -> KHỚP cả hai con số.

   Cột cuối của đường LARS có đúng bằng nghiệm OLS không? lệch 7.999e-11
   Cột cuối của đường lasso có đúng bằng nghiệm OLS không? lệch 8.959e-11
```

Hai bản cài đặt hoàn toàn độc lập cho cùng đường tới $10^{-10}$, và cả hai kết thúc đúng ở
nghiệm bình phương tối thiểu.

### 3.3 Bản cập nhật Cholesky — cách §7 nói phải làm

Bản ở trên giải lại hệ $k\times k$ **từ đầu** ở mỗi bước, nên tốn $O(k^{3})$ mỗi bước và
$O(m^{4})$ cả đường. §7 nói cách đúng là **cập nhật phân tích Cholesky**, tốn $O(k^{2})$ mỗi
bước. Hai chỗ phải sửa, và chỉ hai:

1. **Giữ sẵn một vùng đệm $n\times m$** cho các cột đã đổi dấu, rồi mỗi bước chỉ **thêm một
   cột** thay vì dựng lại $X_{\mathcal A}$. Làm được vì bài báo nói ở tr.9: *"The sign $s_j$ of
   each $x_j$ in (2.4) stays constant as the active set increases"* — **dấu không đổi trong
   LARS gốc**.
2. **Cập nhật $R$ hạng 1**: giải $R^{\top}w = X_{\mathcal A}^{\top}x_{\text{mới}}$, lấy
   $\rho = \sqrt{\lVert x_{\text{mới}}\rVert^{2} - \lVert w\rVert^{2}}$, rồi viền $R$ thành
   $\begin{pmatrix}R & w\\ 0 & \rho\end{pmatrix}$.

Đo hiệu quả của đúng hai chỗ đó, trên $n = 20\,000$:

```
      n      | m    | LARS Cholesky | LARS giải lại  | OLS đầy đủ | LARS/OLS
      20000 |   40 |     0.0591 s  |     0.1024 s  |   0.0107 s |    5.5 lần
      20000 |   80 |     0.2268 s  |     0.5357 s  |   0.0395 s |    5.7 lần
      20000 |  160 |     0.8180 s  |     2.9585 s  |   0.1305 s |    6.3 lần
      (hai bản LARS cho cùng đường: lệch 2.93e-14)
```

**Khoảng 3,6 lần nhanh hơn ở $m = 160$, và hai bản cho cùng một đường tới
$2{,}93\times10^{-14}$.**
Câu thứ hai quan trọng hơn câu thứ nhất: một phép tăng tốc không kèm phép so với đường tính
cũ thì không phải một phép tăng tốc, nó là một thuật toán khác.

---

## 4. Vì sao "cách đều góc" là hướng ĐÚNG

Ba mục trước đã nói LARS **làm gì**. Mục này trả lời câu *"vì sao hướng đó, chứ không phải
hướng khác"* — và câu trả lời có ba tầng, mỗi tầng một phép đo.

### 4.1 Tầng một: nó giữ mọi biến hoạt động ở cùng một mức

Đây là tầng dễ thấy nhất, và nó là **định nghĩa** chứ không phải kết quả: (2.7) đòi
$X_{\mathcal A}^{\top}u_{\mathcal A} = A_{\mathcal A}\mathbf 1$, nên theo (2.15) mọi tương
quan hoạt động tụt cùng tốc độ. Hệ quả: **không biến nào trong tập bị bỏ rơi, và không biến
nào được ưu ái.**

So với hai đối thủ thì thấy ngay chỗ khác:

| Thuật toán | Hướng đi | Điều gì xảy ra với tương quan |
|---|---|---|
| Forward Selection | dọc **một** cột, đi hết cỡ | tương quan của cột đó về **0**; các cột khác bị kéo theo một cách không kiểm soát |
| Forward Stagewise | dọc **một** cột, đi một nhịp $\varepsilon$ | tương quan cột đó giảm một chút; lặp lại nhiều lần thì trung bình ra một hướng ở giữa |
| **LARS** | dọc **hướng cách đều góc** | mọi tương quan hoạt động giảm **cùng nhau, cùng tốc độ $A_{\mathcal A}$** |

Nói cách khác, LARS là **giới hạn của Forward Stagewise khi $\varepsilon\to0$, viết ra thành
công thức đóng**. Hình 18 đo chính điều đó: lệch giữa hai đường tỉ lệ thuận với $\varepsilon$.

### 4.2 Tầng hai: nó là hướng làm $S$ giảm được LÂU nhất

Tầng một chưa đủ, vì "giữ mọi thứ bằng nhau" tự nó không phải một mục tiêu. Tầng hai là chỗ
§5 trả lời, qua (5.38) và (5.43), và nó tinh hơn hẳn:

$$\Delta S = -2\hat C\,\Delta T + L(d)^{2}(\Delta T)^{2}$$

Công thức này nói **đi thêm một lượng ngân sách thì $S$ giảm bao nhiêu**, và nó tách phần
giảm ấy thành hai số hạng có vai trò hoàn toàn khác nhau.

Đọc công thức này theo hai số hạng:

* **Số hạng bậc nhất $-2\hat C\Delta T$ GIỐNG NHAU cho mọi hướng hợp lệ.** Bổ đề 10 nói
  $R(d)\le2\hat C$ với dấu bằng xảy ra cho mọi hướng chỉ dùng biến trong $\mathcal A_{10}$.
  Nên **xét ở tốc độ tức thời, có rất nhiều hướng ngang nhau**.
* **Số hạng bậc hai thì khác nhau**, và nó là chỗ phân thắng bại. $L(d)$ nhỏ nhất bằng
  $A_{\mathcal A}$, đạt được **đúng tại hướng cách đều góc** (Bổ đề 5).

Nghĩa là: **hướng cách đều góc không thắng vì nó xuống dốc nhanh hơn lúc đầu — nó thắng vì nó
giữ được độ dốc ấy lâu hơn.** Đó là một phát biểu về đạo hàm **cấp hai**, và nó giải thích vì
sao trực giác "đi theo biến mạnh nhất" lại sai: đi theo một biến thì tốc độ ban đầu bằng nhau,
nhưng nó tắt rất nhanh.

Phép đo 20 000 hướng ngẫu nhiên ở mục 2.9 chính là phép kiểm của tầng này: không hướng nào
vượt trần $2\hat C$, và hướng cách đều góc chạm đúng trần.

### 4.3 Tầng ba: nó là hướng khiến ba thuật toán gặp nhau

Tầng này là đóng góp thật của bài báo, và nó không phải một tính chất tối ưu mà là một **quan
hệ**:

| Thêm gì vào LARS | Được gì | Chứng minh ở |
|---|---|---|
| ràng buộc "hệ số không đổi dấu" | **đường lasso** | §5, Định lý 1 |
| ràng buộc "hướng đi nằm trong nón lồi" | **đường forward stagewise** | §6, Định lý 2 |
| không thêm gì | LARS gốc | — |

Và khi **điều kiện nón dương (4.11) thoả** thì cả hai ràng buộc tự động đúng, nên ba đường
**trùng nhau**. Đo lại: 6/6 thiết kế ngẫu nhiên thoả (4.11) cho ba đường trùng tới $10^{-8}$;
còn bộ diabetes — vi phạm (4.11) ở 418/1023 tập con — cho ba đường **10, 12 và 13 bước**.

Đó là câu trả lời gọn nhất cho *"vì sao hướng cách đều góc"*: **nó là hướng chung mà cả ba
thuật toán rút về khi bỏ hết ràng buộc riêng đi.** Lasso và Stagewise không phải hai thứ khác
LARS; chúng là LARS **cộng một ràng buộc**, và ràng buộc ấy chỉ cắn khi hình học của dữ liệu
đủ xấu.

### 4.4 Một chỗ trực giác dễ sai, nói thẳng ra

Trực giác hay gặp: *"LARS chậm hơn Forward Selection vì nó đi từng bước nhỏ hơn"*. Sai ở chỗ
**"nhỏ hơn"**. Cột $\rho_k$ ở mục 2.2 cho thấy LARS đi từ **6,33 %** tới **72,59 %** quãng
đường mà Forward Selection đi — tức độ "thận trọng" thay đổi rất mạnh theo bước, và nó không
phải một hệ số cố định ai đặt ra. Nó là **hệ quả của hình học**: bước dài bao nhiêu phụ thuộc
biến ngoài gần nhất còn cách bao xa.

Trực giác thứ hai: *"LARS là một thuật toán tối ưu, nó cực tiểu hoá một hàm mục tiêu"*. Cũng
sai. LARS gốc **không** cực tiểu hoá hàm nào — nó là một **thủ tục hình học**. Cái cực tiểu
hoá một hàm mục tiêu là **lasso**, và mối liên hệ giữa hai thứ là nội dung Định lý 1, tức một
kết quả phải chứng minh chứ không phải một định nghĩa.

---

## 5. Giá phải trả, và khi nào đừng dùng

### 5.1 Bảng giá bốn cột

> | Mặt được | Đo bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
> |---|---|---|---|
> | Toàn bộ đường nghiệm trong $m$ bước | 356 310 bước Stagewise → **10 bước** | mỗi bước phải giải một hệ $k\times k$; tổng chi phí **1,98 lần** một lần khớp OLS | mục 1.2 · 2.11 |
> | Các điểm gãy **chính xác**, không phải xấp xỉ | lệch với nghiệm OLS ở đầu mút: $8{,}0\times10^{-11}$ | **nhạy với suy biến**: $G_{\mathcal A}$ gần suy biến thì hướng đi mất nghĩa | mục 3.2 · 5.2 |
> | Cho luôn đường lasso | thêm **đúng một** quy tắc; 12 bước thay vì 10 | đường lasso **không** dùng được công thức $C_p$ (4.10) | mục 2.3 · 2.7 |
> | Cho luôn đường stagewise | thêm một phép chiếu NNLS; 13 bước | Stagewise tốn **3,92 lần** số bước của LARS trên $m = 64$ | mục 2.4 · 2.5 |
> | Ước lượng bậc tự do miễn phí | $\operatorname{df}\approx k$, **không tốn thêm phép tính nào** | là **xấp xỉ**; hụt có ý nghĩa ở $k = 6,7,8$ trên chính bộ diabetes vì thiết kế vi phạm (4.11) ở **40,86 %** tập con | mục 2.7 |
> | Chọn mô hình bằng $C_p$ | khớp đúng $k = 7$ với $m = 10$ | với $m = 64$, cực tiểu đo được ở $k = 15$ chứ không phải 16; và $C_p$ nhạy với cách ước lượng $\bar\sigma^{2}$ | mục 2.7 |
> | Thay được kiểm định chéo | **KHÔNG** | LARS làm rẻ việc **dựng** đường, không làm rẻ việc **chọn** điểm | mục 5.3 |
> | Là lựa chọn mặc định hôm nay | **không còn** | hạ theo toạ độ với khởi động ấm rẻ hơn cho một lưới $\lambda$ dày | mục 6 |

### 5.2 Năm chỗ hỏng, có số

**1. Suy biến làm hướng đi mất nghĩa.** $A_{\mathcal A}$ được định nghĩa qua
$G_{\mathcal A}^{-1}$. Hai cột trùng nhau thì $G_{\mathcal A}$ suy biến và không có hướng cách
đều góc nào cả. Với $m \gg n$, chuyện này xảy ra **chắc chắn** khi tập hoạt động đạt $n-1$
phần tử — và đó là lý do bản cài đặt ở mục 3.2 phải có điều kiện dừng theo $\hat C$.

**2. `scikit-learn` chế độ `lar` phân kỳ trên chính ví dụ thứ hai của bài báo.** Đây là chỗ
đáng cảnh báo nhất cho người đọc hôm nay:

```
   `sklearn.linear_model.lars_path(method='lar')` trên CHÍNH thiết kế này:
      số bước: 96 (thay vì 64) | Σ|β̂| ở bước cuối: 8.3768e+14 | RSS ở bước cuối: 4.3742e+22
      RSS của nghiệm OLS: 1068217.7577 ; RSS của bản tự viết ở bước 64: 1068217.7577
      tại bước 64, sklearn mới có 51 hệ số khác 0 và RSS = 20424024.5913
      dãy alpha của sklearn có giảm đơn điệu không? KHÔNG
      -> chế độ 'lar' của sklearn PHÂN KỲ trên thiết kế này, và không phát cảnh báo nào.
      Chế độ 'lasso' thì không sao: 104 bước, lệch với bản tự viết 6.90e-06
```

Bốn điều phải đọc cùng nhau:

* Bản viết từ đầu theo (2.4)–(2.13) cho **64 bước** và RSS ở bước cuối **bằng đúng RSS của
  nghiệm OLS**, tới từng chữ số.
* `sklearn` chạy **96 bước**; tại bước 64 nó mới có **51** hệ số khác 0 và RSS lớn hơn **19
  lần**; tới bước cuối thì $\sum_j\lvert\hat\beta_j\rvert$ lên tới $8{,}4\times10^{14}$.
* Dãy `alphas` trả về **không giảm đơn điệu** — một dấu hiệu rõ ràng rằng thuật toán đã ra
  khỏi miền hợp lệ.
* **Không có một cảnh báo nào được phát ra.** `grep -c UserWarning` trên đầu ra: 0.

Ma trận thiết kế này không phải một ca bệnh lý dựng riêng để phá: nó là **mô hình bậc hai của
chính bài báo**, số điều kiện $5{,}47\times10^{3}$ — không lớn. Và chế độ `lasso` của cùng
hàm thì chạy đúng. Nên kết luận thực hành: **nếu cần đường LARS gốc trên một thiết kế có
tương quan cao, phải kiểm lại đầu ra chứ không tin thẳng.** Phép kiểm rẻ nhất là hai dòng: so
RSS ở bước cuối với RSS của nghiệm OLS, và kiểm `alphas` có giảm đơn điệu không.

**3. Thứ tự biến vào không ổn định khi các cột gần trùng nhau.** Một nhiễu nhỏ ở $y$ đổi được
biến nào vào trước. Bài báo tự nói ở tr.40: *"The model sequence, particularly near the
saturated end, tends to be quite variable with respect to small changes in $y$."* Hệ quả: dùng
**thứ tự** các biến làm kết luận khoa học là mong manh; dùng **tập** biến ở điểm được chọn
bằng $C_p$ hoặc CV thì chắc hơn.

**4. Điều kiện "mỗi lần một biến" là giả thiết của Định lý 1, không phải một chi tiết.** Nếu
hai biến cùng bắt kịp thì phải xét cả hai khả năng và chọn cái cho $A_{\mathcal A}$ nhỏ hơn
(tr.33–34) — và bài báo nói thẳng chương trình đi kèm **không cài** phần đó. Cách chữa mà
chính bài báo đề nghị là thêm chút nhiễu vào $y$.

**5. Công thức $C_p$ (4.10) CHỈ áp cho LARS.** Không áp cho lasso, không áp cho stagewise.
Đây là chỗ dễ dùng sai nhất, vì trong thực hành người ta chạy `method="lasso"` rồi vẫn lấy
$k$ = số bước để tính $C_p$.

### 5.3 Quy trình dùng an toàn

1. **Chuẩn hoá đúng quy ước (1.1)**, và chuẩn hoá **bên trong từng mảnh** của kiểm định chéo,
   không phải một lần trên cả dữ liệu.
2. Chạy `lars_path(..., method="lasso")` nếu cần đường lasso; `method="lar"` cho LARS gốc —
   và **kiểm đầu ra** theo hai dòng ở mục 5.2.
3. **Chọn điểm dừng bằng kiểm định chéo**, hoặc bằng $C_p$ (4.10) **nếu và chỉ nếu** đang
   dùng LARS gốc và có ước lượng $\bar\sigma^{2}$ đáng tin.
4. Khớp lại trên train + validation tại điểm đã chọn, rồi báo lỗi test **đúng một lần**.
5. Nếu mục tiêu là **diễn giải tập biến**, lặp lại trên bootstrap và báo tần suất mỗi biến
   được chọn — đừng báo một tập biến duy nhất.

---

## 6. So với họ hàng gần

| Thuật toán | Trả về gì | Chi phí | Khi nào nó thắng |
|---|---|---|---|
| **LARS** | **cả đường**, chính xác từng điểm gãy, trong $m$ bước | $\approx$ **1,98 lần** một lần khớp OLS | cần biết các điểm gãy nằm ở đâu; $m < n$ và $m$ vừa phải |
| **Lasso qua LARS** | cả đường lasso | 12 bước thay vì 10 trên bộ diabetes; 104 thay vì 64 trên $m = 64$ | cần đường lasso **chính xác** |
| **Lasso qua hạ theo toạ độ** | nghiệm tại từng $\lambda$ trên một lưới | rẻ hơn hẳn khi lưới dày và dùng khởi động ấm | dùng thực tế hàng ngày; $m$ rất lớn |
| **Forward Stagewise** | đường xấp xỉ, phụ thuộc $\varepsilon$ | 356 310 bước cho 90 % đường ở $\varepsilon = 0{,}01$ | khi tập biến **vô hạn** (ví dụ: mọi cây) nên không giải được bước tối ưu |
| **Forward Selection** | dãy mô hình lồng nhau | rẻ nhất | khi cần đúng một mô hình nhỏ và chấp nhận rủi ro tham |
| **Ridge** | đường trơn theo $\lambda$ | một lần SVD cho cả đường | khi muốn **giữ hết** biến và chỉ co hệ số |
| **Elastic Net** | đường tuyến tính từng khúc theo $\lambda_1$ | cùng họ với LARS-EN | khi có nhóm biến tương quan cao cần vào cùng nhau |

Ba chỗ so cần nói thẳng:

**LARS so với hạ theo toạ độ.** Đây là chỗ người đọc hôm nay cần câu trả lời rõ. LARS tính
**các điểm gãy** — những giá trị $\lambda$ mà tại đó tập biến đổi. Hạ theo toạ độ tính
**nghiệm tại những $\lambda$ mà người dùng chọn**. Nếu câu hỏi là *"cho tôi 100 mô hình trên
một lưới log"* thì hạ theo toạ độ với khởi động ấm rẻ hơn, và đó là lý do `glmnet` và
`sklearn.linear_model.lasso_path` mặc định dùng nó. Nếu câu hỏi là *"biến thứ năm vào ở đúng
giá trị nào"* thì chỉ LARS trả lời được chính xác.

**LARS so với Forward Stagewise, sau khi đã đọc §8.** Với tập biến hữu hạn, LARS thắng tuyệt
đối: cùng một đường, ít bước hơn bốn bậc độ lớn. Nhưng với tập biến **vô hạn** — mọi cây có
thể khớp — thì (2.13) không tính được, vì không duyệt hết được $\mathcal A^{c}$. Forward
Stagewise vẫn chạy, vì nó chỉ cần **tìm biến tương quan nhất**, và đó là việc một thuật toán
dựng cây làm được. Đó là lý do boosting dùng stagewise chứ không dùng LARS.

**LARS so với chính nó ở hai chế độ.** `method="lar"` và `method="lasso"` cho hai đường khác
nhau, lệch hệ số tới **50,6412** trên thang 792. Chúng không thay thế nhau được, và chọn nhầm
là một lỗi im lặng.

---

## 7. Dùng vào việc gì

**1. Dựng đường nghiệm cho bài toán chọn biến có $m$ vừa phải.** Đây là công dụng trực tiếp,
và là thứ bài báo viết ra để làm. Bộ diabetes 442×10 là ví dụ chuẩn: mười bước cho toàn bộ
câu chuyện.

**2. Biết CHÍNH XÁC biến nào vào ở đâu.** Khi câu hỏi khoa học là *"cần bao nhiêu biến thì mô
hình đủ tốt"*, đường LARS cho câu trả lời dưới dạng một danh sách sự kiện có thứ tự, kèm giá
trị $\hat C$ tại mỗi sự kiện. Không thuật toán nào khác trong họ cho danh sách ấy mà không
phải quét lưới.

**3. Chọn mô hình bằng $C_p$ mà không tốn thêm một phép tính nào.** (4.10) chỉ cần RSS trên
đường — thứ đã có sẵn. Trên bộ diabetes, nó chọn bảy biến `[2, 3, 4, 5, 7, 9, 10]`.

**4. Làm cầu nối để hiểu boosting.** §8 là chỗ bài báo nói ra mối liên hệ, và mục 2.12 đo nó ở
quy mô nhỏ. Với ai đã biết boosting mà chưa biết lasso, đây là đường vào ngắn nhất.

**5. Là bài kiểm chứng cho một bản cài đặt lasso khác.** Đường LARS-lasso cho nghiệm **chính
xác** tại các điểm gãy, nên nó là chuẩn vàng để kiểm một bộ giải lặp. Mục 5.2 cho thấy chuyện
ấy không thừa: một thư viện lớn vẫn có thể phân kỳ ở một chế độ mà vẫn im lặng.

**Chỗ KHÔNG nên dùng:** $m$ rất lớn với lưới $\lambda$ dày; bài toán không phải bình phương
tối thiểu (hồi quy logistic, Cox, Poisson) — hình học tuyến tính từng khúc dựa vào hàm mất
mát bình phương, và không thay được $y - X\beta$ bằng một phần dư khác trong mã LARS; và mọi
tình huống mà điều duy nhất cần là **một** mô hình tốt.

---

## 8. Đối chiếu với bài báo gốc

Luật 6 đòi hai bảng, vì chúng trả lời hai câu hỏi khác nhau: bảng thứ nhất hỏi *"đã lấy hết
hình và bảng chưa"*, bảng thứ hai hỏi *"đã lấy hết NỘI DUNG chưa"*. Một bài có thể lấy đủ
100 % hình mà vẫn bỏ sót cả một mục, vì phần lớn lập luận của một bài báo nằm trong văn xuôi.

Bài này thêm một bảng thứ ba, vì nguồn có **145 công thức đánh số** — nhiều gấp bảy lần một
bài kiến trúc bình thường.

### 8.1 Bảng đối chiếu theo HÌNH và BẢNG — 10 mục, lấy 10

| Hình/Bảng của bài báo | Trang | Nội dung | Vào bài ở đâu | Cách xử lý |
|---|---|---|---|---|
| **Bảng 1** | 3 | dữ liệu 442 bệnh nhân, 10 biến | **Hình 1** | cắt vào bài; dựng lại 8 dòng in ra; phát hiện bảng là bản đã làm tròn |
| **Hình 1** | 4 | đường Lasso và Stagewise trên diabetes | **Hình 5** | cắt vào bài; dựng lại cả hai đường; đo lại câu *"differ … in covariate 8"* bằng ba cách |
| **Hình 2** | 6 | hình học LARS với $m = 2$ | **Hình 6** | cắt vào bài; ví dụ chạy tay mục 0.4 là đúng tình huống này |
| **Hình 3** | 9 | đường LARS gốc và tương quan theo bước | **Hình 11** | cắt vào bài; dựng lại; Hình 9 vẽ lại bảng phải bằng thang log |
| **Hình 4** | 10 | $\hat\mu_k$ tiến gần nhưng không tới $\bar y_k$ | **Hình 12** | cắt vào bài; đo $\rho_k$ cả mười bước |
| **Hình 5** | 15 | mô phỏng 100 lần lặp, bốn thuật toán | **Hình 20** | cắt vào bài; dựng lại toàn bộ (Hình 21); bốn con số đối chiếu |
| **Hình 6** | 19 | bậc tự do, hai mô hình, $B = 500$ | **Hình 28** | cắt vào bài; dựng lại bằng đúng (4.6)–(4.8) (Hình 30) |
| **Hình 7** | 20 | $C_p$ theo $k$, hai mô hình | **Hình 29** | cắt vào bài; dựng lại; $k = 7$ khớp, $k = 16$ đo ra 15 |
| **Hình 8** | 30 | đường $(T, S)$ của lasso | **Hình 43** | cắt vào bài; dựng lại và kiểm cả $\dot S$ lẫn $\ddot S$ (Hình 44) |
| **Hình 9** | 37 | hình học nón lồi của sửa đổi Stagewise | **Hình 47** | cắt vào bài; kiểm Ràng buộc II và III tại đúng chỗ chúng kích hoạt |

**10/10 — không bỏ mục nào.**

### 8.2 Bảng đối chiếu theo MỤC — 8 mục đánh số + phụ lục, lấy đủ

| Mục của bài báo | Trang | Nội dung | Vào bài ở đâu | Vì sao không lấy (nếu có) |
|---|---|---|---|---|
| **1. Introduction** | 2–5 | bài toán, Bảng 1, (1.1)–(1.7), Hình 1 | mục 2.1 | — |
| **2. The LARS algorithm** | 5–10 | (2.1)–(2.22), Hình 2, 3, 4 | mục 2.2 | — |
| **3. Modified versions** | 11 | mở đầu §3 | mục 2.3 | — |
| **3.1 The LARS–Lasso relationship** | 11–12 | (3.1)–(3.6), Định lý 1 | mục 2.3 | — |
| **3.2 The LARS–Stagewise relationship** | 12–14 | (3.7)–(3.14), Định lý 2 | mục 2.4 | — |
| **3.3 Simulation study** | 14–16 | (3.15)–(3.17), Hình 5 | mục 2.5 | — |
| **3.4 Other LARS modifications** | 16–17 | (3.18)–(3.20), positive lasso, hybrid, main-effects-first, backward lasso | mục 2.6 | — |
| **4. Degrees of freedom and $C_p$** | 17–21 | (4.1)–(4.15), Hình 6, 7, Định lý 3, 4 | mục 2.7 | — |
| **4.1 Orthogonal designs** | 21–23 | (4.16)–(4.18), Bổ đề 1 | mục 2.8 | — |
| **4.2 The divergence formula** | 23–25 | (4.19)–(4.24), Bổ đề 2 | mục 2.8 | — |
| **4.3 Proof of Theorem 4** | 25 | Bổ đề 3 | mục 2.8 | — |
| **5. LARS and Lasso properties** | 25–34 | (5.1)–(5.44), Bổ đề 4–10, bốn ràng buộc, Hình 8 | mục 2.9 | — |
| **6. Stagewise properties** | 34–39 | (6.1)–(6.22), Bổ đề 11–12, ba ràng buộc, Hình 9 | mục 2.10 | — |
| **7. Computations** | 39–40 | chi phí $O(m^3+nm^2)$, Cholesky, $m\gg n$ | mục 2.11 · 3.3 | — |
| **8. Boosting procedures** | 40–41 | liên hệ với least squares boosting | mục 2.12 | — |
| **Appendix A.1** | 41–44 | (A.1)–(A.4), Bổ đề 13, 14 | mục 2.13 | — |
| **Appendix A.2** | 44–45 | (A.5)–(A.6), Bổ đề 15 | mục 2.13 | — |
| **Appendix A.3** | 45–47 | Bổ đề 16, 17, Ghi chú A.1, chứng minh Bổ đề 3 | mục 2.13 | — |
| **References** | 47 | 20 tài liệu | mục 9 | trích những tài liệu bài này thật sự dùng |

**18/18 mục có nội dung đều vào bài.**

### 8.3 Bảng đối chiếu theo CÔNG THỨC — 145 công thức, viết ra 145

| Nhóm | Số hiệu | Bao nhiêu | Mức xử lý | Ở mục |
|---|---|---:|---|---|
| §1 Introduction | (1.1)–(1.7) | 7 | đủ ba phần | 2.1 |
| §2 Thuật toán | (2.1)–(2.22) | 22 | đủ ba phần | 2.2 |
| §3.1 Lasso | (3.1)–(3.6) | 6 | đủ ba phần | 2.3 |
| §3.2 Stagewise | (3.7)–(3.14) | 8 | đủ ba phần | 2.4 |
| §3.3 Mô phỏng | (3.15)–(3.17) | 3 | đủ ba phần | 2.5 |
| §3.4 Biến thể | (3.18)–(3.20) | 3 | đủ ba phần, kèm suy lại (3.20) | 2.6 |
| §4 df và $C_p$ | (4.1)–(4.15) | 15 | đủ ba phần, kèm suy lại (4.13) | 2.7 |
| §4.1–4.2 | (4.16)–(4.24) | 9 | (4.16)–(4.19) đủ ba phần; (4.20)–(4.24) bảng gom của chứng minh Bổ đề 2 | 2.8 |
| §5 | (5.1)–(5.44) | 44 | (5.7)–(5.8), (5.12)–(5.17), (5.28)–(5.31), (5.37)–(5.39) đủ ba phần; còn lại trong bốn bảng gom theo từng bổ đề | 2.9 |
| §6 | (6.1)–(6.22) | 22 | (6.3)–(6.7), (6.9), (6.19)–(6.22) đủ ba phần; còn lại trong bảng gom Bổ đề 12 | 2.10 |
| Phụ lục | (A.1)–(A.6) | 6 | bảng gom, mỗi dòng một công thức kèm "nó nói gì" | 2.13 |
| **Tổng** | | **145** | | |

> **Chính sách đã khai ở đầu mục 2, nhắc lại ở đây.** Công thức **phát biểu phương pháp hoặc
> một kết quả** được viết ra kèm đủ ba phần: công thức · mọi ký hiệu kèm chiều · ý nghĩa trừu
> tượng. Công thức chỉ là **một bước biến đổi bên trong chứng minh** nằm trong bảng gom của
> chính chứng minh đó, mỗi dòng một công thức, kèm cột *"nó nói gì"* và *"dùng ở đâu"*. Cả
> 145 số hiệu đều xuất hiện trong bài; phép kiểm tự động ở `tools/_work_lars/ghep.py` đọc
> danh sách số hiệu từ lớp text của PDF rồi đối chiếu với bài, và nó **gãy** nếu thiếu một số.

### 8.4 Bốn định lý và mười bảy bổ đề

| | Nội dung | Ở mục | Kiểm được không |
|---|---|---|---|
| **Định lý 1** | sửa đổi lasso cho ra mọi nghiệm lasso | 2.3 | dựng lại 12 bước, biến 7 rời rồi vào lại — khớp từng chi tiết |
| **Định lý 2** | sửa đổi Stagewise cho ra mọi nghiệm Stagewise | 2.4 | dựng lại 13 bước, bỏ đúng $\{3,7\}$ — khớp |
| **Định lý 3** | thiết kế trực giao $\Rightarrow$ $\operatorname{df} = k$ | 2.7 | bootstrap $B = 2000$, lệch lớn nhất 0,1238 |
| **Định lý 4** | điều kiện nón dương $\Rightarrow$ $\operatorname{df} = k$ | 2.7 | kiểm gián tiếp: bộ diabetes **vi phạm** (4.11) và df hụt có ý nghĩa ở $k=6,7,8$ |
| **Bổ đề 1** | trực giao $\Rightarrow$ ngưỡng mềm | 2.8 | lệch $1{,}78\times10^{-15}$ |
| **Bổ đề 2** | $\nabla\cdot\hat\mu_k = k$ hầu khắp | 2.8 | qua $\operatorname{trace}(M_k) = k$, đúng 10 chữ số |
| **Bổ đề 3** | liên tục và khả vi hầu khắp dưới (4.11) | 2.8 · 2.13 | không kiểm số được — là phát biểu về độ đo |
| **Bổ đề 4** | biến mới vào theo chiều đúng | 2.9 | 10/10 bước khớp |
| **Bổ đề 5** | $A_{\mathcal A}$ = khoảng cách tới đơn hình; đơn điệu | 2.9 | lệch $\le4{,}44\times10^{-16}$; **0/5110** cặp vi phạm đơn điệu |
| **Bổ đề 6** | $S(\gamma)$ là tam thức bậc hai | 2.9 | lệch $\le2{,}63\times10^{-10}$ |
| **Bổ đề 7** | đường lasso đi thẳng theo $u_{\mathcal A}$ | 2.9 | lệch $\le2{,}84\times10^{-14}$ ở 8 điểm thử |
| **Bổ đề 8** | dấu hệ số = dấu tương quan | 2.9 | 0/4001 điểm trên lưới vi phạm |
| **Bổ đề 9** | tương quan hoạt động bằng nhau và lớn nhất | 2.9 | độ trải $1{,}49\times10^{-12}$ |
| **Bổ đề 10** | $R(d)\le2\hat C$ | 2.9 | 20 000 hướng ngẫu nhiên, không hướng nào vượt |
| **Bổ đề 11** | chỉ biến trong $\mathcal A$ mới được chọn | 2.10 | hệ quả của (6.1), kiểm qua Ràng buộc I |
| **Bổ đề 12** | chỉ $v_{\hat{\mathcal B}}$ thoả cả ba ràng buộc | 2.10 | kiểm (6.6), (6.7), (6.9) tại bước 7 — khớp |
| **Bổ đề 13–17** | tính trơn của $\hat\mu_k$ theo $y$ | 2.13 | phát biểu về độ đo; kiểm gián tiếp qua sai phân hữu hạn ở mục 2.7 |

---

## 9. Tài liệu tham khảo

```
[1] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression",
    The Annals of Statistics 32(2):407–451, 2004.  arXiv:math/0406456v2
    -> NGUỒN RUỘT. Mọi hình, bảng, công thức và câu trích trong mục 2 lấy từ đây.
       Bản trong repo là bản arXiv 48 trang, KHÔNG gồm phần thảo luận của
       The Annals of Statistics.
    AI algorithms/_papers/2004 - Efron et al - Least Angle Regression.pdf

[2] R. Tibshirani, "Regression shrinkage and selection via the lasso",
    J. Roy. Statist. Soc. Ser. B 58(1):267–288, 1996.
    -> Bài định nghĩa lasso, tức bài toán (1.5). Bài viết về nó:
       AI algorithms/ai_algorithms_database/Lasso Regression.md
    AI algorithms/_papers/1996 - Tibshirani - Regression Shrinkage and Selection via the Lasso.pdf
    (bản JSTOR, KHÔNG truy cập mở — chỉ có trên máy dựng bài, không đẩy lên repo)

[3] J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized
    Linear Models via Coordinate Descent", J. Statistical Software 33(1), 2010.
    -> Đối thủ hiện đại của LARS, so ở mục 6.
    AI algorithms/_papers/2010 - Friedman, Hastie, Tibshirani - Regularization Paths for GLM via Coordinate Descent (glmnet).pdf

[4] H. Zou, T. Hastie, "Regularization and variable selection via the elastic net",
    J. R. Statist. Soc. B 67(2):301–320, 2005.
    -> Đường nghiệm cùng họ; mục 6. Bài viết về nó:
       AI algorithms/ai_algorithms_database/Elastic Net.md
    AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf

[5] M. Osborne, B. Presnell, B. Turlach, "A new approach to variable selection in
    least squares problems", IMA J. Numer. Anal. 20:389–403, 2000.
    -> Phương pháp đồng luân mà §3.1 và §5 của [1] nói là trùng lặp một phần.
       KHÔNG có bản truy cập mở; bài này chỉ nhắc theo đúng cách [1] nhắc.

[6] C. Mallows, "Some comments on Cp", Technometrics 15:661–675, 1973.
    -> Nguồn của tiêu chí (4.5). KHÔNG có bản truy cập mở trong repo.

[7] C. Stein, "Estimation of the mean of a multivariate normal distribution",
    Ann. Statist. 9:1135–1151, 1981.
    -> Nguồn của công thức (4.12). KHÔNG có bản truy cập mở trong repo.

[8] T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical Learning",
    Springer, 2001 (và bản 2009).
    -> [1] dẫn nó ở §1, §3.2 và §8. Bài này dùng nó gián tiếp qua bài Lasso và
       bài Elastic Net.
    AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf
```

**Hình gốc.** 43 khung cắt từ `_papers/2004 - Efron et al - Least Angle Regression.pdf`, bằng
`tools/pdffig.py`, mỗi khung ghi rõ số trang trong caption.

**Hình tự vẽ và số liệu.** Tám hình, mã sinh in ngay trong bài; bản trích ra nằm ở
[`tools/fig_lars.py`](<../../../tools/fig_lars.py>), sinh bằng
`python extract_fig.py "Least Angle Regression (LARS)" fig_lars.py`. Mọi con số trong bài in
ra từ [`tools/v_lars.py`](<../../../tools/v_lars.py>) — 25 mục, chạy trong **khoảng 25 giây**, đầu ra
lưu ở [`tools/_out_lars.txt`](<../../../tools/_out_lars.txt>).

---

## 10. Kiểm tra trước khi đóng bài

- [x] **Khối IO** — mục 0.1: shape cụ thể, ba đầu ra, năm chỗ dễ hiểu sai
- [x] **Khối KN** — mục 0.3: mười hai từ, định nghĩa tại chỗ, không đẩy sang bài khác
- [x] **Bảng ký hiệu có cột "Đọc là gì"** — mục 0.2, kèm **bảy chỗ trùng tên**, trong đó bốn
      chỗ nằm ngay trong chính bài báo ($A$ · $T$ · $\varepsilon$ · $C$)
- [x] **Ví dụ chạy tay, script in lại TỪNG bước** — mục 0.4, `v_lars.py` mục 24; mọi số trung
      gian của (2.8)–(2.16) đều in ra
- [x] **Khối SS** — mục 1: hai thuật toán cũ, bế tắc nêu bằng SỐ (356 310 bước so với 10), và
      chỗ $\varepsilon = 10$ **không hội tụ**
- [x] **Khối CT — Luật 6, lấy đủ 100 % phần ruột**: 9/9 hình · 1/1 bảng · **145/145 công
      thức** · 18/18 mục · 4/4 định lý · 17/17 bổ đề
- [x] **Mỗi hình ruột có đủ bốn phần**: hình đã cắt kèm số trang · caption gốc nguyên văn +
      dịch · **nhận xét của chính tác giả nguyên văn + dịch** · "Cách đọc hình" kèm điều hình
      KHÔNG nói ra
- [x] **Khối VS** — mục 4: ba tầng cơ chế, mỗi tầng một phép đo; và hai trực giác sai được
      nói thẳng
- [x] **Khối KQ** — mục 2: mọi bảng số đều dựng lại từ nguồn, kèm số hiệu Hình/Bảng và trang
- [x] **Khối GIÁ** — mục 5: bảng bốn cột, năm chỗ hỏng kèm số, quy trình dùng an toàn
- [x] **Hai bảng đối chiếu** — mục 8.1 theo hình/bảng, mục 8.2 theo mục, cộng mục 8.3 theo
      công thức và 8.4 theo định lý/bổ đề
- [x] **Mọi con số sinh từ script** — `v_lars.py`, 25 mục, khoảng 25 giây, EXIT 0, **0
      UserWarning**
- [x] **Số luồng BLAS đặt cứng bằng 1** trước khi nạp `numpy`, và khai báo rõ ở mục 0.5 rằng
      mọi con số thời gian là một lần chạy
- [x] **Tám hình tự vẽ có mã sinh in ngay trong bài**, mỗi khối đúng một `savefig()`
- [x] **Bản cài đặt LARS viết TỪ ĐẦU** theo (2.4)–(2.13), đối chiếu `sklearn` (lệch
      $\le1{,}7\times10^{-10}$ trên $m = 10$) và đối chiếu bản cập nhật Cholesky (lệch
      $2{,}93\times10^{-14}$)
- [x] **Sáu chỗ nguồn in sai** được nêu kèm bằng chứng: (3.20) · (4.13) · câu về đơn vị sau
      (1.1) · dẫn "Lemmas 3 and 4" ở tr.31 · dẫn (3.12)/(3.13) ở tr.13 · dẫn (2.3) ở tr.17 ·
      dẫn (4.12) ở tr.40
- [x] **Bảy chỗ đo ra khác con số được in** được nêu kèm số: 103 so với 104 bước · 255 so với
      251 bước · $C_p$ tại $k=16$ so với 15 · 0,963 so với 0,966422 · 0,950 so với 0,956009 ·
      33,23 so với 35,59 · "covariate 8" so với ba phép đo đều chỉ chỗ khác
- [x] **Một cảnh báo thực hành có bằng chứng**: `sklearn` chế độ `lar` phân kỳ trên chính mô
      hình bậc hai của bài báo, không phát cảnh báo nào

### Bốn chỗ còn bỏ ngỏ

1. **$C_p$ cực tiểu ở $k = 15$ hay $k = 16$ với $m = 64$.** Chênh lệch phụ thuộc cách ước
   lượng $\bar\sigma^{2}$, và bài báo không nói nó dùng mẫu số nào. Bài này dùng
   $\mathrm{RSS}_{\text{đầy đủ}}/(n-m-1)$. Chưa quét hết các quy ước hợp lý để xem quy ước nào
   cho $k = 16$.
2. **Số bước Stagewise: 251 so với 255, và số hệ số khác 0 ở bước 40: 35,59 so với 33,23.**
   Cả hai lệch cùng một chiều và cùng dính tới bản cài đặt sửa đổi Stagewise. Nghi ngưỡng coi
   $P_j$ bằng 0 trong phép chiếu NNLS, hoặc quy tắc phá hoà — chưa quét.
3. **Số bước Lasso trên $m = 64$: 104 so với 103.** Lệch đúng một bước. Chưa truy được nguyên
   nhân; có thể là một sự kiện hoà mà bản cài đặt này đếm thành hai bước.
4. **Phụ lục A chỉ được đọc và tóm tắt, không kiểm số.** Bổ đề 13–17 là các phát biểu về độ
   đo và tính Lipschitz; chúng kiểm được gián tiếp qua sai phân hữu hạn ở mục 2.7, nhưng
   không có phép kiểm trực tiếp nào trong `v_lars.py`.
