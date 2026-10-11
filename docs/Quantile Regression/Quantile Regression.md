# Quantile Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 430
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Roger Koenker, Gilbert Bassett
Date: January 1, 1978
url: https://doi.org/10.2307/1913643

> **Quantile Regression — Hồi quy phân vị.** Bình phương tối thiểu mô hình hoá **trung bình
> có điều kiện** $\mathbb{E}[y\mid x]$. Hồi quy phân vị mô hình hoá **phân vị có điều kiện**
> $Q_\theta(y\mid x)$, cho bất kỳ mức $\theta\in(0,1)$. Khác biệt không nằm ở chỗ "bền vững
> hơn"; khác biệt là nó **trả lời một câu hỏi khác**.
>
> Bài báo gốc là Koenker & Bassett, *"Regression Quantiles"*, **Econometrica 46(1):33–50,
> 1978**. Bài báo đó **không có một hình vẽ nào**; phần ruột của nó là **hai bảng số**, **tám
> định lý** và **22 công thức được đánh số**. Mục 2 dưới đây đi qua đủ cả ba nhóm, theo đúng
> thứ tự bài báo in.
>
> Điều đáng nhớ nhất của bài viết này: **Bảng II trang 42 in nghiệm cho mọi $\theta$ trên
> một ví dụ năm quan sát, nhưng không in năm quan sát đó.** Mục 2.14 dựng lại chúng — và cả
> mười con số đều là số nguyên.

---

## 0. Trước khi đọc

### 0.1 Nhận gì, trả ra gì

Hồi quy phân vị nhận **đúng những gì hồi quy tuyến tính nhận**, cộng thêm một con số
$\theta$. Chỗ khác biệt nằm ở **ý nghĩa của đầu ra**: không phải "giá trị trung bình", mà
"mức mà $\theta$ phần dữ liệu nằm dưới".

| | Cách hiểu thông thường | Ký hiệu của **bài báo** | Ví dụ ở mục 5 |
|---|---|---|---|
| **Đầu vào** | một bảng gồm $T$ mẫu và $K$ cột số, **cột đầu toàn số 1** | $X\in\mathbb{R}^{T\times K}$ | $400\times2$ |
| **Đáp án khi huấn luyện** | một giá trị số cho mỗi mẫu | $y\in\mathbb{R}^{T}$ | 400 số |
| **Siêu tham số** | "hỏi về mức phần trăm nào" | $\theta\in(0,1)$ | 0,10 · 0,25 · 0,50 · 0,75 · 0,90 |
| **Đầu ra của việc học** | một hệ số cho mỗi cột, **riêng cho $\theta$ đó** | $\beta^*(\theta)\in\mathbb{R}^{K}$ | 2 số, **mỗi $\theta$ một bộ** |
| **Dự đoán** | mức mà $\theta$ phần dữ liệu nằm dưới | $\hat Q_\theta(y\mid x)=x_t\beta^*(\theta)$ | 400 số |

```
X       shape (T, K)      y   shape (T,)

θ = 0.10  →  β*(0.10)   một mô hình
θ = 0.50  →  β*(0.50)   một mô hình KHÁC — không suy ra được từ cái trên
θ = 0.90  →  β*(0.90)   một mô hình KHÁC nữa

muốn cả một dải  →  phải giải lại một bài quy hoạch tuyến tính cho từng θ
```

Năm điều dễ hiểu sai, nói rõ ngay ở đây:

- **Mỗi $\theta$ là một mô hình riêng, phải giải lại từ đầu.** Không có "một mô hình trả về
  mọi phân vị". Muốn 9 mức thì giải 9 bài. Đây là chi phí thật, ghi ở bảng mục 6.
- **$\theta$ không phải thứ học được** — nó là câu hỏi người dùng đặt ra.
- **$\theta=1/2$ cho trung vị, KHÔNG cho trung bình.** Hai thứ chỉ trùng nhau khi phân phối
  đối xứng. Bài báo gọi trường hợp này là *least absolute error* (LAE) và dành cho nó một vị
  trí riêng — xem mục 2.6.
- **Nghiệm có thể KHÔNG duy nhất.** Đây không phải chuyện lý thuyết xa xôi: Bảng II của
  chính bài báo có **ba** giá trị $\theta$ mà bài toán có nhiều nghiệm, và mục 2.14 đếm lại
  được đúng ba chỗ đó.
- **Các đường phân vị có thể cắt nhau.** Giải riêng từng $\theta$ nên không có gì bảo đảm
  đường $\theta=0{,}9$ luôn nằm trên đường $\theta=0{,}8$ ở mọi $x$. Bài báo 1978 **không
  nhắc tới hiện tượng này**; nó là phần đời sau — mục 6.

### 0.2 Bảng ký hiệu

Bảng thuật ngữ ở mục 0.3 giải nghĩa các **từ**; bảng này giải nghĩa các **ký hiệu**, kèm cách
đọc thành tiếng. Ký hiệu không đọc được thành tiếng là ký hiệu không nghĩ được.

**Cột "bài báo" là ký hiệu Koenker & Bassett 1978 dùng thật.** Bài viết này giữ nguyên ký
hiệu đó ở mục 2 để người đọc đối chiếu được với bản in, và chỉ đổi sang ký hiệu quen thuộc
hơn ở các mục còn lại — mỗi lần đổi đều nói rõ.

| Ký hiệu bài báo | Đọc là gì | Nghĩa | Chiều / kiểu | Tên thường gặp hôm nay |
|---|---|---|---|---|
| $\theta$ | **"thê-ta"** | **núm điều khiển của cả bài**: hỏi về mức phần trăm nào | vô hướng, $0<\theta<1$ | $\tau$ (tau), `quantile` của thư viện |
| $T$ | "tê" | số quan sát | số nguyên | $n$ |
| $K$ | "ca" | số cột của $X$, tức số hệ số | số nguyên | $p$ |
| $x_t$ | "ích chỉ số t" | **một DÒNG** của $X$ — vectơ đặc trưng của quan sát thứ $t$ | $\mathbb{R}^{K}$ (vectơ hàng) | $x_i$ |
| $b$ | "bê" | **biến chạy** của bài toán tối ưu, chưa phải nghiệm | $\mathbb{R}^{K}$ | $\beta$ trong `argmin` |
| $\beta$ | "bê-ta" | tham số **THẬT**, chưa biết, của mô hình (1.1) | $\mathbb{R}^{K}$ | — |
| $\beta^*(\theta)$ | "bê-ta sao của thê-ta" | **nghiệm** ước lượng được, riêng cho một $\theta$ | $\mathbb{R}^{K}$ | $\hat\beta(\tau)$ |
| $B^*(\theta)$ | "bê lớn sao của thê-ta" | **TẬP** mọi nghiệm. Viết hoa vì nó có thể có nhiều hơn một phần tử | tập con của $\mathbb{R}^{K}$ | — |
| $u_t$ | "u chỉ số t" | $y_t-x_t\beta$ — sai số của mô hình THẬT | vô hướng | nhiễu $\varepsilon_t$ |
| $u^*(\theta)$ | "u sao của thê-ta" | $y-X\beta^*(\theta)$ — **phần dư của NGHIỆM** | $\mathbb{R}^{T}$ | residual |
| $F$, $f$ | "ép lớn", "ép nhỏ" | hàm phân phối và mật độ của sai số | hàm | — |
| $\xi(\theta)$ | **"xi của thê-ta"** | $F^{-1}(\theta)$ — phân vị thứ $\theta$ của **phân phối**, không phải của mẫu | vô hướng | — |
| $h$, $\mathcal{H}$, $H$ | "hát", "hát hoa", "hát lớn" | một **tập con $K$ quan sát**; tập mọi tập con như thế; và những tập con làm $X(h)$ đủ hạng | — | "tập cơ sở" của quy hoạch tuyến tính |
| $X(h)$, $y(h)$ | "ích của hát" | phần của $X$, $y$ **chỉ gồm** các dòng trong $h$ | $K\times K$, $\mathbb{R}^K$ | — |
| $\bar h$ | "hát gạch" | phần bù: $\mathcal{T}-h$, tức mọi quan sát KHÔNG trong $h$ | — | — |
| $\iota$ | **"iota"** | vectơ **toàn số 1**. Rất dễ đọc nhầm thành chữ *l* hoặc số *1* | $\mathbb{R}^{T}$ hoặc $\mathbb{R}^K$ | `np.ones` |
| $P$, $N$, $Z$ | "pê", "en", "dét" | số phần dư **dương**, **âm**, **bằng 0** | số nguyên | — |
| $\psi(b;\theta,y,X)$ | "xi hàm mục tiêu" | **giá trị hàm mục tiêu** tại $b$ | vô hướng | tổng pinball loss |
| $\Omega$, $\omega_{ij}$ | "ô-mê-ga lớn", "ô-mê-ga i j" | ma trận hiệp phương sai tiệm cận của $M$ phân vị mẫu | $M\times M$ | — |
| $Q$ | "quy" | $\lim T^{-1}X'X$ — ma trận thiết kế đã chuẩn hoá | $K\times K$ | Gram/moment matrix |
| $\pi(\theta)$ | "pi của thê-ta" | **bộ trọng số** để trộn nhiều phân vị (Định lý 4.3) | vectơ | — |
| $\alpha$ | "an-pha" | tỉ lệ **cắt bỏ** ở mỗi đuôi (tr.47) | $0<\alpha<1/2$ | `trim` |
| $r^+$, $r^-$ | "e cộng", "e trừ" | phần **đoán thiếu** và phần **đoán thừa** trong Phụ lục | $\mathbb{R}^T_+$ | $u$, $v$ |
| $d$, $\Delta$ | "đê", "đen-ta" | biến **đối ngẫu**, và bản đã dịch $\Delta=d+1-\theta$ | $\mathbb{R}^{T}$ | dual variable |

**Toán tử và ký hiệu nền:**

| Ký hiệu | Đọc là gì | Nghĩa |
|---|---|---|
| $\mathbb{E}[y\mid x]$ | "kỳ vọng của y khi biết ích" | **trung bình** có điều kiện — thứ bình phương tối thiểu ước lượng. Gạch đứng $\mid$ đọc là "khi biết" |
| $\operatorname{sgn}(u)$ | "dấu của u" | bằng $1$, $0$, $-1$ tuỳ $u>0$, $=0$, $<0$ (bài báo tr.39) |
| $\operatorname{sgn}^*(u;z)$ | "dấu sao của u, cho z" | bằng $\operatorname{sgn}(u)$ nếu $u\ne0$, bằng $\operatorname{sgn}(z)$ nếu $u=0$ — cách xử lý chỗ **không khả vi** (tr.40) |
| $\iota'_K$ | "iota phẩy ca" | vectơ hàng $K$ số 1; nhân với vectơ cột là lấy **tổng** |
| $A\otimes B$ | "a tích Kronecker b" | ghép hai ma trận thành một ma trận lớn: ô $(i,j)$ của $A$ được thay bằng cả khối $a_{ij}B$ |
| $\lvert X(h)\rvert$ | "định thức của ích của hát" | định thức — bằng 0 nghĩa là $K$ quan sát đó **không xác định nổi** một siêu phẳng |
| $\rho_\theta(u)$ | "rô thê-ta của u" | hàm mất mát **pinball**. **Ký hiệu này KHÔNG có trong bài báo 1978** — xem mục 2.5 |

> **Sáu chỗ trùng tên, và BỐN chỗ nằm ngay trong chính bài báo này:**
>
> 1. **$u$ mang hai nghĩa trong cùng một bài báo.** Ở trang 38, $u_t=y_t-x_t\beta$ là sai số
>    của **mô hình thật**. Ở trang 41, $u^*(\theta)=y-X\beta^*(\theta)$ là **phần dư của
>    nghiệm**. Dấu sao là thứ duy nhất phân biệt hai cái, và nó rất dễ trôi mất khi đọc lướt.
> 2. **$\pi$ vừa là trọng số vừa là số $3{,}14159$, cách nhau một trang.** Công thức (4.9)
>    trang 45 viết $(2\pi)^{-K}$; Định lý 4.3 trang 46 viết $\pi(\theta)$ là **bộ trọng số**.
>    Cùng chữ, hai trang liền nhau.
> 3. **$D$ vừa là ma trận đường chéo vừa là tên bài toán đối ngẫu.** Định lý 3.5 trang 41
>    dùng $D$ cho một ma trận $T\times T$; Phụ lục trang 48 gọi bài toán đối ngẫu là **(D)**.
> 4. **$\theta$ đứng cạnh $\Theta$.** Công thức (4.9) trang 45 có ma trận $\Theta$ viết hoa,
>    khác hẳn $\theta$ thường. Hai ký hiệu cách nhau đúng một nét chữ.
> 5. **$\psi$ ở bài này KHÔNG phải $\psi$ ở bài [Huber Regression](<../Huber Regression/Huber Regression.md>).**
>    Huber 1964 dùng $\psi=\rho'$ — **đạo hàm** của hàm mất mát, một hàm của một biến số.
>    Koenker & Bassett dùng $\psi(b;\theta,y,X)$ cho **toàn bộ tổng mất mát**, một hàm của
>    vectơ tham số. Chép công thức từ bài này sang bài kia là hỏng ngay.
> 6. **$\theta$ của bài báo là `quantile` của thư viện, KHÔNG phải `alpha`.** `alpha` trong
>    `QuantileRegressor` của `scikit-learn` là mức phạt $L_1$ — một thứ bài báo 1978 không
>    có. Đặt nhầm là ra một mô hình khác hẳn.

### 0.3 Mười hai từ cần biết trước khi đọc tiếp

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Trong bài này | Nghĩa |
|---|---|---|
| **Bình phương tối thiểu** (least squares) | thứ đem ra so | chọn hệ số làm $\sum(y_t-x_tb)^2$ nhỏ nhất; kết quả ước lượng **trung bình có điều kiện** |
| **LAE** (least absolute error) | trường hợp $\theta=1/2$ | chọn hệ số làm $\sum\lvert y_t-x_tb\rvert$ nhỏ nhất. Bài báo gọi nó là **trung vị hồi quy** |
| **Phân vị** (quantile) | $\theta=0{,}9$ | mức mà $\theta$ phần của phân phối nằm dưới. $\theta=0{,}5$ là **trung vị** |
| **Có điều kiện** (conditional) | $Q_\theta(y\mid x)$ | hỏi riêng cho những quan sát có cùng $x$, không gom toàn bộ dữ liệu lại |
| **Thống kê thứ tự** (order statistic) | tr.38, tr.39 | dữ liệu đã sắp xếp tăng dần. Đây là thứ **có trong mô hình vị trí mà KHÔNG có trong mô hình tuyến tính** — chính là bế tắc ở mục 1 |
| **Mô hình vị trí** (location model) | tr.34, chú thích 5 | trường hợp $K=1$, $x_t=1$ với mọi $t$: chỉ ước lượng một con số, không có biến giải thích |
| **Phần dư** (residual) | $u^*=y-X\beta^*$ | giá trị thật trừ dự đoán |
| **Quy hoạch tuyến tính** (linear program) | Phụ lục tr.48 | lớp bài toán tối ưu có hàm mục tiêu và mọi ràng buộc đều tuyến tính |
| **Bài đối ngẫu** (dual) | (D), (D') tr.48 | một bài toán thứ hai, dựng từ bài gốc, có cùng giá trị tối ưu nhưng biến khác hẳn |
| **Đồng biến đổi** (equivariance) | Định lý 3.2 | đổi dữ liệu theo một phép biến đổi thì nghiệm đổi theo đúng phép đó. **Khác với "bất biến"**: bất biến là *không đổi gì cả* |
| **Hiệu suất tiệm cận** (asymptotic efficiency) | tr.36, tr.46 | tỉ số giữa phương sai nhỏ nhất có thể và phương sai thật, khi $T\to\infty$. Càng gần 1 càng tốt |
| **Nhiễm bẩn** (contamination) | Bảng I | một phần nhỏ dữ liệu đến từ một phân phối khác, thường là tản hơn nhiều |

Ba từ xuất hiện muộn hơn và được giải nghĩa ngay tại chỗ dùng: **dưới vi phân** (mục 2.10),
**tích Kronecker** (mục 2.17) và **vị trí tổng quát** (mục 2.7).

### 0.4 Một ví dụ chạy tay — và nó là ví dụ của chính bài báo

Bài báo có một ví dụ năm quan sát ở Bảng II trang 42. Mục 2.14 chứng minh rằng năm quan sát
đó là:

$$(x_t, y_t) \;=\; (1,3),\;(2,2),\;(4,7),\;(7,8),\;(9,6)$$

Chạy tay được toàn bộ với $\theta=0{,}2$. Bài báo bảo nghiệm là $\beta^*=(6/7,\,4/7)$, tức
đường thẳng $\hat y = \tfrac67 + \tfrac47 x$. Lập bảng phần dư:

| $t$ | $x_t$ | $y_t$ | $\hat y_t=\tfrac67+\tfrac47x_t$ | $u_t=y_t-\hat y_t$ | dấu | đóng góp $\rho_{0{,}2}(u_t)=u_t(0{,}2-\mathbb{1}[u_t<0])$ |
|---|---|---|---|---|---|---|
| 1 | 1 | 3 | $10/7$ | $+11/7$ | dương | $0{,}2\times 11/7 = 11/35$ |
| 2 | 2 | 2 | $2$ | $0$ | **bằng 0** | $0$ |
| 3 | 4 | 7 | $22/7$ | $+27/7$ | dương | $0{,}2\times 27/7 = 27/35$ |
| 4 | 7 | 8 | $34/7$ | $+22/7$ | dương | $0{,}2\times 22/7 = 22/35$ |
| 5 | 9 | 6 | $6$ | $0$ | **bằng 0** | $0$ |

Tổng $= (11+27+22)/35 = 60/35 = 12/7 = 1{,}7142857143$.

Toàn bộ bảng trên do [`tools/v_quantile.py`](<../../../tools/v_quantile.py>) mục 6 bước 7 in
lại từng bước bằng **số hữu tỉ chính xác** — ví dụ chạy tay là chỗ dễ sai nhất của cả bài
vì nó được gõ bằng tay theo định nghĩa, nên nó phải có một đường kiểm riêng.

Ba điều đọc được ngay từ bảng này, và cả ba là nội dung của ba định lý ở mục 2:

* **Đúng $K=2$ phần dư bằng 0** — quan sát 2 và 5. Đường thẳng đi **qua** hai điểm dữ liệu.
  Đó là Định lý 3.1 (mục 2.7).
* **Không có phần dư âm nào**, tức $N=0$. Điều kiện (3.7) đòi $N\le T\theta\le N+Z$, tức
  $0\le 5\times0{,}2\le 0+2$, tức $0\le 1\le 2$ — đúng. Đó là Định lý 3.4 (mục 2.12).
* **Kéo ba điểm dương lên cao thêm bao nhiêu cũng không làm đường đổi chỗ**, vì chúng vẫn ở
  cùng một phía. Đó là Định lý 3.5 (mục 2.13).

Đổi $\theta$ thì bảng trên đổi **chỉ ở cột cuối** — cột hệ số phạt. Với $\theta=0{,}9$, ba
phần dư dương bị phạt $0{,}9$ thay vì $0{,}2$, nên đường bị đẩy lên và nghiệm thành
$(17/3,\,1/3)$.

### 0.5 Nguồn của bài này, và một hạn chế phải nói trước

Bài báo gốc **Koenker & Bassett 1978** không có bản truy cập mở hợp pháp: hỏi Unpaywall cho
DOI `10.2307/1913643` thì được `is_oa: false`, `oa_status: "closed"`, `has_repository_copy:
false`, `oa_locations: []`. Bản dùng để viết bài này là **bản quét JSTOR** do một trang môn
học đăng lại; nó nằm trong `AI algorithms/_papers/` của máy dựng bài nhưng **được khai trong
`.gitignore` và không được đẩy lên repo công khai**, theo đúng cách đã xử lý bài
[Lasso Regression](<../Lasso Regression/Lasso Regression.md>).

Hệ quả kỹ thuật, phải ghi ra vì nó giới hạn mọi phép kiểm tự động:

| Đo được | Con số |
|---|---:|
| số trang của file PDF | 24 |
| số trang **thân bài báo** (tr.33–50) | 18 |
| số trang thân bài có lớp chữ máy đọc được | **0** |
| số ký tự trung bình mỗi trang thân bài | **0,0** |
| số trang CÓ lớp chữ (bìa JSTOR + phụ lục dẫn liên kết) | 6 |

Nói cách khác: **mười tám trang nội dung là ảnh quét thuần tuý.** Mọi đoạn trích nguyên văn
tiếng Anh trong bài này được chép tay từ ảnh render ở tỉ lệ 2,2, và **không có công cụ nào
trong `tools/` đối chiếu lại được**. `kiemtrich.py` báo đúng hạn chế đó thay vì báo đỏ hàng
loạt — xem ô `[ ]` ở checklist cuối bài.

> **Bản vá công cụ sinh ra từ chính bài này.** `kiemtrich.py` trước đó dò bản quét bằng
> **mật độ ký tự trung bình cả file**. File này đánh bại phép dò đó: trang bìa JSTOR và năm
> trang dẫn liên kết ở cuối có lớp chữ đầy đủ, kéo trung bình lên **352,8 ký tự mỗi trang**
> — vượt xa ngưỡng 100 — trong khi **18/18 trang thân bài có 0 ký tự**. Đã đổi sang đếm
> **số trang rỗng**: quá nửa số trang dưới 100 ký tự thì báo là bản quét. Đo bán kính hai
> chiều: LARS giữ nguyên 80 đoạn trích kiểm được và 0 chỗ không khớp, WGAN 52/0, LSGAN 45/0,
> Autoencoder 29/0 — chiều mất bằng 0; chiều thêm bắt đúng một bài,
> [Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>), vốn cũng là bản quét
> và vốn làm `kiemtrich.py` chạy hơn mười hai phút không xong.

Ngược lại, **mọi con số trong bài này đều dựng lại được** bằng
[`tools/v_quantile.py`](<../../../tools/v_quantile.py>) — script chạy trong khoảng **bốn phút**, không
cần dữ liệu ngoài, không cần huấn luyện gì, và không in ra một `UserWarning` nào. Chạy:
`cd tools && python v_quantile.py`.

---

## 1. Trước 1978, người ta làm gì — và tắc ở đâu

### 1.1 Mô hình vị trí thì đã có lời giải

Với **một con số** cần ước lượng — bài toán mà bài báo gọi là *mô hình vị trí*, tức trường
hợp $K=1$, $x_t=1$ với mọi $t$ (chú thích 5 trang 34) — thì thống kê đã có cả một họ công cụ
bền vững: trung vị, trung bình cắt đuôi, và các tổ hợp tuyến tính của **thống kê thứ tự**.

Bài báo dành hẳn mục 2 (trang 34–37) cho lịch sử này và chốt nó bằng **Bảng I trang 37**.

![Bảng I của Koenker & Bassett 1978](<paper_kb1978_bang_I.png>)

**Hình 1 — "TABLE I. Empirical Variances of Some Alternative Location Estimators (Sample
Size 20)", Koenker & Bassett 1978, trang 37.** Nguồn: bản quét JSTOR của
`1978 - Koenker, Bassett - Regression Quantiles.pdf`, không nằm trong repo (mục 0.5).

**Chú thích gốc, chép nguyên văn:**

*"TABLE I. EMPIRICAL VARIANCES OF SOME ALTERNATIVE LOCATION ESTIMATORS (Sample Size 20).
a Abstracted from Exhibit 5 in Andrews, et al. [3]. b Gaussian Mixture: .9Φ(1)+.1Φ(3).
c Gaussian Mixture: .9Φ(1)+.1Φ(10). d β̂ = .3β\*(1/3)+.4β\*(1/2)+.3β\*(2/3), where β\*(θ) is
the θth sample quantile. e β̂ = 1/4β\*(1/4)+1/2β\*(1/2)+1/4β\*(3/4)."*

Dịch: *"Bảng I. Phương sai thực nghiệm của một số ước lượng vị trí thay thế (cỡ mẫu 20).
a Trích từ Biểu 5 trong Andrews và cộng sự [3]. b Hỗn hợp Gauss: .9Φ(1)+.1Φ(3). c Hỗn hợp
Gauss: .9Φ(1)+.1Φ(10). d β̂ = .3β\*(1/3)+.4β\*(1/2)+.3β\*(2/3), trong đó β\*(θ) là phân vị
mẫu thứ θ. e β̂ = 1/4β\*(1/4)+1/2β\*(1/2)+1/4β\*(3/4)."*

**Nhận xét của chính tác giả, chép nguyên văn từ thân bài trang 36:**

*"In order to provide some quantitative evidence on the performance of some alternative
estimates of location we have abstracted a small subset of estimators and a small subset of
distributions from those considered in the Princeton Robustness Study [3]. Table I gives
Monte-Carlo variances for six selected estimators and five selected distributions. We have
purposely chosen simple estimators which are nonadaptive. The table clearly illustrates that
estimators of location are available which, while making a small sacrifice of efficiency to
the mean at the Gaussian distribution, are greatly superior to the mean for non-Gaussian
distributions."*

Dịch: *"Để có bằng chứng định lượng về hiệu năng của một số ước lượng vị trí thay thế, chúng
tôi đã trích ra một tập con nhỏ các ước lượng và một tập con nhỏ các phân phối từ những thứ
được xem xét trong Nghiên cứu Bền vững Princeton [3]. Bảng I cho phương sai Monte-Carlo của
sáu ước lượng được chọn và năm phân phối được chọn. Chúng tôi cố ý chọn những ước lượng đơn
giản, không thích nghi. Bảng này minh hoạ rõ rằng có sẵn những ước lượng vị trí mà, trong khi
hy sinh một chút hiệu suất so với trung bình ở phân phối Gauss, lại vượt trội hẳn so với
trung bình ở các phân phối không Gauss."*

**Cách đọc bảng.** Mỗi ô là phương sai thực nghiệm, chuẩn hoá sao cho ô "Mean / Normal" bằng
đúng $1{,}00$. Càng nhỏ càng tốt.

* Đọc theo **hàng đầu**: trung bình mẫu đi từ $1{,}00$ ở phân phối Gauss lên $11{,}54$ ở hỗn
  hợp $10\%$ tại $10\sigma$, rồi $12\,548{,}0$ ở Cauchy. Hàng này là **toàn bộ lý do bài báo
  tồn tại**.
* Đọc theo **hàng cuối bốn hàng**: chúng trả giá $1{,}06$ đến $1{,}50$ ở phân phối Gauss —
  tức mất 6 % đến 50 % hiệu suất — và đổi lại không bao giờ vượt quá $7{,}3$ ở bất kỳ cột
  nào.
* **Điều bảng KHÔNG nói ra:** cả sáu ước lượng này chỉ dùng được cho **một con số**. Không
  dòng nào của Bảng I áp được cho một mô hình có biến giải thích — và đó chính là chỗ trống
  mà bài báo lấp.

Số $12\,548{,}0$ đáng đọc kỹ. Phương sai lý thuyết của trung bình mẫu dưới phân phối Cauchy
là **vô cực**; con số hữu hạn ấy là kết quả của một lần chạy Monte-Carlo cụ thể, không hội tụ
về đâu cả. Mục 2.4 đo lại điều đó và cho thấy nó thay đổi **hơn bốn bậc độ lớn** giữa các hạt
giống.

### 1.2 Mô hình tuyến tính thì không

Bế tắc được chính bài báo nêu, ở trang 35 và nhắc lại ở trang 38. Chép nguyên văn trang 35:

*"There was a parallel early recognition of the need for robust alternatives to the least
squares estimator for the linear model. Wild observations, or "outliers" as they came to be
called, were more difficult to identify in such models and the fruitful notion from the
location model of an ordering of sample observations had no simple analogue in the more
complicated models."*

Dịch: *"Song song với đó, nhu cầu về những lựa chọn bền vững thay cho ước lượng bình phương
tối thiểu trong mô hình tuyến tính cũng được nhận ra từ sớm. Những quan sát bất thường, hay
"ngoại lai" như chúng được gọi sau này, khó nhận diện hơn trong các mô hình như thế, và khái
niệm hữu ích từ mô hình vị trí — việc sắp thứ tự các quan sát mẫu — không có một tương tự
đơn giản nào trong những mô hình phức tạp hơn."*

Đó là bế tắc, nêu bằng đúng một câu: **sắp thứ tự dữ liệu là chuyện hiển nhiên khi chỉ có một
con số, và vô nghĩa khi có nhiều biến giải thích.** Sắp theo $y$ thì bỏ qua $x$; sắp theo
phần dư thì phần dư phụ thuộc chính cái hệ số đang đi tìm.

Bài báo trang 38 nói thẳng rằng đây là chỗ bài toán mơ hồ, và nói luôn cách gỡ:

*"Huber's [26] observation that outliers are difficult to identify in the regression context
underlines the essential ambiguity involved in extending to the linear model the ordinary
notions of sample quantiles based on an ordering of sample observations. A direct
generalization of the minimization problem posed above resolves this ambiguity."*

Dịch: *"Nhận xét của Huber [26] rằng ngoại lai khó nhận diện trong bối cảnh hồi quy làm nổi
bật sự mơ hồ cốt lõi khi mở rộng sang mô hình tuyến tính các khái niệm thông thường về phân
vị mẫu vốn dựa trên việc sắp thứ tự các quan sát mẫu. Một sự tổng quát hoá trực tiếp của bài
toán cực tiểu hoá nêu ở trên gỡ được sự mơ hồ này."*

**Toàn bộ đóng góp của bài báo nằm trong hai chữ "direct generalization".** Không sắp thứ tự
nữa; thay vào đó, định nghĩa phân vị bằng **một bài toán cực tiểu hoá** — và bài toán đó thì
thay $b$ bằng $x_tb$ là xong. Mục 2.5 và 2.6 là hai định nghĩa ấy, đặt cạnh nhau.

### 1.3 Bế tắc đo bằng số

Một đường bình phương tối thiểu duy nhất không mô tả nổi dữ liệu có độ tản mát thay đổi theo
$x$. Đo trên 400 điểm sinh từ $y=1+0{,}5x+(0{,}3+0{,}3x)\varepsilon$ với
$\varepsilon\sim N(0,1)$ — tức độ lệch chuẩn của nhiễu tăng từ $0{,}3$ ở $x=0$ lên $3{,}3$ ở
$x=10$:

| Khúc $x$ | Số điểm | Khoảng phân vị $5\%$–$95\%$ phủ được | Khoảng $\hat y\pm1{,}96s$ của bình phương tối thiểu phủ được |
|---|---:|---:|---:|
| $[0,2)$ | 76 | $0{,}894737$ | $1{,}000000$ |
| $[2,4)$ | 88 | $0{,}886364$ | $1{,}000000$ |
| $[4,6)$ | 79 | $0{,}936709$ | $0{,}974684$ |
| $[6,8)$ | 77 | $0{,}922078$ | $0{,}896104$ |
| $[8,10)$ | 80 | $0{,}887500$ | $\mathbf{0{,}812500}$ |
| **toàn bộ** | 400 | $0{,}905000$ | $0{,}937500$ |

Dòng cuối là cái bẫy: nhìn con số tổng thì bình phương tối thiểu trông **tốt hơn** —
$0{,}937500$ gần $0{,}95$ hơn $0{,}905000$. Nhưng nó đúng trung bình và **sai ở mọi chỗ**:
rộng quá ở nơi dữ liệu chụm (phủ $1{,}000000$ ở hai khúc đầu) và hẹp quá ở nơi dữ liệu tản
(phủ $0{,}812500$ ở khúc cuối, tức **một trong năm** trường hợp rơi ra ngoài một khoảng được
quảng cáo là chứa 90 %).

Đo độ chênh giữa ô tệ nhất và ô tốt nhất của mỗi cách:

* khoảng phân vị: $0{,}050345$ (từ $0{,}886364$ tới $0{,}936709$)
* khoảng bình phương tối thiểu: $0{,}187500$ (từ $0{,}812500$ tới $1{,}000000$)
* tỉ số hai độ rộng: **$3{,}7242857143$ lần**

Mọi con số ở bảng này do [`tools/v_quantile.py`](<../../../tools/v_quantile.py>) mục 17 in ra.

---

## 2. Ruột bài báo, theo đúng thứ tự bản in

Bài báo Koenker & Bassett 1978 có **0 hình vẽ**, **2 bảng số**, **5 mục đánh số cộng một phụ
lục**, **8 định lý** và **22 công thức được đánh số**. Mục này đi qua đủ cả bốn nhóm, mỗi
phần có: khung cắt từ bản in · đoạn trích nguyên văn · bản dịch · phần "cách đọc" của người
viết, kèm điều bài báo **không** nói ra.

Một quy ước đọc, nói trước để khỏi vấp: bài báo viết $x_t b$ chứ không viết $x_t^\top b$, vì
$x_t$ ở đây là một **dòng** của $X$. Nhân một dòng với một cột ra một số.

### 2.1 Tóm tắt và mô hình (1.1) — trang 33

![Tóm tắt bài báo](<paper_kb1978_tom_tat.png>)

**Hình 2 — Tiêu đề, tác giả và tóm tắt, Koenker & Bassett 1978, trang 33.**

**Trích nguyên văn:**

*"A simple minimization problem yielding the ordinary sample quantiles in the location model
is shown to generalize naturally to the linear model generating a new class of statistics we
term "regression quantiles." The estimator which minimizes the sum of absolute residuals is
an important special case. Some equivariance properties and the joint asymptotic distribution
of regression quantiles are established. These results permit a natural generalization to the
linear model of certain well-known robust estimators of location. Estimators are suggested,
which have comparable efficiency to least squares for Gaussian linear models while
substantially out-performing the least-squares estimator over a wide class of non-Gaussian
error distributions."*

Dịch: *"Một bài toán cực tiểu hoá đơn giản, vốn cho ra các phân vị mẫu thông thường trong mô
hình vị trí, được chỉ ra là tổng quát hoá một cách tự nhiên sang mô hình tuyến tính, sinh ra
một lớp thống kê mới mà chúng tôi gọi là "phân vị hồi quy". Ước lượng cực tiểu hoá tổng trị
tuyệt đối phần dư là một trường hợp riêng quan trọng. Một số tính chất đồng biến đổi và phân
phối tiệm cận đồng thời của các phân vị hồi quy được thiết lập. Những kết quả này cho phép
một sự tổng quát hoá tự nhiên sang mô hình tuyến tính của một số ước lượng vị trí bền vững
nổi tiếng. Các ước lượng được đề xuất có hiệu suất tương đương bình phương tối thiểu với mô
hình tuyến tính Gauss, đồng thời vượt trội hẳn ước lượng bình phương tối thiểu trên một lớp
rộng các phân phối sai số không Gauss."*

**Cách đọc.** Tóm tắt này là một bản đồ chính xác của cả bài báo, và đọc kỹ nó thì biết mục
nào ở đâu: "bài toán cực tiểu hoá" là mục 3 · "tính chất đồng biến đổi" là Định lý 3.2 ·
"phân phối tiệm cận đồng thời" là Định lý 4.2 · "tổng quát hoá các ước lượng bền vững" là
Định lý 4.3 và đoạn trang 47.

**Điều tóm tắt KHÔNG nói ra, và nó đáng chú ý:** không có chữ *quy hoạch tuyến tính* nào.
Cách giải — thứ mà mọi thư viện hôm nay dùng — bị đẩy hẳn xuống **Phụ lục** (mục 2.21). Bài
báo tự xếp đóng góp của mình là **định nghĩa và lý thuyết tiệm cận**, không phải thuật toán.

![Mô hình (1.1)](<paper_kb1978_mo_hinh_1_1.png>)

**Hình 3 — Công thức (1.1) và điều kiện về $F$, trang 33.**

$$P(Y_t < y) = F(y - x_t\beta) \qquad (t = 1,\ldots,T) \tag{1.1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $Y_t$ | biến ngẫu nhiên quan sát thứ $t$ | vô hướng |
| $x_t$ | dòng thứ $t$ của ma trận thiết kế $(T\times K)$ đã biết | $\mathbb{R}^{K}$ |
| $\beta$ | vectơ tham số **chưa biết** | $\mathbb{R}^{K}$ |
| $F$ | hàm phân phối của sai số, **dạng không biết chính xác** | hàm $\mathbb{R}\to[0,1]$ |

**Ý nghĩa trừu tượng — nó nói điều gì.** Công thức này **không** nói "$y$ bằng $x\beta$ cộng
nhiễu" theo lối thông thường; nó nói mạnh hơn: **toàn bộ phân phối của $Y_t$ là một bản tịnh
tiến của cùng một hình dạng $F$**, và thứ duy nhất thay đổi theo $t$ là chỗ đặt nó, $x_t\beta$.
Hình dạng giữ nguyên; chỉ vị trí trượt đi.

Hệ quả trực tiếp, và nó là thứ làm cả bài báo chạy được: nếu hình dạng không đổi thì **mọi
phân vị của $Y_t$ đều dịch cùng một lượng**, nên phân vị thứ $\theta$ của $Y_t$ bằng
$x_t\beta + \xi(\theta)$ với $\xi(\theta)=F^{-1}(\theta)$. Tức là **mọi đường phân vị đều
song song**, chỉ khác nhau ở hệ số chặn.

Đó cũng là điều bài báo **không** nói ra và người đọc hôm nay dễ đọc ngược: ứng dụng nổi
tiếng nhất của hồi quy phân vị — mô tả dữ liệu mà độ tản mát **đổi theo $x$**, như hình quạt
ở mục 5 — nằm **ngoài** mô hình (1.1). Lý thuyết tiệm cận của mục 4 được chứng minh cho mô
hình sai số đồng nhất; phần "đường phân vị toả ra hình quạt" là phần đời sau.

Chữ in nghiêng ngay sau công thức nói rõ chỗ khó: *"and the shape of F is not precisely
known"* — *"và hình dạng của $F$ không được biết chính xác"*. Chú thích 4 giải thích rằng
cách đặt vấn đề này chỉ nhắm vào *"robustness to distributional assumptions"* và gác lại các
vấn đề về phụ thuộc giữa các quan sát hay dạng hàm phi tuyến.

### 2.2 Câu chuyện Laplace 1818 và điều kiện $[2f(0)]^{-1} < \sigma$ — trang 35

![Đoạn về Laplace 1818](<paper_kb1978_laplace_1818.png>)

**Hình 4 — Mục 2, đoạn về ước lượng LAE và kết quả của Laplace năm 1818, trang 35.**

**Trích nguyên văn:**

*"In 1818 Laplace proved that in the simple model of bivariate regression through the origin,
this least absolute error (LAE) estimator had smaller asymptotic variance than the least
squares estimator if the error law of the model had variance, $\sigma^2$, and density at the
median, $f(0)$, satisfying $[2f(0)]^{-1} < \sigma$."*

Dịch: *"Năm 1818 Laplace chứng minh rằng trong mô hình đơn giản của hồi quy hai biến qua gốc
toạ độ, ước lượng sai số tuyệt đối nhỏ nhất (LAE) này có phương sai tiệm cận nhỏ hơn ước
lượng bình phương tối thiểu nếu luật sai số của mô hình có phương sai $\sigma^2$ và mật độ
tại trung vị $f(0)$ thoả $[2f(0)]^{-1} < \sigma$."*

**Ý nghĩa trừu tượng.** Đây là **điều kiện đổi phe**, và nó nói một điều gọn: đại lượng quyết
định LAE thắng hay thua **không phải** độ nặng của đuôi phân phối, mà là **mật độ tại trung
vị**. Vế trái $[2f(0)]^{-1}$ là căn của phương sai tiệm cận của trung vị; vế phải $\sigma$ là
căn của phương sai của trung bình. Phân phối nào "nhọn ở giữa" thì $f(0)$ lớn, vế trái nhỏ,
LAE thắng.

Tính ra cho bốn phân phối:

| $F$ | $[2f(0)]^{-1}$ | $\sigma$ | LAE thắng? | tỉ số phương sai bình phương tối thiểu / LAE |
|---|---:|---:|---|---:|
| Chuẩn $N(0,1)$ | $1{,}2533141373$ | $1{,}0000000000$ | **không** | $0{,}6366197724$ |
| Laplace | $1{,}0000000000$ | $1{,}4142135624$ | có | $2{,}0000000000$ |
| Logistic | $2{,}0000000000$ | $1{,}8137993642$ | **không** | $0{,}8224670334$ |
| Cauchy | $1{,}5707963268$ | vô cực | có | vô cực |

Hai điều đáng nhớ từ bảng này. Một: **phân phối logistic làm LAE thua**, dù nó có đuôi nặng
hơn phân phối chuẩn — đúng như câu trên dự báo, chuyện không nằm ở đuôi. Hai: cái giá của
LAE ở phân phối chuẩn là hệ số $\pi/2 = 1{,}5707963268$, tức **mất 57,0796 % hiệu suất**. Đó
chính là "phí bảo hiểm" mà mục 5 của bài báo nói tới (mục 2.20).

**Điều bài báo KHÔNG nói ra:** nó không nói kết quả của Laplace áp cho hồi quy **qua gốc toạ
độ** — tức không có hệ số chặn — trong khi toàn bộ mục 4 của chính bài báo lại **đòi** ma
trận $X$ có một cột toàn số 1 (điều kiện (ii) của Định lý 4.2). Hai thiết lập khác nhau, và
bài báo đặt chúng cạnh nhau mà không nối.

### 2.3 Lời hứa "gần tám mươi phần trăm" — trang 36

![Đoạn về ước lượng Gastwirth](<paper_kb1978_gastwirth_80.png>)

**Hình 5 — Đoạn giới thiệu ước lượng Gastwirth, trang 36.**

**Trích nguyên văn:**

*"For example, the weighted average of the 1/3, 1/2, and 2/3 quantiles with weights .3, .4,
.3 has asymptotic efficiency of nearly eighty per cent for the Gaussian, Laplace, logistic,
and Cauchy distributions. In contrast, the sample mean has asymptotic efficiency of one in
the Gaussian case, but is less than half as efficient as the median for the Laplace
distribution and has zero efficiency for the Cauchy distribution."*

Dịch: *"Chẳng hạn, trung bình có trọng số của các phân vị 1/3, 1/2 và 2/3 với trọng số .3,
.4, .3 có hiệu suất tiệm cận gần tám mươi phần trăm cho các phân phối Gauss, Laplace,
logistic và Cauchy. Ngược lại, trung bình mẫu có hiệu suất tiệm cận bằng một trong trường hợp
Gauss, nhưng kém hơn một nửa so với trung vị ở phân phối Laplace, và có hiệu suất bằng không
ở phân phối Cauchy."*

**Câu này kiểm được, và phải đọc nó là một SÀN chứ không phải một khoảng.** "Gần tám mươi
phần trăm" áp cho **cả bốn** phân phối cùng lúc, nên nó nói về giá trị **nhỏ nhất**. Tính
bằng Định lý 4.1 (mục 2.16) — chính bài báo trang 43 mời làm đúng thế:

| $F$ | Thông tin Fisher $I(F)$ | Phương sai tiệm cận của Gastwirth | Hiệu suất $1/[I(F)\,V]$ |
|---|---:|---:|---:|
| Chuẩn | $1{,}0000000000$ | $1{,}2566824519$ | $\mathbf{0{,}7957459726}$ |
| Laplace | $1{,}0000000000$ | $1{,}1800000000$ | $0{,}8474576271$ |
| Logistic | $0{,}3333333333$ | $3{,}2950000000$ | $0{,}9104704097$ |
| Cauchy | $0{,}5000000000$ | $2{,}5002997816$ | $0{,}7999040814$ |

Giá trị nhỏ nhất trong bốn là **$0{,}7957459726$**, tức $79{,}57\%$ — câu của bài báo đúng,
và đúng sát sàn. Bốn giá trị nằm trong khoảng rộng $0{,}1147244371$.

Đối chiếu với trung bình mẫu trên cùng thước đo: $1{,}0000$ · $0{,}5000$ · $0{,}9119$ ·
**$0$**. Đó là điểm của cả đoạn văn: trung bình hoàn hảo ở đúng một chỗ và mất sạch ở một chỗ
khác, còn Gastwirth giữ $0{,}7957$–$0{,}9105$ trên cả bốn.

**Một chi tiết bài báo KHÔNG nói ra, và phép đo lộ ra:** ở phân phối logistic, trung bình mẫu
có hiệu suất $0{,}9118906528$, **cao hơn** Gastwirth ($0{,}9104704097$). Chênh lệch nhỏ —
$0{,}0014$ — nhưng nó có thật, và nó nhắc rằng "gần 80 % ở khắp nơi" không có nghĩa "tốt hơn
ở khắp nơi".

### 2.4 Bảng I đọc lại bằng phép đo — trang 37

Bảng I đã dán ở **Hình 1**. Mục này dựng lại nó.

Trước hết một điều về nguồn, và nó quyết định cách đọc mọi chênh lệch dưới đây: chú thích
**a** của Bảng I ghi *"Abstracted from Exhibit 5 in Andrews, et al. [3]"* — **Bảng I không
phải phép đo của Koenker & Bassett.** Nó được chép từ Nghiên cứu Bền vững Princeton. Nên mọi
chỗ không khớp là chênh với Andrews và cộng sự, không phải với bài báo này.

**Bước 1 — điều kiện duy nhất của chính bài báo hỏng ở $T=20$.** Trang 41 nói nghiệm duy nhất
đòi $T\theta$ **không nguyên**. Với $T=20$:

| Ước lượng | Các mức $\theta$ | $T\theta$ | Duy nhất? |
|---|---|---|---|
| Trung vị | $0{,}5$ | $10{,}00$ | **không** |
| Gastwirth | $1/3$ · $1/2$ · $2/3$ | $6{,}67$ · $10{,}00$ · $13{,}33$ | hai trong ba thì có |
| Trimean | $1/4$ · $1/2$ · $3/4$ | $5{,}00$ · $10{,}00$ · $15{,}00$ | **cả ba đều không** |

Nghĩa là: ở đúng cỡ mẫu mà Bảng I dùng, việc chọn giá trị nào trong đoạn nghiệm là một **lựa
chọn quy ước**, không phải một định nghĩa. Thử cả hai quy ước rồi để tổng sai lệch quyết định
(luật "thử cả hai cách đọc"):

| Quy ước | Tổng sai lệch tuyệt đối trên 24 ô ngoài cột Cauchy |
|---|---:|
| lấy **trung bình** hai thống kê thứ tự kề nhau | $\mathbf{1{,}742388}$ |
| lấy thống kê thứ tự **dưới** | $1{,}999683$ |

Quy ước trung bình khớp hơn, nên bảng dựng lại dùng nó.

**Bước 2 — Bảng I dựng lại**, $T=20$, 120 000 lần lặp, trung bình ba hạt giống:

| Ước lượng | Normal | 10 % 3σ | 10 % 10σ | Laplace | Cauchy |
|---|---:|---:|---:|---:|---:|
| Mean | $1{,}0016$ | $1{,}7961$ | $10{,}9091$ | $1{,}9958$ | $7\,318\,896{,}7$ |
| 10 % trimmed mean | $1{,}0562$ | $1{,}3295$ | $\mathbf{2{,}0006}$ | $1{,}5522$ | $8{,}4$ |
| 25 % trimmed mean | $1{,}1870$ | $1{,}3983$ | $1{,}5128$ | $1{,}3255$ | $3{,}2$ |
| Median | $1{,}4695$ | $1{,}7016$ | $1{,}8028$ | $1{,}3329$ | $2{,}8$ |
| Gastwirth | $1{,}2202$ | $1{,}4346$ | $1{,}5454$ | $1{,}3447$ | $3{,}2$ |
| Trimean | $1{,}1525$ | $1{,}3785$ | $1{,}5409$ | $1{,}4016$ | $4{,}1$ |

Kết quả trên 24 ô ngoài cột Cauchy: lệch tuyệt đối **trung bình $0{,}072600$**, lớn nhất
$0{,}630925$; **21/24 ô lệch dưới $0{,}10$** và **22/24 ô lệch dưới $0{,}20$**; chỉ
**6/24 ô** khớp đến đúng hai chữ số thập phân bài báo in.

**Hai ô lệch nhiều nhất, và chúng lệch vì hai lý do khác hẳn nhau:**

| Ô | Bảng I in | Dựng lại | Lệch |
|---|---:|---:|---:|
| Mean / 10 % 10σ | $11{,}54$ | $10{,}9091$ | $-0{,}6309$ |
| 10 % trimmed mean / 10 % 10σ | $1{,}46$ | $2{,}0006$ | $+0{,}5406$ |

Ô thứ nhất có một **giá trị lý thuyết chính xác**, và đó là thứ phân xử. Phương sai của trung
bình mẫu dưới một hỗn hợp Gauss $0{,}9\Phi(1)+0{,}1\Phi(c)$ là $0{,}9\times1+0{,}1\times c^2$
**đúng**, với mọi $T$. Tính cho cả hai cột hỗn hợp:

| Cột | Lý thuyết | Bảng I in | Bảng I lệch lý thuyết | Dựng lại | Dựng lại lệch lý thuyết |
|---|---:|---:|---:|---:|---:|
| 10 % 3σ | $1{,}800000$ | $1{,}88$ | $+0{,}080000$ ($+4{,}4444\%$) | $1{,}796074$ | $-0{,}003926$ ($-0{,}2181\%$) |
| 10 % 10σ | $10{,}900000$ | $11{,}54$ | $+0{,}640000$ ($+5{,}8716\%$) | $10{,}909075$ | $+0{,}009075$ ($+0{,}0833\%$) |

Đọc bảng này: phép dựng lại khớp lý thuyết trong **0,22 %**, còn hai ô của Bảng I lệch
**4,44 %** và **5,87 %**. Chúng là nhiễu Monte-Carlo của nghiên cứu Princeton, không phải một
đại lượng khác.

Ô thứ hai thì có một chẩn đoán cấu trúc. Cắt $10\%$ của $T=20$ là **2 quan sát mỗi đầu**;
số điểm bị nhiễm là $\text{Binomial}(20;\,0{,}1)$, kỳ vọng đúng 2. Tính:

$$P(\text{số điểm nhiễm} \ge 3) = 0{,}3230731948$$

tức **32,31 % số mẫu vẫn còn điểm nhiễm sau khi cắt**, và mỗi điểm sót lại ở $10\sigma$ kéo
phương sai lên rất mạnh. Đó là lý do phép đo cho $2{,}0006$ chứ không phải $1{,}46$. Con số
$1{,}46$ của Andrews và cộng sự đòi một cách cắt mạnh tay hơn định nghĩa dùng ở đây; bài viết
này **không dựng lại được nó**, và ghi ra như thế.

**Bước 3 — cột Cauchy phải đọc riêng, vì lý thuyết và số đo khác nhau về BẢN CHẤT.**

| Ước lượng | Bảng I in | Ba lần chạy | Độ rộng khoảng | Độ rộng / giá trị bài báo |
|---|---:|---|---:|---:|
| Mean | $12\,548{,}0$ | $2\,660\,671{,}83$ · $1\,294\,552{,}54$ · $18\,001\,465{,}84$ | $16\,706\,913{,}30$ | $\mathbf{1\,331{,}440333}$ |
| 10 % trimmed mean | $7{,}3$ | $8{,}59$ · $8{,}24$ · $8{,}28$ | $0{,}34$ | $0{,}047030$ |
| 25 % trimmed mean | $3{,}1$ | $3{,}18$ · $3{,}18$ · $3{,}18$ | $0{,}00$ | $0{,}000655$ |
| Median | $2{,}9$ | $2{,}79$ · $2{,}79$ · $2{,}79$ | $0{,}01$ | $0{,}002361$ |
| Gastwirth | $3{,}1$ | $3{,}15$ · $3{,}15$ · $3{,}15$ | $0{,}00$ | $0{,}000754$ |
| Trimean | $3{,}9$ | $4{,}10$ · $4{,}08$ · $4{,}10$ | $0{,}02$ | $0{,}005732$ |

Năm hàng dưới ổn định tới hai chữ số; hàng đầu đổi **hơn một bậc độ lớn** giữa ba hạt giống,
và độ rộng khoảng của nó lớn gấp **$1\,331{,}440333$ lần** chính con số bài báo in.
Không có gì sai ở đây: phương sai của trung bình mẫu dưới Cauchy là **vô cực**, nên không con
số hữu hạn nào tái lập được. Ô $12\,548{,}0$ của Bảng I là một lần chạy cụ thể của nghiên cứu
Princeton, và **nó không được đọc như một phép đo**.

**Bước 4 — so giá trị tiệm cận với giá trị $T=20$.** Trang 43 nói dịch danh: *"The asymptotic
variance of the Gastwirth or trimean estimator of Table I can be easily calculated in a
similar manner from Theorem 4.1 for arbitrary distributions."* Làm đúng thế:

| Ước lượng | $F$ | Tiệm cận | Bảng I ($T=20$) | Chênh |
|---|---|---:|---:|---:|
| Mean | Normal | $1{,}0000000000$ | $1{,}00$ | $+0{,}000000$ |
| Mean | Laplace | $2{,}0000000000$ | $2{,}10$ | $+0{,}100000$ |
| Median | Normal | $1{,}5707963268$ | $1{,}50$ | $-0{,}070796$ |
| Median | Laplace | $1{,}0000000000$ | $1{,}37$ | $+0{,}370000$ |
| Gastwirth | Normal | $1{,}2566824519$ | $1{,}23$ | $-0{,}026682$ |
| Gastwirth | Laplace | $1{,}1800000000$ | $1{,}35$ | $+0{,}170000$ |
| Trimean | Normal | $1{,}1951616383$ | $1{,}15$ | $-0{,}045162$ |
| Trimean | Laplace | $1{,}2500000000$ | $1{,}43$ | $+0{,}180000$ |

Đọc cột "chênh" theo phân phối: ở **Normal** cả bốn giá trị $T=20$ đều **thấp hơn** tiệm cận;
ở **Laplace** cả bốn đều **cao hơn**. Đó không phải ngẫu nhiên mà là một hiệu ứng hữu hạn mẫu
có dấu rõ ràng, và nó không được bài báo bình luận.

Cột Cauchy, giá trị tiệm cận: Mean **vô cực** (Bảng I in $12\,548{,}0$) · Median
$2{,}4674011003$ (in $2{,}9$) · Gastwirth $2{,}5002997816$ (in $3{,}1$) · Trimean
$3{,}0842513753$ (in $3{,}9$). Tỉ số "trung bình / trung vị" ở cột Cauchy: Bảng I cho
$4\,326{,}8966$; lý thuyết cho **vô cực**. Đó là ô duy nhất của cả bảng mà lý thuyết và số đo
khác nhau về **bản chất** chứ không về độ lớn.

![Đoạn dẫn vào Bảng I](<paper_kb1978_bang_I_dan.png>)

**Hình 6 — đoạn thân bài dẫn vào Bảng I, trang 36.** Đây là "nhận xét của chính tác giả" cho
Hình 1, đã dịch ở mục 1.1.

![Chú thích 13, trang 37](<paper_kb1978_hampel_te_nhat.png>)

**Hình 7 — Chú thích 13 và 14, trang 37.**

**Trích nguyên văn chú thích 13:** *"Based on the full study of 65 different estimators and
ten distributions, Hampel concludes that the sample mean is the "clear candidate for ... worst
estimator of the study." Andrews, et al., [3, p. 239]."*

Dịch: *"Dựa trên nghiên cứu đầy đủ gồm 65 ước lượng khác nhau và mười phân phối, Hampel kết
luận rằng trung bình mẫu là "ứng viên rõ ràng cho ... ước lượng tệ nhất của nghiên cứu."
Andrews và cộng sự, [3, tr. 239]."*

**Cách đọc.** Câu mạnh nhất về bình phương tối thiểu trong cả bài báo nằm ở một **chú thích
chân trang**, và nó là lời của người khác. Bảng I chỉ trưng 6 trên 65 ước lượng và 5 trên 10
phân phối — nên con số "65 và 10" ở chú thích là thứ cho biết Bảng I nhỏ tới mức nào so với
nguồn của nó.

### 2.5 Định nghĩa phân vị mẫu — mục 3, trang 38

![Định nghĩa phân vị mẫu](<paper_kb1978_dinh_nghia_phan_vi_mau.png>)

**Hình 8 — Mở đầu mục 3 và định nghĩa phân vị mẫu thứ $\theta$, trang 38.**

**Trích nguyên văn:**

*"Our point of departure is an elementary definition of the sample quantiles which, by
circumventing the usual reliance on an ordered set of sample observations, is readily
extended to the linear model. As above, let $\{y_t: t=1,\ldots,T\}$ be a random sample on a
random variable $Y$ having distribution function $F$. Then the $\theta$th sample quantile,
$0<\theta<1$, may be defined as any solution to the minimization problem:"*

Dịch: *"Điểm xuất phát của chúng tôi là một định nghĩa sơ cấp về các phân vị mẫu mà, bằng
cách tránh né sự phụ thuộc thông thường vào một tập quan sát đã sắp thứ tự, mở rộng được dễ
dàng sang mô hình tuyến tính. Như trên, gọi $\{y_t: t=1,\ldots,T\}$ là một mẫu ngẫu nhiên
trên biến ngẫu nhiên $Y$ có hàm phân phối $F$. Khi đó phân vị mẫu thứ $\theta$, với
$0<\theta<1$, định nghĩa được là **nghiệm bất kỳ** của bài toán cực tiểu hoá:"*

$$\min_{b\in\mathbb{R}}\left[\sum_{t\in\{t:\,y_t\ge b\}}\theta\lvert y_t-b\rvert
+\sum_{t\in\{t:\,y_t< b\}}(1-\theta)\lvert y_t-b\rvert\right]$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $b$ | biến chạy — ứng viên cho giá trị phân vị | vô hướng |
| $\theta$ | mức phân vị cần hỏi | vô hướng, $0<\theta<1$ |
| $\{t: y_t\ge b\}$ | tập chỉ số các quan sát **nằm trên hoặc bằng** $b$ | tập con của $\{1,\ldots,T\}$ |
| $\lvert y_t-b\rvert$ | khoảng cách từ quan sát tới ứng viên | vô hướng $\ge0$ |

**Ý nghĩa trừu tượng — nó nói điều gì.** Định nghĩa này **định nghĩa phân vị mà không hề sắp
xếp gì cả**, và đó là toàn bộ mẹo. Nó nói: phân vị thứ $\theta$ là điểm mà, nếu ta phải trả
tiền cho mọi khoảng cách tới nó, với **giá khác nhau cho hai phía** — $\theta$ cho phía trên,
$1-\theta$ cho phía dưới — thì tổng tiền là nhỏ nhất.

Vì sao chỗ cực tiểu rơi đúng vào phân vị: dịch $b$ lên một lượng nhỏ $\delta$ làm mỗi quan
sát ở **phía trên** rẻ đi $\theta\delta$ và mỗi quan sát ở **phía dưới** đắt thêm
$(1-\theta)\delta$. Cân bằng khi $\theta\cdot(\text{số điểm trên}) = (1-\theta)\cdot(\text{số
điểm dưới})$, tức khi tỉ lệ điểm nằm dưới đúng bằng $\theta$. Không cần sắp xếp, không cần
thống kê thứ tự — chỉ cần **đếm hai phía**.

Hai chữ **"any solution"** (nghiệm bất kỳ) là chỗ bài báo cẩn thận và người đọc hay lướt qua:
tập nghiệm có thể là **một đoạn**, không phải một điểm. Với $T$ chẵn và $\theta=1/2$, mọi giá
trị giữa hai thống kê thứ tự giữa đều là nghiệm.

Chính bài báo nói câu tiếp theo, và nó đáng ghi lại:

*"The case of the median ($\theta = 1/2$) is, of course, well known, but the general result
has languished in the status of curiosum—appearing for example as an exercise in Ferguson
[13]."*

Dịch: *"Trường hợp trung vị ($\theta = 1/2$) dĩ nhiên đã nổi tiếng, nhưng kết quả tổng quát
thì mòn mỏi ở địa vị của một chuyện lạ — xuất hiện chẳng hạn như một bài tập trong Ferguson
[13]."*

**Điều đó nói rằng đóng góp của mục 3 KHÔNG phải định nghĩa này** — định nghĩa đã có, dưới
dạng một bài tập trong sách. Đóng góp là nhận ra rằng **chính vì nó không sắp xếp gì** nên nó
mở rộng được, trong khi mọi định nghĩa dựa trên thống kê thứ tự thì không.

Kiểm bằng số: giải bài toán trên bằng quy hoạch tuyến tính rồi so với `np.quantile` trên
$T=19$ điểm (chọn 19 để $T\theta$ không nguyên ở cả năm mức, tránh đúng chỗ nghiệm không duy
nhất):

| $\theta$ | Nghiệm bài toán của bài báo | `np.quantile(..., method="inverted_cdf")` | Lệch |
|---|---:|---:|---:|
| $0{,}10$ | $-1{,}0217387112$ | $-1{,}0217387112$ | $0$ |
| $0{,}25$ | $-0{,}5480597228$ | $-0{,}5480597228$ | $0$ |
| $0{,}50$ | $0{,}1641357216$ | $0{,}1641357216$ | $0$ |
| $0{,}75$ | $0{,}8684699198$ | $0{,}8684699198$ | $0$ |
| $0{,}90$ | $1{,}9737383178$ | $1{,}9737383178$ | $0$ |

Lệch lớn nhất trên năm mức: $0{,}000\times10^{0}$ — trùng **từng bit**.

### 2.6 Định nghĩa phân vị hồi quy — mục 3, trang 38

![Định nghĩa phân vị hồi quy](<paper_kb1978_dinh_nghia_phan_vi_hoi_quy.png>)

**Hình 9 — Định nghĩa phân vị hồi quy thứ $\theta$, trang 38.**

**Trích nguyên văn:**

*"Letting $\{x_t: t=1,\ldots,T\}$ denote a sequence of (row) $K$-vectors of a known design
matrix, suppose $\{y_t: t=1,\ldots,T\}$ is a random sample on the regression process
$u_t = y_t - x_t\beta$ having distribution function $F$. The $\theta$th regression quantile,
$0<\theta<1$, is defined as any solution to the minimization problem:"*

Dịch: *"Gọi $\{x_t: t=1,\ldots,T\}$ là dãy các vectơ (hàng) $K$ chiều của một ma trận thiết
kế đã biết, giả sử $\{y_t: t=1,\ldots,T\}$ là một mẫu ngẫu nhiên trên quá trình hồi quy
$u_t = y_t - x_t\beta$ có hàm phân phối $F$. **Phân vị hồi quy thứ $\theta$**, với
$0<\theta<1$, được định nghĩa là nghiệm bất kỳ của bài toán cực tiểu hoá:"*

$$\min_{b\in\mathbb{R}^{K}}\left[\sum_{t\in\{t:\,y_t\ge x_tb\}}\theta\lvert y_t-x_tb\rvert
+\sum_{t\in\{t:\,y_t< x_tb\}}(1-\theta)\lvert y_t-x_tb\rvert\right]$$

**Ý nghĩa trừu tượng.** Đặt hai công thức cạnh nhau thì thấy toàn bộ bài báo nằm ở **một phép
thay**: $b$ thành $x_tb$, và $\mathbb{R}$ thành $\mathbb{R}^K$. Không có gì khác. Đó chính là
nghĩa của hai chữ *"direct generalization"* ở mục 1.2.

Điều đáng nhớ là **vì sao phép thay đó hợp lệ**: định nghĩa ở mục 2.5 chỉ nói về **khoảng
cách từ mỗi quan sát tới một mức**, và "một mức" thì thay bằng "một siêu phẳng" được ngay.
Nếu định nghĩa phân vị bằng thống kê thứ tự thì không có phép thay nào tương ứng, vì không
có cách nào sắp thứ tự các điểm theo một siêu phẳng chưa biết.

Câu chốt của mục, chép nguyên văn:

*"In the location model ($K = 1$, $x_t = 1$, for all $t$) the two minimization problems
coincide. The least absolute error estimator is the regression median, i.e., the regression
quantile for $\theta = 1/2$."*

Dịch: *"Trong mô hình vị trí ($K = 1$, $x_t = 1$ với mọi $t$) hai bài toán cực tiểu hoá trùng
nhau. Ước lượng sai số tuyệt đối nhỏ nhất chính là **trung vị hồi quy**, tức phân vị hồi quy
cho $\theta = 1/2$."*

**Ký hiệu $\rho_\theta$ KHÔNG có trong bài báo này.** Bài báo viết hai tổng tách theo dấu
phần dư. Cách viết gọn thành một dòng mà mọi tài liệu hôm nay dùng —

$$\rho_\theta(u) = u\bigl(\theta - \mathbb{1}[u<0]\bigr)$$

— là cách viết của đời sau. Hai cách hoàn toàn tương đương; kiểm trên lưới 100 001 điểm
trong $[-5,5]$ cho lệch lớn nhất $0{,}000\times10^{0}$ ở cả ba mức $\theta = 0{,}1$; $0{,}5$;
$0{,}9$. Giá trị cụ thể:

| $u$ | $\rho_{0{,}1}(u)$ | $\rho_{0{,}5}(u)$ | $\rho_{0{,}9}(u)$ |
|---:|---:|---:|---:|
| $-4$ | $3{,}600000$ | $2{,}000000$ | $0{,}400000$ |
| $-1$ | $0{,}900000$ | $0{,}500000$ | $0{,}100000$ |
| $0$ | $0$ | $0$ | $0$ |
| $+1$ | $0{,}100000$ | $0{,}500000$ | $0{,}900000$ |
| $+4$ | $0{,}400000$ | $2{,}000000$ | $3{,}600000$ |

Đọc hàng $u=-4$ và $u=+4$ ở cột $\theta=0{,}9$: đoán **thiếu** 4 đơn vị bị phạt $3{,}6$; đoán
**thừa** 4 đơn vị chỉ bị phạt $0{,}4$. Gấp **9 lần**, đúng bằng $0{,}9/0{,}1$. Đó là toàn bộ
cơ chế đẩy đường lên cao.

### 2.7 Định lý 3.1 — nghiệm đi qua đúng $K$ quan sát — trang 39

![Định lý 3.1](<paper_kb1978_dinh_ly_3_1.png>)

**Hình 10 — Định nghĩa tập $H$ và Định lý 3.1, trang 39.**

**Trích nguyên văn:**

*"$H=\{h\in\mathcal{H}\mid \text{rank } X(h)=K\}$. THEOREM 3.1: If $X$ has rank $K$ then the
set of regression quantiles, $B^*(\theta)$, has at least one element of the form
$\beta^*(\theta) = X(h)^{-1}y(h)$ for some $h\in H$. Moreover, $B^*(\theta)$, is the convex
hull of all solutions having this form."*

Dịch: *"$H=\{h\in\mathcal{H}\mid \text{hạng } X(h)=K\}$. **ĐỊNH LÝ 3.1**: Nếu $X$ có hạng
$K$ thì tập các phân vị hồi quy $B^*(\theta)$ có ít nhất một phần tử dạng
$\beta^*(\theta) = X(h)^{-1}y(h)$ với một $h\in H$ nào đó. Hơn nữa, $B^*(\theta)$ là **bao
lồi** của mọi nghiệm có dạng này."*

Chứng minh của bài báo dài đúng một câu: *"This result follows immediately from the linear
programming formulation of the defining minimization problem; see Appendix 1 and the recent
paper by Abdelmalek [1] for details."* — *"Kết quả này suy ra ngay từ cách phát biểu bài toán
cực tiểu hoá dưới dạng quy hoạch tuyến tính; xem Phụ lục 1 và bài báo gần đây của Abdelmalek
[1] để biết chi tiết."*

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\mathcal{H}$ | tập **mọi** tập con gồm $K$ chỉ số của $\{1,\ldots,T\}$ | có $\binom{T}{K}$ phần tử |
| $h$ | một tập con như thế | — |
| $X(h)$ | ma trận vuông gồm đúng $K$ dòng của $X$ ứng với $h$ | $K\times K$ |
| $y(h)$ | vectơ gồm đúng $K$ phần tử của $y$ ứng với $h$ | $\mathbb{R}^{K}$ |
| $H$ | những $h$ làm $X(h)$ **khả nghịch** | tập con của $\mathcal{H}$ |

**Ý nghĩa trừu tượng — nó nói điều gì.** $X(h)^{-1}y(h)$ là nghiệm của hệ $K$ phương trình
$K$ ẩn $X(h)b = y(h)$, tức là **siêu phẳng đi chính xác qua $K$ điểm dữ liệu đó**. Định lý
nói: dù bài toán cực tiểu hoá chạy trên cả $\mathbb{R}^K$ liên tục, nghiệm của nó luôn tìm
được trong một tập **HỮU HẠN** gồm nhiều nhất $\binom{T}{K}$ ứng viên.

Đó là điều làm bài toán giải được. Và nó là khác biệt cấu trúc lớn nhất so với bình phương
tối thiểu: đường bình phương tối thiểu nói chung **không đi qua quan sát nào cả**, còn đường
phân vị hồi quy luôn **neo vào đúng $K$ quan sát**.

Bài báo nói rõ điều này tương ứng với cái gì ở mô hình vị trí:

![Remark về thống kê thứ tự](<paper_kb1978_remark_thong_ke_thu_tu.png>)

**Hình 11 — Remark sau Định lý 3.1, trang 39.**

*"REMARK: Sample quantiles in the location model are either identified with a single order
statistic from the observed sample, or, for example in the case of the median from an even
sample, they are identified with a closed interval between two adjacent order statistics.
Theorem 1 generalizes this feature to regression quantiles where normals to hyperplanes
defined by subsets of $K$ observations play the role of order statistics."*

Dịch: *"NHẬN XÉT: Phân vị mẫu trong mô hình vị trí hoặc được đồng nhất với **một** thống kê
thứ tự của mẫu quan sát, hoặc — chẳng hạn trong trường hợp trung vị của một mẫu chẵn — được
đồng nhất với một **đoạn đóng** giữa hai thống kê thứ tự kề nhau. Định lý 1 tổng quát hoá
đặc điểm này sang phân vị hồi quy, nơi các pháp tuyến của siêu phẳng xác định bởi các tập con
$K$ quan sát đóng vai trò của thống kê thứ tự."*

**Đây là câu trả lời cho bế tắc ở mục 1.2.** Thống kê thứ tự không mở rộng được sang mô hình
tuyến tính; thứ **thay thế** nó là các siêu phẳng qua $K$ điểm. Bài báo không nói "ta sắp xếp
dữ liệu theo một cách khác" — nó nói **đối tượng cơ sở đã đổi**, từ một điểm dữ liệu thành
một tổ hợp $K$ điểm dữ liệu.

Kiểm bằng số trên 15 cấu hình $(T,K,\theta)$ khác nhau — giải bằng quy hoạch tuyến tính rồi
đi tìm tập con $h$:

| $T$ | $K$ | $\theta$ | Số phần dư bằng 0 | Bằng $K$? | Lệch $\lVert X(h)^{-1}y(h)-\beta^*\rVert_\infty$ |
|---:|---:|---:|---:|---|---:|
| 30 | 2 | $0{,}15$ | 2 | có | $0{,}000\times10^{0}$ |
| 30 | 2 | $0{,}50$ | 2 | có | $2{,}220\times10^{-16}$ |
| 40 | 3 | $0{,}85$ | 3 | có | $1{,}332\times10^{-15}$ |
| 55 | 4 | $0{,}50$ | 4 | có | $5{,}551\times10^{-16}$ |
| 120 | 3 | $0{,}15$ | 3 | có | $1{,}145\times10^{-16}$ |

(Bảng trên là 5 dòng tiêu biểu; script in đủ 15.) Kết quả: **15/15 trường hợp** có
$X(h)^{-1}y(h)$ trùng nghiệm quy hoạch tuyến tính tới $10^{-8}$, và **15/15 trường hợp** có
đúng $K$ phần dư bằng 0.

**Điều định lý KHÔNG nói, và phải nói rõ kẻo đọc thành định luật:** "đúng $K$ phần dư bằng 0"
là điều đúng **trong vị trí tổng quát** — tức khi không có điểm trùng, không có ràng buộc phụ
thuộc và nghiệm không suy biến. Cụm *"vị trí tổng quát"* nghĩa là: không có sự trùng hợp
tình cờ nào giữa các điểm dữ liệu. Chính Bảng II của bài báo cho ba phản ví dụ, ở đúng ba mốc
$\theta$ mà nghiệm không duy nhất (mục 2.15).

### 2.8 Định lý 3.2 — bốn tính chất đồng biến đổi — trang 39

![Định lý 3.2](<paper_kb1978_dinh_ly_3_2.png>)

**Hình 12 — Định lý 3.2 và bốn tính chất, trang 39.**

**Trích nguyên văn:** *"THEOREM 3.2: If $\beta^*(\theta, y, X)\in B^*(\theta, y, X)$ then the
following are elements of the solution of the specified transformed problems:"*

$$
\begin{aligned}
&\text{(i)} && \beta^*(\theta,\lambda y,X) = \lambda\beta^*(\theta,y,X), && \lambda\in[0,\infty)\\
&\text{(ii)} && \beta^*(1-\theta,\lambda y,X) = \lambda\beta^*(\theta,y,X), && \lambda\in(-\infty,0]\\
&\text{(iii)} && \beta^*(\theta,y+X\gamma,X) = \beta^*(\theta,y,X)+\gamma, && \gamma\in\mathbb{R}^{K}\\
&\text{(iv)} && \beta^*(\theta,y,XA) = A^{-1}\beta^*(\theta,y,X), && A_{K\times K}\ \text{không suy biến}
\end{aligned}
$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\lambda$ | hệ số nhân vào $y$ — đổi **thang đo** của biến phụ thuộc | vô hướng |
| $\gamma$ | vectơ cộng vào qua $X\gamma$ — đổi **vị trí** | $\mathbb{R}^{K}$ |
| $A$ | ma trận đổi tham số của **thiết kế** | $K\times K$ khả nghịch |

**Ý nghĩa trừu tượng — bốn dòng nói bốn điều khác nhau, và dòng (ii) là dòng lạ nhất.**

* **(i)** Nhân $y$ với một số dương thì nghiệm nhân theo. Tức đổi đơn vị của $y$ — từ đồng
  sang nghìn đồng — không đổi bản chất mô hình.
* **(ii)** Nhân $y$ với một số **âm** thì $\theta$ **lật thành $1-\theta$**. Đây là điều
  không có tương tự ở bình phương tối thiểu, và nó hợp lý ngay khi nghĩ kỹ: lật dấu $y$ là
  lật ngược trên thành dưới, nên "mức mà 90 % nằm dưới" thành "mức mà 90 % nằm trên", tức
  mức mà 10 % nằm dưới.
* **(iii)** Cộng vào $y$ một thứ **nằm trong không gian cột của $X$** thì nghiệm cộng đúng
  lượng đó. Chỗ quan trọng: phải là $X\gamma$, không phải một vectơ bất kỳ.
* **(iv)** Đổi tham số hoá của thiết kế — ví dụ đổi đơn vị một cột, hay xoay hai cột — thì
  nghiệm đổi theo phép nghịch đảo. Tức **mô hình không đổi**, chỉ cách viết hệ số đổi.

Bài báo tóm bốn dòng đó ở trang 40: *"Note that (i) and (ii) imply $\beta^*(1/2)$ is scale
equivariant, (iii) states $\beta^*(\theta)$ is location (or "shift" or "regression")
equivariant, and (iv) states that $\beta^*(\theta)$ is equivariant to reparameterization of
design."*

Kiểm cả bốn bằng số trên $T=60$, $K=3$, nhiễu Laplace, $\theta=0{,}35$, với
$\lambda=2{,}75$ và $\lambda=-1{,}60$:

| Tính chất | Lệch |
|---|---:|
| (i) | $2{,}220\times10^{-16}$ |
| (ii) | $4{,}996\times10^{-15}$ |
| (iii) | $4{,}441\times10^{-16}$ |
| (iv) | $4{,}441\times10^{-16}$ |

Lệch lớn nhất trên cả bốn: $4{,}996\times10^{-15}$.

**Định lý này có một hệ quả mà bài báo KHÔNG rút ra, và mục 2.15 sẽ cần tới nó:** vì (i) và
(iii) nói nghiệm đổi **đúng theo** phép biến đổi, nên ngược lại, **một bảng nghiệm cố định
ghim chặt dữ liệu**: nếu hai bộ dữ liệu cho cùng một bảng nghiệm thì chúng không thể khác
nhau bởi một phép tịnh tiến $X\gamma$ hay một phép co dãn $\lambda$ nào. Đó là lý do phép
dựng lại ở mục 2.15 có cơ hội cho ra một đáp án duy nhất.

### 2.9 Hàm $\psi$ và ký hiệu $\operatorname{sgn}^*$ — trang 39–40

![Hàm psi](<paper_kb1978_ham_psi.png>)

**Hình 13 — Hàm mục tiêu $\psi$ viết lại thành một tổng, trang 39.**

**Trích nguyên văn:** *"PROOF: Let $\psi(b;\theta,y,X)=\sum_{\{t:y_t>x_tb\}}\theta\lvert
y_t-x_tb\rvert+\sum_{\{t:y_t<x_tb\}}(1-\theta)\lvert y_t-x_tb\rvert
=\sum_{t=1}^{T}[\theta-1/2+1/2\operatorname{sgn}(y_t-x_tb)][y_t-x_tb]$ where
$\operatorname{sgn}(u)$ takes values $1$, $0$, $-1$ as $u\gtreqless 0$."*

$$\psi(b;\theta,y,X)=\sum_{t=1}^{T}\Bigl[\theta-\tfrac12+\tfrac12\operatorname{sgn}(y_t-x_tb)\Bigr]\,[y_t-x_tb]$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\psi(b;\theta,y,X)$ | **giá trị** hàm mục tiêu tại ứng viên $b$ | vô hướng $\ge0$ |
| $\operatorname{sgn}(u)$ | $1$, $0$, $-1$ tuỳ $u>0$, $=0$, $<0$ | vô hướng |

**Ý nghĩa trừu tượng.** Dòng này gộp **hai tổng thành một**, và đó không phải chuyện làm gọn
cho đẹp: nó là bước bắt buộc để lấy đạo hàm theo hướng ở Định lý 3.3. Hệ số
$\theta-\tfrac12+\tfrac12\operatorname{sgn}(u)$ bằng $\theta$ khi $u>0$, bằng $\theta-1$ khi
$u<0$, và bằng $\theta-\tfrac12$ khi $u=0$ — tức nó **tự chọn đúng nhánh phạt**, không cần
viết hai tổng nữa. Chỗ $u=0$ nhận giá trị giữa, và đó là dấu hiệu đầu tiên rằng bài toán
không khả vi tại đó.

Lưu ý ký hiệu: **$\psi$ ở đây là TOÀN BỘ TỔNG mất mát**, không phải đạo hàm của hàm mất mát.
Bài [Huber Regression](<../Huber Regression/Huber Regression.md>) dùng $\psi$ cho đúng thứ ngược lại — hàm ảnh
hưởng $\psi=\rho'$, một hàm của một biến số. Đây là chỗ trùng tên nguy hiểm nhất giữa hai
bài, đã ghi ở mục 0.2.

![Ký hiệu sgn sao và công thức (3.2)](<paper_kb1978_sgn_sao.png>)

**Hình 14 — Công thức (3.2) và định nghĩa $\operatorname{sgn}^*$, trang 40.**

$$\psi'(b;w)=\sum_{t=1}^{T}\bigl[\tfrac12-\tfrac12\operatorname{sgn}^*(y_t-x_tb;\,-x_tw)-\theta\bigr]x_tw \tag{3.2}$$

$$\operatorname{sgn}^*(u;z)=\begin{cases}\operatorname{sgn}u & \text{nếu } u\ne0\\ \operatorname{sgn}z & \text{nếu } z=0\end{cases}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $w$ | **hướng** đạo hàm theo hướng, một vectơ khác 0 | $\mathbb{R}^{K}$ |
| $\psi'(b;w)$ | đạo hàm theo hướng $w$ tại $b$ | vô hướng |
| $\operatorname{sgn}^*(u;z)$ | phiên bản của $\operatorname{sgn}$ có **quy tắc phá hoà** khi $u=0$ | $\{-1,0,1\}$ |

**Ý nghĩa trừu tượng — và đây là chỗ cả bài báo xử lý điểm không khả vi.** Hàm mục tiêu có
góc nhọn tại mọi chỗ mà một phần dư bằng 0, nên **không có gradient**. Cách gỡ của bài báo
không phải làm trơn hàm, cũng không phải dùng dưới vi phân trừu tượng, mà là một mẹo gọn:
**hỏi đạo hàm theo TỪNG HƯỚNG một.**

Ở một điểm trơn, đạo hàm theo hướng $w$ và theo hướng $-w$ chỉ khác nhau dấu. Ở một góc nhọn
thì không — và chính chỗ đó, $\operatorname{sgn}^*$ quyết định quan sát nằm góc đó sẽ tính
như "ở trên" hay "ở dưới", **tuỳ theo ta đang đi về hướng nào**. Đối số thứ hai $-x_tw$ chính
là tốc độ mà phần dư thứ $t$ thay đổi khi ta bước theo $w$.

Cụm **"dưới vi phân"** nghĩa là: tại một góc nhọn, thay cho một tiếp tuyến duy nhất, hàm có
cả một tập tiếp tuyến hợp lệ. Bài báo tránh phải dựng tập đó bằng cách chỉ hỏi *"đi theo
hướng này thì hàm tăng hay giảm"*, và điều kiện tối ưu thành: **hàm tăng theo MỌI hướng**.

### 2.10 Định lý 3.3 và công thức (3.1) — điều kiện nghiệm duy nhất — trang 40

![Định lý 3.3](<paper_kb1978_dinh_ly_3_3.png>)

**Hình 15 — Định lý 3.3 và công thức (3.1), trang 40.**

**Trích nguyên văn:** *"THEOREM 3.3: If $F$ is continuous then with probability one:
$\beta^*=X(h)^{-1}y(h)$ is a unique solution to Problem $(P)$ if and only if,"*

$$(\theta-1)\iota'_K<\sum_{t\in\bar h}\bigl[\tfrac12-\tfrac12\operatorname{sgn}(y_t-x_t\beta^*)-\theta\bigr]x_tX(h)^{-1}<\theta\iota'_K \tag{3.1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\bar h$ | mọi quan sát **KHÔNG** thuộc $h$, tức $T-K$ quan sát còn lại | tập chỉ số |
| $x_tX(h)^{-1}$ | dòng $x_t$ viết lại trong **hệ toạ độ của $K$ điểm neo** | vectơ hàng $\mathbb{R}^{K}$ |
| $\iota'_K$ | vectơ hàng $K$ số 1 | $\mathbb{R}^{K}$ |

**Ý nghĩa trừu tượng — nó nói điều gì.** Bất đẳng thức kép này là **điều kiện tối ưu bậc
không** của bài toán, viết cho một tập neo $h$ cụ thể. Vế giữa là một vectơ $K$ chiều; mỗi toạ
độ của nó đo *"nếu xoay siêu phẳng quanh $K-1$ điểm neo còn lại, thả điểm neo thứ $k$ ra, thì
tổng mất mát thay đổi bao nhiêu"*. Nghiệm là duy nhất khi và chỉ khi **mọi hướng xoay như thế
đều làm hàm tăng**, tức mọi toạ độ nằm hẳn trong khoảng $(\theta-1,\ \theta)$.

Chỗ đáng nhớ: khoảng $(\theta-1,\ \theta)$ có **độ rộng đúng bằng 1**, không phụ thuộc
$\theta$. Đổi $\theta$ chỉ **trượt** khoảng đó dọc trục số. Đó lại là cùng một sự thật đã gặp
ở mục 2.6 — $\theta$ chỉ vào bài toán qua đúng một chỗ.

Kiểm bằng số: tính vế giữa tại nghiệm quy hoạch tuyến tính trên 12 cấu hình:

| $T$ | $K$ | $\theta$ | Vế giữa, từng toạ độ | Trong khoảng? | Biên gần nhất |
|---:|---:|---:|---|---|---:|
| 30 | 2 | $0{,}20$ | $-0{,}67777$ · $0{,}07777$ | đúng | $0{,}122230$ |
| 45 | 3 | $0{,}80$ | $0{,}74989$ · $0{,}38164$ · $0{,}26847$ | đúng | $0{,}050106$ |
| 70 | 2 | $0{,}50$ | $-0{,}35810$ · $0{,}35810$ | đúng | $0{,}141900$ |
| 90 | 3 | $0{,}20$ | $-0{,}78707$ · $-0{,}55189$ · $-0{,}06104$ | đúng | $0{,}012928$ |

Kết quả đầy đủ: **(3.1) đúng ở 12/12 trường hợp.**

**Phép thử ngược, và nó là phép thử sắc nhất của cả mục 2.** Nếu (3.1) chỉ là một bất đẳng
thức lỏng thì mọi tập neo đều thoả, và nó chẳng nói gì. Lấy $T=40$, $K=2$, $\theta=0{,}3$,
rồi tính vế giữa cho **mọi** tập con hai quan sát khác:

> thử **779** tập $h$ khác: **779** tập **VI PHẠM** (3.1), **0** tập thoả.

Tức (3.1) chọn ra **đúng một** tập neo trong 780 ứng viên. Đó là một điều kiện **cần và đủ**
thật sự, không phải một bất đẳng thức trang trí.

### 2.11 Công thức (3.6) — vì sao $T\theta$ phải không nguyên — trang 40–41

![Công thức (3.6)](<paper_kb1978_cong_thuc_3_6.png>)

**Hình 16 — Remark và công thức (3.6), trang 40.**

**Trích nguyên văn:** *"REMARK: It is perhaps instructive to pause to consider the ordinary
sample quantiles in the light of Theorem 3.3. If $x_t=1$, $t=1,\ldots,T$, so
$H=\mathcal{T}$ and $F$ is continuous, then Theorem 3.3 asserts that $\beta^*(\theta)=y(h)$
is a unique $\theta$th sample quantile if and only if,"*

$$\theta-1<\sum_{t\in\bar h}\bigl[\tfrac12-\tfrac12\operatorname{sgn}(y_t-y(h))-\theta\bigr]<\theta \tag{3.6}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $y(h)$ | giá trị của **một** quan sát — ở mô hình vị trí, $h$ chỉ gồm một chỉ số | vô hướng |
| $\bar h$ | $T-1$ quan sát còn lại | tập chỉ số |
| $\operatorname{sgn}(y_t-y(h))$ | $+1$ nếu $y_t$ nằm trên ứng viên, $-1$ nếu nằm dưới | $\{-1,+1\}$ |

**Ý nghĩa trừu tượng — đây là (3.1) đã rút hết ma trận ra khỏi nó.** Khi $x_t=1$ với mọi $t$
thì $X(h)^{-1}=1$, nên thừa số $x_tX(h)^{-1}$ ở (3.1) biến mất và vế giữa từ một vectơ $K$
chiều tụt xuống thành **một con số**. Con số ấy là một **phép đếm hai phía**, đúng cùng phép
đếm đã thấy ở mục 2.5: mỗi quan sát nằm dưới đóng góp $1-\theta$, mỗi quan sát nằm trên đóng
góp $-\theta$.

Nói cách khác, (3.6) là chỗ toàn bộ bộ máy của mục 3 **quay về đúng thứ ai cũng biết** — và
việc nó quay về được chính là bằng chứng rằng định nghĩa ở mục 2.6 là một mở rộng đúng, chứ
không phải một đối tượng mới ngẫu nhiên trùng tên.

Và bài báo đọc nó thành lời ngay ở trang 41:

*"The expression in brackets takes the values of $-\theta$ if $y_t>y(h)$ and $1-\theta$ if
$y_t<y(h)$, so (3.6) reduces to the requirement that the number of $y_t$'s less that $y(h)$
be strictly between $T\theta-1$ and $T\theta$. This in turn demands that $T\theta$ be
nonintegral."*

Dịch: *"Biểu thức trong ngoặc nhận giá trị $-\theta$ nếu $y_t>y(h)$ và $1-\theta$ nếu
$y_t<y(h)$, nên (3.6) thu gọn thành yêu cầu rằng **số các $y_t$ nhỏ hơn $y(h)$ phải nằm
NGHIÊM NGẶT giữa $T\theta-1$ và $T\theta$**. Điều này đến lượt nó đòi $T\theta$ phải **không
nguyên**."*

**Ý nghĩa trừu tượng.** Đây là chỗ một bất đẳng thức ma trận trang 40 biến thành một **phép
đếm** mà ai cũng kiểm được bằng tay. Và câu cuối là một phát biểu rất gọn về khi nào phân vị
mẫu **không** duy nhất: đúng khi $T\theta$ là số nguyên, vì khi ấy không có số nguyên nào nằm
nghiêm ngặt giữa $T\theta-1$ và $T\theta$.

Kiểm bằng số:

| $T$ | $\theta$ | $T\theta$ | Nguyên? | Số $y_t$ nhỏ hơn phân vị mẫu | Nằm nghiêm ngặt trong $(T\theta-1,\,T\theta)$? |
|---:|---:|---:|---|---:|---|
| 19 | $0{,}25$ | $4{,}75$ | không | 4 | đúng |
| 19 | $0{,}50$ | $9{,}50$ | không | 9 | đúng |
| 19 | $0{,}90$ | $17{,}10$ | không | 17 | đúng |
| 20 | $0{,}25$ | $5{,}00$ | **có** | 5 | **trượt** |
| 20 | $0{,}50$ | $10{,}00$ | **có** | 9 | **trượt** |
| 40 | $0{,}50$ | $20{,}00$ | **có** | 20 | **trượt** |

Cả ba dòng có $T\theta$ nguyên đều trượt bất đẳng thức ngặt — **đúng như bài báo báo trước**.
Đó không phải lỗi; nó là điều bài báo nói ra một trang trước khi nó xảy ra.

Và nó là lý do mục 2.4 phải chọn một quy ước khi dựng lại Bảng I: cỡ mẫu $T=20$ của Bảng I
rơi đúng vào trường hợp $T\theta$ nguyên ở cả ba mức của trimean.

### 2.12 Định lý 3.4 và công thức (3.7) — và một chỗ in sai ở Remark ngay sau nó — trang 41

![Định lý 3.4](<paper_kb1978_dinh_ly_3_4.png>)

**Hình 17 — Định lý 3.4 và công thức (3.7), trang 41.**

**Trích nguyên văn:** *"THEOREM 3.4: Let $P(u^*(\theta))$, $N(u^*(\theta))$, and
$Z(u^*(\theta))$ denote the number of positive, negative, and zero elements in the vector
$u^*(\theta)=y-X\beta^*(\theta)$. Then if $X$ contains a column of ones,"*

$$N(u^*)\le T\theta\le T-P(u^*)=N(u^*)+Z(u^*) \tag{3.7}$$

*"for every $\beta^*(\theta)\in B^*(\theta)$. If $\beta^*$ is unique, i.e., $\beta^*=B^*$,
then the inequalities hold strictly."*

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $P(u^*)$ | số phần dư **dương** — quan sát nằm **trên** siêu phẳng | số nguyên |
| $N(u^*)$ | số phần dư **âm** — quan sát nằm **dưới** | số nguyên |
| $Z(u^*)$ | số phần dư **bằng 0** — quan sát nằm **trên** siêu phẳng | số nguyên |

**Ý nghĩa trừu tượng.** Đây là phát biểu chặt chẽ của câu nói nôm na *"khoảng $\theta$ phần
dữ liệu nằm dưới đường"*, và điều đáng nhớ là **nó là một bất đẳng thức KẸP, không phải một
đẳng thức**. Không có công thức đếm chính xác nào cả; cái có là $T\theta$ bị kẹp giữa $N$ và
$N+Z$, với $Z$ thường bằng $K$.

Đây cũng là chỗ điều kiện **"$X$ có một cột toàn số 1"** lần đầu trở nên bắt buộc. Lý do nằm
ngay trong chứng minh: nó lấy $w^+=(1,0,\ldots,0)$ và $w^-=(-1,0,\ldots,0)$, và chỉ với cột
toàn 1 thì $x_tw^+=1$ với **mọi** $t$ — tức dịch cả siêu phẳng lên hoặc xuống một lượng như
nhau ở khắp nơi.

Kiểm trên 20 cấu hình:

| $T$ | $K$ | $\theta$ | $N$ | $Z$ | $P$ | $T\theta$ | (3.7) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 2 | $0{,}10$ | 2 | 2 | 21 | $2{,}50$ | đúng |
| 40 | 2 | $0{,}50$ | 19 | 2 | 19 | $20{,}00$ | đúng |
| 60 | 3 | $0{,}37$ | 21 | 3 | 36 | $22{,}20$ | đúng |
| 100 | 4 | $0{,}90$ | 89 | 4 | 7 | $90{,}00$ | đúng |

(4 dòng tiêu biểu; script in đủ 20.) Kết quả: **(3.7) đúng ở 20/20 trường hợp**, và
$Z(u^*)=K$ ở **20/20 trường hợp** — đúng như Remark ngay sau đó nói.

**Phép thử ngược:** bỏ cột toàn 1 khỏi $X$ thì (3.7) gãy ngay — đo được $N=152$ trong khi
$T\theta=90{,}0$, tức $N>T\theta$.

![Remark và Định lý 3.5](<paper_kb1978_remark_va_dinh_ly_3_5.png>)

**Hình 18 — Remark sau Định lý 3.4, và Định lý 3.5, trang 41.**

**Trích nguyên văn của Remark:** *"REMARK: Note that if $F$ is continuous then $Z(u^*)=K$
with probability one, so there are at least $T\theta$ observations below the $\theta$th
regression quantile hyperplane and at most $T\theta+K$ observations above it."*

Dịch: *"NHẬN XÉT: Lưu ý rằng nếu $F$ liên tục thì $Z(u^*)=K$ với xác suất một, nên có **ít
nhất $T\theta$** quan sát nằm **dưới** siêu phẳng phân vị hồi quy thứ $\theta$ và **nhiều
nhất $T\theta+K$** quan sát nằm **trên** nó."*

**Nhận xét này không đọc được theo nghĩa đen, và phép đo cho biết phải đọc nó thế nào.** Thử
cả hai cách đọc trên đúng 20 cấu hình của bảng trên:

| Cách đọc | Phát biểu | Số trường hợp SAI |
|---|---|---:|
| (A) theo đúng chữ in | số quan sát **dưới** $=N \ge T\theta$ | **20/20** |
| (A) theo đúng chữ in | số quan sát **trên** $=P \le T\theta+K$ | **8/20** |
| (B) suy thẳng từ (3.7) | $N+Z \ge T\theta$ (dưới **hoặc nằm trên** siêu phẳng) | **0/20** |
| (B) suy thẳng từ (3.7) | $P \le T(1-\theta)$ | **0/20** |

Đọc bảng này cho hai kết luận khác nhau về hai vế:

* **Vế "below" đúng nếu hiểu "below" gồm cả các quan sát NẰM TRÊN siêu phẳng**, tức
  $N+Z\ge T\theta$ — điều này suy trực tiếp từ vế phải của (3.7). Đây gần như chắc chắn là ý
  tác giả, vì $Z=K$ điểm neo nằm đúng trên siêu phẳng và không thuộc phía nào.
* **Vế "above" thì không cứu được bằng cách đọc lại.** Chặn đúng suy từ (3.7) là
  $P\le T(1-\theta)$, và nó khác hẳn $T\theta+K$: ở $T=40$, $\theta=0{,}10$, đo được $P=35$
  trong khi $T\theta+K=4+2=6$. Thay $\theta$ bằng $1-\theta$ thì chặn thành đúng — đó là dấu
  vân tay của **một chỗ in sai một ký hiệu**.

Viết cho chính xác: **không kết luận nào của bài báo đổi vì chỗ này.** (3.7) là thứ được
chứng minh và được dùng ở mọi chỗ về sau; Remark chỉ là một câu đọc lại (3.7) thành lời, và
câu đó lệch ở một chữ.

### 2.13 Định lý 3.5 — đẩy quan sát ra xa thì đường không nhúc nhích — trang 41–42

Định lý 3.5 đã dán ở **Hình 18**, cùng khung với Remark ở mục trên.

**Trích nguyên văn:** *"THEOREM 3.5: If $\beta^*(\theta)\in B^*(\theta,y,X)$, then
$\beta^*(\theta)\in B^*(\theta, X\beta^*+Du^*, X)$ where $u^*=y-X\beta^*$ and $D$ is any
$T\times T$ diagonal matrix with nonnegative elements."*

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $D$ | ma trận **đường chéo**, mọi phần tử $\ge0$ | $T\times T$ |
| $d_t$ | phần tử đường chéo thứ $t$: hệ số kéo dãn phần dư của quan sát $t$ | vô hướng $\ge0$ |
| $X\beta^*+Du^*$ | bộ dữ liệu **mới**, giữ nguyên dự đoán và nhân phần dư lên | $\mathbb{R}^{T}$ |

Bài báo giải thích ý nghĩa ngay ở Remark trang 42:

![Remark hình học](<paper_kb1978_remark_hinh_hoc.png>)

**Hình 19 — Remark giải thích ý nghĩa hình học của Định lý 3.5, và đoạn dẫn vào Bảng II,
trang 42.**

*"REMARK: This result has a simple geometric interpretation. Imagine a scatter of sample
observations in $\mathbb{R}$ with the $\theta$th regression quantile line slicing through the
scatter. Now consider the effect (on the position of the $\theta$th $RQ$ line) of moving
observations up or down in the scatter. The result states that as long as these movements
leave observations on the same side of the original line its position is unaffected. This
property is obvious in the location model, sample quantile context, but its generalization is
perhaps somewhat less obvious."*

Dịch: *"NHẬN XÉT: Kết quả này có một cách diễn giải hình học đơn giản. Hãy hình dung một đám
điểm quan sát với đường phân vị hồi quy thứ $\theta$ cắt ngang qua. Bây giờ xét tác động (lên
vị trí của đường $RQ$ thứ $\theta$) của việc dịch các quan sát lên hoặc xuống trong đám điểm.
Kết quả nói rằng **chừng nào các dịch chuyển ấy còn để quan sát ở cùng phía so với đường ban
đầu thì vị trí của nó không đổi**. Tính chất này hiển nhiên trong mô hình vị trí, bối cảnh
phân vị mẫu, nhưng sự tổng quát hoá của nó thì có lẽ kém hiển nhiên hơn đôi chút."*

**Ý nghĩa trừu tượng — đây là phát biểu chính xác của "bền vững với ngoại lai theo $y$".**
Không phải "ít nhạy hơn", không phải "giảm ảnh hưởng" — mà là **bằng KHÔNG**: kéo một quan
sát xa gấp bao nhiêu lần cũng được, miễn không cho nó vượt sang phía kia, thì đường đứng
nguyên tại chỗ.

Đó là khác biệt cấu trúc so với bình phương tối thiểu, nơi mọi điểm kéo theo đường với trọng
số tỉ lệ khoảng cách, và cũng khác [Huber](<../Huber Regression/Huber Regression.md>), nơi ảnh hưởng bị **chặn**
nhưng không bằng 0.

Kiểm bằng số trên $T=45$, $K=2$, $\theta=0{,}30$:

| Hệ số $d_t$ của ma trận $D$ | $\beta^*$ sau khi đẩy | Lệch |
|---|---|---:|
| $d_t=1$ (không đổi) | $[0{,}81123\ \ 0{,}49303]$ | $0{,}000\times10^{0}$ |
| $d_t=3$ (đẩy xa 3 lần) | $[0{,}81123\ \ 0{,}49303]$ | $0{,}000\times10^{0}$ |
| $d_t=25$ | $[0{,}81123\ \ 0{,}49303]$ | $0{,}000\times10^{0}$ |
| $d_t$ ngẫu nhiên trong $[0,50]$ | $[0{,}81123\ \ 0{,}49303]$ | $0{,}000\times10^{0}$ |

**Phép thử ngược** — đổi dấu một hệ số $d_t$ để $D$ không còn không âm, tức cho một quan sát
**vượt sang phía kia**: đường dịch $0{,}030621$. Điều kiện "không âm" của định lý là điều
kiện thật, không phải giả thiết kỹ thuật thừa.

### 2.14 Bảng II — dựng lại cả bảng, và dựng lại cả dữ liệu bài báo không in — trang 42

Đây là chỗ bài báo trưng một ví dụ hoàn chỉnh, và cũng là chỗ nó **bỏ trống nhiều nhất**.

![Bảng II](<paper_kb1978_bang_II.png>)

**Hình 20 — Đoạn dẫn và "TABLE II", Koenker & Bassett 1978, trang 42.**

**Trích nguyên văn đoạn dẫn:**

*"We conclude this section with an illustration in Table II of the regression quantiles for
all $\theta\in(0,1)$ in a simple bivariate model with five observations. Note that for
$\theta\in\{7/22, 1/2, 3/4\}$ the problem admits multiple solutions."*

Dịch: *"Chúng tôi kết thúc mục này bằng một minh hoạ ở Bảng II về các phân vị hồi quy cho
**mọi** $\theta\in(0,1)$ trong một mô hình hai biến đơn giản với **năm quan sát**. Lưu ý rằng
với $\theta\in\{7/22,\,1/2,\,3/4\}$ bài toán có **nhiều nghiệm**."*

Bảng in ra:

| | $0<\theta\le 7/22$ | $7/22\le\theta\le 1/2$ | $1/2\le\theta\le 3/4$ | $3/4\le\theta<1$ |
|---|---|---|---|---|
| $\beta^*_1(\theta)$ | $6/7$ | $21/8$ | $13/6$ | $17/3$ |
| $\beta^*_2(\theta)$ | $4/7$ | $3/8$ | $5/6$ | $1/3$ |

**Vấn đề: bài báo KHÔNG in năm quan sát đó.** Không ở trang 42, không ở phụ lục, không ở chỗ
nào khác trong 18 trang. Bảng đứng trơ một mình với 8 phân số và 3 mốc.

Nhưng 11 con số ấy là đủ. Sau đây là cách dựng lại.

**Bước 1 — hai nghiệm liền nhau phải CHUNG đúng một quan sát.** Theo Định lý 3.1, mỗi nghiệm
là đường thẳng qua $K=2$ điểm dữ liệu. Tại một mốc $\theta$, hai nghiệm cùng tối ưu, và hai
tập neo của chúng lệch nhau đúng **một** phần tử — đó là cấu trúc của một bước xoay trong quy
hoạch tuyến tính. Vậy điểm **chung** của hai đường liền nhau là một quan sát thật, và nó chính
là **giao điểm** của hai đường:

| Cặp nghiệm | Giao điểm $(x,y)$ |
|---|---|
| $(6/7,\,4/7)$ và $(21/8,\,3/8)$ | $(9,\ 6)$ |
| $(21/8,\,3/8)$ và $(13/6,\,5/6)$ | $(1,\ 3)$ |
| $(13/6,\,5/6)$ và $(17/3,\,1/3)$ | $(7,\ 8)$ |

**Cả ba đều là số nguyên** — và chưa cần giả thiết gì để có chúng, chỉ cần giải ba hệ hai
phương trình hai ẩn bằng số hữu tỉ chính xác.

**Bước 2 — hai quan sát còn lại nằm trên đường đầu và đường cuối.** Quét mọi $x$ hữu tỉ dạng
$p/q$ với $\lvert p/q\rvert\le20$ và $q\le6$ — **481** giá trị cho mỗi đầu, tức **231 361**
cặp — rồi dựng lại toàn bộ đường nghiệm cho từng cặp bằng số hữu tỉ chính xác.

Cách dựng đường nghiệm: với một tập neo $h$ cố định, giá trị hàm mục tiêu
$\psi(h,\theta)=\theta A_h+C_h$ là một **hàm tuyến tính của $\theta$**, với
$A_h=\sum_t u_t$ và $C_h=\sum_{u_t<0}\lvert u_t\rvert$. Nên nghiệm tối ưu theo $\theta$ chính
là **bao dưới của $\binom{5}{2}=10$ đường thẳng**, và mọi mốc là giao điểm của hai đường
trong số đó. Không cần giải một bài quy hoạch tuyến tính nào.

**Kết quả của phép quét:**

| Đo | Số bộ dữ liệu |
|---|---:|
| cho ra **đúng bốn nghiệm** của Bảng II, đúng thứ tự | **5041** |
| đồng thời cho **đúng ba mốc** $7/22$, $1/2$, $3/4$ | **1** |

Dòng này là bài học của cả mục: **bốn nghiệm KHÔNG đủ để xác định dữ liệu** — hơn năm nghìn
bộ dữ liệu khác nhau cho ra cùng bốn nghiệm ấy. **Ba mốc mới là bằng chứng phân xử.** Nếu bài
báo chỉ in bảng mà không in câu *"for $\theta\in\{7/22,1/2,3/4\}$ the problem admits multiple
solutions"*, phép dựng lại này sẽ không kết luận được gì.

**Năm quan sát của bài báo:**

$$
\begin{array}{c|ccccc}
t & 1 & 2 & 3 & 4 & 5\\\hline
x_t & 1 & 2 & 4 & 7 & 9\\
y_t & 3 & 2 & 7 & 8 & 6
\end{array}
$$

Cả mười con số đều **nguyên** — và đó không phải giả thiết đưa vào, mà là kết quả rơi ra từ
một phép quét trên số hữu tỉ.

**Bước 3 — dựng lại toàn bộ Bảng II từ năm điểm đó:**

| Khoảng $\theta$ | $\beta^*_1$ | $\beta^*_2$ | Số nghiệm cơ sở |
|---|---|---|---:|
| $0\le\theta\le 7/22$ | $6/7$ | $4/7$ | 1 |
| $7/22\le\theta\le 1/2$ | $21/8$ | $3/8$ | 1 |
| $1/2\le\theta\le 3/4$ | $13/6$ | $5/6$ | 1 |
| $3/4\le\theta\le 1$ | $17/3$ | $1/3$ | 1 |

**8/8 ô khớp. 3/3 mốc khớp.**

**Bước 4 — kiểm câu "nhiều nghiệm" tại đúng ba mốc.** Đếm số tập neo tối ưu tại chính các giá
trị $\theta$ ấy:

| $\theta$ | Số nghiệm cơ sở tối ưu | Chúng là |
|---|---:|---|
| $7/22$ | **2** | $(6/7,\,4/7)$ và $(21/8,\,3/8)$ |
| $1/2$ | **2** | $(13/6,\,5/6)$ và $(21/8,\,3/8)$ |
| $3/4$ | **2** | $(13/6,\,5/6)$ và $(17/3,\,1/3)$ |

Cả ba mốc đều có nhiều hơn một nghiệm — đúng như bài báo nói, và không mốc nào khác có. Theo
Định lý 3.1, tập nghiệm tại mỗi mốc là **bao lồi** của hai điểm đó, tức cả một đoạn thẳng
trong $\mathbb{R}^2$.

**Bước 5 — phép thử ngược.** Nếu phép dựng lại lỏng thì nhiều bộ dữ liệu gần nhau cũng khớp.
Đổi từng toạ độ $y_t$ đi $\pm1$ đơn vị — mười bộ dữ liệu lệch một đơn vị — rồi dựng lại:
**không bộ nào còn khớp cả bốn nghiệm lẫn ba mốc.** Năm điểm ấy là duy nhất trong miền quét.

**Bước 6 — đối chiếu với bộ giải số thực**, để chắc phép tính hữu tỉ không tự lừa mình:

| $\theta$ | $\beta^*_1$ (quy hoạch tuyến tính) | Bảng II | $\beta^*_2$ (quy hoạch tuyến tính) | Bảng II | Lệch |
|---:|---:|---:|---:|---:|---:|
| $0{,}20$ | $0{,}8571428571$ | $0{,}8571428571$ | $0{,}5714285714$ | $0{,}5714285714$ | $0$ |
| $0{,}40$ | $2{,}6250000000$ | $2{,}6250000000$ | $0{,}3750000000$ | $0{,}3750000000$ | $0$ |
| $0{,}60$ | $2{,}1666666667$ | $2{,}1666666667$ | $0{,}8333333333$ | $0{,}8333333333$ | $1{,}11\times10^{-16}$ |
| $0{,}90$ | $5{,}6666666667$ | $5{,}6666666667$ | $0{,}3333333333$ | $0{,}3333333333$ | $5{,}55\times10^{-17}$ |

Lệch lớn nhất: $1{,}11\times10^{-16}$.

![Bảng II dựng lại](<quantile_bang_II.png>)

**Hình 21 — Bảng II dựng lại (hình tự vẽ).** Bảng trái: năm quan sát và bốn đường nghiệm.
Bảng phải: hai hệ số theo $\theta$, dạng bậc thang với ba mốc.

**Cách đọc hình.**

- **Bảng trái.** Năm chấm là năm quan sát dựng lại được. Bốn đường màu là bốn nghiệm của
  Bảng II. Điều cần nhìn: **mỗi đường đi qua đúng hai chấm** — đó là Định lý 3.1 vẽ ra thành
  hình. Và ba chấm ở giữa, $(1,3)$, $(7,8)$, $(9,6)$, mỗi chấm được **hai** đường liền nhau
  cùng đi qua; đó là bước 1 của phép dựng lại.
- **Bảng phải.** Hai đường bậc thang là $\beta^*_1(\theta)$ và $\beta^*_2(\theta)$. Ba vạch
  đứng là ba mốc $7/22$, $1/2$, $3/4$. Điều cần nhìn: **hai hệ số nhảy ngược chiều nhau** ở
  mỗi mốc — hệ số chặn tăng thì hệ số góc giảm, và ngược lại. Đường phải xoay quanh đám dữ
  liệu chứ không tịnh tiến.
- **Điều hình KHÔNG nói ra:** $\beta^*_1$ **không đơn điệu theo $\theta$** — nó đi
  $6/7 \to 21/8 \to 13/6 \to 17/3$, tức tăng, giảm, rồi tăng. Chỉ có **đường** là dịch lên
  đều đặn; từng hệ số riêng lẻ thì không. Đó là lý do không đọc được xu hướng từ một cột hệ
  số.

<details>
<summary><b>Mã sinh Hình 21</b> — khối tự chứa, chạy trong dưới một giây</summary>

```python
# tools/fig_quantile.py — phần A
# Mọi con số dưới đây do tools/v_quantile.py mục 6 in ra; khối này chỉ VẼ.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# nam quan sat dung lai duoc tu Bang II (v_quantile.py muc 6, BUOC 2)
BII_X = [1.0, 2.0, 4.0, 7.0, 9.0]
BII_Y = [3.0, 2.0, 7.0, 8.0, 6.0]
# bon nghiem cua Bang II tr.42
BII_B0 = [6 / 7, 21 / 8, 13 / 6, 17 / 3]
BII_B1 = [4 / 7, 3 / 8, 5 / 6, 1 / 3]
BII_MOC = [7 / 22, 1 / 2, 3 / 4]
NHAN = [r"$0<\theta\leq 7/22$", r"$7/22\leq\theta\leq 1/2$",
        r"$1/2\leq\theta\leq 3/4$", r"$3/4\leq\theta<1$"]
MAU = ["#2563eb", "#0891b2", "#d97706", "#dc2626"]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.0))

xs = np.linspace(0, 10, 200)
for b0, b1, nh, c in zip(BII_B0, BII_B1, NHAN, MAU):
    ax[0].plot(xs, b0 + b1 * xs, lw=1.7, color=c, label=nh)
ax[0].scatter(BII_X, BII_Y, s=70, color="#111827", zorder=5)
for x, y in zip(BII_X, BII_Y):
    ax[0].annotate("(%d, %d)" % (x, y), (x, y), textcoords="offset points",
                   xytext=(6, -12), fontsize=8.2, color="#111827")
ax[0].set_xlim(0, 10); ax[0].set_ylim(0, 10)
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7.6, loc="upper left")
ax[0].set_title("Năm quan sát dựng lại được và bốn nghiệm của Bảng II\n"
                "mỗi đường đi qua ĐÚNG hai điểm — Định lý 3.1", fontsize=9)

canh = [0.0] + BII_MOC + [1.0]
for gt, ten, c, nl in ((BII_B0, r"$\beta^*_1$ (hệ số chặn)", "#be185d", "-"),
                       (BII_B1, r"$\beta^*_2$ (hệ số góc)", "#0f766e", "--")):
    for i, v in enumerate(gt):
        ax[1].hlines(v, canh[i], canh[i + 1], color=c, lw=2.2, linestyle=nl,
                     label=ten if i == 0 else None)
for m in BII_MOC:
    ax[1].axvline(m, color="#9ca3af", lw=0.9, linestyle=":")
    ax[1].annotate("%.4f" % m, (m, 0.12), fontsize=7.4, color="#6b7280",
                   ha="center", va="bottom",
                   bbox=dict(fc="white", ec="none", pad=1.0))
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 7.0)
ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("giá trị hệ số")
ax[1].legend(fontsize=8, loc="upper left")
ax[1].set_title("Hai hệ số theo θ: bậc thang, ba mốc\n"
                "hai hệ số nhảy NGƯỢC chiều nhau ở mỗi mốc", fontsize=9)

fig.savefig("quantile_bang_II.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

### 2.15 Định lý 4.1 — hiệp phương sai của nhiều phân vị mẫu — trang 42

![Định lý 4.1](<paper_kb1978_dinh_ly_4_1.png>)

**Hình 22 — Mở đầu mục 4 và Định lý 4.1, trang 42.**

**Trích nguyên văn:** *"THEOREM 4.1: Let $\{\xi^*_T(\theta_1),\ldots,\xi^*_T(\theta_M)\}$
with $0<\theta_1<\theta_2<\ldots<\theta_M<1$, denote a sequence of unique sample quantiles
from random samples of size $T$ from a population with inverse distribution function
$\xi(\theta)=F^{-1}(\theta)$. If $F$ is continuous and has continuous and positive density,
$f$, at $\xi(\theta_i)$, $i=1,\ldots,M$, then $\sqrt{T}[\xi^*_T(\theta_1)-\xi(\theta_1),
\ldots,\xi^*_T(\theta_M)-\xi(\theta_M)]$ converges in distribution to an $(M)$-variate
Gaussian random vector with mean, $0$, and covariance matrix
$\Omega(\theta_1,\ldots,\theta_M;F)$ with typical element,"*

$$\omega_{ij}=\frac{\theta_i(1-\theta_j)}{f(\xi(\theta_i))\,f(\xi(\theta_j))},\qquad i\le j \tag{$\omega$}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\xi^*_T(\theta_i)$ | phân vị **mẫu** thứ $\theta_i$ từ mẫu cỡ $T$ | vô hướng |
| $\xi(\theta_i)$ | phân vị **quần thể** $F^{-1}(\theta_i)$ | vô hướng |
| $f(\xi(\theta_i))$ | mật độ **tại đúng chỗ phân vị nằm** | vô hướng $>0$ |
| $\Omega$ | ma trận hiệp phương sai tiệm cận | $M\times M$ |

**Ý nghĩa trừu tượng — nó nói điều gì.** Công thức này nói phương sai của một phân vị mẫu
phụ thuộc **hai** thứ, và cách phụ thuộc rất khác nhau:

* tử số $\theta(1-\theta)$ là phương sai của một phép đếm — nó **lớn nhất ở giữa** ($\theta=0{,}5$)
  và tiến về 0 ở hai đuôi;
* mẫu số $f(\xi(\theta))^2$ là mật độ **bình phương** tại đó — và ở đuôi phân phối mật độ rất
  nhỏ, nên mẫu số **sụp về 0** nhanh hơn tử số.

Kết quả là **phân vị ở đuôi luôn kém ổn định hơn phân vị ở giữa**, và đó là lời giải thích
đúng cho câu "đừng ước lượng phân vị 1 % từ 200 điểm" ở mục 8. Không phải vì "ít dữ liệu ở
đó" theo nghĩa mơ hồ, mà vì **$f$ ở mẫu số**.

Chú ý ràng buộc $i\le j$: công thức này **không đối xứng** như viết, và phải hiểu là ô $(i,j)$
với $i\le j$ rồi lấy đối xứng. Vì $\theta_i\le\theta_j$ nên $\theta_i(1-\theta_j)$ luôn là
"min nhân một trừ max".

Tính ra cho $N(0,1)$ với $\theta=0{,}25$; $0{,}50$; $0{,}75$:

$$\Omega=\begin{pmatrix}
1{,}85676747 & 0{,}98600262 & 0{,}61892249\\
0{,}98600262 & 1{,}57079633 & 0{,}98600262\\
0{,}61892249 & 0{,}98600262 & 1{,}85676747
\end{pmatrix}$$

Kiểm bằng Monte-Carlo, $T=4000$, 40 000 lần lặp — lệch tương đối lớn nhất **1,160629 %**.

Bài báo rút ra ngay trường hợp riêng ở trang 43: trung vị có phương sai tiệm cận
$[2f(\xi(1/2))]^{-2}$. Tính ra:

| $F$ | $[2f(0)]^{-2}$ |
|---|---:|
| Chuẩn $N(0,1)$ | $1{,}5707963268$ |
| Laplace | $1{,}0000000000$ |
| Cauchy | $2{,}4674011003$ |

![Đoạn về phương sai trung vị](<paper_kb1978_phuong_sai_trung_vi.png>)

**Hình 23 — Đoạn nối Định lý 4.1 sang Định lý 4.2, trang 43.**

**Trích nguyên văn:** *"This theorem provides the foundation for a large-sample theory of
so-called "systematic statistics"—estimators which are linear combinations of a few sample
quantiles. The median is the most important special case having asymptotic variance
$[2f(\xi(1/2))]^{-2}$. As we have noted above, this value will be less than the variance of
the mean for a large class of long-tailed distributions. The asymptotic variance of the
Gastwirth or trimean estimator of Table I can be easily calculated in a similar manner from
Theorem 4.1 for arbitrary distributions. We merely require the evaluation of the density
function of the distribution $F$ at specified quantiles."*

Dịch: *"Định lý này cung cấp nền tảng cho một lý thuyết mẫu lớn về cái gọi là "thống kê hệ
thống" — những ước lượng là tổ hợp tuyến tính của một vài phân vị mẫu. Trung vị là trường hợp
riêng quan trọng nhất, có phương sai tiệm cận $[2f(\xi(1/2))]^{-2}$. Như đã lưu ý ở trên, giá
trị này sẽ nhỏ hơn phương sai của trung bình với một lớp rộng các phân phối đuôi dài. **Phương
sai tiệm cận của ước lượng Gastwirth hay trimean của Bảng I tính được dễ dàng theo cách tương
tự từ Định lý 4.1 cho các phân phối tuỳ ý.** Ta chỉ cần đánh giá hàm mật độ của phân phối $F$
tại các phân vị được chỉ định."*

**Câu in đậm là một lời mời tính, và mục 2.3 cùng mục 2.4 đã nhận lời.** Đó cũng là điều đáng
chú ý: bài báo **nói rằng tính được** nhưng **không in ra một con số nào** cho Gastwirth hay
trimean ở bất kỳ phân phối nào. Bảng I chỉ có giá trị Monte-Carlo ở $T=20$; cột tiệm cận
tương ứng không tồn tại trong bài báo.

### 2.16 Định lý 4.2 — trái tim của bài báo — trang 43

![Định lý 4.2](<paper_kb1978_dinh_ly_4_2.png>)

**Hình 24 — Định lý 4.2, trang 43.**

**Trích nguyên văn:** *"THEOREM 4.2: Let $\{\beta^*_T(\theta_1),\ldots,\beta^*_T(\theta_M)\}$
with $0<\theta_1<\ldots<\theta_M<1$ denote a sequence of unique regression quantiles from
model (1.1). ... Assume: (i) $F$ is continuous and has continuous and positive density, $f$,
at $\xi(\theta_i)$, $i=1,\ldots,M$, and (ii) $x_{1t}=1$: $t=1,2,\ldots$ and
$\lim_{T\to\infty}T^{-1}X'X=Q$, a positive definite matrix. Then,
$\sqrt{T}[\boldsymbol{\xi}^*_T(\theta_1)-\boldsymbol{\xi}(\theta_1),\ldots]$ converges in
distribution to an $(MK)$-variate Gaussian random vector with mean $0$ and covariance matrix
$\Omega(\theta_1,\ldots,\theta_M;F)\otimes Q^{-1}$, where $\Omega$ is the covariance matrix
of the corresponding $M$ ordinary sample quantiles from random samples from distribution $F$."*

$$\sqrt{T}\bigl[\boldsymbol{\xi}^*_T(\theta_1)-\boldsymbol{\xi}(\theta_1),\ldots,
\boldsymbol{\xi}^*_T(\theta_M)-\boldsymbol{\xi}(\theta_M)\bigr]\ \xrightarrow{d}\
N\bigl(0,\ \Omega\otimes Q^{-1}\bigr)$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $Q=\lim T^{-1}X'X$ | ma trận thiết kế đã chuẩn hoá, **xác định dương** | $K\times K$ |
| $\Omega$ | **đúng** ma trận của Định lý 4.1 — không đổi gì cả | $M\times M$ |
| $\otimes$ | tích Kronecker | kết quả $MK\times MK$ |
| điều kiện (ii) | cột đầu của $X$ **toàn số 1** | — |

**Ý nghĩa trừu tượng — và đây là câu quan trọng nhất của cả bài báo.** Tích Kronecker
$\Omega\otimes Q^{-1}$ nghĩa là ma trận hiệp phương sai **tách hẳn làm hai phần độc lập**:

* $\Omega$ — **chỉ phụ thuộc phân phối sai số $F$ và các mức $\theta$**, không biết gì về
  $X$;
* $Q^{-1}$ — **chỉ phụ thuộc ma trận thiết kế**, không biết gì về $F$ hay $\theta$.

Cụm **"tích Kronecker"** nghĩa là: ô $(i,j)$ của $\Omega$ được thay bằng cả khối
$\omega_{ij}Q^{-1}$. Nên khối chéo thứ $i$ — hiệp phương sai của $\beta^*(\theta_i)$ với
chính nó — đúng bằng $\omega_{ii}Q^{-1}$.

Hệ quả, và nó là thứ mà mục 2.17 dùng tới: **mọi câu so sánh hiệu suất giữa hai ước lượng
đều KHÔNG phụ thuộc thiết kế**, vì $Q^{-1}$ là nhân tử chung và triệt tiêu khi lấy tỉ số.

Chính bài báo nói câu này gọn hơn: *"$\Omega$ là ma trận hiệp phương sai của $M$ phân vị mẫu
thông thường tương ứng"* — tức là **bài toán hồi quy không sinh ra một ma trận mới nào cả**;
nó dùng lại đúng ma trận của mô hình vị trí.

Kiểm bằng Monte-Carlo: $T=300$, $K=3$, $M=2$ với $\theta=0{,}3$ và $0{,}7$, nhiễu chuẩn,
3000 lần lặp. Ma trận lý thuyết $\Omega\otimes Q^{-1}$ là $6\times6$.

| $i$ | $j$ | Lý thuyết | Đo được | Lệch tương đối |
|---:|---:|---:|---:|---:|
| 0 | 0 | $1{,}7394092819$ | $1{,}6670211423$ | $4{,}1617\%$ |
| 1 | 1 | $1{,}7902822401$ | $1{,}7686650674$ | $1{,}2075\%$ |
| 2 | 2 | $1{,}9692339849$ | $1{,}9520124081$ | $0{,}8745\%$ |
| 0 | 3 | $0{,}7454611208$ | $0{,}7629515345$ | $2{,}3463\%$ |
| 1 | 4 | $0{,}7672638172$ | $0{,}7829686977$ | $2{,}0469\%$ |
| 2 | 5 | $0{,}8439574221$ | $0{,}8439656082$ | $0{,}0010\%$ |

**Phải báo hai thước đo, không được gộp.** Trên 16 ô có giá trị lý thuyết lớn hơn 10 % ô lớn
nhất: lệch tương đối **trung bình 1,7149 %**, **lớn nhất 4,1617 %** — so với sai số
Monte-Carlo dự kiến của một phương sai là $\sqrt{2/3000}=2{,}5820\%$. Trên **toàn bộ 36 ô**:
lệch **tuyệt đối** lớn nhất $0{,}072388$, trong khi ô lớn nhất là $1{,}969234$.

Vì sao phải báo cả hai: có một ô mà giá trị lý thuyết là $0{,}0004$, cho lệch **tương đối
488 %** trong khi lệch tuyệt đối chỉ $0{,}002$. Con số tương đối ở đó **vô nghĩa**, và nếu
chỉ báo "lệch tương đối lớn nhất" thì cả phép kiểm trông như hỏng.

**Phép thử ngược cho điều kiện (ii):** bỏ cột toàn 1 khỏi $X$ thì kết luận của Định lý 3.4
gãy ngay — $N=152$ trong khi $T\theta=90{,}0$.

### 2.17 Câu in nghiêng ở trang 46 — và vì sao nó đúng trong một dòng

![Câu in nghiêng trang 46](<paper_kb1978_cau_in_nghieng.png>)

**Hình 25 — Remark mở đầu trang 46, chứa câu in nghiêng.**

**Trích nguyên văn:** *"REMARK: An extremely important special case is the regression median,
or least absolute error estimator, $\beta^*(1/2)$. Without loss of generality $\beta$ may be
located so that $F(0)=1/2$, so $\xi(1/2)=0$. Then the asymptotic distribution of the random
variable $\sqrt{T}(\beta^*(1/2)-\beta)$ is $K$-variate Gaussian with mean zero and covariance
matrix $[2f(0)]^{-2}Q^{-1}$. See our detailed treatment of this special case in [6]. Thus,
**the regression median is seen to be more efficient than the least squares estimator in the
linear model for any distribution for which the median is more efficient than the mean in the
location model.** While such a result has been the subject of conjecture among a number of
authors we believe ours to be its first formal statement."*

Dịch: *"NHẬN XÉT: Một trường hợp riêng cực kỳ quan trọng là trung vị hồi quy, hay ước lượng
sai số tuyệt đối nhỏ nhất, $\beta^*(1/2)$. Không mất tính tổng quát, có thể đặt $\beta$ sao
cho $F(0)=1/2$, tức $\xi(1/2)=0$. Khi đó phân phối tiệm cận của biến ngẫu nhiên
$\sqrt{T}(\beta^*(1/2)-\beta)$ là Gauss $K$ chiều với trung bình không và ma trận hiệp phương
sai $[2f(0)]^{-2}Q^{-1}$. Xem xử lý chi tiết trường hợp riêng này trong [6]. Vậy, **trung vị
hồi quy tỏ ra hiệu quả hơn ước lượng bình phương tối thiểu trong mô hình tuyến tính, với bất
kỳ phân phối nào mà trung vị hiệu quả hơn trung bình trong mô hình vị trí.** Dù kết quả như
thế đã là đối tượng phỏng đoán của một số tác giả, chúng tôi tin rằng phát biểu của mình là
phát biểu chính thức đầu tiên."*

**Vì sao câu đó đúng, trong một dòng.** Hiệp phương sai tiệm cận của trung vị hồi quy là
$[2f(0)]^{-2}Q^{-1}$; của bình phương tối thiểu là $\sigma^2Q^{-1}$. **$Q^{-1}$ là nhân tử
chung**, nên khi lấy tỉ số nó triệt tiêu:

$$\frac{\text{phương sai bình phương tối thiểu}}{\text{phương sai trung vị hồi quy}}
=\frac{\sigma^2Q^{-1}}{[2f(0)]^{-2}Q^{-1}}=\frac{\sigma^2}{[2f(0)]^{-2}}$$

Vế phải **không chứa $X$**. Nó đúng bằng tỉ số của mô hình vị trí. Đó là toàn bộ nội dung
câu in nghiêng — và cũng là lý do bài báo có quyền nói "any distribution", không kèm điều
kiện nào về thiết kế.

**Kiểm bằng số, và phép kiểm phải tách làm hai câu hỏi.**

*Câu hỏi 1 — tỉ số có tiến tới giá trị tiệm cận không?* Đo trong mô hình vị trí với nhiễu
Laplace (tỉ số lý thuyết $=2{,}0000000000$):

| $T$ | $T\cdot\text{Var}$(trung vị) | $T\cdot\text{Var}$(trung bình) | Tỉ số |
|---:|---:|---:|---:|
| 100 | $1{,}1472854890$ | $1{,}9973488265$ | $1{,}7409344454$ |
| 300 | $1{,}0908493902$ | $2{,}0032182230$ | $1{,}8363838683$ |
| 1 000 | $1{,}0495099444$ | $2{,}0097253592$ | $1{,}9149178814$ |
| 10 000 | $1{,}0176868042$ | $2{,}0055131780$ | $1{,}9706585265$ |

Đơn điệu tăng, tiến tới 2. Định lý 4.2 là phát biểu **tiệm cận**, và dãy này là bằng chứng nó
đúng theo nghĩa đó.

*Câu hỏi 2 — tỉ số có phụ thuộc thiết kế không?* Đo ở $T=300$, $K=3$, nhiễu Laplace, trên bốn
ma trận thiết kế rất khác nhau:

| Thiết kế | Số điều kiện của $Q$ | $T\cdot\text{Var}$(trung vị) | $T\cdot\text{Var}$(OLS) | Tỉ số |
|---|---:|---:|---:|---:|
| cột 1 + chuẩn độc lập | $1{,}1307$ | $1{,}2332909661$ | $2{,}0132426695$ | $1{,}6324149977$ |
| cột 1 + đều $[0,10]$ | $413{,}2403$ | $0{,}1421365742$ | $0{,}2440732158$ | $1{,}7171739028$ |
| cột 1 + tương quan $0{,}95$ | $41{,}0451$ | $11{,}9128473776$ | $21{,}0200437769$ | $1{,}7644852746$ |
| cột 1 + thang đo lệch 100 lần | $11\,860{,}6211$ | $1{,}4996277235$ | $2{,}3868531559$ | $1{,}5916304550$ |

Số điều kiện chạy qua **$10\,489{,}3279$ lần**, mà bốn tỉ số nằm trong một khoảng rộng
$0{,}1728548196$.

**Nhưng phải đo độ nhiễu trước khi kết luận.** Chạy **cùng một** thiết kế với bốn hạt giống
khác nhau cho các tỉ số $1{,}4342067517$ · $1{,}6484974954$ · $1{,}5943967172$ ·
$1{,}6659206426$, tức độ rộng **$0{,}2317138908$** — **lớn hơn** độ rộng theo thiết kế. Nên
câu đọc được là: **không kết luận được rằng thiết kế có ảnh hưởng.** Đó là một phép không bác
bỏ, không phải một phép chứng minh — và viết mạnh hơn thế là viết quá tay.

*Câu hỏi 3 — vì sao cả bốn tỉ số đều thấp hơn mô hình vị trí ở cùng $T$?* Giả thuyết: số tham
số $K$ ăn bớt bậc tự do, và hiệu ứng đó có bậc $K/T$ nên ở $T$ nhỏ phải rõ hơn. Đo ở $T=150$,
mỗi $K$ hai hạt giống:

| $K$ | Tỉ số trung bình | Độ rộng hai hạt giống |
|---:|---:|---:|
| 1 | $1{,}7069619970$ | $0{,}0234298454$ |
| 2 | $1{,}6804583603$ | $0{,}0927724955$ |
| 3 | $1{,}6861831693$ | $0{,}0517008371$ |
| 6 | $1{,}4562825031$ | $0{,}0644297235$ |
| 12 | $1{,}3803871586$ | $0{,}0165609531$ |

Chênh giữa $K=1$ và $K=12$ là **$0{,}3265748385$**, trong khi độ rộng hạt giống lớn nhất là
**$0{,}0927724955$** — lớn hơn **ba lần rưỡi**. **Hiệu ứng $K$ đo được**, và tỉ số giảm đơn
điệu ở 3/4 bước liên tiếp.

Và $K=1$ **chính là** mô hình vị trí. Nên chênh lệch ở câu hỏi 2 không phải bằng chứng chống
lại Định lý 4.2; nó là một hiệu ứng hữu hạn mẫu theo $K$, và Định lý 4.2 không nói gì về $T$
hữu hạn.

### 2.18 Định lý 4.3 — trộn nhiều phân vị thành một ước lượng — trang 46

![Định lý 4.3](<paper_kb1978_dinh_ly_4_3.png>)

**Hình 26 — Định lý 4.3 và Remark nêu hai bộ trọng số, trang 46.**

**Trích nguyên văn:** *"THEOREM 4.3: Let $\pi(\theta)=(\pi(\theta_1),\ldots,\pi(\theta_M))'$
be a discrete, symmetric probability measure on $(0,1)$ concentrating mass on
$\{\theta_i: i=1,\ldots,M\mid 0<\theta_1<\theta_2<\ldots<\theta_M<1\}$. Suppose $F(0)=1/2$,
so $\xi(1/2)\equiv F^{-1}(1/2)=0$, and conditions (i) and (ii) of Theorem 4.2 hold. Then
$\tilde\beta_T(\pi)=\sum\pi(\theta)\beta^*_T(\theta)$ is invariant to location, scale and
reparameterization of design in the sense of Theorem 3.2, and
$\sqrt{T}(\tilde\beta_T(\pi)-\beta)$ converges in distribution to a $(K)$-variate Gaussian
random vector with mean zero and covariance matrix $\pi'\Omega\pi\cdot Q^{-1}$."*

$$\tilde\beta_T(\pi)=\sum_\theta \pi(\theta)\,\beta^*_T(\theta),
\qquad \text{hiệp phương sai tiệm cận } = \pi'\Omega\pi\cdot Q^{-1}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\pi(\theta)$ | bộ trọng số, **cộng lại bằng 1** và **đối xứng** quanh $1/2$ | $\mathbb{R}^{M}$ |
| $\pi'\Omega\pi$ | một **số**, không phải ma trận | vô hướng |
| $\tilde\beta_T(\pi)$ | ước lượng trộn | $\mathbb{R}^{K}$ |

**Ý nghĩa trừu tượng.** Định lý này là chỗ bài báo trả nợ cho tóm tắt: *"These results permit
a natural generalization to the linear model of certain well-known robust estimators of
location."* Mọi ước lượng vị trí ở Bảng I mà là tổ hợp phân vị — trung vị, Gastwirth,
trimean — đều có bản hồi quy, và bản hồi quy đó có **đúng cùng** phương sai tiệm cận, chỉ
nhân thêm $Q^{-1}$.

Chỗ đắt là chữ $\pi'\Omega\pi$: nó là một **dạng toàn phương**, nên trộn nhiều phân vị có thể
cho phương sai **nhỏ hơn** từng phân vị riêng lẻ — đúng như trộn nhiều ước lượng tương quan
không hoàn toàn thì giảm được phương sai.

**Remark ngay sau, chép nguyên văn:** *"REMARK: Obvious examples of weight functions
$\pi(\theta)$, in addition to the regression median example $(\pi(1/2)=1)$ already discussed,
are the Gastwirth: $(\pi(1/3),\pi(1/2),\pi(2/3))=(.3,.4,.3)$ and the trimean:
$(\pi(1/4),\pi(1/2),\pi(3/4))=(1/4,1/2,1/4)$. When $F$ is non-Gaussian, weight functions
$\pi(\theta)$ are readily found which make $\pi'\Omega\pi$ smaller than the variance of $F$,
i.e., yielding estimators $\tilde\beta(\pi)$ having superior asymptotic efficiency to the
least squares estimator."*

Tính ra cho bốn phân phối, bằng đúng công thức $\pi'\Omega\pi$:

| $F$ | Trung bình | Trung vị | Gastwirth | Trimean |
|---|---:|---:|---:|---:|
| Chuẩn | $1{,}0000000000$ | $1{,}5707963268$ | $1{,}2566824519$ | $1{,}1951616383$ |
| Laplace | $2{,}0000000000$ | $1{,}0000000000$ | $1{,}1800000000$ | $1{,}2500000000$ |
| Logistic | $3{,}2898681337$ | $4{,}0000000000$ | $3{,}2950000000$ | $3{,}2222222222$ |
| Cauchy | vô cực | $2{,}4674011003$ | $2{,}5002997816$ | $3{,}0842513753$ |

Đọc bảng này theo đúng câu của Remark — *"$\pi'\Omega\pi$ nhỏ hơn phương sai của $F$"*:

* **Laplace**: cả ba cột phân vị ($1{,}00$; $1{,}18$; $1{,}25$) đều nhỏ hơn $2{,}00$ — câu
  đúng.
* **Cauchy**: phương sai của $F$ vô cực, nên mọi cột đều nhỏ hơn — câu đúng theo nghĩa mạnh
  nhất.
* **Logistic**: $3{,}2950$ và $3{,}2222$ so với $3{,}2899$ — **trimean thắng, Gastwirth
  thua**, và trung vị ($4{,}0000$) thua rõ. Câu của Remark nói *"weight functions are readily
  found"*, tức nói về **sự tồn tại** một bộ trọng số tốt hơn, không nói mọi bộ đều tốt hơn.
  Đọc nó thành "phân vị luôn thắng" là đọc quá tay.

Kiểm Gastwirth bằng Monte-Carlo trên phân phối chuẩn, $T=4000$, 20 000 lần lặp: lý thuyết
$1{,}2566824519$, đo được $1{,}2572191811$, lệch **$0{,}0427\%$**.

### 2.19 Bình phương tối thiểu cắt dưới — đề xuất thuật toán của bài báo — trang 47

![Bình phương tối thiểu cắt dưới](<paper_kb1978_binh_phuong_cat_duoi.png>)

**Hình 27 — Đoạn đề xuất ước lượng bình phương tối thiểu cắt dưới, trang 47.**

**Trích nguyên văn:** *"An obvious analogue to the trimmed mean of the location model is
suggested by regression quantiles. Let $\alpha$ be the desired trimming proportion. Calculate
$\beta^*(\alpha)$ and $\beta^*(1-\alpha)$ for the sample. By Theorem 3.4 there will be
approximately $T\alpha$ observations below the former and above the latter. These observations
are set aside, and the remaining observations are subjected to a least squares fit. ... This
estimator, say $\hat\beta_\alpha$, is location and scale invariant, and will be asymptotically
$K$-variate Gaussian with covariance matrix $\sigma^2(\alpha,F)Q^{-1}$ where
$\sigma^2(\alpha,F)$ denotes the asymptotic variance of the corresponding $\alpha$-trimmed
mean from a population with distribution $F$."*

Dịch: *"Phân vị hồi quy gợi ra một tương tự hiển nhiên của trung bình cắt đuôi trong mô hình
vị trí. Gọi $\alpha$ là tỉ lệ cắt mong muốn. Tính $\beta^*(\alpha)$ và $\beta^*(1-\alpha)$
cho mẫu. Theo Định lý 3.4 sẽ có **xấp xỉ $T\alpha$** quan sát nằm dưới cái thứ nhất và trên
cái thứ hai. Những quan sát này được gạt sang một bên, và các quan sát còn lại được đem khớp
bình phương tối thiểu. ... Ước lượng này, gọi là $\hat\beta_\alpha$, bất biến với vị trí và
thang đo, và sẽ tiệm cận Gauss $K$ chiều với ma trận hiệp phương sai
$\sigma^2(\alpha,F)Q^{-1}$, trong đó $\sigma^2(\alpha,F)$ là phương sai tiệm cận của trung
bình cắt $\alpha$ tương ứng từ một quần thể có phân phối $F$."*

**Ý nghĩa trừu tượng — đây là chỗ bài báo trả lời câu hỏi "để làm gì".** Toàn bộ mục 3 và mục
4 dựng ra một đối tượng mới; đoạn này nói cuối cùng dùng nó thế nào trong thực tế. Và câu trả
lời rất tiết kiệm: **hai bài quy hoạch tuyến tính là đủ** để biết quan sát nào nên gạt đi;
phần còn lại là bình phương tối thiểu thông thường.

Đáng chú ý là chỗ hai phân vị hồi quy được dùng: chúng **không** phải ước lượng cuối cùng, mà
là **công cụ chọn mẫu**. Đó là một vai trò khác hẳn với mọi mục trước.

Đo trên $T=200$, $K=2$, 400 lần lặp cho mỗi mức nhiễu:

| Nhiễu | OLS: $T\cdot\text{Var}$(hệ số góc) | cắt $10\%$ | cắt $25\%$ | Lợi/hại của OLS |
|---|---:|---:|---:|---:|
| Chuẩn | $0{,}1289638257$ | $0{,}1379624408$ | $0{,}1611059531$ | $0{,}934775$ |
| Laplace | $0{,}2194952124$ | $0{,}1732591227$ | $0{,}1402596802$ | $1{,}564920$ |
| $10\%$ tại $10\sigma$ | $1{,}3444140361$ | $0{,}1711421367$ | $0{,}1807821260$ | $\mathbf{7{,}855541}$ |

Cột cuối là tỉ số "phương sai OLS / phương sai bản cắt tốt nhất". Lớn hơn 1 nghĩa là cắt có
lợi; nhỏ hơn 1 nghĩa là cắt phải trả giá.

Ba dòng nói ba chuyện: ở nhiễu chuẩn, tỉ số là $0{,}934775$ — **cắt làm tệ đi**; ở Laplace,
tỉ số $1{,}564920$ nghiêng về phía cắt; ở dữ liệu có $10\%$ nhiễm bẩn nặng, tỉ số vọt lên
$\mathbf{7{,}855541}$. Đó đúng là chữ *"insurance"* — bảo hiểm — mà mục 5 của bài báo dùng:
trả một khoản nhỏ ở cột đầu để mua một khoản lớn ở cột cuối.

### 2.20 Mục 5 — kết luận, và chữ "bảo hiểm" — trang 48

![Kết luận](<paper_kb1978_ket_luan.png>)

**Hình 28 — Mục 5 Kết luận, trang 48.**

**Trích nguyên văn:** *"We argue that the conventional least squares estimator may be
seriously deficient in linear models with non-Gaussian errors. In the absence of a
well-specified prior on the set of plausible distribution functions it is useful, following
Huber [25] and others, to view the problem of estimation in terms of insurance. It seems
reasonable to pay a small premium in the form of sacrificed efficiency "at the Gaussian
distribution" (if that is the hypothesized parametric model), in order to achieve a
substantial improvement over least squares in the event of a non-Gaussian situation."*

Dịch: *"Chúng tôi lập luận rằng ước lượng bình phương tối thiểu thông thường có thể thiếu sót
nghiêm trọng trong các mô hình tuyến tính với sai số không Gauss. Khi thiếu một tiên nghiệm
được xác định rõ trên tập các hàm phân phối có thể xảy ra, sẽ hữu ích — theo Huber [25] và
những người khác — khi nhìn bài toán ước lượng **dưới góc độ bảo hiểm**. Có vẻ hợp lý khi trả
một khoản phí nhỏ dưới dạng hy sinh hiệu suất "ở phân phối Gauss" (nếu đó là mô hình tham số
được giả định), để đạt được một cải thiện đáng kể so với bình phương tối thiểu trong trường
hợp xảy ra tình huống không Gauss."*

**Cách đọc.** Mục kết luận không tuyên bố phương pháp mới thắng; nó tuyên bố một **cách đánh
đổi**. Và cả bài viết này có một con số cho "phí bảo hiểm" ấy: ở phân phối chuẩn, trung vị
hồi quy trả $\pi/2=1{,}5707963268$ lần phương sai, tức mất $57{,}0796\%$ hiệu suất (mục 2.2);
còn bản cắt $10\%$ ở mục 2.19 cho tỉ số $0{,}934775$ ở nhiễu chuẩn — tức phí bảo hiểm rất
nhỏ — và đổi lại tỉ số $7{,}855541$ khi có nhiễm bẩn.

**Một chỗ bài báo tự bất đồng với chính nó, và nó tự lộ bằng phép đếm.** Trang 37 viết:

*"In Sections 4 through 6 below we propose new classes of robust alternatives to the least
squares estimator for the linear model and show that members of these classes have, unlike
the least squares estimator, high efficiency over a wide range of error distributions."*

Dịch: *"Trong các **mục 4 đến 6** dưới đây, chúng tôi đề xuất những lớp mới các lựa chọn bền
vững thay cho ước lượng bình phương tối thiểu cho mô hình tuyến tính, và chỉ ra rằng các phần
tử của những lớp này có, khác với ước lượng bình phương tối thiểu, hiệu suất cao trên một dải
rộng các phân phối sai số."*

Nhưng bài báo chỉ có **năm mục**:

| Mục | Trang | Tiêu đề |
|---|---:|---|
| 1 | 33 | INTRODUCTION |
| 2 | 34 | BACKGROUND AND MOTIVATION |
| 3 | 38 | REGRESSION QUANTILES |
| 4 | 42 | THE ASYMPTOTIC DISTRIBUTION THEORY OF REGRESSION QUANTILES |
| 5 | 48 | CONCLUSION |
| — | 48 | APPENDIX: COMPUTATIONAL ASPECTS OF REGRESSION QUANTILES |

**Mục 6 không tồn tại.** Nội dung được hứa thì có thật — Định lý 4.3 và đoạn trang 47 đúng là
"những lớp mới các lựa chọn bền vững" — nhưng chúng nằm gọn trong mục 4 và mục 5. Dấu vết của
một lần gộp mục khi sửa bản thảo: trang 48 ghi *"Manuscript received February, 1975; revision
received January, 1977"* — **hai năm** giữa hai mốc.

Đây là cùng lớp với chuyện phụ lục không tồn tại ở bài [YOLOv7](<../../Deep Learning/YOLOv7/YOLOv7.md>), chỉ nhẹ hơn:
ở đó phần được trỏ tới **không có**; ở đây phần được trỏ tới **có**, chỉ mang số khác.

### 2.21 Phụ lục — bài toán (P), (D) và (D') — trang 48–49

![Phụ lục: (P) và (D)](<paper_kb1978_phu_luc_P_D.png>)

**Hình 29 — Cuối mục 5, Phụ lục, và hai bài toán (P), (D), trang 48.**

**Trích nguyên văn:** *"The regression quantile minimization problem posed above in Section 3
is equivalent to the linear program"*

$$\text{(P)}\qquad \min\ \bigl[\theta\,\iota'r^+ + (1-\theta)\,\iota'r^-\bigr]
\quad\text{với}\quad y = Xb + r^+ - r^-,\ \ (b,r^+,r^-)\in\mathbb{R}^K\times\mathbb{R}^{2T}_+$$

*"where $\iota'=(1,1,\ldots,1)$, a $T$ vector of ones. The dual to (P) is the bounded
variables problem,"*

$$\text{(D)}\qquad \max\ [\,y'd\,]\quad\text{với}\quad X'd=0,\ \ d\in[\theta-1,\theta]^T$$

**Ý nghĩa trừu tượng của (D), đọc ngay trước khi sang (D').** Ràng buộc $X'd=0$ nói rằng
vectơ $d$ phải **trực giao với mọi cột của $X$** — kể cả cột toàn số 1, nên $\sum_t d_t=0$.
Và ràng buộc $d\in[\theta-1,\theta]^T$ nói mỗi toạ độ bị kẹp trong một khoảng **rộng đúng
bằng 1**, đúng khoảng đã gặp ở công thức (3.1) của mục 2.10. Hai ràng buộc ấy gộp lại là một
cách phát biểu khác của điều kiện tối ưu: **tìm một cách gán "phiếu" cho từng quan sát, mỗi
phiếu trong $[\theta-1,\theta]$, sao cho tổng phiếu theo mọi hướng của $X$ bằng 0 và tổng
$y'd$ lớn nhất.**

Chỗ đáng nhớ: $\theta$ vào (D) qua **chặn của biến**, trong khi ở (P) nó vào qua **hệ số của
hàm mục tiêu**. Cùng một tham số, hai vai trò khác hẳn — và phép đổi biến sang (D') dưới đây
là để biến chặn phụ thuộc $\theta$ ấy thành chặn cố định $[0,1]$.

![Phụ lục: (D')](<paper_kb1978_phu_luc_D_phay.png>)

**Hình 30 — Ràng buộc của (D') và câu về biến thiên tham số, trang 49.**

$$\text{(D')}\qquad \max\ [\,y'\Delta\,]\quad\text{với}\quad
X'\Delta=(1-\theta)X'\iota,\ \ \Delta\in[0,1]^T,\qquad \Delta=d+1-\theta$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $r^+$, $r^-$ | phần **dương** và phần **âm** của phần dư, cả hai $\ge0$ | $\mathbb{R}^{T}_+$ |
| $d$ | biến đối ngẫu, bị chặn hai đầu bởi $\theta-1$ và $\theta$ | $\mathbb{R}^{T}$ |
| $\Delta=d+1-\theta$ | bản đã dịch, bị chặn trong $[0,1]$ | $\mathbb{R}^{T}$ |

**Ý nghĩa trừu tượng của (P).** Tách phần dư thành hai nửa không âm là mẹo trung tâm: hàm
$\lvert u\rvert$ không tuyến tính, nhưng $u=r^+-r^-$ với $r^+,r^-\ge0$ thì
$\lvert u\rvert=r^++r^-$ **tại nghiệm tối ưu**, và cả hai vế đều tuyến tính. Cái giá là số
biến tăng từ $K$ lên $K+2T$ — và trên thực tế lên $2K+2T$, vì $b$ tự do về dấu nên cũng phải
tách đôi trong dạng chuẩn.

**Ý nghĩa trừu tượng của (D), và đây là phần bài viết cũ của database bỏ qua hoàn toàn.** Bài
đối ngẫu có **$T$ biến và $K$ ràng buộc** — trong khi bài gốc có $2K+2T$ biến và $T$ ràng
buộc. Với $T$ lớn và $K$ nhỏ, đó là một bài toán dễ hơn hẳn.

Và mỗi biến $\Delta_t$ có một nghĩa cụ thể. **Đo ra chứ không đoán:**

| Phần dư của quan sát $t$ | Chặn hoạt động ở $d_t$ | Giá trị $\Delta_t$ |
|---|---|---|
| **dương** ($r^+_t>0$) | chặn trên $\theta$ | $\Delta_t=1$ |
| **âm** ($r^-_t>0$) | chặn dưới $\theta-1$ | $\Delta_t=0$ |
| **bằng 0** | không chặn nào hoạt động | $0<\Delta_t<1$ |

Kiểm ở $T=50$, $K=3$, $\theta=0{,}35$: **0/50** quan sát vi phạm quan hệ trên; cách đọc
**ngược lại** cho **47/50** vi phạm. Và số toạ độ nằm hẳn trong $(0,1)$ là **3**, đúng bằng
$K$ — tức chúng chính là $K$ điểm neo của Định lý 3.1, nhìn từ phía đối ngẫu.

Giá trị hai bài toán **không bằng nhau**, và đó không phải lỗi: phép đổi biến $\Delta=d+1-\theta$
làm hàm mục tiêu lệch đúng một hằng số, $y'\Delta=y'd+(1-\theta)\sum_t y_t$. Kiểm:

| $\theta$ | Giá trị (P) | Giá trị (D') | Hiệu | $(1-\theta)\sum y_t$ | $\Delta=0$ | $\Delta=1$ | $0<\Delta<1$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $0{,}10$ | $11{,}1291849034$ | $66{,}8003320704$ | $55{,}6711471670$ | $55{,}6711471670$ | 3 | 44 | 3 |
| $0{,}25$ | $21{,}2813946627$ | $67{,}6740173019$ | $46{,}3926226391$ | $46{,}3926226391$ | 11 | 36 | 3 |
| $0{,}50$ | $26{,}1498570738$ | $57{,}0782721665$ | $30{,}9284150928$ | $30{,}9284150928$ | 23 | 24 | 3 |
| $0{,}75$ | $23{,}7758664638$ | $39{,}2400740102$ | $15{,}4642075464$ | $15{,}4642075464$ | 36 | 11 | 3 |
| $0{,}90$ | $13{,}6064126495$ | $19{,}7920956681$ | $6{,}1856830186$ | $6{,}1856830186$ | 44 | 3 | 3 |

Với $\sum y_t = 61{,}8568301855$. Lệch lớn nhất giữa hiệu đo được và
$(1-\theta)\sum y_t$: $7{,}105\times10^{-15}$.

**Câu cuối của phụ lục là câu có giá trị thực hành cao nhất trong cả bài báo:**

*"This translated dual formulation proves to be extremely convenient for computational
purposes. Standard linear programming algorithms provide quite efficient algorithms for such
bounded variables problems. So-called "parametric variation of the right-hand-side" of (D')
permits economical solution for many, or indeed all, $\theta\in[0,1]$."*

Dịch: *"Dạng đối ngẫu đã dịch này tỏ ra cực kỳ tiện cho mục đích tính toán. Các thuật toán
quy hoạch tuyến tính chuẩn cung cấp những thuật toán khá hiệu quả cho các bài toán biến bị
chặn như thế. Cái gọi là "biến thiên tham số của vế phải" của (D') cho phép giải một cách
tiết kiệm cho **nhiều, hay thực ra là tất cả**, $\theta\in[0,1]$."*

**Đây là câu trả lời cho "mỗi $\theta$ một mô hình riêng" ở mục 0.1 — và bài báo đã có nó từ
1978.** Nhìn vào (D'): $\theta$ chỉ xuất hiện ở **vế phải** $(1-\theta)X'\iota$, và vế phải
đó là một **đường thẳng theo $\theta$**. Ma trận ràng buộc $X'$ không đổi. Nên đi từ $\theta$
này sang $\theta$ khác không cần giải lại từ đầu; chỉ cần đẩy vế phải và xoay cơ sở khi nó
đổi. Đó chính là thứ mục 2.14 khai thác để dựng cả đường nghiệm của Bảng II — và cũng là thứ
`scikit-learn` **không** làm, xem mục 3.

---

## 3. Từ bài báo 1978 tới thư viện hôm nay

Bài báo định nghĩa bằng $\theta$, giải bằng (P) hoặc (D'), và đo bằng $\Omega\otimes Q^{-1}$.
Thư viện hôm nay đổi tên, đổi một chỗ trong công thức, và **bỏ mất một thứ**. Bảng đối chiếu:

| Bài báo 1978 | `scikit-learn` `QuantileRegressor` | Ghi chú |
|---|---|---|
| $\theta$ | tham số `quantile` | cùng nghĩa, chỉ đổi tên |
| $T$, $K$ | `n_samples`, `n_features` | — |
| bài (P), tr.48 | `scipy.optimize.linprog(method="highs")` | **cùng một dạng**, xem dưới |
| $b$ tự do về dấu | tách đôi $\beta=\beta^+-\beta^-$ | thư viện làm, bài báo không nói |
| — | tham số `alpha`: phạt $L_1$ lên hệ số | **KHÔNG có trong bài báo 1978** |
| (D') với vế phải tham số hoá, tr.49 | **không dùng** | mỗi `quantile` là một lần gọi `linprog` riêng |

Ba điều đọc ra từ bảng này:

1. **`alpha` không phải $\alpha$ của bài báo.** Trong bài báo, $\alpha$ ở trang 47 là **tỉ lệ
   cắt đuôi**; trong thư viện, `alpha` là **mức phạt $L_1$** kiểu [Lasso](<../Lasso Regression/Lasso Regression.md>).
   Hai thứ không liên quan gì nhau. Muốn đúng bài báo thì đặt `alpha=0`.
2. **Dạng (P) là dạng thư viện dùng.** Dựng lại (P) bằng `scipy.optimize.linprog` rồi so với
   `QuantileRegressor(alpha=0, solver="highs")` trên cùng dữ liệu, ở năm mức $\theta$: lệch
   $0{,}00\times10^{0}$ ở **cả năm** mức, tức trùng từng bit. Điều đó không chứng minh độ
   chính xác — hai đường cùng gọi HiGHS trên cùng bài toán — nhưng nó chứng minh **bài toán
   được dựng đúng**: sai một dấu trong ma trận ràng buộc là con số khác ngay.
3. **Thứ bị bỏ mất là câu cuối của Phụ lục.** Bài báo nói rõ (D') cho phép giải *"for many,
   or indeed all, $\theta\in[0,1]$"* một cách tiết kiệm; `scikit-learn` giải lại từ đầu cho
   mỗi $\theta$. Gói `quantreg` của R thì có — hàm `rq` với `tau=-1` trả về **cả đường
   nghiệm**, đúng theo Phụ lục. Đó là chỗ một câu ở trang 49 trở thành một khác biệt thực
   hành 47 năm sau.

Đo trên $T=400$, $K=2$, dữ liệu $y=1+0{,}5x+(0{,}3+0{,}3x)\varepsilon$:

| $\theta$ | $\beta^*_1$ | $\beta^*_2$ | Lệch với `sklearn` | Tỉ lệ nằm dưới | $N$ | $Z$ |
|---:|---:|---:|---:|---:|---:|---:|
| $0{,}10$ | $0{,}3597812245$ | $0{,}2064511239$ | $0{,}00\times10^{0}$ | $0{,}097500$ | 39 | 2 |
| $0{,}25$ | $0{,}6467400754$ | $0{,}3323229047$ | $0{,}00\times10^{0}$ | $0{,}247500$ | 99 | 2 |
| $0{,}50$ | $0{,}8819308829$ | $0{,}5543104328$ | $0{,}00\times10^{0}$ | $0{,}497500$ | 199 | 2 |
| $0{,}75$ | $1{,}0799527250$ | $0{,}7376227500$ | $0{,}00\times10^{0}$ | $0{,}747500$ | 299 | 2 |
| $0{,}90$ | $1{,}3078586526$ | $0{,}8781836986$ | $0{,}00\times10^{0}$ | $0{,}897500$ | 359 | 2 |

Đọc cột $N$ và $Z$ bằng công thức (3.7) của mục 2.12: ở $\theta=0{,}10$ thì $N=39$, $Z=2$,
$T\theta=40$, và $39\le40\le41$ — **đúng, sát cả hai biên**. Cột "tỉ lệ nằm dưới" vì thế bằng
$\theta-1/400$ ở cả năm mức, không phải vì một công thức đếm nào mà vì $N$ chạm đúng biên
dưới của (3.7).

Bình phương tối thiểu trên cùng dữ liệu cho **đúng một** đường: hệ số chặn $0{,}8263081793$,
hệ số góc $0{,}5409838544$. Trong khi hệ số góc của năm đường phân vị đi từ $0{,}2064511239$
tới $0{,}8781836986$, tức **gấp $4{,}2537123647$ lần**.

---

## 4. Vì sao đổi hàm mất mát lại đổi câu trả lời

Mục 2.5 đã chỉ ra cơ chế bằng một câu: dịch đường lên một chút thì mỗi điểm **phía trên** rẻ
đi $\theta$ và mỗi điểm **phía dưới** đắt thêm $1-\theta$, nên cân bằng rơi đúng vào chỗ tỉ
lệ điểm nằm dưới bằng $\theta$. Mục này đo cơ chế đó.

![Hàm mất mát pinball](<quantile_ham_pinball.png>)

**Hình 31 — Hàm mất mát của bài báo, vẽ cho ba mức $\theta$ (hình tự vẽ).** Bảng trái: hàm
$\rho_\theta(u)$. Bảng phải: **độ dốc** của nó, tức thứ quyết định đường bị đẩy về đâu.

**Cách đọc hình.**

- **Bảng trái.** Ba chữ V lệch, đỉnh tại $u=0$. Không phải parabol như $u^2$ của bình phương
  tối thiểu, và chỉ đối xứng khi $\theta=0{,}5$. Với $\theta=0{,}9$, nhánh phải (đoán thiếu)
  dốc hơn nhánh trái (đoán thừa) đúng **9 lần**.
- **Bảng phải.** Đây mới là bảng nói được cơ chế: độ dốc là một **hàm bậc thang**, nhận đúng
  hai giá trị $\theta$ và $\theta-1$, nhảy tại $u=0$. So với bình phương tối thiểu, nơi độ
  dốc là $2u$ — **tỉ lệ với khoảng cách và không bị chặn**.
- Điều cần nhìn: **bề cao của bậc thang luôn bằng 1**, ở cả ba mức $\theta$. Đổi $\theta$ chỉ
  **trượt** bậc thang lên xuống, không làm nó cao hơn hay thấp hơn. Đó đúng là điều mục 2.10
  đọc ra từ công thức (3.1): khoảng $(\theta-1,\ \theta)$ có độ rộng đúng bằng 1.
- **Điều hình KHÔNG nói ra:** độ dốc bị chặn giải thích vì sao phân vị hồi quy bền với ngoại
  lai theo $y$, nhưng **không** nói gì về ngoại lai theo $x$. Ở mục 2.12, công thức (3.2) có
  thừa số $x_tw$ đứng ngoài ngoặc, và thừa số đó **không bị chặn**. Đó là giới hạn ghi ở mục 6.

<details>
<summary><b>Mã sinh Hình 31</b> — khối tự chứa, chạy trong dưới một giây</summary>

```python
# tools/fig_quantile.py — phần B
# Hình này thuần toán: không có dữ liệu nào, không có con số nào phải đi mượn.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Ham mat mat cua bai bao tr.38, viet gon mot dong (cach viet doi sau).
def rho(u, th):
    u = np.asarray(u, dtype=float)
    return u * (th - (u < 0))

u = np.linspace(-4, 4, 1601)
BO = [(0.1, "#2563eb"), (0.5, "#111827"), (0.9, "#dc2626")]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))

for th, c in BO:
    ax[0].plot(u, rho(u, th), lw=1.9, color=c, label=r"$\theta$ = %.1f" % th)
ax[0].axvline(0, color="#9ca3af", lw=0.8)
ax[0].set_xlabel("phần dư  u = y − xb"); ax[0].set_ylabel(r"$\rho_\theta(u)$")
ax[0].legend(fontsize=8.4)
ax[0].set_title("Chữ V LỆCH, đỉnh tại u = 0\n"
                "θ = 0,9: đoán thiếu bị phạt gấp 9 lần đoán thừa", fontsize=9)

for th, c in BO:
    doc = np.where(u >= 0, th, th - 1.0)
    ax[1].step(u, doc, where="post", lw=1.9, color=c, label=r"$\theta$ = %.1f" % th)
    ax[1].annotate("%.1f" % th, (0.35, th), fontsize=7.6, color=c, va="bottom", ha="left",
                   bbox=dict(fc="white", ec="none", pad=0.8))
    ax[1].annotate("%.1f" % (th - 1), (-0.35, th - 1), fontsize=7.6, color=c,
                   va="top", ha="right", bbox=dict(fc="white", ec="none", pad=0.8))
ax[1].plot(u, 2 * u / 8.0, "--", color="#6b7280", lw=1.2,
           label="bình phương tối thiểu, chia 8")
ax[1].axvline(0, color="#9ca3af", lw=0.8)
ax[1].axhline(0, color="#9ca3af", lw=0.8)
ax[1].set_ylim(-1.15, 1.15)
ax[1].set_xlabel("phần dư  u"); ax[1].set_ylabel("độ dốc")
ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("Độ dốc là BẬC THANG, bề cao luôn bằng 1\n"
                "đổi θ chỉ TRƯỢT bậc thang, không làm nó cao hơn", fontsize=9)

fig.savefig("quantile_ham_pinball.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

**Ba cơ chế, đo riêng từng cái:**

| Cơ chế | Đo ở đâu | Con số |
|---|---|---|
| Độ dốc bị chặn ⟹ kéo một điểm ra xa không làm đường đổi | mục 2.13 | lệch $0{,}000\times10^{0}$ với $d_t$ tới 50 |
| Phạt bất đối xứng ⟹ tỉ lệ điểm nằm dưới bị kẹp quanh $\theta$ | mục 2.12 | (3.7) đúng 20/20 |
| $\theta$ chỉ vào bài toán qua hai trọng số ⟹ đường nghiệm là bậc thang | mục 2.14 | Bảng II, 4 bậc, 3 mốc |

---

## 5. Kết quả — đo trên dữ liệu có phương sai đổi theo $x$

![Năm đường phân vị](<quantile_nam_phan_vi.png>)

**Hình 32 — Năm đường phân vị trên dữ liệu phương sai không đều, và độ phủ theo từng khúc
$x$ (hình tự vẽ).**

**Cách đọc hình.**

- **Bảng trái.** 400 chấm xám là dữ liệu; năm đường màu là năm đường phân vị hồi quy; đường
  đen đứt là bình phương tối thiểu. Điều cần nhìn: các đường phân vị **toả ra hình quạt** khi
  $x$ tăng, trong khi bình phương tối thiểu chỉ cho một đường duy nhất đi giữa.
- **Bảng phải.** Hai cột cho mỗi khúc $x$: độ phủ của khoảng phân vị $5\%$–$95\%$ và của
  khoảng $\hat y\pm1{,}96s$. Vạch ngang là mức $90\%$ nhắm tới. Điều cần nhìn: cột phân vị
  bám sát vạch ở cả năm khúc; cột bình phương tối thiểu **vọt lên $1{,}000000$** ở hai khúc
  đầu rồi **tụt xuống $0{,}812500$** ở khúc cuối.
- **Điều hình KHÔNG nói ra, và nó quan trọng:** dữ liệu này **vi phạm mô hình (1.1)** của
  chính bài báo. Công thức (1.1) đòi mọi $Y_t$ là bản tịnh tiến của cùng một $F$, tức mọi
  đường phân vị **song song**. Hình quạt ở đây thì không. Nên đây là một minh hoạ cho **cách
  dùng hiện đại**, không phải một minh hoạ cho lý thuyết của mục 4 — và lý thuyết tiệm cận ở
  mục 2.16 **không áp được** cho nó nguyên xi.

<details>
<summary><b>Mã sinh Hình 32</b> — khối tự chứa; dữ liệu sinh lại từ cùng hạt giống</summary>

```python
# tools/fig_quantile.py — phần C
# Du lieu sinh lai tu dung hat giong cua v_quantile.py muc 17, nen khoi nay tu chua.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def giai_lp(y, X, th):
    """Bai (P) cua Phu luc tr.48; beta tach doi vi no tu do ve dau."""
    T, K = X.shape
    c = np.r_[np.zeros(2 * K), th * np.ones(T), (1.0 - th) * np.ones(T)]
    A = np.hstack([X, -X, np.eye(T), -np.eye(T)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(0, None)] * (2 * K + 2 * T), method="highs")
    return r.x[:K] - r.x[K:2 * K]


rng = np.random.default_rng(430430)
T = 400
x = np.sort(rng.uniform(0, 10, T))
y = 1 + 0.5 * x + (0.3 + 0.3 * x) * rng.normal(size=T)
X = np.c_[np.ones(T), x]
ols = np.linalg.lstsq(X, y, rcond=None)[0]

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0))

ax[0].scatter(x, y, s=7, color="#9ca3af", zorder=2)
for th, c in ((0.10, "#2563eb"), (0.25, "#0891b2"), (0.50, "#111827"),
              (0.75, "#d97706"), (0.90, "#dc2626")):
    b = giai_lp(y, X, th)
    ax[0].plot(x, X @ b, lw=1.5, color=c, zorder=4,
               label=r"$\theta$ = %.2f  (hệ số góc %.4f)" % (th, b[1]))
# Duong OLS gan TRUNG duong theta = 0.50 (he so goc 0.5410 so voi 0.5543), nen phai
# doi mau va ve SAU CUNG, neu khong no bi che kin — loi da gap o Hinh 34 bai Huber.
ax[0].plot(x, X @ ols, "--", color="#7c3aed", lw=2.6, zorder=8, dashes=(5, 3),
           label="bình phương tối thiểu (%.4f)" % ols[1])
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].legend(fontsize=7.2, ncol=2, loc="upper left")
ax[0].set_title("Phương sai tăng theo x: năm đường phân vị TOẢ RA\n"
                "bình phương tối thiểu chỉ cho MỘT đường", fontsize=9)

b05, b95 = giai_lp(y, X, 0.05), giai_lp(y, X, 0.95)
s = float(np.std(y - X @ ols, ddof=X.shape[1]))
trong_pv = (y >= X @ b05) & (y <= X @ b95)
trong_ols = np.abs(y - X @ ols) <= 1.96 * s
lo = np.arange(0, 10, 2)
pv = [float(np.mean(trong_pv[(x >= a) & (x < a + 2)])) for a in lo]
ls = [float(np.mean(trong_ols[(x >= a) & (x < a + 2)])) for a in lo]
w = 0.38
ax[1].bar(np.arange(5) - w / 2, pv, w, color="#0891b2", label="phân vị 5 %–95 %")
ax[1].bar(np.arange(5) + w / 2, ls, w, color="#9ca3af",
          label=r"bình phương tối thiểu $\pm 1{,}96\,s$")
ax[1].axhline(0.90, color="#dc2626", lw=1.2, linestyle="--", label="mức 90 % nhắm tới")
for i, (a, b) in enumerate(zip(pv, ls)):
    ax[1].annotate("%.3f" % a, (i - w / 2, a), ha="center", va="bottom", fontsize=7)
    ax[1].annotate("%.3f" % b, (i + w / 2, b), ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(np.arange(5))
ax[1].set_xticklabels(["[%d, %d)" % (a, a + 2) for a in lo])
ax[1].set_ylim(0, 1.12); ax[1].set_xlabel("khúc x"); ax[1].set_ylabel("tỉ lệ phủ")
ax[1].legend(fontsize=7.6, loc="lower left")
ax[1].set_title("Độ phủ theo từng khúc x\n"
                "cột xám phủ QUÁ ở đầu này và phủ THIẾU ở đầu kia", fontsize=9)

fig.savefig("quantile_nam_phan_vi.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

Số liệu đã in ở mục 1.3 và mục 3. Ba con số đáng nhớ:

* Hệ số góc đi từ $0{,}2064511239$ ($\theta=0{,}10$) tới $0{,}8781836986$ ($\theta=0{,}90$),
  tức **gấp $4{,}2537123647$ lần**.
* Độ phủ ở khúc $x\in[8,10)$: phân vị $0{,}887500$, bình phương tối thiểu $0{,}812500$.
* Chênh giữa ô tệ nhất và ô tốt nhất: phân vị $0{,}050345$, bình phương tối thiểu
  $0{,}187500$ — **gấp $3{,}7242857143$ lần**.

---

## 6. Giá phải trả

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
|---|---|---|---|
| Khoảng đúng ở mọi vùng $x$ | phủ $0{,}812500\to0{,}887500$ ở khúc $x$ lớn; độ chênh giữa các khúc nhỏ hơn $3{,}7242857143$ lần | phải giải **một bài quy hoạch tuyến tính cho mỗi $\theta$** | mục 1.3 · mục 3 |
| Trả lời được câu hỏi về đuôi phân phối | 5 hệ số góc khác nhau, chênh nhau $4{,}2537123647$ lần | **mất nghiệm đóng**; một dải 9 mức tốn từ $1\,790{,}78$ tới $81\,452{,}38$ lần thời gian một lần bình phương tối thiểu | mục 3 · bảng dưới |
| Bền với ngoại lai theo $y$ — **tuyệt đối**, không phải "ít nhạy hơn" | đẩy phần dư xa 50 lần, đường lệch $0{,}000\times10^{0}$ | **không bền với ngoại lai theo $x$**: thừa số $x_tw$ trong (3.2) không bị chặn | mục 2.13 · mục 4 |
| Không cần giả định dạng phân phối nhiễu | Định lý 4.2 chỉ đòi $F$ liên tục, mật độ dương tại $\xi(\theta_i)$ | ước lượng ở $\theta$ cực đoan **rất kém ổn định**, vì $f(\xi(\theta))^2$ ở mẫu số | mục 2.15 |
| Hiệu suất cao trên nhiều phân phối | Gastwirth giữ $0{,}7957459726$–$0{,}9104704097$ trên bốn phân phối | ở phân phối chuẩn, trung vị hồi quy mất **$57{,}0796\%$** hiệu suất | mục 2.3 · mục 2.2 |
| Nghiệm neo vào $K$ quan sát, hình học rõ ràng | $Z(u^*)=K$ ở 20/20 phép đo | **nghiệm có thể KHÔNG duy nhất**: Bảng II của chính bài báo có 3 mốc $\theta$ nhiều nghiệm | mục 2.7 · mục 2.14 |
| — | — | **các đường phân vị có thể cắt nhau**, và bài báo 1978 không nhắc tới vấn đề này | mục 7 |
| — | — | **suy luận thống kê khó hơn**: sai số chuẩn cần ước lượng $f(\xi(\theta))$, thứ không có công thức mẫu gọn | mục 2.15 |

**Chi phí thời gian, đo trên một lần chạy** — con số này **không tái lập được** giữa các máy
và các lần chạy, và phải đọc như thế:

| $T$ | $K$ | Bình phương tối thiểu (ms) | 1 phân vị (ms) | 9 phân vị (ms) | 9 phân vị / bình phương tối thiểu |
|---:|---:|---:|---:|---:|---:|
| 200 | 3 | $0{,}0239$ | $5{,}0497$ | $42{,}7809$ | $1\,790{,}78$ |
| 1 000 | 3 | $0{,}0451$ | $53{,}7777$ | $459{,}4488$ | $10\,196{,}13$ |
| 5 000 | 3 | $0{,}1063$ | $998{,}0714$ | $8\,657{,}3298$ | $81\,452{,}38$ |
| 1 000 | 10 | $0{,}0922$ | $73{,}3285$ | $661{,}5291$ | $7\,177{,}07$ |

**Đọc bảng này cho đúng.** Nó đo **dạng (P) dựng dày đặc bằng `scipy.optimize.linprog`**, với
$2K+2T$ biến — ở $T=5000$, $K=3$ là hơn mười nghìn biến cho ba hệ số. Nó **không** đo giới hạn của hồi quy
phân vị: chính Phụ lục trang 49 của bài báo chỉ ra rằng bài đối ngẫu (D') chỉ có $T$ biến và
$K$ ràng buộc, và rằng biến thiên tham số của vế phải cho phép lấy **cả dải $\theta$** với
chi phí gần như một lần giải. Cột cuối vì thế là **chặn trên của một cách cài đặt ngây thơ**,
không phải một tính chất của phương pháp.

---

## 7. Họ hàng và phần đời sau

* [Linear Regression](<../Linear Regression/Linear Regression.md>) — ước lượng **trung bình** có điều kiện thay vì
  phân vị. Chính là thứ bài báo đem ra so ở mọi mục.
* [Huber Regression](<../Huber Regression/Huber Regression.md>) — Huber 1964 là tài liệu tham khảo **[24]** của
  bài báo này, và Huber 1973 là **[26]**. Hai cách bền vững khác nhau: Huber **chặn** ảnh
  hưởng của ngoại lai, phân vị hồi quy làm nó **bằng không** khi điểm không đổi phía. Cẩn
  thận: ký hiệu $\psi$ ở hai bài mang hai nghĩa ngược nhau, xem mục 0.2.
* [Lasso Regression](<../Lasso Regression/Lasso Regression.md>) — `alpha` của `QuantileRegressor` chính là ý
  tưởng phạt $L_1$ của Lasso ghép vào, và nó **không có trong bài báo 1978**.
* [Gradient Boosting (GBM)](<../../Ensemble/Gradient Boosting (GBM)/Gradient Boosting (GBM).md>) — `loss='quantile'` dùng đúng hàm
  mất mát của mục 2.6 nhưng bỏ ràng buộc tuyến tính, cho phân vị **phi tuyến**. Đây là cách
  dùng phổ biến nhất hôm nay.
* [Random Forest](<../../Ensemble/Random Forest/Random Forest.md>) — quantile regression forest cho cả phân phối có điều
  kiện từ **một** mô hình, tránh được chuyện "mỗi $\theta$ một mô hình" của mục 0.1.
* [Quantile và Percentile](<../../Nền tảng/Quantile và Percentile/Quantile và Percentile.md>) — khái niệm phân vị ở dạng không có
  biến giải thích.

**Ba thứ nổi tiếng gắn với hồi quy phân vị mà KHÔNG có trong bài báo 1978:**

1. **Ký hiệu $\rho_\tau$ và chữ "pinball loss".** Bài báo viết hai tổng tách theo dấu, dùng
   chữ $\theta$, và không đặt tên cho hàm mất mát.
2. **Vấn đề các đường phân vị cắt nhau (*quantile crossing*).** Không xuất hiện trong 18
   trang. Và điều đó hợp lý: dưới mô hình (1.1) mọi đường phân vị **song song** nên không thể
   cắt nhau. Vấn đề chỉ sinh ra khi người ta dùng phương pháp **ngoài** mô hình của bài báo —
   đúng thứ mục 5 làm.
3. **Ứng dụng cho dữ liệu phương sai không đều.** Cũng vì lý do trên: mô hình (1.1) là mô
   hình sai số đồng nhất.

---

## 8. Dùng khi nào, không dùng khi nào

**Dùng khi:**

* Câu hỏi nghiệp vụ vốn là về **đuôi phân phối**: mức dự trữ rủi ro (VaR chính là một phân
  vị), thời gian giao hàng cam kết cho 95 % đơn, mức tiêu thụ đỉnh.
* Cần **khoảng dự đoán** chứ không chỉ một điểm, và độ tản mát của $y$ **đổi theo $x$** —
  mục 1.3 và mục 5 đo đúng chỗ này.
* $y$ lệch mạnh hoặc bị chặn (thu nhập, thời gian chờ): **trung vị có điều kiện** nói được
  điều mà trung bình có điều kiện không nói.
* Muốn một ước lượng **bền với ngoại lai theo $y$** mà vẫn giữ được ý nghĩa thống kê rõ ràng
  — mục 2.13.

**Không dùng khi:**

* Chỉ cần trung bình có điều kiện và mô hình Gauss đồng phương sai là đúng: ở đó bình phương
  tối thiểu tốt hơn, và bài viết này có con số — trung vị hồi quy mất $57{,}0796\%$ hiệu
  suất (mục 2.2).
* $T$ nhỏ và $\theta$ cực đoan: công thức $\omega$ ở mục 2.15 có $f(\xi(\theta))^2$ ở mẫu số,
  và ở đuôi phân phối mẫu số đó gần 0.
* Ngoại lai nằm ở **$x$**, không phải ở $y$: phương pháp này không bảo vệ gì cả.
* Cần tốc độ trên dữ liệu rất lớn **và** dùng một cài đặt giải lại từ đầu cho mỗi $\theta$ —
  xem bảng thời gian ở mục 6, và xem lối thoát ở Phụ lục trang 49 (mục 2.21).

**Quy trình dùng thật:**

1. Chọn $\theta$ từ quyết định nghiệp vụ, không chọn vì đường trông đẹp.
2. Đặt `alpha=0` nếu muốn đúng bài báo; `alpha>0` là thêm một mô hình khác (mục 3).
3. Kiểm $T\theta$ có nguyên không. Nếu có, nghiệm không duy nhất và giá trị trả về phụ thuộc
   bộ giải — mục 2.11.
4. Đo độ phủ **ngoài mẫu** và **theo từng vùng $X$**, không chỉ đo tổng thể: mục 1.3 cho thấy
   con số tổng thể giấu được chỗ hỏng.
5. Kiểm xem các đường phân vị có cắt nhau không. Nếu có, đó là dấu hiệu dữ liệu nằm ngoài mô
   hình (1.1), và phải xử lý riêng.
6. Muốn sai số chuẩn thì đừng dùng công thức của bình phương tối thiểu. Công thức đúng là
   $\omega_{ii}Q^{-1}/T$ ở mục 2.16, và nó đòi ước lượng $f(\xi(\theta))$.

---

## 9. Đối chiếu với bài báo gốc

Luật 6 đòi ba bảng: theo **hình/bảng**, theo **mục**, và theo **công thức đánh số**. Bài báo
này có **0 hình**, nên bảng thứ nhất chỉ có hai dòng bảng — và chính con số 0 ấy là một dữ
kiện định hình cả bài viết.

### 9.1 Theo hình và bảng

| # | Hình/Bảng của bài báo | Trang | Vào bài chưa | Ở mục nào · vì sao không lấy |
|---|---|---:|---|---|
| — | **hình vẽ** | — | — | **bài báo không có hình vẽ nào trong 18 trang** |
| Bảng I | Empirical Variances of Some Alternative Location Estimators | 37 | **có** | Hình 1 · dựng lại 30 ô ở mục 2.4 |
| Bảng II | Regression quantiles cho mọi $\theta$, mô hình hai biến 5 quan sát | 42 | **có** | Hình 20 · dựng lại 8 ô **và cả dữ liệu** ở mục 2.14 |

**2/2 bảng đã vào bài, và cả hai đã được dựng lại bằng số.**

### 9.2 Theo mục

| Mục | Trang | Tiêu đề | Vào bài chưa | Ở mục nào · vì sao không lấy |
|---|---:|---|---|---|
| 1 | 33 | Introduction | **có** | mục 2.1 (tóm tắt + (1.1)) |
| 2 | 34–37 | Background and Motivation | **có** | mục 1.1 · 1.2 · 2.2 · 2.3 · 2.4 |
| 3 | 38–42 | Regression Quantiles | **có** | mục 2.5 → mục 2.14 (đủ 5 định lý) |
| 4 | 42–47 | The Asymptotic Distribution Theory of Regression Quantiles | **có** | mục 2.15 → mục 2.19 (đủ 3 định lý) |
| 5 | 48 | Conclusion | **có** | mục 2.20 |
| Phụ lục | 48–49 | Computational Aspects of Regression Quantiles | **có** | mục 2.21 ((P), (D), (D')) |
| Tham khảo | 49–50 | 43 mục | **một phần** | mục 10 dẫn 6 mục có liên quan trực tiếp; 37 mục còn lại là bối cảnh lịch sử của mục 2, không mang nội dung kỹ thuật vào bài |

**6/6 mục (5 mục đánh số + 1 phụ lục) đã vào bài.**

Và một dòng phải ghi ra: **câu ở trang 37 hứa "Sections 4 through 6"** trong khi bài báo chỉ
có 5 mục — mục 2.20 đo và giải thích chỗ này.

### 9.3 Theo công thức đánh số

| Nhãn | Trang | Nội dung | Vào bài chưa | Ở mục nào |
|---|---:|---|---|---|
| công thức (1.1) | 33 | $P(Y_t<y)=F(y-x_t\beta)$ | **có** | mục 2.1 |
| — | 38 | định nghĩa phân vị mẫu (không đánh số) | **có** | mục 2.5 |
| — | 38 | định nghĩa phân vị hồi quy (không đánh số) | **có** | mục 2.6 |
| — | 39 | $\psi(b;\theta,y,X)$, hai cách viết | **có** | mục 2.9 |
| công thức (3.1) | 40 | điều kiện cần và đủ cho nghiệm duy nhất | **có** | mục 2.10 · kiểm 12/12 + thử ngược 779/779 |
| công thức (3.2) | 40 | đạo hàm theo hướng $\psi'(b;w)$ | **có** | mục 2.9 |
| công thức (3.3) | 40 | $\psi'(\beta^*;w)$ tách hai tổng | **một phần** | mục 2.10 — bước trung gian của chứng minh Định lý 3.3, nội dung đã gói trong công thức (3.1) |
| công thức (3.4) | 40 | điều kiện viết theo $v=X(h)w$ | **một phần** | như trên |
| công thức (3.5) | 40 | dạng trước khi rút gọn thành công thức (3.1) | **một phần** | như trên |
| công thức (3.6) | 40 | bản thu gọn về mô hình vị trí | **có** | mục 2.11 · kiểm 9 cấu hình |
| công thức (3.7) | 41 | $N\le T\theta\le T-P=N+Z$ | **có** | mục 2.12 · kiểm 20/20 + thử ngược |
| công thức (3.8) | 41 | $\psi'(\beta^*;w)\ge0$ với mọi $w$ | **một phần** | mục 2.12 — bước chứng minh Định lý 3.4 |
| công thức (3.9) | 41 | dạng cho $w^+$ và $w^-$ | **một phần** | như trên |
| ($\omega$) | 42 | $\omega_{ij}=\theta_i(1-\theta_j)/[f\,f]$ | **có** | mục 2.15 · kiểm Monte-Carlo 1,160629 % |
| công thức (4.1) | 43 | mật độ đồng thời $g_T(\delta_1,\delta_2)$ | **không** | bước đầu của chứng minh Định lý 4.2; kết luận đã kiểm trực tiếp ở mục 2.16 |
| công thức (4.2) | 43 | $\beta^*(\theta,u,X)=\beta^*(\theta,y,X)-\beta$ | **có** | mục 2.8 — chính là Định lý 3.2 (iii) |
| công thức (4.3) tới công thức (4.8) | 44 | các bước trong chứng minh Định lý 4.2 | **không** | sáu bước kỹ thuật của một chứng minh giải tích; kết luận kiểm ở mục 2.16 |
| công thức (4.9) tới công thức (4.12) | 45 | hội tụ của $Z_{Ti}$, ma trận $\Theta$, định thức | **không** | như trên |
| (P) | 48 | bài quy hoạch tuyến tính gốc | **có** | mục 2.21 · dựng lại, khớp `sklearn` từng bit |
| (D) | 48 | bài đối ngẫu | **có** | mục 2.21 |
| (D′) | 49 | bài đối ngẫu đã dịch biến | **có** | mục 2.21 · kiểm quan hệ dấu 0/50 vi phạm |

**Tổng: 22 công thức đánh số + 3 nhãn (P)/(D)/(D′) + 3 công thức không đánh số.** Trong đó
**11 vào bài đầy đủ**, **5 vào một phần** (các bước trung gian của hai chứng minh, với kết
luận đã kiểm trực tiếp), **10 không lấy** — tất cả đều là bước giải tích bên trong chứng minh
Định lý 4.2, và ô "vì sao" ghi rõ từng dòng.

### 9.4 Bốn thứ KHÔNG kiểm được, và vì sao

| Không kiểm được | Vì sao | Hệ quả |
|---|---|---|
| **Mọi đoạn trích nguyên văn** | bài báo là bản quét, 18/18 trang thân bài có **0 ký tự** lớp chữ (mục 0.5) | `kiemtrich.py` báo đúng hạn chế thay vì báo đỏ; mọi đoạn trích chép tay từ ảnh render |
| **Bảng I có đúng là Biểu 5 của Andrews et al. không** | không có bản *Robust Estimates of Location* (Princeton 1972) trên máy | mục 2.4 chỉ so được với **lý thuyết** và với phép đo lại, không so được với nguồn của Bảng I |
| **Ô "10 % trimmed mean / 10 % 10σ"** | định nghĩa chính xác của "10 % trimmed mean" trong nghiên cứu Princeton không có trong bài báo | dựng lại cho $2{,}0006$ thay vì $1{,}46$; mục 2.4 ghi chẩn đoán và **không** kết luận ai sai |
| **Ô "Mean / Cauchy"** | phương sai lý thuyết là **vô cực** | con số $12\,548{,}0$ không tái lập được; ba hạt giống cho độ rộng khoảng gấp $1\,331{,}440333$ lần chính nó |

Và một thứ nữa, khác loại: **bài báo Koenker & Bassett 1978 không có bản truy cập mở**, nên
bản PDF dùng để viết bài **chỉ có trên máy dựng bài** và được khai trong `.gitignore`. Người
đọc repo không mở được nó; mọi đoạn trích trong bài kèm số trang để đối chiếu với bản in
JSTOR.

---

## 10. Tài liệu tham khảo

```
[1] R. Koenker, G. Bassett, "Regression Quantiles", Econometrica 46(1):33-50, 1978.
    → BÀI GỐC, nguồn ruột của toàn bộ mục 2.
    https://doi.org/10.2307/1913643
    Unpaywall: is_oa = false, oa_status = "closed", has_repository_copy = false.
    Bản dùng để viết bài: bản quét JSTOR, đặt tại
    AI algorithms/_papers/1978 - Koenker, Bassett - Regression Quantiles.pdf
    KHÔNG đẩy lên repo (khai trong .gitignore) — xem mục 0.5.

[2] D. F. Andrews et al., "Robust Estimates of Location", Princeton University Press, 1972.
    → Tham khảo [3] của bài báo; nguồn của Bảng I. KHÔNG có trên máy dựng bài,
      nên mục 9.4 ghi rõ chỗ này không kiểm được.

[3] P. J. Huber, "Robust Estimation of a Location Parameter",
    The Annals of Mathematical Statistics 35(1):73-101, 1964.
    → Tham khảo [24] của bài báo. So sánh hai lối bền vững, mục 7.
    AI algorithms/_papers/1964 - Huber - Robust Estimation of a Location Parameter.pdf

[4] N. N. Abdelmalek, "On the Discrete Linear L1 Approximation and L1 Solutions of
    Overdetermined Linear Equations", Journal of Approximation Theory 11:38-53, 1974.
    → Tham khảo [1] của bài báo; chứng minh Định lý 3.1 trỏ thẳng sang đây.

[5] F. Mosteller, "On Some Useful 'Inefficient' Statistics",
    The Annals of Mathematical Statistics 17(4):377-408, 1946.
    → Tham khảo [32] của bài báo; nguồn của Định lý 4.1.

[6] J. L. Gastwirth, "On Robust Procedures",
    Journal of the American Statistical Association 65:946-973, 1966.
    → Tham khảo [14] của bài báo; nguồn của bộ trọng số (.3, .4, .3) ở mục 2.3.
```

**Hình gốc.** 29 khung ở bài này được cắt từ bản in của [1] bằng
[`tools/pdffig.py`](<../../../tools/pdffig.py>) ở 300 DPI, với cờ `--sat` nới khung tới dải
trắng gần nhất để không cắt ngang chữ. Bài báo **không có hình vẽ nào**, nên mọi khung đều là
bảng số, phát biểu định lý, hoặc đoạn thân bài.

**Hình tự vẽ.** Ba hình — Hình 21, Hình 31, Hình 32 — do ba khối mã in ngay trong bài sinh
ra, mỗi khối tự chứa và kết thúc bằng đúng một `savefig()`.

**Số liệu.** Mọi con số trong bài do
[`tools/v_quantile.py`](<../../../tools/v_quantile.py>) in ra. Chạy:
`cd tools && python v_quantile.py`. Script chạy trong khoảng **bốn phút**, không cần dữ liệu
ngoài, và không in ra `UserWarning` nào. Thời gian chạy là một con số đồng hồ, nên bài cố ý
**không** in giá trị chính xác của nó: một con số như thế chỉ tái lập được trong đúng một lần
chạy, và khai nó ra là tạo một phép kiểm xanh giả dựa vào cache.

---

## Kiểm tra trước khi đóng bài

- [x] **Nguồn ruột là BÀI BÁO GỐC, không phải bài tổng quan.** Bản trước của bài này dùng
      Koenker 2000 (*Quantile Regression*, bài tổng quan) làm nguồn hình vì repo không có bài
      1978. Bản này tìm được bản quét JSTOR của chính bài 1978, đưa vào `_papers/` cùng một
      dòng `.gitignore`, và viết lại toàn bộ mục 2 từ nó.
- [x] **Bảng đối chiếu theo HÌNH/BẢNG** — mục 9.1, 2/2 bảng, và ghi rõ bài báo có **0 hình**
- [x] **Bảng đối chiếu theo MỤC** — mục 9.2, 6/6 mục kể cả phụ lục
- [x] **Bảng đối chiếu theo CÔNG THỨC ĐÁNH SỐ** — mục 9.3, đủ 22 công thức cộng 3 nhãn, mỗi
      dòng có ô "vì sao không lấy"
- [x] **Bảng ký hiệu có cột "Đọc là gì"** — mục 0.2, kèm **sáu chỗ trùng tên** trong đó
      **bốn chỗ nằm ngay trong chính bài báo** ($u$ hai nghĩa · $\pi$ vừa là trọng số vừa là
      $3{,}14159$ · $D$ vừa là ma trận vừa là tên bài toán · $\theta$ cạnh $\Theta$)
- [x] **Dựng lại Bảng II VÀ dữ liệu bài báo không in** — mục 2.14: 8/8 ô, 3/3 mốc, năm quan
      sát $(1,3)$ $(2,2)$ $(4,7)$ $(7,8)$ $(9,6)$ đều nguyên, **duy nhất** trong 231 361 cặp
      ứng viên
- [x] **Phép thử ngược cho mọi chi tiết dò ra** — 5041 bộ dữ liệu khớp bốn nghiệm nhưng chỉ
      **1** khớp cả ba mốc; 10 bộ lệch một đơn vị đều **không** khớp
- [x] **Dựng lại Bảng I** — mục 2.4: 21/24 ô lệch dưới $0{,}10$; hai ô lệch nhiều được chẩn
      đoán riêng, và cả hai đều truy về nguồn của Bảng I chứ không về bài báo
- [x] **Tám định lý đều có phát biểu nguyên văn, bản dịch, ý nghĩa và phép kiểm bằng số** —
      3.1 (15/15) · 3.2 (4/4) · 3.3 (12/12 + thử ngược 779/779) · 3.4 (20/20) · 3.5 (lệch 0)
      · 4.1 (1,16 %) · 4.2 (trung bình 1,71 %) · 4.3 (0,0427 %)
- [x] **Phụ lục (P), (D), (D′) vào bài đầy đủ** — mục 2.21, gồm cả quan hệ dấu của biến đối
      ngẫu (0/50 vi phạm, cách đọc ngược 47/50 vi phạm) — bản trước của bài **bỏ hoàn toàn**
      phần đối ngẫu
- [x] **Một chỗ bài báo tự bất đồng, đo được** — trang 37 hứa "Sections 4 through 6" trong
      khi bài báo có 5 mục (mục 2.20)
- [x] **Một chỗ in sai ở Remark trang 41, thử CẢ HAI cách đọc rồi để số chỗ sai quyết định** —
      cách đọc theo chữ sai 20/20 ở vế "below" và 8/20 ở vế "above"; cách suy từ (3.7) đúng
      20/20 ở cả hai vế (mục 2.12)
- [x] **Kết quả âm được giữ lại kèm chẩn đoán** — ô Bảng I không dựng lại được (mục 2.4);
      tỉ số ở mục 2.17 thấp hơn mô hình vị trí, và **đo được** rằng đó là hiệu ứng theo $K$
      chứ không phải mâu thuẫn với Định lý 4.2
- [x] **Mọi phép so đều in ĐỘ RỘNG của nhiễu cạnh phần chênh** — mục 2.17: độ rộng hạt giống
      $0{,}2317138908$ **lớn hơn** độ rộng theo thiết kế $0{,}1728548196$, nên kết luận là
      *không kết luận được*, không phải *thiết kế không ảnh hưởng*
- [x] **Ba hình tự vẽ có khối mã tự chứa, mỗi khối đúng một `savefig()`**
- [x] **Mọi hình có caption, nguồn kèm số trang, và mục "Cách đọc"**
- [x] Script đo chạy trong khoảng **bốn phút**, **0 `UserWarning`**, ghim số luồng BLAS về 1
      trước khi `import numpy`; mọi con số đồng hồ ở mục 6 được khai là một lần chạy
- [ ] **Không kiểm được đoạn trích nguyên văn bằng công cụ.** Bài báo là bản quét: 18/18
      trang thân bài có 0 ký tự lớp chữ. `kiemtrich.py` (đã vá trong đợt này để dò theo **số
      trang rỗng** thay vì mật độ trung bình) báo đúng hạn chế này. Mọi đoạn trích được chép
      tay từ ảnh render ở tỉ lệ 2,2 và **chưa có đường kiểm tự động**. Trả nợ được nếu sau này
      có bản có lớp chữ, hoặc nếu cài được OCR trên máy.
- [ ] **Chưa đối chiếu được Bảng I với nguồn gốc của nó.** Andrews et al. 1972 không có trên
      máy; mục 2.4 chỉ so được với lý thuyết và với phép đo lại. Ô "10 % trimmed mean /
      10 % 10σ" vì thế còn mở.
