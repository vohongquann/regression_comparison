# Elastic Net

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 400
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Hui Zou, Trevor Hastie
Date: April 1, 2005
url: https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf

> **Elastic Net** khớp một đường thẳng qua dữ liệu, tự loại bớt biến, **và không bắt cả một
> nhóm biến gần trùng nhau phải cử một đại diện duy nhất**. Nó cộng hai khoản phạt lên cùng
> một bài toán bình phương tối thiểu: một khoản $\ell_1$ tạo ra hệ số bằng đúng 0, và một
> khoản $\ell_2$ kéo hệ số của các cột tương quan lại gần nhau.
>
> Bài báo gốc ra đời cho dữ liệu vi mảng gen: hàng nghìn cột, vài chục hàng, và các gen cùng
> một đường trao đổi chất tương quan rất mạnh. Ở đó lasso vấp hai chỗ mà bài này đo được bằng
> số: nó **không thể giữ quá $n$ biến** khi số cột nhiều hơn số hàng, và trong một nhóm cột
> gần trùng nhau nó **chia hệ số một cách không xác định**.
>
> Bài đọc được một mình: mục 1 nhắc lại đủ dùng về ridge và lasso, mục 0.3 định nghĩa mọi từ
> mới tại chỗ. Liên kết ra ngoài chỉ để đào sâu.

---

## 0. Bắt đầu từ đây

### 0.1 Nhận gì, trả ra gì

| | Nói bằng lời | Ký hiệu | Ví dụ ở mục 3.11 |
|---|---|---|---|
| **Đầu vào** | một bảng số: $n$ hàng là mẫu, $p$ cột là đặc trưng | $X \in \mathbb{R}^{n\times p}$ | $67 \times 8$ — 67 bệnh nhân, 8 phép đo lâm sàng |
| **Đáp án khi huấn luyện** | một số thực cho mỗi hàng | $y \in \mathbb{R}^{n}$ | 67 giá trị log PSA |
| **Núm vặn 1** | phạt phần $\ell_1$ mạnh tới đâu — núm tạo ra số 0 | $\lambda_1 \ge 0$ | chọn gián tiếp qua $s = 0{,}26$ |
| **Núm vặn 2** | phạt phần $\ell_2$ mạnh tới đâu — núm gom nhóm | $\lambda_2 \ge 0$ | $1000$ |
| **Đầu ra của việc học** | một hệ số cho mỗi cột, một số ô bằng **đúng** 0 | $\hat\beta \in \mathbb{R}^{p}$ | 8 số, 3 số bằng 0 |
| **Dự đoán** | nhân từng hàng dữ liệu với các hệ số rồi cộng | $\hat y = X\hat\beta$ | một số cho mỗi bệnh nhân mới |

Hai đầu mút của bài toán là hai thứ đã có tên riêng, và đó là cách nhớ rẻ nhất:

```
λ₁ = 0  ─────────────────────────────────────────────────────────  λ₂ = 0
 RIDGE                    ELASTIC NET                               LASSO
 không cột nào bị loại    loại được cột, giữ cả nhóm                loại được cột
 giữ đủ p hệ số           có thể giữ hơn n hệ số khi p > n          giữ tối đa n hệ số
 nghiệm duy nhất          nghiệm duy nhất                          có thể không duy nhất
```

**Điều dễ hiểu sai nhất, nói ngay ở dòng đầu:** chữ $\alpha$ trong tài liệu về elastic net
**không có một nghĩa cố định**. Bài báo gốc 2005 đặt $\alpha = 1$ là *ridge*; thư viện
`glmnet` và `scikit-learn` đặt $\alpha = 1$ là *lasso* — ngược hẳn. Quyển *The Elements of
Statistical Learning* dùng **cả hai** quy ước ở hai chương khác nhau. Mục 0.2 có bảng đối
chiếu đủ năm nguồn; đọc nó trước khi chép một con số $\alpha$ từ bất kỳ đâu.

### 0.2 Bảng ký hiệu

| Ký hiệu | Đọc là gì | Nghĩa | Chiều / kiểu |
|---|---|---|---|
| $n$, $p$ | "en", "pê" | số hàng (mẫu) và số cột (đặc trưng) của $X$ | số nguyên dương |
| $X$, $y$ | "ích lớn", "y" | bảng dữ liệu vào; giá trị thật cần dự đoán | $\mathbb{R}^{n\times p}$, $\mathbb{R}^{n}$ |
| $x_j$ | "ích chỉ số j" | **cột** thứ $j$ của $X$ — một đặc trưng đo trên cả $n$ mẫu | $\mathbb{R}^{n}$ |
| $\beta$, $\hat\beta$ | "bê-ta", "bê-ta mũ" | hệ số thật (không quan sát được); hệ số ước lượng được | $\mathbb{R}^{p}$ |
| $\lambda_1$, $\lambda_2$ | "lam-đa một", "lam-đa hai" | hai núm vặn của bài báo gốc: mức phạt $\ell_1$ và mức phạt $\ell_2$ | vô hướng $\ge 0$ |
| $\alpha$ | "an-pha" | tỉ lệ trộn giữa hai khoản phạt — **năm nguồn, hai nghĩa ngược nhau**, xem khối dưới | vô hướng trong $[0,1]$ |
| $s$ | "ét" | phần trăm của độ dài $\ell_1$ so với đầu cuối đường nghiệm: $\lVert\hat\beta\rVert_1/\max\lVert\hat\beta\rVert_1$ | vô hướng trong $[0,1]$ |
| $t$ | "tê" | cận trên của lượng phạt ở dạng ràng buộc, công thức (5) | vô hướng $> 0$ |
| $k$ | "ca" | số bước đã chạy của thuật toán LARS-EN — núm vặn thứ ba, mục 3.10 | số nguyên |
| $\rho$ | "rô" | tương quan mẫu giữa hai cột, $\rho = x_i^{\top}x_j$ sau chuẩn hoá (2) | vô hướng trong $[-1,1]$ |
| $\hat\beta_i$, $\hat\beta_j$ | "bê-ta mũ chỉ số i / j" | hệ số của **hai cột khác nhau** — Định lý 1 nói về khoảng cách giữa chúng | vô hướng |
| $X^{*}$, $y^{*}$ | "ích sao", "y sao" | bộ dữ liệu **nhân tạo** cỡ $(n+p)\times p$ của Bổ đề 1, mục 3.3 | $\mathbb{R}^{(n+p)\times p}$, $\mathbb{R}^{n+p}$ |
| $\hat\beta^{*}$ | "bê-ta mũ sao" | nghiệm lasso trên bộ nhân tạo đó | $\mathbb{R}^{p}$ |
| $\gamma$ | "ga-ma" | mức phạt $\ell_1$ **sau khi đổi sang bài toán nhân tạo**: $\gamma = \lambda_1/\sqrt{1+\lambda_2}$ | vô hướng $\ge 0$ |
| $\hat\Sigma$ | "xích-ma mũ" | ma trận tương quan mẫu $X^{\top}X$ sau chuẩn hoá (2) | $\mathbb{R}^{p\times p}$ |
| $R$, $R^{*}$ | "e-rờ", "e-rờ sao" | toán tử ridge và bản đã **khử tương quan** của nó, công thức (13) | $\mathbb{R}^{p\times n}$ |
| $J(\cdot)$ | "giê" | một hàm phạt bất kỳ, dùng ở công thức (7) để phát biểu Bổ đề 2 | hàm $\mathbb{R}^{p}\to\mathbb{R}_{\ge0}$ |
| $D_{\lambda_1,\lambda_2}(i,j)$ | "đê chỉ số lam-đa" | đại lượng **không thứ nguyên** đo chênh lệch hai hệ số, Định lý 1 | vô hướng $\ge 0$ |
| $\lVert\beta\rVert_1$ | "chuẩn một của bê-ta" | $\sum_j\lvert\beta_j\rvert$ — khoản phạt **tạo ra số 0** | vô hướng |
| $\lvert\beta\rvert^2$ | "bê-ta bình phương" | $\sum_j\beta_j^2$ — cách bài báo gốc viết chuẩn 2 bình phương | vô hướng |
| $(z)_{+}$ | "dét cộng" | phần dương: bằng $z$ nếu $z>0$, bằng 0 nếu không | vô hướng |
| $\operatorname{sgn}(z)$ | "ét-gờ-en của dét" | dấu của $z$: $+1$, $-1$ hoặc 0 | vô hướng |
| $\arg\min_\beta$ | "ác-min theo bê-ta" | **giá trị $\beta$** làm biểu thức nhỏ nhất, không phải giá trị nhỏ nhất | $\mathbb{R}^{p}$ |

> **Chỗ trùng tên đắt nhất của cả GĐ 1: chữ $\alpha$.** Năm nguồn dưới đây đều nói về **cùng
> một hàm phạt**, và chúng chia làm hai phe ngược nhau. Bảng này do mục 17 của
> [`tools/v_enet.py`](<../../../tools/v_enet.py>) in ra.
>
> | Nguồn | Hàm phạt | $\alpha = 1$ nghĩa là |
> |---|---|---|
> | Zou & Hastie 2005, công thức (5) tr.3 | $(1-\alpha)\lVert\beta\rVert_1 + \alpha\lvert\beta\rvert^2$, với $\alpha = \lambda_2/(\lambda_1+\lambda_2)$ | **RIDGE** |
> | ESL 2009, công thức (3.54) tr.73 | $\lambda\sum_j\big(\alpha\beta_j^2 + (1-\alpha)\lvert\beta_j\rvert\big)$ | **RIDGE** |
> | ESL 2009, công thức (18.20) tr.662 | $\sum_j\big(\alpha\lvert\beta_j\rvert + (1-\alpha)\beta_j^2\big)$ | **LASSO** |
> | glmnet 2010, công thức (2) tr.3 | $(1-\alpha)\tfrac12\lVert\beta\rVert_2^2 + \alpha\lVert\beta\rVert_1$ | **LASSO** |
> | `sklearn.linear_model.ElasticNet` | `l1_ratio`$\cdot\lVert\beta\rVert_1 + \tfrac12(1-$`l1_ratio`$)\lVert\beta\rVert_2^2$ | **LASSO** |
>
> Hai dòng giữa nằm trong **cùng một quyển sách**, cách nhau 589 trang, và Trevor Hastie là
> đồng tác giả của cả năm dòng. Mục 3.15 đo cụ thể: tại $\alpha = 0{,}2$, hai công thức của
> ESL cho hai giá trị phạt khác nhau — $1{,}3200$ và $1{,}0800$ trên cùng một điểm
> $\beta = (0{,}6\,;\,-0{,}8)$ — và chúng chỉ bằng nhau đúng tại $\alpha = 0{,}5$.
>
> **Vì thế bài này KHÔNG dùng chữ $\alpha$ làm núm vặn.** Mọi công thức ở đây viết theo
> $(\lambda_1, \lambda_2)$ như bài báo gốc; chữ $\alpha$ chỉ xuất hiện khi đang trích một
> nguồn cụ thể, và luôn kèm tên nguồn đó.

Ba chỗ trùng tên nhỏ hơn, cũng cần nhớ:

* **$t$** ở bài này là cận trên của phạt trong công thức (5); ở bài [Lasso](<../Lasso Regression/Lasso Regression.md>)
  và ở Efron 2004, $t$ là $\sum_j\lvert\beta_j\rvert$ — hai đại lượng khác nhau.
* **$\lambda$** trong Bảng 1 của bài báo gốc là $\lambda_2$, không phải tổng mức phạt: dòng
  *"Elastic net $\lambda = 1000$, $s = 0{,}26$"* nghĩa là $\lambda_2 = 1000$.
* **$\alpha$** của `scikit-learn` là **tổng** mức phạt (vai trò của $\lambda$), không phải tỉ
  lệ trộn. Tỉ lệ trộn ở đó tên là `l1_ratio`. Nhầm hai tham số này là lỗi phổ biến nhất khi
  dùng elastic net, và mục 3.16 cho phép quy đổi đã kiểm bằng số.

### 0.3 Mười từ cần biết trước khi đọc tiếp

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Nghĩa dùng trong bài này |
|---|---|
| **OLS** (*ordinary least squares*) | bình phương tối thiểu thường: chọn $\beta$ làm $\lVert y - X\beta\rVert^2$ nhỏ nhất, không phạt gì |
| **Phạt** (*penalty*) | một số hạng cộng thêm vào hàm mục tiêu để trừng phạt hệ số lớn; nó kéo nghiệm về gần 0 |
| **Chuẩn 1** ($\ell_1$) | $\lVert\beta\rVert_1 = \sum_j\lvert\beta_j\rvert$. Phạt bằng nó **tạo ra hệ số bằng đúng 0** |
| **Chuẩn 2 bình phương** ($\ell_2^2$) | $\sum_j\beta_j^2$. Phạt bằng nó co mọi hệ số lại nhưng **không** đưa cái nào về đúng 0 |
| **Thưa** (*sparse*) | nghiệm có nhiều hệ số bằng đúng 0 |
| **Giá đỡ** (*support*) | tập chỉ số các cột có hệ số khác 0 — tức "mô hình đã chọn những biến nào" |
| **Lồi ngặt** (*strictly convex*) | hàm cong hẳn lên, không có đoạn thẳng nào trên đồ thị. Hệ quả dùng được: cực tiểu **duy nhất** |
| **Hiệu ứng nhóm** (*grouping effect*) | tính chất: hai cột càng tương quan thì hai hệ số càng buộc phải gần nhau. Định lý 1 cho một chặn cụ thể |
| **Naive elastic net** | bản cộng thẳng hai khoản phạt. Bài báo gốc cho rằng nó co hệ số hai lần nên đặt tên "ngây thơ" và đề xuất nhân lại — mục 3.6 |
| **Đường nghiệm** (*solution path*) | tập hợp nghiệm khi cho núm vặn chạy từ phạt tối đa tới phạt bằng 0; với lasso và elastic net nó **tuyến tính từng khúc** |

Ba từ xuất hiện muộn hơn, được giải nghĩa ngay tại chỗ dùng: **UST** (mục 3.8), **LARS-EN**
(mục 3.9) và **kiểm định chéo** (mục 3.10).

### 0.4 Một ví dụ chạy tay, đủ nhỏ để bấm máy tính

Lấy trường hợp đơn giản nhất mà bài báo gốc cũng lấy: các cột **trực giao** với nhau và đã
chuẩn hoá. Khi đó nghiệm có công thức đóng — đó chính là công thức (6) ở mục 3.4.

Giả sử một cột cho hệ số OLS $\hat\beta(\mathrm{OLS}) = 2{,}5$, và chọn
$\lambda_1 = 2$, $\lambda_2 = 1$ (đúng bộ tham số mà chú thích Hình 2 của bài báo dùng):

| Bước | Phép tính | Kết quả |
|---|---|---|
| 1. Trừ đi ngưỡng $\lambda_1/2$ | $2{,}5 - 2/2 = 2{,}5 - 1$ | $1{,}5$ |
| 2. Nếu kết quả âm thì đặt bằng 0 | $1{,}5 > 0$ nên giữ nguyên | $1{,}5$ |
| 3. Chia cho $1 + \lambda_2$ | $1{,}5 / (1+1)$ | $\hat\beta = \mathbf{0{,}75}$ |

Cùng cột đó, ba phương pháp cho ba số khác nhau:

| Phương pháp | Phép tính | Kết quả |
|---|---|---|
| OLS | không đụng gì | $2{,}5$ |
| Ridge ($\lambda_2 = 1$) | $2{,}5/(1+1)$ | $1{,}25$ |
| Lasso ($\lambda_1 = 2$) | $2{,}5 - 1$ | $1{,}5$ |
| **Naive elastic net** | $(2{,}5-1)/2$ | $\mathbf{0{,}75}$ |
| **Elastic net** (đã nhân lại $1+\lambda_2$) | $0{,}75 \times 2$ | $\mathbf{1{,}5}$ |

Hai dòng cuối là toàn bộ nội dung mục 3.6 và 3.7, viết bằng số học lớp 6: bản "ngây thơ" co
hệ số **hai lần** — một lần bởi ngưỡng, một lần bởi phép chia — nên bài báo nhân trả lại hệ
số $1+\lambda_2$. Ở thiết kế trực giao này, phép nhân lại đưa elastic net về **đúng bằng
lasso**; ở dữ liệu thật, các cột không trực giao nên hai thứ khác nhau.

Và nếu hệ số OLS chỉ là $0{,}8$: bước 1 cho $0{,}8 - 1 = -0{,}2 < 0$, nên bước 2 đặt nó về
**đúng 0**. Đó là chỗ tính thưa sinh ra, và nó đến từ khoản $\ell_1$ chứ không từ khoản
$\ell_2$ — ridge với cùng dữ liệu cho $0{,}8/2 = 0{,}4$, nhỏ nhưng khác 0.

### 0.5 Nguồn của bài, và những gì việc dựng lại tìm ra

Bài này dựng lại **toàn bộ** phần phương pháp của năm nguồn bằng
[`tools/v_enet.py`](<../../../tools/v_enet.py>) — 20 mục, chạy hết trong **khoảng 25 giây**. Mỗi con số
trong bài đều lấy từ `stdout` của script đó.

| Nguồn | Vai trò | Đã lấy gì |
|---|---|---|
| Zou & Hastie 2005 (JRSS-B 67(2):301–320) | **bài gốc** | toàn bộ: 7 hình, 4 bảng, 21 công thức đánh số, 19 mục |
| ESL 2009, §3.4.3 và §18.4 | cùng tác giả, hai quy ước $\alpha$ | công thức (3.54), (3.91), (18.20), (18.21); Hình 3.13, Hình 18.5 |
| Friedman, Hastie & Tibshirani 2010 (glmnet) | thuật toán đang được dùng nhiều nhất | công thức (1)–(3), chú thích 1 tr.3, Hình 1 |
| Tibshirani 1996 | lasso — ba ví dụ mô phỏng mà bài gốc dùng lại | §7.2, định nghĩa sai số mô hình |
| Efron và cộng sự 2004 | LARS — nền của thuật toán LARS-EN | §3 |

**Mọi con số THỜI GIAN trong bài là một lần chạy trên máy dựng bài** (6 lõi, `torch` đặt một
luồng) và **đổi theo máy**; không kết luận nào của bài đứng trên một con số thời gian.

**Mười ba chỗ việc dựng lại tìm ra.** Không chỗ nào lật đổ kết luận trung tâm của elastic
net; chúng là lỗi in, chỗ đọc dễ nhầm, và vài chỗ bài báo nói mạnh hơn số liệu của chính nó.

| # | Chỗ | Bài báo viết | Dựng lại cho | Ở mục |
|---|---|---|---|---|
| 1 | Bảng 1 tr.11 | 5 dòng sai số kiểm tra trên dữ liệu prostate | khớp **5/5 dòng cả sai số lẫn SE** — nhưng chỉ khi dùng **bản dữ liệu CÓ LỖI** trước 2012 | 3.11 |
| 2 | §3.5 tr.10 | "chọn $\lambda_2$ cho sai số CV nhỏ nhất", lưới tới 100 | thủ tục đó chọn $\lambda_2 = 0{,}01$; Bảng 1 báo cáo $\lambda_2 = 1000$, **ngoài lưới** | 3.10 · 3.11 |
| 3 | Bảng 2 tr.13, ví dụ 4 | "giảm 27 %" | $(46{,}60-34{,}50)/46{,}60 = 25{,}97\ \%$ → **26 %**; ba ví dụ kia làm tròn đúng | 3.12 |
| 4 | Bảng 2 tr.13, ví dụ 1 | naive EN 5,70 so với lasso 3,06 | naive EN với $\lambda_2 = 0$ **chính là** lasso, nên trung vị không thể tệ hơn 86 % | 3.12 |
| 5 | Định lý 1 tr.6 | chia cho $\lVert y\rVert_1$ | chứng minh ở Phụ lục A.2 chia cho $\lVert y\rVert_2$ — phát biểu lỏng hơn **11,47 lần** ở ca đo được | 3.5 |
| 6 | tr.6, trường hợp $p = 2$ | "$\lvert\hat\beta_1 - \hat\beta_2\rvert = \lvert\cos\theta\rvert$" | đo được $20{,}35$ trong khi $\lvert\cos\theta\rvert = 0{,}126$; công thức đúng có thêm thừa số $\sqrt2\lVert y\rVert/\sqrt{1-\rho}$ | 3.5 |
| 7 | tr.5 | "ridge-type direct shrinkage **followed by** lasso-type thresholding" | đọc đúng thứ tự đó với ngưỡng $\lambda_1/2$ thì lệch công thức (6) tới **0,5**; hai bước chỉ giao hoán khi chia ngưỡng cho $1+\lambda_2$ | 3.4 |
| 8 | tr.11 | "elastic net thắng ... cả về độ thưa" | lasso 5 biến, elastic net 5 biến — **hoà** | 3.11 |
| 9 | tr.13 | "elastic net chính xác hơn hẳn lasso ở mọi ví dụ" | đúng so với lasso, nhưng **ridge** thắng elastic net ở ví dụ 2 và 3 theo chính Bảng 2 | 3.12 |
| 10 | tr.12 | "test error (the mean-squared error) on the test data set" | số in khớp **sai số mô hình** của Tibshirani 1996 (3,039 so với 3,06), không khớp sai số dự đoán trên tập kiểm tra (3,610) | 3.12 |
| 11 | tr.15 | "7129 genes" rồi "those **7219** genes" | hai hoán vị chữ số của nhau; 7129 là con số của Golub 1999 | 3.13 |
| 12 | tr.12 | "the highest is 0.76 (between pgg45 and gleason)" | 0,7519 trên cả 97 mẫu; **0,7571 trên 67 mẫu huấn luyện** → con số của bài báo tính trên tập huấn luyện | 3.11 |
| 13 | ESL (3.54) tr.73 và (18.20) tr.662 | hai công thức elastic net trong cùng một quyển | $\alpha$ mang nghĩa **ngược nhau**; chúng chỉ trùng giá trị tại $\alpha = 0{,}5$ | 3.15 |

Bốn chỗ **không** dựng lại được, ghi rõ để không ai đi tìm: giá trị $s = 0{,}39$ mà Bảng 1
gán cho lasso (kiểm định chéo cho trung vị 0,670); ba trong bốn cột của Bảng 2 lệch khỏi số
in dù thứ tự các phương pháp giữ nguyên; toàn bộ §6 về dữ liệu bạch cầu, vì bộ Golub 1999
không có trong repo; và Hình 6 – Hình 7 đi kèm nó.

---

## 1. Nhắc lại đủ dùng về ridge và lasso

Elastic net là phép cộng hai thứ có sẵn, nên phải biết hai thứ đó đã. Mục này nhắc lại đủ để
đọc hết bài mà không cần mở bài nào khác.

Cả ba phương pháp đều bắt đầu từ cùng một đại lượng: **tổng bình phương phần dư**

$$\mathrm{RSS}(\beta) = \lVert y - X\beta\rVert^2 = \sum_{i=1}^{n}\Big(y_i - \sum_{j=1}^{p} x_{ij}\beta_j\Big)^2 .$$

Chọn $\beta$ làm RSS nhỏ nhất là OLS. Khi $p$ lớn hoặc các cột tương quan mạnh, OLS cho hệ số
rất lớn và rất nhạy: đổi vài mẫu thì hệ số nhảy hẳn. Hai cách chữa cổ điển đều là **cộng thêm
một khoản phạt**:

| | Ridge (Hoerl & Kennard 1970) | Lasso (Tibshirani 1996) |
|---|---|---|
| Hàm mục tiêu | $\mathrm{RSS} + \lambda_2\sum_j\beta_j^2$ | $\mathrm{RSS} + \lambda_1\sum_j\lvert\beta_j\rvert$ |
| Nghiệm đóng | **có**: $\hat\beta = (X^{\top}X+\lambda_2 I)^{-1}X^{\top}y$ | không, trừ khi các cột trực giao |
| Có hệ số bằng đúng 0 không | **không** | **có** |
| Số hệ số khác 0 tối đa khi $p>n$ | $p$ | $n$ |
| Nhóm cột gần trùng nhau | chia hệ số gần đều nhau | chia không xác định |

Chỗ khác nhau nằm ở **hình dạng của khoản phạt tại điểm 0**. Hàm $\beta^2$ trơn tại 0: đạo hàm
của nó bằng 0 ở đó, nên nó không đủ sức giữ một hệ số đứng yên tại 0. Hàm $\lvert\beta\rvert$
thì **gãy** tại 0: đạo hàm nhảy từ $-1$ sang $+1$, và cú nhảy đó tạo ra một khoảng lực đủ để
ghim hệ số ở đúng 0. Mục 3.3 đo chính xác khoảng đó — nó rộng đúng $\lambda_1$.

Cả hai đều cần dữ liệu được đưa về cùng thang trước khi phạt, vì cả hai đều phạt theo **độ lớn
số học** của hệ số. Bài báo gốc viết hẳn quy ước đó thành công thức (2) — mục 3.2.

---

## 2. Hai chỗ lasso tắc, đo bằng số

![Ba kịch bản mà bài báo nêu ở phần mở đầu](<paper_enet_zh2005_tr301_ba_kich_ban.png>)

**Hình 1 — Zou & Hastie (2005), trang 302 của tạp chí (trang 2 của PDF).** Cắt từ
`AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf`.

**Tác giả nói gì.** Nguyên văn trang 302: *"Although the lasso has shown success in many
situations, it has some limitations. Consider the following three scenarios. (a) In the p>n
case, the lasso selects at most n variables before it saturates, because of the nature of the
convex optimization problem. This seems to be a limiting feature for a variable selection
method. (b) If there is a group of variables among which the pairwise correlations are very
high, then the lasso tends to select only one variable from the group and does not care which
one is selected. (c) For usual n>p situations, if there are high correlations between
predictors, it has been empirically observed that the prediction performance of the lasso is
dominated by ridge regression."* — Dịch: *"Tuy lasso đã thành công trong nhiều tình huống, nó
có một số hạn chế. Xét ba kịch bản sau. (a) Khi $p>n$, lasso chọn tối đa $n$ biến rồi bão hoà,
do bản chất của bài toán tối ưu lồi. Đây có vẻ là một đặc điểm gây giới hạn cho một phương pháp
chọn biến. (b) Nếu có một nhóm biến mà tương quan từng cặp giữa chúng rất cao, lasso có xu
hướng chỉ chọn một biến trong nhóm và không quan tâm chọn biến nào. (c) Với tình huống $n>p$
thường gặp, nếu các biến dự báo tương quan cao, người ta đã quan sát bằng thực nghiệm rằng khả
năng dự đoán của lasso bị hồi quy ridge lấn át."*

**Cách đọc.** Ba kịch bản này là toàn bộ lý do bài báo tồn tại, và chúng **không cùng loại**:
(a) là một định lý về bài toán tối ưu, (b) là một quan sát về tính không duy nhất của nghiệm,
còn (c) là một kết quả **thực nghiệm mượn từ Tibshirani 1996** — bài báo không tự đo lại nó.
Mục 3.12 sẽ cho thấy chính bảng mô phỏng của bài báo xác nhận (c) mạnh tới mức ridge thắng cả
elastic net ở hai trong bốn ví dụ.

**Điều đoạn này KHÔNG nói ra.** Nó không nói *vì sao* $n$ là trần của (a); phần giải thích nằm
ở Bổ đề 1 trang 4, và ở đó câu trả lời hoá ra rất gọn: bài toán lasso trên $n$ hàng không thể
có quá $n$ ràng buộc hoạt động. Mục 3.3 đo trần đó.

### 2.1 Kịch bản (a): trần cứng $n$ biến

Dựng một bài toán đúng vùng $p \gg n$ — $n = 30$ hàng, $p = 200$ cột, 8 cột thật sự có hệ số
khác 0 — rồi chạy đường nghiệm tới tận cùng:

| Phương pháp | Số hàng bài toán thật sự giải | Số hệ số khác 0 **lớn nhất** đạt được |
|---|---:|---:|
| Lasso | 30 | **29** |
| Elastic net, $\lambda_2 = 0{,}01$ | 230 | **199** |
| Elastic net, $\lambda_2 = 1$ | 230 | **200** |

Con số 29 đáng chú ý hơn con số 30 mà văn bản bài báo nêu: trần đo được là $n - 1$, không phải
$n$, vì $y$ đã được trừ trung bình nên mất một bậc tự do. Với $\lambda_2 = 1$, elastic net đi
tới đủ **200 = p** biến.

Lý do nằm ở hạng của ma trận: $\operatorname{rank}(X) = 29$ còn
$\operatorname{rank}(X^{*}) = 200$, với $X^{*}$ là bộ dữ liệu nhân tạo của Bổ đề 1 (mục 3.3).

![Số biến chọn được theo từng bước của LARS và LARS-EN](<enet_tran_n.png>)

**Hình 2 — Trần $n$ biến của lasso, và cách elastic net vượt qua nó (hình tự vẽ).** Trục ngang
là số bước đã chạy của thuật toán đường nghiệm, trục dọc là số hệ số khác 0.

**Cách đọc hình.** Đường đỏ (lasso) leo đều rồi **dừng hẳn** ở 29, dưới vạch đứt $n = 30$: từ
bước đó trở đi, có thêm bước cũng không thêm được biến nào. Hai đường elastic net đi xuyên qua
vạch đó và chạm trần $p = 200$. Đường $\lambda_2 = 0{,}01$ (xanh lá) cần nhiều bước hơn đường
$\lambda_2 = 1$ (xanh dương) vì phạt $\ell_2$ yếu hơn thì bài toán gần lasso hơn.

**Điều hình KHÔNG nói ra.** Vượt trần không có nghĩa là chọn **đúng**: ở đây chỉ 8 trong 200
cột có hệ số thật khác 0, nên một mô hình 200 biến là mô hình đã nhận hết cả nhiễu. Giá trị của
việc vượt trần chỉ hiện ra khi số biến thật sự cần lớn hơn $n$ — đúng tình huống dữ liệu gen mà
bài báo nhắm tới.

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

tran_lasso = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 26, 26, 26, 25, 26, 27, 28, 28, 28, 28, 28, 29])
tran_en_0_01 = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 33, 34, 35, 36, 37, 38, 39, 40, 40, 40, 41, 42, 43, 44, 45, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199])
tran_en_1_0 = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200])
tran_n, tran_p = 30, 200

fig, ax = plt.subplots(figsize=(7.8, 4.4))
ax.plot(np.arange(len(tran_lasso)), tran_lasso, color="#dc2626", lw=1.8, label="lasso")
ax.plot(np.arange(len(tran_en_0_01)), tran_en_0_01, color="#059669", lw=1.8,
        label=r"elastic net, $\lambda_2 = 0{,}01$")
ax.plot(np.arange(len(tran_en_1_0)), tran_en_1_0, color="#2563eb", lw=1.8,
        label=r"elastic net, $\lambda_2 = 1$")
ax.axhline(tran_n, color="#111827", ls="--", lw=1.0)
ax.text(2, tran_n + 3, "n = %d — trần của lasso" % tran_n, fontsize=8)
ax.axhline(tran_p, color="#6b7280", ls=":", lw=1.0)
ax.text(2, tran_p - 12, "p = %d" % tran_p, fontsize=8, color="#6b7280")
ax.set_xlabel("bước của LARS / LARS-EN")
ax.set_ylabel("số hệ số khác 0")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Kịch bản (a) của bài báo: lasso bão hoà ở %d = n − 1 biến,\n"
             "còn elastic net đi tới %d = p" % (int(tran_lasso.max()), int(tran_en_1_0.max())), fontsize=9)
fig.tight_layout()
fig.savefig("enet_tran_n.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

### 2.2 Kịch bản (b): hai cột trùng khít thì lasso chia hệ số **không xác định**

Đây là chỗ mạnh nhất, và nó chứng minh được chứ không chỉ quan sát được. Lấy hai cột **trùng
khít nhau** ($x_1 = x_2$, sai khác $0$) rồi giải cả ba bài toán:

| Phương pháp | $\hat\beta_1$ | $\hat\beta_2$ | $\lvert\hat\beta_1 - \hat\beta_2\rvert$ |
|---|---:|---:|---:|
| Elastic net ($\lambda_1 = 0{,}4$, $\lambda_2 = 1$) | 3,705786 | 3,705786 | $2{,}7\times10^{-15}$ |
| Ridge ($\lambda_2 = 1$) | 3,777738 | 3,777738 | $3{,}6\times10^{-15}$ |
| **Lasso** ($\lambda_1 = 0{,}4$) | **11,654039** | **0,000000** | **11,65** |

Và điều đáng nói không phải là lasso chọn cột 1: nó **không chọn gì cả**. Mọi cách chia tổng
$11{,}654039$ cho hai cột đều là nghiệm, với giá trị hàm mục tiêu **giống nhau tới chữ số thứ
mười**:

| Chia theo tỉ lệ $s$ | $\hat\beta_1$ | $\hat\beta_2$ | Giá trị hàm mục tiêu |
|---|---:|---:|---:|
| $s = 0{,}00$ | 0,000000 | 11,654039 | 116,7424224940 |
| $s = 0{,}25$ | 2,913510 | 8,740530 | 116,7424224940 |
| $s = 0{,}50$ | 5,827020 | 5,827020 | 116,7424224940 |
| $s = 0{,}75$ | 8,740530 | 2,913510 | 116,7424224940 |
| $s = 1{,}00$ | 11,654039 | 0,000000 | 116,7424224940 |

Năm dòng, một giá trị. Bài toán lasso ở đây có **vô số nghiệm**, và con số cụ thể mà một thư
viện trả về chỉ là dấu vết của thuật toán bên trong nó, không phải một phát biểu về dữ liệu.
Đó chính là Bổ đề 2 của bài báo — mục 3.5 trích nguyên văn và dán cả chứng minh.

Elastic net không có chỗ để tuỳ tiện: phạt của nó **lồi ngặt**, nên nghiệm là duy nhất, và
với hai cột trùng khít thì nghiệm duy nhất đó buộc phải chia đều.

---

## 3. Phần ruột — đọc theo bài báo gốc, từng công thức một

Mục này đi hết Zou & Hastie 2005 theo đúng thứ tự của bài báo: **7 hình, 4 bảng, 21 công thức
đánh số, 19 mục**. Mỗi hình và bảng có đủ bốn phần — khung đã cắt, chú thích gốc, nhận xét
nguyên văn của chính tác giả, và phần đọc hình của bài này. Mỗi công thức có bảng ký hiệu và
một đoạn nói **ý nghĩa** của nó.

### 3.1 §2.1 — công thức (2): dữ liệu phải được đưa về cùng thang

![Mục 2.1 và công thức (2) của bài báo](<paper_enet_zh2005_tr302_2_1_dinh_nghia_ct2.png>)

**Hình 3 — Zou & Hastie (2005), §2.1 và công thức (2), trang 303 của tạp chí.** Cắt từ
`AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf`.

**Khung chứa gì.** Mở đầu §2.1: ký hiệu $y$, $X$, $x_j$, rồi quy ước chuẩn hoá đánh số (2).

**Tác giả nói gì.** Nguyên văn trang 303: *"After a location and scale transformation, we can
assume that the response is centred and the predictors are standardized"* — *"Sau một phép dời
và phép đổi thang, ta có thể giả định rằng biến phản hồi đã được đưa về trung bình 0 và các
biến dự báo đã được chuẩn hoá."*

$$\sum_{i=1}^{n} y_i = 0, \qquad \sum_{i=1}^{n} x_{ij} = 0 \quad\text{và}\quad \sum_{i=1}^{n} x_{ij}^{2} = 1, \quad j = 1,\dots,p. \tag{2}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $y_i$ | giá trị cần dự đoán của mẫu thứ $i$ | vô hướng |
| $x_{ij}$ | ô hàng $i$ cột $j$ của bảng dữ liệu | vô hướng |
| $n$, $p$ | số mẫu, số cột | số nguyên |

**Ý nghĩa.** Hai điều kiện đầu bỏ hằng số chặn ra khỏi bài toán: khi cả $y$ và mọi cột đều có
trung bình 0, hệ số chặn tối ưu bằng 0, nên không cần viết nó ra và cũng không bao giờ phạt
nhầm vào nó. Điều kiện thứ ba là điều kiện **đắt** nhất về hệ quả: nó nói mọi cột phải có cùng
độ dài, vì khoản phạt trừng phạt theo **giá trị số** của hệ số. Một cột đo bằng mét và một cột
đo bằng milimét, cùng một đại lượng vật lý, sẽ nhận hai mức phạt chênh nhau một nghìn lần nếu
bỏ qua bước này — nghĩa là chính đơn vị đo quyết định biến nào bị loại.

Chú ý một chỗ dễ đọc nhầm: $\sum_i x_{ij}^2 = 1$ **không** phải "độ lệch chuẩn bằng 1". Nó là
**độ dài Euclid của cột bằng 1**, tức nhỏ hơn quy ước độ lệch chuẩn đúng $\sqrt{n}$ lần. Hệ quả
thực hành: một giá trị $\lambda_2 = 1$ theo quy ước này tương ứng với một mức phạt **rất mạnh**,
vì đường chéo của $X^{\top}X$ cũng bằng 1. Bảng 1 của bài báo dùng $\lambda_2 = 1000$, và con số
đó chỉ đọc được khi biết thang này.

Kiểm bằng số trên dữ liệu dựng riêng: $\max_j\lvert\sum_i x_{ij}\rvert = 5{,}00\times10^{-16}$,
$\max_j\lvert\sum_i x_{ij}^2 - 1\rvert = 3{,}33\times10^{-16}$,
$\lvert\sum_i y_i\rvert = 3{,}55\times10^{-15}$.

### 3.2 §2.1 — công thức (3), (4), (5): định nghĩa, và chữ $\alpha$ của bài báo

![Công thức (3), (4), (5) của bài báo](<paper_enet_zh2005_tr302_ct3_ct4_ct5.png>)

**Hình 4 — Zou & Hastie (2005), công thức (3)–(5), trang 303.** Cắt từ cùng PDF.

**Khung chứa gì.** Tiêu chí naive elastic net (3), định nghĩa ước lượng (4), và dạng ràng buộc
tương đương (5) cùng phép đổi biến $\alpha = \lambda_2/(\lambda_1+\lambda_2)$.

**Tác giả nói gì.** Nguyên văn trang 303: *"For any fixed non-negative"* $\lambda_1$ và
$\lambda_2$, *"we define the naive elastic net criterion"*.

$$L(\lambda_1,\lambda_2,\beta) \;=\; \lvert y - X\beta\rvert^{2} \;+\; \lambda_2\lvert\beta\rvert^{2} \;+\; \lambda_1\lvert\beta\rvert_1 \tag{3}$$

$$\hat\beta \;=\; \arg\min_{\beta}\,\{L(\lambda_1,\lambda_2,\beta)\} \tag{4}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\lvert y - X\beta\rvert^2$ | tổng bình phương phần dư — phần "khớp dữ liệu" | vô hướng |
| $\lambda_2\lvert\beta\rvert^2$ | khoản phạt ridge, $\lvert\beta\rvert^2 = \sum_j\beta_j^2$ | vô hướng |
| $\lambda_1\lvert\beta\rvert_1$ | khoản phạt lasso, $\lvert\beta\rvert_1 = \sum_j\lvert\beta_j\rvert$ | vô hướng |

**Ý nghĩa của (3).** Ba số hạng này đại diện cho ba đòi hỏi mâu thuẫn nhau, và hai núm vặn là
hai cái cân giữa chúng: số hạng đầu muốn khớp dữ liệu càng sát càng tốt, kể cả khớp nhiễu; số
hạng $\ell_1$ muốn mô hình ít biến; số hạng $\ell_2$ muốn các hệ số **đừng chênh lệch nhau** khi
các cột giống nhau. Điểm đáng nhớ: hai khoản phạt **không** cùng loại công việc — cái thứ nhất
cắt biến, cái thứ hai chia đều. Bỏ khoản $\ell_1$ thì không cắt được biến nào; bỏ khoản $\ell_2$
thì quay lại đúng hai chỗ hỏng ở mục 2.

**Ý nghĩa của (4).** Ký hiệu $\arg\min$ trả về **giá trị $\beta$** đạt cực tiểu, không phải giá
trị cực tiểu. Với $\lambda_2 > 0$, cực tiểu đó là duy nhất — đây là điều lasso không có, và mục
2.2 đã cho thấy hậu quả của việc không có.

$$\hat\beta = \arg\min_{\beta}\lvert y - X\beta\rvert^{2}, \quad\text{với ràng buộc}\quad (1-\alpha)\lvert\beta\rvert_1 + \alpha\lvert\beta\rvert^{2} \le t \tag{5}$$

**Ý nghĩa của (5).** Đây là **cùng một bài toán** viết theo cách khác: thay vì cộng một khoản
phạt vào hàm mục tiêu, ta giữ nguyên hàm mục tiêu và vẽ một hàng rào quanh gốc toạ độ. Hàng rào
đó là hình gì thì mục 3.3 vẽ ra. Hai cách viết tương đương nhau: chạy phép chia đôi trên hệ số
tỉ lệ của $(\lambda_1,\lambda_2)$ để hai bên có cùng giá trị $t$, kết quả ra hệ số tỉ lệ
$c = 1{,}000000$ và hai nghiệm lệch nhau $1{,}8\times10^{-15}$.

**Và đây là chỗ sinh ra toàn bộ nỗi rối về chữ $\alpha$.** Bài báo đặt

$$\alpha \;=\; \frac{\lambda_2}{\lambda_1 + \lambda_2},$$

tức $\alpha$ đo **phần ridge** trong tổng lượng phạt. Nguyên văn trang 303:
*"When α=1, the naive elastic net becomes simple ridge regression."* — *"Khi $\alpha=1$, naive
elastic net trở thành hồi quy ridge thường."* Và ngay sau đó: *"Note that the lasso penalty
(α = 0) is convex but not strictly convex."* — *"Chú ý rằng phạt lasso ($\alpha=0$) là lồi
nhưng không lồi ngặt."*

Kiểm cả hai đầu mút bằng số: đặt $\lambda_1 = 0$ (tức $\alpha = 1$) rồi so với nghiệm đóng của
ridge — lệch $8{,}9\times10^{-16}$; đặt $\lambda_2 = 0$ (tức $\alpha = 0$) rồi so với nghiệm
lasso lấy từ đường LARS — lệch $5{,}3\times10^{-15}$. Quy ước của bài báo gốc **ngược hẳn** với
quy ước của `glmnet` và `scikit-learn`, nơi $\alpha = 1$ là lasso.

![Đoạn bài báo bình luận về α và tính lồi ngặt](<paper_enet_zh2005_tr302_alpha_lasso_ridge.png>)

**Hình 5 — Zou & Hastie (2005), đoạn cuối §2.1, trang 303.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 303: *"We call the function"* $(1-\alpha)\lvert\beta\rvert_1 + \alpha\lvert\beta\rvert^2$ *"the elastic net penalty, which is a convex combination of the lasso and ridge penalty."* — *"Ta gọi hàm ... là phạt elastic net, một tổ hợp lồi của phạt lasso và phạt ridge."* Và: *"In this paper, we consider only α<1."* — *"Trong bài này, chúng tôi chỉ xét $\alpha<1$."*

**Cách đọc.** Câu cuối là một ràng buộc dễ bỏ qua nhưng có hệ quả: bài báo **loại hẳn** trường
hợp ridge thuần khỏi định nghĩa của mình, vì ridge không tạo được hệ số 0 nên không phải một
phương pháp chọn biến. Nhưng Bảng 1 và Bảng 2 vẫn **so sánh** với ridge như một đối thủ — và ở
Bảng 2, ridge thắng elastic net ở hai trong bốn ví dụ (mục 3.12).

**Điều đoạn này KHÔNG nói ra.** Nó không nói rõ rằng từ $\alpha$ sẽ được dùng lại với nghĩa
**ngược lại** trong chính hệ sinh thái của các tác giả năm năm sau — xem mục 3.15 và 3.16.

### 3.3 Hình 1 và Bổ đề 1 — hàng rào có đỉnh nhọn, và mẹo dữ liệu nhân tạo

![Hình 1 của bài báo: ba miền phạt](<paper_enet_zh2005_tr303_hinh1_mien_phat.png>)

**Hình 6 — Hình 1 của Zou & Hastie (2005), trang 304.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 1. Two-dimensional contour plots (level 1) ( - - -, shape of the ridge
penalty; ———, contour of the lasso penalty; , contour of the elastic net penalty with α=0.5):
we see that singularities at the vertices and the edges are strictly convex; the strength of
convexity varies with α"* — *"Hình 1. Đồ thị đường mức hai chiều (mức 1) (- - - dạng của phạt
ridge; ——— đường mức của phạt lasso; đường mức của phạt elastic net với $\alpha = 0{,}5$): ta
thấy các điểm kỳ dị ở các đỉnh, và các cạnh thì lồi ngặt; độ mạnh của tính lồi thay đổi theo
$\alpha$."*

**Cách đọc hình.** Đây là mặt cắt hai chiều của **hàng rào** trong công thức (5): tập các
$\beta = (\beta_1,\beta_2)$ có cùng mức phạt. Cả ba đường đi qua bốn điểm $(\pm1,0)$ và
$(0,\pm1)$ — kiểm được: cả ba hàm phạt đều cho giá trị $1{,}000000$ tại $(1,0)$. Chỗ chúng
khác nhau là phần giữa, đo tại điểm $(0{,}707\,;\,0{,}707)$: ridge $1{,}000000$, elastic net
$1{,}207107$, lasso $1{,}414214$.

Hai đặc điểm của đường elastic net, mỗi đặc điểm chịu trách nhiệm một tính chất:

| Đặc điểm hình học | Đo được | Tính chất nó tạo ra |
|---|---|---|
| **đỉnh nhọn** trên bốn trục | hệ số góc một phía tại $(0,1)$ là $-0{,}333352$, một số **hữu hạn khác 0** | hệ số bằng **đúng** 0 |
| **cạnh cong ra ngoài** | đạo hàm bậc hai của biên nằm trong $[-16{,}88\,;\,-0{,}75]$, luôn âm | **lồi ngặt** ⇒ nghiệm duy nhất |

So với hai đường kia: lasso có hệ số góc $-1{,}000000$ tại đỉnh (cũng nhọn) nhưng đạo hàm bậc
hai bằng $0$ tới $1{,}1\times10^{-8}$ — cạnh **thẳng**, nên không lồi ngặt; ridge có đạo hàm
bậc hai âm (lồi ngặt) nhưng hệ số góc tại $(0,1)$ là $-2{,}5\times10^{-5}$, tức **trơn**, nên
không có đỉnh và không tạo được số 0. Elastic net là đường duy nhất có **cả hai**.

Phần phồng ra: so với đường thẳng của lasso, biên elastic net ở góc phần tư thứ nhất lồi ra
ngoài nhiều nhất $0{,}236068$ tại $\beta_1 = 0{,}6180$; ridge lồi ra $0{,}414214$.

**Điều hình KHÔNG nói ra.** Hình chỉ vẽ **một** giá trị $\alpha = 0{,}5$. Nó không cho thấy
đỉnh nhọn xuất hiện ở mọi $\alpha < 1$ và biến mất đột ngột đúng tại $\alpha = 1$ — đó là một
bước nhảy, không phải một chuyển tiếp trơn. Hình tự vẽ dưới đây bù đúng chỗ đó.

![Miền phạt biến dạng theo α, và hệ số góc tại đỉnh](<enet_mien_phat.png>)

**Hình 7 — Miền phạt theo $\alpha$, và độ nhọn của đỉnh (hình tự vẽ).** Bảng trái vẽ lại
Hình 1 của bài báo cho **năm** giá trị $\alpha$ theo đúng quy ước của bài báo ($\alpha=1$ là
ridge). Bảng phải đo hệ số góc một phía của biên khi tiến dần về đỉnh $(0,1)$.

**Cách đọc hình.** Ở bảng trái, đường đỏ ngoài cùng là lasso ($\alpha = 0$), đường xanh dương
trong cùng là ridge ($\alpha = 1$); ba đường giữa biến dạng liên tục. Ở bảng phải, mỗi đường
cho biết biên dốc bao nhiêu khi cách đỉnh một khoảng $\beta_1$: bốn đường $\alpha < 1$ **hội tụ
về một hằng số khác 0** (lasso về $-1$, $\alpha=0{,}5$ về $-1/3$, …), còn đường $\alpha = 1$
tụt về $0$. Hằng số khác 0 đó chính là "đỉnh nhọn"; giá trị 0 là "biên trơn".

**Điều hình KHÔNG nói ra.** Nó vẽ hàng rào chứ không vẽ nghiệm. Việc nghiệm có rơi vào đỉnh hay
không còn phụ thuộc hướng của các đường đồng mức RSS, tức phụ thuộc dữ liệu; hình chỉ nói rằng
**có chỗ để rơi vào**.

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# Miền phạt mức 1 của (1−α)|β|₁ + α|β|² — quy ước α của [Z] 2005: α = 1 là RIDGE.
g = np.linspace(0.0, 1.0, 20001)
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))

for a, col, nhan in [(0.0, "#dc2626", "α = 0 — lasso"), (0.25, "#ea580c", "α = 0,25"),
                     (0.5, "#059669", "α = 0,5 — Hình 1 của bài báo"),
                     (0.75, "#0891b2", "α = 0,75"), (1.0, "#2563eb", "α = 1 — ridge")]:
    if a == 0.0:
        b2 = 1.0 - g
    else:
        d = (1 - a) ** 2 + 4 * a * (1 - (1 - a) * g - a * g ** 2)
        b2 = (-(1 - a) + np.sqrt(np.maximum(d, 0))) / (2 * a)
    x = np.r_[g, -g[::-1], -g, g[::-1]]
    y = np.r_[b2, b2[::-1], -b2, -b2[::-1]]
    ax[0].plot(x, y, color=col, lw=1.5, label=nhan)
ax[0].set_aspect("equal")
ax[0].legend(fontsize=7.6, loc="upper right")
ax[0].set_xlabel(r"$\beta_1$")
ax[0].set_ylabel(r"$\beta_2$")
ax[0].set_title("Miền phạt mức 1 biến dạng liên tục theo α\n"
                "(quy ước của Zou & Hastie 2005: α = 1 là ridge)", fontsize=9)

# hệ số góc một phía tại đỉnh (0,1) — đỉnh NHỌN khi α < 1, TRƠN khi α = 1
eps = np.logspace(-8, -1, 200)
for a, col in [(0.0, "#dc2626"), (0.25, "#ea580c"), (0.5, "#059669"), (0.75, "#0891b2"), (1.0, "#2563eb")]:
    if a == 0.0:
        d1 = np.full_like(eps, -1.0)
    elif a == 1.0:
        d1 = (np.sqrt(np.maximum(1 - eps ** 2, 0)) - 1) / eps
    else:
        d1 = ((-(1 - a) + np.sqrt((1 - a) ** 2 + 4 * a * (1 - (1 - a) * eps - a * eps ** 2))) / (2 * a) - 1) / eps
    ax[1].semilogx(eps, d1, color=col, lw=1.5)
    ax[1].text(1.3e-8, d1[0] + 0.03, "α = %.2f → %.3f" % (a, d1[0]), fontsize=7.4, color=col)
ax[1].set_xlabel(r"khoảng cách tới đỉnh, $\beta_1$")
ax[1].set_ylabel("hệ số góc một phía của biên")
ax[1].set_title("Đỉnh NHỌN khi α < 1, TRƠN khi α = 1\n"
                "đỉnh nhọn là thứ tạo ra hệ số bằng đúng 0", fontsize=9)
fig.tight_layout()
fig.savefig("enet_mien_phat.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

![Bổ đề 1 và bộ dữ liệu nhân tạo](<paper_enet_zh2005_tr303_bode1_du_lieu_tang_cuong.png>)

**Hình 8 — Bổ đề 1 của Zou & Hastie (2005), trang 304.** Cắt từ cùng PDF.

**Khung chứa gì.** Phát biểu Bổ đề 1: bộ dữ liệu nhân tạo $(y^{*}, X^{*})$, phép đổi biến, và
kết luận rằng bài toán elastic net trở thành một bài toán lasso.

$$X^{*}_{(n+p)\times p} = (1+\lambda_2)^{-1/2}\begin{pmatrix} X \\ \sqrt{\lambda_2}\,I \end{pmatrix}, \qquad y^{*}_{(n+p)} = \begin{pmatrix} y \\ 0 \end{pmatrix}$$

$$\gamma = \frac{\lambda_1}{\sqrt{1+\lambda_2}}, \qquad \beta^{*} = \sqrt{1+\lambda_2}\,\beta, \qquad \hat\beta = \frac{1}{\sqrt{1+\lambda_2}}\,\hat\beta^{*}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $X^{*}$ | bảng dữ liệu gốc, **nối thêm** một khối $\sqrt{\lambda_2}\,I$ ở dưới, rồi chia cả hai cho $\sqrt{1+\lambda_2}$ | $\mathbb{R}^{(n+p)\times p}$ |
| $y^{*}$ | vectơ $y$ nối thêm $p$ số 0 | $\mathbb{R}^{n+p}$ |
| $I$ | ma trận đơn vị cỡ $p\times p$ | $\mathbb{R}^{p\times p}$ |
| $\gamma$ | mức phạt $\ell_1$ của bài toán lasso mới | vô hướng |

**Ý nghĩa.** Đây là mẹo trung tâm của cả bài báo, và nó nói một điều rất gọn: **phạt $\ell_2$
chính là thêm dữ liệu giả**. Mỗi hàng mới $\sqrt{\lambda_2}\,e_j$ đi kèm đáp án $0$ là một mẫu
nhân tạo nói *"cột thứ $j$ nên có hệ số bằng 0"*, và trọng lượng của lời khuyên đó là
$\lambda_2$. Ghép $p$ mẫu giả như vậy vào rồi giải một bài toán lasso thường là xong.

Hai hệ quả, và cả hai đều là lý do bài báo tồn tại:

1. **Bài toán mới có $n+p$ hàng và $X^{*}$ có hạng $p$**, nên trần $n$ biến của kịch bản (a)
   biến mất. Nguyên văn trang 304: *"Note that the sample size in the augmented problem is n+p
   and X* has rank p, which means that the naive elastic net can potentially select all p
   predictors in all situations."* — *"Chú ý rằng cỡ mẫu của bài toán tăng cường là $n+p$ và
   $X^{*}$ có hạng $p$, nghĩa là naive elastic net về nguyên tắc có thể chọn cả $p$ biến dự báo
   trong mọi tình huống."* Kiểm: $\operatorname{rank}(X) = 29$ còn
   $\operatorname{rank}(X^{*}) = 200$ trên bài toán $30\times200$ của mục 2.1.
2. **Mọi thuật toán lasso dùng lại được ngay**, không cần viết lại gì. Đó là nền của LARS-EN ở
   mục 3.9.

Kiểm bằng số: giải bài toán lasso trên $(y^{*},X^{*})$ rồi chia cho $\sqrt{1+\lambda_2}$, so với
nghiệm hạ theo toạ độ của (3) — lệch $2{,}1\times10^{-15}$.

**Điều Bổ đề 1 KHÔNG nói ra.** Nó không nói cái giá: bài toán mới có $n+p$ hàng, và khi
$p = 5000$ thì đó là 5000 hàng giả cho mỗi bài toán. Chính bài báo thừa nhận điều này ở §3.4
trang 309 — *"for pn the augmented data set has p+n 'observations' and p variables, which can
slow the computation considerably"* — rồi dành cả mục đó để nói cách né nó bằng cấu trúc thưa
của $X^{*}$.

### 3.4 §2.2 và Hình 2 — công thức (6): nghiệm chính xác khi các cột trực giao

![Hình 2 của bài báo: nghiệm chính xác trong thiết kế trực giao](<paper_enet_zh2005_tr304_hinh2_thiet_ke_truc_giao.png>)

**Hình 9 — Hình 2 của Zou & Hastie (2005), trang 305.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 2. Exact solutions for the lasso (———), ridge regression ( - - -) and
the naive elastic net ( ) in an orthogonal design (·······, OLS): the shrinkage parameters are
λ1 = 2 and λ2 = 1"* — *"Hình 2. Nghiệm chính xác của lasso (———), hồi quy ridge (- - -) và naive
elastic net ( ) trong thiết kế trực giao (······· là OLS): các tham số co là $\lambda_1 = 2$ và
$\lambda_2 = 1$."*

**Tác giả nói gì về hình này.** Nguyên văn trang 305: *"Fig. 2 shows the operational
characteristics of the three penalization methods in an orthogonal design, where the naive
elastic net can be viewed as a two-stage procedure: a ridge-type direct shrinkage followed by a
lasso-type thresholding."* — *"Hình 2 cho thấy đặc tính vận hành của ba phương pháp phạt trong
một thiết kế trực giao, trong đó naive elastic net có thể được xem như một thủ tục hai giai
đoạn: một phép co trực tiếp kiểu ridge, tiếp theo là một phép lấy ngưỡng kiểu lasso."*

**Cách đọc hình.** Trục ngang là hệ số OLS, trục dọc là hệ số sau khi phạt. Đường chấm là đường
chéo $y = x$ (không phạt gì). Ridge là một đường thẳng **thoải hơn** đường chéo, đi qua gốc —
nó co mọi hệ số theo cùng một tỉ lệ và không bao giờ chạm 0. Lasso là đường chéo **dịch xuống**
một đoạn, có một **đoạn phẳng** quanh gốc: mọi hệ số OLS nhỏ hơn ngưỡng đều bị đưa về đúng 0.
Naive elastic net có cả hai: đoạn phẳng của lasso, cộng độ dốc thoải của ridge.

![Công thức (6) và đoạn bình luận về thiết kế trực giao](<paper_enet_zh2005_tr304_ct6_nghiem_chinh_xac.png>)

**Hình 10 — Zou & Hastie (2005), §2.2 và công thức (6), trang 305.** Cắt từ cùng PDF.

**Khung chứa gì.** Công thức (6) cho nghiệm naive elastic net trong thiết kế trực giao, cùng hai
nghiệm đối chiếu: ridge $\hat\beta(\mathrm{OLS})/(1+\lambda_2)$ và lasso
$(\lvert\hat\beta_i(\mathrm{OLS})\rvert-\lambda_1/2)_{+}\operatorname{sgn}\{\hat\beta_i(\mathrm{OLS})\}$.

$$\hat\beta_i(\text{naive elastic net}) \;=\; \frac{\big(\lvert\hat\beta_i(\mathrm{OLS})\rvert - \lambda_1/2\big)_{+}}{1+\lambda_2}\;\operatorname{sgn}\!\big\{\hat\beta_i(\mathrm{OLS})\big\} \tag{6}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta_i(\mathrm{OLS})$ | hệ số bình phương tối thiểu; khi các cột trực chuẩn thì bằng $x_i^{\top}y$ | vô hướng |
| $(z)_{+}$ | phần dương của $z$ | vô hướng |
| $\lambda_1/2$ | **ngưỡng**: hệ số OLS nào nhỏ hơn nó thì bị đưa về 0 | vô hướng |
| $1+\lambda_2$ | **hệ số co** áp cho phần còn lại | vô hướng |

**Ý nghĩa.** Công thức này là cả bài báo thu vào một dòng, và nó đọc được từ phải sang trái như
một dây chuyền hai máy: máy thứ nhất (ngưỡng $\lambda_1/2$) **quyết định biến nào sống sót**;
máy thứ hai (chia cho $1+\lambda_2$) **quyết định phần sống sót bị thu nhỏ bao nhiêu**. Hai núm
vặn không tranh việc nhau — mỗi núm điều khiển đúng một máy. Đó cũng là lý do bài toán có hai
tham số chứ không gộp được thành một.

Kiểm bằng số, thiết kế trực giao 80 hàng 8 cột: công thức (6) so với hạ theo toạ độ lệch
$2{,}2\times10^{-16}$; đặt $\lambda_1 = 0$ cho ridge $\hat\beta(\mathrm{OLS})/(1+\lambda_2)$,
lệch $4{,}4\times10^{-16}$; đặt $\lambda_2 = 0$ cho ngưỡng mềm của lasso, lệch
$8{,}9\times10^{-16}$. Với $\lambda_1 = 2$ thì ngưỡng là $1{,}0000$, và $5/8$ hệ số bị đưa về 0.

> **Một chỗ câu văn và công thức không khớp thứ tự.** Câu trích ở trên nói *ridge trước, ngưỡng
> sau*. Đọc đúng thứ tự đó — co trước bằng $1/(1+\lambda_2)$ rồi mới lấy ngưỡng $\lambda_1/2$ —
> thì kết quả **lệch công thức (6) tới 0,500000** trên lưới hệ số OLS từ $-4$ tới $4$. Công
> thức (6) lấy ngưỡng **trước** rồi mới chia.
>
> Hai bước có giao hoán được, nhưng chỉ khi ngưỡng ở bước sau được chia theo: dùng ngưỡng
> $\lambda_1/\{2(1+\lambda_2)\}$ thì hai đường **trùng khít**, lệch $0{,}00\times10^{0}$.
>
> Nên câu của bài báo đúng về **ý** — hai loại co ghép nối tiếp — và lỏng về **thứ tự cùng
> ngưỡng**. Điều này không đổi kết luận nào của bài báo, nhưng ai cài đặt lại theo đúng chữ
> trong câu đó sẽ ra một ước lượng khác.

### 3.5 §2.3 — Bổ đề 2 và Định lý 1: hiệu ứng nhóm, phát biểu và chứng minh

![Công thức (7) và Bổ đề 2](<paper_enet_zh2005_tr305_bode2.png>)

**Hình 11 — Zou & Hastie (2005), công thức (7) và Bổ đề 2, trang 306.** Cắt từ cùng PDF.

$$\hat\beta = \arg\min_{\beta}\;\lvert y - X\beta\rvert^{2} + \lambda\,J(\beta) \tag{7}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $J(\cdot)$ | một hàm phạt **bất kỳ**, dương với mọi $\beta \ne 0$ | hàm $\mathbb{R}^p\to\mathbb{R}_{\ge0}$ |
| $\lambda$ | mức phạt chung | vô hướng $>0$ |

**Ý nghĩa của (7).** Bài báo cố ý viết một dạng **tổng quát** ở đây, không gắn với elastic net.
Lý do: Bổ đề 2 sắp phát biểu một điều đúng cho *mọi* phạt lồi ngặt, chứ không riêng phạt của
bài báo. Đó là cách tách "tính chất đến từ đâu" khỏi "phương pháp cụ thể nào" — và câu trả lời
hoá ra là: hiệu ứng nhóm đến từ **tính lồi ngặt**, không từ chi tiết của công thức.

**Tác giả nói gì.** Nguyên văn trang 306: *"Qualitatively speaking, a regression method exhibits
the grouping effect if the regression coefficients of a group of highly correlated variables
tend to be equal (up to a change of sign if negatively correlated)."* — *"Nói một cách định
tính, một phương pháp hồi quy thể hiện hiệu ứng nhóm nếu các hệ số hồi quy của một nhóm biến
tương quan cao có xu hướng bằng nhau (sai khác một phép đổi dấu nếu tương quan âm)."*

Và phát biểu Bổ đề 2, nguyên văn trang 306: *"Assume that xi =xj"*. Phần (a): *"If J(·) is
strictly convex, then"* $\hat\beta_i = \hat\beta_j$ với mọi $\lambda > 0$. Phần (b): nếu
$J(\beta) = \lvert\beta\rvert_1$ thì $\hat\beta_i\hat\beta_j \ge 0$ và **mọi** cách chia tổng
theo một tỉ lệ $s \in [0,1]$ cũng là nghiệm.

**Cách đọc.** Đây chính là bảng năm dòng ở mục 2.2, viết thành định lý: với hai cột trùng khít,
phạt lồi ngặt **buộc** hai hệ số bằng nhau, còn phạt lasso để lại **cả một đoạn thẳng** nghiệm.
Nguyên văn câu kết của bài báo, trang 306: *"Lemma 2 shows a clear distinction between strictly
convex penalty functions and the lasso penalty. Strict convexity guarantees the grouping effect
in the extreme situation with identical predictors. In contrast the lasso does not even have a
unique solution."* — *"Bổ đề 2 cho thấy sự phân biệt rõ ràng giữa các hàm phạt lồi ngặt và phạt
lasso. Tính lồi ngặt bảo đảm hiệu ứng nhóm trong tình huống cực đoan với các biến dự báo trùng
khít. Ngược lại, lasso thậm chí không có nghiệm duy nhất."*

![Định lý 1 và chặn của hiệu ứng nhóm](<paper_enet_zh2005_tr305_dinh_ly1_chan_nhom.png>)

**Hình 12 — Định lý 1 của Zou & Hastie (2005), trang 306.** Cắt từ cùng PDF.

Bổ đề 2 chỉ nói về trường hợp **trùng khít**, tức một trường hợp không bao giờ gặp trong dữ
liệu thật. Định lý 1 là bản dùng được: nó cho một chặn theo tương quan $\rho$.

$$D_{\lambda_1,\lambda_2}(i,j) \;=\; \frac{1}{\lVert y\rVert_1}\,\big\lvert\hat\beta_i(\lambda_1,\lambda_2) - \hat\beta_j(\lambda_1,\lambda_2)\big\rvert \;\le\; \frac{1}{\lambda_2}\sqrt{2(1-\rho)}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $D_{\lambda_1,\lambda_2}(i,j)$ | chênh lệch hai hệ số, chia cho $\lVert y\rVert_1$ để **không phụ thuộc thang đo** | vô hướng $\ge0$ |
| $\rho = x_i^{\top}x_j$ | tương quan mẫu giữa hai cột, sau chuẩn hoá (2) | $[-1,1]$ |
| điều kiện | định lý đòi $\hat\beta_i\hat\beta_j > 0$, tức hai hệ số **cùng dấu và cùng khác 0** | — |

**Ý nghĩa.** Vế phải là tích của hai thứ, và mỗi thứ nói một câu riêng. Thừa số
$\sqrt{2(1-\rho)}$ chính là **khoảng cách giữa hai cột**: khi $\rho \to 1$ hai cột chập lại và
thừa số này về 0, kéo theo hai hệ số phải sát nhau. Thừa số $1/\lambda_2$ nói **cái giá**: muốn
chặn chặt hơn thì phải phạt $\ell_2$ mạnh hơn, mà phạt mạnh hơn thì co hệ số nhiều hơn. Đó là
đánh đổi trung tâm của phương pháp, và nó nằm gọn trong một phép nhân.

Nguyên văn trang 306: *"The unitless quantity"* $D_{\lambda_1,\lambda_2}(i,j)$ *"describes the
difference between the coefficient paths of predictors i and j. If xi and xj are highly
correlated, i.e."* $\rho \doteq 1$ *"then theorem 1 says that the difference between the
coefficient paths of predictor i and predictor j is almost 0."*

Dựng lại trên bốn mức tương quan và hai mức $\lambda_2$:

| $\rho$ đo được | $\lambda_2$ | $D$ theo $\lVert y\rVert_1$ | $D$ theo $\lVert y\rVert_2$ | chặn | chặn lỏng gấp mấy lần |
|---:|---:|---:|---:|---:|---:|
| 0,748706 | 0,5 | $7{,}713\times10^{-4}$ | $8{,}749\times10^{-3}$ | $1{,}418\times10^{0}$ | 1838,3 |
| 0,748706 | 5,0 | $1{,}256\times10^{-4}$ | $1{,}425\times10^{-3}$ | $1{,}418\times10^{-1}$ | 1129,0 |
| 0,962724 | 0,5 | $6{,}228\times10^{-5}$ | $7{,}214\times10^{-4}$ | $5{,}461\times10^{-1}$ | 8768,6 |
| 0,962724 | 5,0 | $3{,}708\times10^{-5}$ | $4{,}295\times10^{-4}$ | $5{,}461\times10^{-2}$ | 1472,8 |
| 0,995583 | 0,5 | $2{,}052\times10^{-5}$ | $2{,}294\times10^{-4}$ | $1{,}880\times10^{-1}$ | 9161,3 |
| 0,995583 | 5,0 | $2{,}330\times10^{-5}$ | $2{,}604\times10^{-4}$ | $1{,}880\times10^{-2}$ | 806,9 |
| 0,999291 | 0,5 | $1{,}683\times10^{-4}$ | $1{,}931\times10^{-3}$ | $7{,}534\times10^{-2}$ | 447,5 |
| 0,999291 | 5,0 | $2{,}515\times10^{-5}$ | $2{,}884\times10^{-4}$ | $7{,}534\times10^{-3}$ | 299,6 |

Tám phép đo, tám lần bất đẳng thức đúng. Nhưng **chặn rất lỏng** — lỏng từ 300 tới 9161 lần —
nên nó dùng để chứng minh hiệu ứng nhóm *tồn tại*, không dùng để chọn $\lambda_2$.

> **Một chỗ phát biểu và chứng minh dùng hai chuẩn khác nhau.** Phát biểu ở trang 306 chia cho
> $\lVert y\rVert_1$ (chuẩn 1). Chứng minh ở Phụ lục A.2, trang 318–319, chia cho
> $\lVert y\rVert$ (chuẩn 2): nó viết *"So"* $\lvert\hat r(\lambda_1,\lambda_2)\rvert\le\lvert y\rvert$
> rồi dùng tỉ số $\lvert\hat r\rvert/\lvert y\rvert \le 1$ để kết thúc.
>
> Bất đẳng thức **vẫn đúng như đã phát biểu**, vì $\lVert y\rVert_1 \ge \lVert y\rVert_2$ với
> mọi vectơ, nên chia cho chuẩn 1 chỉ làm vế trái nhỏ đi. Nhưng nó lỏng hơn bản mà chứng minh
> thực sự cho: ở hàng cuối bảng trên, $\lVert y\rVert_1/\lVert y\rVert_2 = 11{,}4702$, nên bản
> phát biểu lỏng hơn bản chứng minh đúng **11,47 lần**. Bài này in cả hai cột để người đọc tự
> thấy.

![Đoạn bài báo về lasso và trường hợp p = 2](<paper_enet_zh2005_tr305_lasso_khong_co_hieu_ung_nhom.png>)

**Hình 13 — Zou & Hastie (2005), cuối §2.3, trang 306.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 306: *"The lasso does not have the grouping effect."* Và
tiếp: *"For a simpler illustration, let us consider the linear model with p = 2. Tibshirani
(1996) gave the explicit expression for"* $(\hat\beta_1,\hat\beta_2)$ *", from which we easily
obtain that"* $\lvert\hat\beta_1 - \hat\beta_2\rvert = \lvert\cos(\theta)\rvert$ *", where θ is
the angle between y and x1 −x2. It is easy to construct examples such that ρ=corr(x1, x2)→1 but
cos(θ) does not vanish."*

**Dựng lại thì đẳng thức đó không đúng như in.** Trừ hai điều kiện dừng của lasso cho nhau khi
hai hệ số cùng dấu, ta được $(x_1-x_2)^{\top}\hat r = 0$, tức

$$\hat\beta_1 - \hat\beta_2 \;=\; \frac{(x_1-x_2)^{\top}y}{1-\rho} \;=\; \frac{\sqrt{2}\,\lVert y\rVert\cos\theta}{\sqrt{1-\rho}} .$$

Đo trên năm mức tương quan, cột "cùng dấu" là công thức vừa dẫn:

| $\rho$ | $\cos\theta$ | $\lvert\hat\beta_1-\hat\beta_2\rvert$ đo được | $\lvert\cos\theta\rvert$ | $\sqrt2\lVert y\rVert\cos\theta/\sqrt{1-\rho}$ |
|---:|---:|---:|---:|---:|
| 0,770318 | 0,126291 | 20,347942 | 0,126291 | 20,347942 |
| 0,958349 | 0,047982 | 17,897092 | 0,047982 | 17,897092 |
| 0,995904 | 0,012845 | 15,672214 | 0,012845 | 15,672214 |
| 0,999478 | $-0{,}003545$ | 11,240157 | 0,003545 | 11,240157 |
| 0,999941 | $-0{,}000600$ | 5,751260 | 0,000600 | 5,751260 |

Cột 3 và cột 5 khớp từng chữ số ở cả năm dòng; cột 4 — công thức như bài báo in — lệch tới
**161 lần** ở dòng đầu. Vậy đẳng thức của bài báo thiếu thừa số $\sqrt2\lVert y\rVert/\sqrt{1-\rho}$.

**Và kết luận của bài báo không những đứng vững mà còn mạnh hơn.** Dựng đúng thứ câu văn đòi —
giữ $\cos\theta$ gần như không đổi rồi cho $\rho \to 1$ — thì chênh lệch hệ số **phân kỳ**:

| $\rho$ | $\cos\theta$ | $\lvert\hat\beta_1-\hat\beta_2\rvert$ | $\hat\beta_1$ | $\hat\beta_2$ | số điều kiện của $X$ |
|---:|---:|---:|---:|---:|---:|
| 0,9081533 | 0,096539 | 27,3681 | 44,6276 | 17,2596 | $4{,}56$ |
| 0,9904194 | 0,096511 | 82,6412 | 71,6294 | $-11{,}0117$ | $1{,}44\times10^{1}$ |
| 0,9990378 | 0,098296 | 251,6494 | 156,0763 | $-95{,}5731$ | $4{,}56\times10^{1}$ |
| 0,9999037 | 0,098134 | 650,5374 | 355,4580 | $-295{,}0794$ | $1{,}44\times10^{2}$ |
| 0,9999904 | 0,098600 | 653,9886 | 357,2316 | $-296{,}7569$ | $4{,}56\times10^{2}$ |

Từ dòng thứ hai trở đi hai hệ số **trái dấu**, nên công thức cùng dấu không còn áp dụng; công
thức tổng quát (trừ hai điều kiện dừng với hai dấu khác nhau) khớp đúng cả năm dòng. Số điều
kiện lớn nhất là 456, nên đây không phải hiện tượng của sai số máy tính.

![Chứng minh Bổ đề 2 và Định lý 1](<paper_enet_zh2005_tr317_chung_minh_bode2_dinh_ly1.png>)

**Hình 14 — Phụ lục A.1 và đầu A.2 của Zou & Hastie (2005), trang 318.** Cắt từ cùng PDF.

![Phần cuối chứng minh Định lý 1](<paper_enet_zh2005_tr318_chung_minh_dinh_ly1_tiep.png>)

**Hình 15 — Phụ lục A.2 và A.3 của Zou & Hastie (2005), trang 319.** Cắt từ cùng PDF.

**Khung chứa gì.** Chứng minh Bổ đề 2 (hai phần), chứng minh Định lý 1 qua các công thức
(17)–(20), và chứng minh Định lý 2 qua công thức (21).

**Cách đọc.** Chứng minh Định lý 1 chỉ có bốn bước, và bước nào cũng kiểm được: viết điều kiện
dừng (17) cho mọi hệ số khác 0; viết nó ra cho $i$ và $j$ thành (18) và (19); trừ hai phương
trình đó cho nhau ra (20), tức
$\hat\beta_i - \hat\beta_j = (1/\lambda_2)(x_i - x_j)^{\top}\hat r$; rồi chặn
$\lvert\hat r\rvert \le \lvert y\rvert$ bằng cách so giá trị hàm mục tiêu tại nghiệm với giá trị
tại $\beta = 0$. Bước cuối là chỗ chuẩn 2 xuất hiện, và là chỗ lệch với phát biểu.

**Điều chứng minh KHÔNG nói ra.** Nó không cho biết chặn có chặt hay không — và bảng tám dòng ở
trên cho thấy nó lỏng ba tới bốn bậc độ lớn. Bước
$\lvert\hat r\rvert \le \lvert y\rvert$ là bước phóng tay nhất: nó thay chuẩn phần dư tại nghiệm
bằng chuẩn của toàn bộ $y$, tức giả định mô hình **không giải thích được gì**.

### 3.6 §2.4 — công thức (8) và (9): góc nhìn Bayes, và vì sao $1<q<2$ bị loại

![Công thức (8) và (9): tiên nghiệm](<paper_enet_zh2005_tr306_ct8_ct9_tien_nghiem.png>)

**Hình 16 — Zou & Hastie (2005), §2.4 và công thức (8)–(9), trang 307.** Cắt từ cùng PDF.

$$p_{\lambda,q}(\beta) \;=\; C(\lambda,q)\exp\big(-\lambda\lvert\beta\rvert_q^{q}\big) \tag{8}$$

$$p_{\lambda,\alpha}(\beta) \;=\; C(\lambda,\alpha)\exp\big[-\lambda\{\alpha\lvert\beta\rvert^{2} + (1-\alpha)\lvert\beta\rvert_1\}\big] \tag{9}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $p_{\lambda,q}(\beta)$ | mật độ tiên nghiệm đặt lên hệ số | hàm $\mathbb{R}^p\to\mathbb{R}_{>0}$ |
| $C(\lambda,q)$ | hằng số chuẩn hoá để tích phân bằng 1 | vô hướng |
| $\lvert\beta\rvert_q^q$ | $\sum_j\lvert\beta_j\rvert^{q}$ — phạt "cầu" (*bridge*) bậc $q$ | vô hướng |

**Ý nghĩa.** Hai công thức này đổi cách nhìn chứ không đổi phép tính: **phạt chính là tiên
nghiệm**. Cực tiểu hoá "RSS cộng phạt" giống hệt cực đại hoá hậu nghiệm với tiên nghiệm
$\exp(-\text{phạt})$. Nhìn thế thì $q = 2$ là tiên nghiệm Gauss — tin rằng hệ số nhỏ và phân bố
đều quanh 0; $q = 1$ là tiên nghiệm Laplace — tin rằng **phần lớn hệ số bằng 0** và vài hệ số thì
lớn. Công thức (9) là một niềm tin ở giữa hai cái đó.

Nguyên văn trang 307: *"Ridge regression (q=2) corresponds to a Gaussian prior and the lasso
(q=1) a Laplacian (or double-exponential) prior. The elastic net penalty corresponds to a new
prior given by"* (9), *"a compromise between the Gaussian and Laplacian priors."*

**Và đây là chỗ bài báo giải thích vì sao không đơn giản dùng $1<q<2$.** Nguyên văn trang 307:
*"Although bridge regression with 1 <q< 2 will have many similarities with the elastic net,
there is a fundamental difference between them. The elastic net produces sparse solutions,
whereas bridge regression does not."* — *"Tuy hồi quy cầu với $1<q<2$ có nhiều điểm giống elastic
net, giữa chúng có một khác biệt căn bản. Elastic net cho nghiệm thưa, còn hồi quy cầu thì
không."*

Dựng lại lời khẳng định đó bằng một bộ giải riêng cho phạt bậc $q$:

| $q$ | Số hệ số **đúng** bằng 0 (trên 8) | $\lvert\beta\rvert$ nhỏ nhất |
|---:|---:|---:|
| 1,0 | **2** | $0$ |
| 1,2 | 0 | $4{,}099\times10^{-4}$ |
| 1,5 | 0 | $2{,}947\times10^{-2}$ |
| 2,0 | 0 | $1{,}015\times10^{-1}$ |

Lý do nằm ở đạo hàm và có thể nói không cần chạy gì: với $q>1$, đạo hàm của $\lambda\lvert t\rvert^q$
tại $t=0$ **bằng 0**, nên điều kiện dừng một chiều $2\lVert x_j\rVert^2 t - 2\lvert z\rvert + \lambda q t^{q-1}$
âm tại $t = 0$ mỗi khi $z \ne 0$ — nghiệm luôn nằm ở phía dương, không bao giờ rơi đúng vào 0.
Chỉ $q = 1$ mới có cú nhảy đạo hàm đủ để ghim hệ số tại 0.

**Điều mục này KHÔNG nói ra.** Nó không nói phạt elastic net có phải là một tiên nghiệm **hợp
lệ** hay không cho suy diễn Bayes đầy đủ — bài báo chỉ dùng nó ở mức *mode hậu nghiệm*, và không
chỗ nào trong bài tính hậu nghiệm thật. Đó là một mục ngắn có tính dẫn nhập, không phải một
đóng góp được đo.

### 3.7 §3.1 — vì sao bản "ngây thơ" chưa dùng được

![Mục 3.1: khuyết điểm của naive elastic net](<paper_enet_zh2005_tr306_3_1_naive_co_hai_lan.png>)

**Hình 17 — Zou & Hastie (2005), §3.1, trang 307.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 307: *"However, empirical evidence (see Sections 4 and 5)
shows that the naive elastic net does not perform satisfactorily unless it is very close to
either ridge regression or the lasso. This is why we call it naive."* — *"Tuy nhiên, bằng chứng
thực nghiệm (xem mục 4 và 5) cho thấy naive elastic net không hoạt động thoả đáng trừ khi nó
rất gần với hồi quy ridge hoặc với lasso. Đó là lý do chúng tôi gọi nó là ngây thơ."*

Và chẩn đoán, nguyên văn cùng trang: *"The naive elastic net estimator is a two-stage procedure:
for each fixed λ2 we first find the ridge regression coefficients, and then we do the lasso-type
shrinkage along the lasso coefficient solution paths. It appears to incur a double amount of
shrinkage. Double shrinkage does not help to reduce the variances much and introduces unnecessary
extra bias, compared with pure lasso or ridge shrinkage."* — *"Ước lượng naive elastic net là một
thủ tục hai giai đoạn: với mỗi $\lambda_2$ cố định, trước hết ta tìm các hệ số hồi quy ridge, rồi
sau đó co kiểu lasso dọc theo đường nghiệm hệ số lasso. Nó có vẻ phải chịu một lượng co gấp đôi.
Co hai lần không giúp giảm phương sai được bao nhiêu mà lại đưa vào chệch thừa không cần thiết,
so với co thuần lasso hoặc thuần ridge."*

**Cách đọc.** Ví dụ chạy tay ở mục 0.4 chính là câu này bằng số: hệ số OLS $2{,}5$ bị ngưỡng cắt
còn $1{,}5$, rồi bị chia đôi còn $0{,}75$. Hai lần co liên tiếp, và bài báo cho rằng lần thứ hai
là lần thừa.

**Điều mục này KHÔNG nói ra, và nó quan trọng.** Câu *"empirical evidence (see Sections 4 and 5)"*
trỏ tới Bảng 1 và Bảng 2. Nhưng ở **Bảng 1**, dòng naive elastic net trùng khít dòng ridge —
$0{,}566$ so với $0{,}566$ — chỉ vì kiểm định chéo chọn $s = 1$, tức **không dùng phần lasso
chút nào**; nó không phải bằng chứng về "co hai lần". Còn ở **Bảng 2**, naive elastic net
**thắng** elastic net ở hai trong bốn ví dụ (2,73 so với 3,16 và 41,0 so với 56,6). Mục 3.12 đo
lại cả hai chỗ đó.

### 3.8 §3.2 — công thức (10)–(15): phép nhân lại, và cách đọc nó thành KHỬ TƯƠNG QUAN

![Mục 3.2 và công thức (10), (11), (12)](<paper_enet_zh2005_tr306_3_2_ct10_ct11_ct12.png>)

**Hình 18 — Zou & Hastie (2005), §3.2 và công thức (10)–(12), trang 307.** Cắt từ cùng PDF.

$$\hat\beta^{*} = \arg\min_{\beta^{*}}\;\lvert y^{*} - X^{*}\beta^{*}\rvert^{2} + \frac{\lambda_1}{\sqrt{1+\lambda_2}}\lvert\beta^{*}\rvert_1 \tag{10}$$

$$\hat\beta(\text{elastic net}) = \sqrt{1+\lambda_2}\;\hat\beta^{*} \tag{11}$$

$$\hat\beta(\text{elastic net}) = (1+\lambda_2)\,\hat\beta(\text{naive elastic net}) \tag{12}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\hat\beta^{*}$ | nghiệm lasso trên bộ dữ liệu nhân tạo của Bổ đề 1 | $\mathbb{R}^p$ |
| $1+\lambda_2$ | hệ số nhân lại — đúng bằng hệ số co ở mẫu số của công thức (6) | vô hướng |

**Ý nghĩa.** Bản sửa chỉ khác bản ngây thơ đúng **một phép nhân**, và phép nhân đó chọn sao cho
nó **huỷ đúng** bước co thứ hai. Điều đáng chú ý là nó **không đổi tập biến được chọn**: nhân
một vectơ với một số dương thì các ô bằng 0 vẫn bằng 0. Nguyên văn trang 307: *"Such a scaling
transformation preserves the variable selection property of the naive elastic net and is the
simplest way to undo shrinkage."* Kiểm bằng số: tập biến khác 0 của cả hai bản đều là
$\{0,1,2,3,4\}$, và $\lVert\hat\beta(\text{EN}) - (1+\lambda_2)\hat\beta(\text{naive})\rVert_\infty = 2{,}8\times10^{-14}$.

Bài báo đưa **ba** lý lẽ cho hệ số $1+\lambda_2$, chứ không chỉ một. Lý lẽ thứ hai, nguyên văn
trang 308: *"Consider the exact solution of the naive elastic net when the predictors are
orthogonal. The lasso is known to be minimax optimal (Donoho et al., 1995) in this case, which
implies that the naive elastic net is not optimal. After scaling by 1+λ2, the elastic net
automatically achieves minimax optimality."*

![Công thức (13): phân rã toán tử ridge](<paper_enet_zh2005_tr307_ct13_khu_tuong_quan.png>)

**Hình 19 — Zou & Hastie (2005), công thức (13) và đoạn "decorrelation", trang 308.** Cắt từ
cùng PDF.

Lý lẽ thứ ba là lý lẽ dạy được nhiều nhất, vì nó nói **phạt $\ell_2$ thật ra làm gì**:

$$R = (X^{\top}X + \lambda_2 I)^{-1}X^{\top} = \frac{1}{1+\lambda_2}R^{*}, \qquad R^{*} = \begin{pmatrix} 1 & \tfrac{\rho_{12}}{1+\lambda_2} & \cdots \\ & 1 & \\ & & 1\end{pmatrix}^{-1}X^{\top} \tag{13}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $R$ | toán tử ridge: nhân nó với $y$ ra ngay hệ số ridge | $\mathbb{R}^{p\times n}$ |
| $R^{*}$ | cùng toán tử, nhưng đã bỏ hệ số $1/(1+\lambda_2)$ ra ngoài | $\mathbb{R}^{p\times n}$ |
| $\rho_{ij}/(1+\lambda_2)$ | tương quan mẫu **đã bị co** — đây là phần "khử tương quan" | vô hướng |

**Ý nghĩa.** Công thức này tách ridge thành hai việc riêng: (i) **làm cho các cột bớt giống
nhau** — mọi tương quan ngoài đường chéo bị nhân với $1/(1+\lambda_2)$; và (ii) **thu nhỏ toàn
bộ nghiệm** theo cùng hệ số đó. Nguyên văn trang 308: *"Hence from equation (13) we can interpret
the ridge operator as decorrelation followed by direct scaling shrinkage."*

Và đây là chỗ lập luận khép lại, nguyên văn trang 308: *"This decomposition suggests that the
grouping effect of ridge regression is caused by the decorrelation step. When we combine the
grouping effect of ridge regression with the lasso, the direct 1/(1+λ2) shrinkage step is not
needed and is removed by rescaling."* — *"Phân rã này gợi ý rằng hiệu ứng nhóm của hồi quy ridge
là do bước khử tương quan gây ra. Khi ta kết hợp hiệu ứng nhóm của ridge với lasso, bước co trực
tiếp $1/(1+\lambda_2)$ là không cần thiết và bị phép nhân lại loại bỏ."*

Nói gọn: **elastic net muốn phần khử tương quan của ridge, không muốn phần thu nhỏ của nó** —
vì việc thu nhỏ đã có khoản $\ell_1$ lo rồi. Kiểm bằng số: $\lVert R - R^{*}/(1+\lambda_2)\rVert_\infty = 2{,}8\times10^{-17}$,
và với $\lambda_2 = 2$ thì tương quan $\rho_{12} = 0{,}944489$ trong $R^{*}$ tụt xuống
$0{,}314830$, đúng bằng một phần ba.

![Công thức (14), (15) và Định lý 2](<paper_enet_zh2005_tr308_ct14_ct15_dinh_ly2.png>)

**Hình 20 — Định lý 2 và công thức (14)–(15) của Zou & Hastie (2005), trang 309.** Cắt từ cùng PDF.

$$\hat\beta = \arg\min_{\beta}\;\beta^{\top}\Big(\frac{X^{\top}X+\lambda_2 I}{1+\lambda_2}\Big)\beta - 2y^{\top}X\beta + \lambda_1\lvert\beta\rvert_1 \tag{14}$$

$$\hat\beta(\text{lasso}) = \arg\min_{\beta}\;\beta^{\top}(X^{\top}X)\beta - 2y^{\top}X\beta + \lambda_1\lvert\beta\rvert_1 \tag{15}$$

**Ý nghĩa.** Đặt hai công thức cạnh nhau thì khác biệt duy nhất nằm ở **ma trận ở giữa**. Lasso
dùng $\hat\Sigma = X^{\top}X$ đúng như dữ liệu cho; elastic net thay nó bằng một bản đã kéo về
phía ma trận đơn vị. Nguyên văn trang 309: *"Hence theorem 2 interprets the elastic net as a
stabilized version of the lasso."* Với $\gamma = \lambda_2/(1+\lambda_2)$,

$$\frac{X^{\top}X + \lambda_2 I}{1+\lambda_2} = (1-\gamma)\hat\Sigma + \gamma I ,$$

và bài báo nối luôn sang một việc quen thuộc khác, nguyên văn trang 309: *"In linear discriminant
analysis, the prediction accuracy can often be improved by replacing"* $\hat\Sigma$ *"by a
shrunken estimate (Friedman, 1989; Hastie et al., 2001). Likewise we improve the lasso by
regularizing"* $\hat\Sigma$ *"in equation (15)."*

Kiểm cả ba: giải (14) bằng hạ theo toạ độ rồi so với công thức (12) — lệch
$2{,}5\times10^{-14}$; giải (15) rồi so với nghiệm lasso từ LARS — lệch $1{,}1\times10^{-13}$;
và $\lVert(X^{\top}X+\lambda_2I)/(1+\lambda_2) - \{(1-\gamma)\hat\Sigma+\gamma I\}\rVert_\infty = 5{,}6\times10^{-17}$
với $\gamma = 0{,}666667$.

**Điều Định lý 2 KHÔNG nói ra.** Nó không nói phải chọn $\gamma$ bao nhiêu. Và nó làm lộ một
điều bài báo không bình luận: khi $\lambda_2 \to \infty$ thì $\gamma \to 1$, ma trận ở giữa
thành $I$, và bài toán **quên sạch** mọi tương quan giữa các cột. Đó chính là mục tiếp theo.

### 3.9 §3.3 — công thức (16): đầu mút $\lambda_2\to\infty$ là ngưỡng mềm một biến

![Mục 3.3 và công thức (16)](<paper_enet_zh2005_tr308_3_3_ct16_ust.png>)

**Hình 21 — Zou & Hastie (2005), §3.3 và công thức (16), trang 309.** Cắt từ cùng PDF.

$$\hat\beta(\infty)_i = \Big(\lvert y^{\top}x_i\rvert - \frac{\lambda_1}{2}\Big)_{+}\operatorname{sgn}(y^{\top}x_i), \qquad i = 1,\dots,p \tag{16}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $y^{\top}x_i$ | hệ số hồi quy của $y$ lên **một mình** cột $i$ (vì $\lVert x_i\rVert = 1$) | vô hướng |
| $\hat\beta(\infty)$ | giới hạn của nghiệm elastic net khi $\lambda_2\to\infty$ | $\mathbb{R}^p$ |

**Ý nghĩa.** Đây là đầu mút đối diện của lasso trên cùng một trục. Lasso là elastic net với
$\lambda_2 = 0$; khi $\lambda_2 \to \infty$ thì mọi tương quan bị xoá hết, nên mỗi cột được xét
**một mình**: lấy hệ số hồi quy đơn biến rồi cắt ngưỡng. Bài báo gọi nó là **UST** (*univariate
soft thresholding* — ngưỡng mềm một biến) và nói thẳng nó bỏ qua điều gì, nguyên văn trang 309:
*"UST totally ignores the dependence between predictors and treats them as independent
variables. Although this may be considered illegitimate, UST and its variants are used in other
methods such as significance analysis of microarrays (Tusher et al., 2001) and the nearest
shrunken centroids classifier (Tibshirani et al., 2002), and have shown good empirical
performance. The elastic net naturally bridges the lasso and UST."*

Đo tốc độ hội tụ về đầu mút đó:

| $\lambda_2$ | $\lVert\hat\beta(\text{EN}) - \text{UST}\rVert_\infty$ | $\lVert\hat\beta(\text{EN})\rVert_1$ |
|---:|---:|---:|
| 1 | $1{,}268\times10^{1}$ | 73,415289 |
| 10 | $3{,}112$ | 92,768331 |
| 100 | $3{,}648\times10^{-1}$ | 98,941850 |
| 1000 | $3{,}712\times10^{-2}$ | 99,712972 |
| $10^{4}$ | $3{,}718\times10^{-3}$ | 99,791596 |
| $10^{6}$ | $3{,}719\times10^{-5}$ | 99,800261 |

Sai lệch giảm đúng theo $1/\lambda_2$: mỗi lần nhân $\lambda_2$ với 10 thì lệch chia 10. Con số
này có hệ quả trực tiếp cho mục 3.11: Bảng 1 của bài báo chọn $\lambda_2 = 1000$, và ở mức đó
nghiệm đã cách UST **dưới 4 %** — bài báo tự nhận ra điều đó và viết ra.

### 3.10 §3.4 và §3.5 — LARS-EN, và ba cách đặt núm vặn thứ hai

![Mục 3.4: thuật toán LARS-EN](<paper_enet_zh2005_tr308_3_4_lars_en.png>)

**Hình 22 — Zou & Hastie (2005), §3.4, trang 309–310.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 309: *"By lemma 1, for each fixed λ2 the elastic net problem
is equivalent to a lasso problem on the augmented data set. So algorithm LARS can be directly
used to create the entire elastic net solution path efficiently with the computational efforts
of a single OLS fit. Note, however, that for pn the augmented data set has p+n 'observations'
and p variables, which can slow the computation considerably."*

Và cách né chi phí đó, nguyên văn trang 310: *"we can save computations by using the simple fact
that"* $X^{*}_j$ *"has p − 1 zero elements."*

**Cách đọc.** Ba ý, mỗi ý kiểm được:

1. **Dùng lại LARS y nguyên.** Không cần thuật toán mới; chỉ cần dựng $(y^{*},X^{*})$ rồi gọi
   LARS. Đường nghiệm vì thế **tuyến tính từng khúc**, giống lasso.
2. **Cột của $X^{*}$ rất thưa.** Đếm thật trên bài toán $30\times200$: mỗi cột của $X^{*}$ có
   $199 = p-1$ phần tử bằng 0 trên $230 = n+p$ hàng. Nên mọi tích vô hướng trong LARS chỉ cần
   chạm phần $X$ gốc cộng đúng một ô.
3. **Ma trận Gram có dạng gọn.** Bài báo dùng
   $G_A = \frac{1}{1+\lambda_2}(X_A^{\top}X_A + \lambda_2 I)$; kiểm trên tập $A$ mười cột đầu:
   lệch $2{,}2\times10^{-16}$.

Số bước chạy thật, cùng bài toán $n=30$, $p=200$:

| Phương pháp | Cỡ bài toán thật sự giải | Số bước | Số biến khác 0 lớn nhất |
|---|---|---:|---:|
| Lasso | $30\times200$ | 41 | 29 |
| LARS-EN, $\lambda_2 = 0{,}01$ | $230\times200$ | 207 | 199 |
| LARS-EN, $\lambda_2 = 0{,}1$ | $230\times200$ | 202 | 200 |
| LARS-EN, $\lambda_2 = 1$ | $230\times200$ | 200 | 200 |
| LARS-EN, $\lambda_2 = 10$ | $230\times200$ | 200 | 200 |

Và chi phí khi dừng sớm, nguyên văn trang 310: *"If we stop the algorithm after m steps, then it
requires O(m3 +pm2) operations."* Với $p = 200$ và $m = 200$, con số đó là
$1{,}600\times10^{7}$ phép.

![Mục 3.5: chọn tham số điều chỉnh](<paper_enet_zh2005_tr309_3_5_chon_tham_so.png>)

**Hình 23 — Zou & Hastie (2005), §3.5, trang 310.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 310, về việc có ba cách đặt núm vặn thứ hai:
*"Although we defined the elastic net by using (λ1, λ2), it is not the only choice as the tuning
parameter. In the lasso, the conventional tuning parameter is the L1-norm of the coefficients
(t) or the fraction of the L1-norm (s). ... The advantage of using (λ2,s) is that s is always
valued within [0, 1]."* Và cách chọn, nguyên văn cùng trang: *"Note that there are two tuning
parameters in the elastic net, so we need to cross-validate on a two-dimensional surface.
Typically we first pick a (relatively small) grid of values for λ2, say (0, 0.01, 0.1, 1, 10,
100). Then, for each λ2, algorithm LARS-EN produces the entire solution path of the elastic net.
The other tuning parameter (λ1, s or k) is selected by tenfold CV. The chosen λ2 is the one
giving the smallest CV error."*

**Cách đọc.** Ba cách tham số hoá cùng một đường nghiệm — $(\lambda_2,\lambda_1)$,
$(\lambda_2,s)$, $(\lambda_2,k)$ — và cách thứ ba, đếm **số bước** của thuật toán, là cách duy
nhất dùng được khi phải dừng sớm, vì $s$ chỉ tính được sau khi đã chạy hết đường.

**Điều mục này KHÔNG nói ra, và mục 3.11 sẽ va vào nó.** Lưới đề nghị dừng ở $100$. Bảng 1 báo
cáo $\lambda = 1000$ — một giá trị **nằm ngoài lưới** — và chạy đúng thủ tục vừa trích thì
$\lambda_2$ được chọn là $0{,}01$, không phải $1000$.

### 3.11 §4 và Bảng 1 — dữ liệu prostate, và bản dữ liệu đã bị sửa sau đó

![Bảng 1 của bài báo: so sánh trên dữ liệu prostate](<paper_enet_zh2005_tr310_bang1_prostate.png>)

**Hình 24 — Bảng 1 của Zou & Hastie (2005), trang 311.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Table 1. Prostate cancer data: comparing different methods"* — *"Bảng 1. Dữ
liệu ung thư tuyến tiền liệt: so sánh các phương pháp khác nhau."*

**Tác giả nói gì về bảng này.** Nguyên văn trang 311: *"Table 1 clearly shows that the elastic
net is the winner among all the competitors in terms of both prediction accuracy and sparsity.
OLS is the worst method. The naive elastic net performs identically to ridge regression in this
example and fails to do variable selection."* — *"Bảng 1 cho thấy rõ elastic net là bên thắng
trong tất cả các đối thủ, xét cả về độ chính xác dự đoán lẫn độ thưa. OLS là phương pháp tệ nhất.
Naive elastic net cho kết quả y hệt hồi quy ridge trong ví dụ này và không chọn được biến."*

**Cách đọc bảng.** Tám biến lâm sàng, 97 bệnh nhân, chia 67 huấn luyện và 30 kiểm tra. Cột thứ
hai ghi tham số đã chọn; ở đây $\lambda$ nghĩa là $\lambda_2$, và $s$ là phần trăm độ dài
$\ell_1$. Cột thứ ba là sai số bình phương trung bình trên 30 mẫu kiểm tra, số trong ngoặc là
sai số chuẩn của nó. Cột cuối liệt kê chỉ số các biến được giữ.

**Dựng lại thì bảng này không khớp dữ liệu prostate hiện hành — nhưng khớp tuyệt đối bản CÓ
LỖI.** Bộ prostate lưu hành đến 2012 có một ô sai: `lweight` của đối tượng 32 bị nhập là
$6{,}1$, tức $\ln 449$, thay vì $\ln 44{,}9 = 3{,}804438$ — lệch đúng $\ln 10 = 2{,}302585$, một
lỗi dấu thập phân. Chú thích trang 3 của ESL kể lại chuyện đó. Chạy cả hai bản:

| Phương pháp | Bài báo in | Bản dữ liệu ĐÃ SỬA | Bản CÓ LỖI | Khớp bản nào |
|---|---:|---:|---:|---|
| OLS | 0,586 (0,184) | 0,5213 (0,1787) | **0,5863 (0,1839)** | bản lỗi |
| Ridge $\lambda = 1$ | 0,566 (0,188) | 0,5311 (0,1882) | **0,5655 (0,1880)** | bản lỗi |
| Lasso $s = 0{,}39$ | 0,499 (0,161) | 0,4723 (0,1618) | **0,4987 (0,1613)** | bản lỗi |
| Naive EN $\lambda=1$, $s=1$ | 0,566 (0,188) | 0,5311 (0,1882) | **0,5655 (0,1880)** | bản lỗi |
| Elastic net $\lambda=1000$, $s=0{,}26$ | 0,381 (0,105) | 0,3754 (0,1052) | **0,3805 (0,1049)** | bản lỗi |

**Năm trên năm dòng, cả sai số lẫn sai số chuẩn**, và cả hai tập biến cũng khớp: lasso giữ
$(1,2,4,5,8)$, elastic net giữ $(1,2,5,6,8)$. Bài báo năm 2005 dùng bản dữ liệu chưa sửa — điều
đó bình thường, vì bản sửa mãi 2012 mới có; nhưng ai dựng lại Bảng 1 hôm nay trên dữ liệu hiện
hành sẽ thấy lệch và không hiểu vì sao.

Kết luận của bài báo **không** đổi khi dùng bản đúng: elastic net vẫn tốt nhất (0,3754 so với
lasso 0,4723), và thứ tự năm phương pháp giữ nguyên.

**Ba chỗ nữa trong cùng trang, kiểm được:**

* **Dòng naive elastic net trùng dòng ridge là chuyện tất yếu, không phải bằng chứng.** Tham số
  của nó là $s = 1$, tức **không dùng phần lasso chút nào**; với $s = 1$ thì (3) rút về đúng
  ridge. Nên câu *"fails to do variable selection"* đúng theo nghĩa hiển nhiên: kiểm định chéo đã
  chọn một điểm mà ở đó phương pháp không còn là elastic net nữa.
* **Câu "thắng cả về độ thưa" không có bên thắng.** Lasso giữ 5 biến, elastic net giữ 5 biến —
  **hoà**. Bài báo thắng về sai số dự đoán, không thắng về độ thưa.
* **Tương quan lớn nhất 0,76.** Trên cả 97 mẫu, $\operatorname{corr}(\text{pgg45},\text{gleason}) = 0{,}7519$;
  trên 67 mẫu huấn luyện, $0{,}7571$ — làm tròn thành 0,76. Vậy con số của bài báo tính trên tập
  huấn luyện, và nó cũng là tương quan lớn nhất ngoài đường chéo trong cả hai trường hợp.

![Đoạn bình luận Bảng 1 và câu về UST](<paper_enet_zh2005_tr311_binh_luan_bang1.png>)

**Hình 25 — Zou & Hastie (2005), đoạn bình luận Bảng 1, trang 312.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 312: *"The prediction error of the elastic net is about 24%
lower than that of the lasso. We also see in this case that the elastic net is actually UST,
because the λ selected is very big (1000)."* Và: *"If we check the correlation matrix of these
eight predictors, we see that there are some medium correlations, although the highest is 0.76
(between pgg45 and gleason). We have seen that the elastic net dominates the lasso by a good
margin. In other words, the lasso is hurt by the high correlation. We conjecture that, whenever
ridge regression improves on OLS, the elastic net will improve the lasso."*

**Kiểm ba con số.** Theo đúng số in trong Bảng 1: $(0{,}499-0{,}381)/0{,}499 = 0{,}2365$, tức
**23,65 %** — làm tròn thành 24 %, đúng. Theo bản dựng lại:
$(0{,}4987-0{,}3805)/0{,}4987 = 23{,}70\ \%$.

Còn câu *"the elastic net is actually UST"*: so nghiệm elastic net ở $\lambda_2 = 1000$, $s = 0{,}26$
với nghiệm UST có cùng độ dài $\ell_1$:

```
β̂(EN λ=1000, s=0,26) = [3.6762  1.2445  0.  0.  1.9458  1.2807  0.  0.8778]
UST cùng ‖β‖₁          = [3.6758  1.2429  0.  0.  1.9462  1.2820  0.  0.8782]
```

Lệch lớn nhất $1{,}582\times10^{-3}$, tương đối $4{,}303\times10^{-4}$. Câu của bài báo đúng, và
đúng tới bốn chữ số.

**Câu cuối là một phỏng đoán, và bài báo gọi đúng tên nó.** *"We conjecture that..."* — bài báo
không chứng minh, chỉ nêu. Mục 3.12 cho thấy Bảng 2 của chính bài báo **không** ủng hộ phỏng đoán
đó ở mọi ví dụ: ở ví dụ 2 và 3, ridge cải thiện rất mạnh so với OLS, nhưng elastic net lại thua
ridge chứ không thắng.

![Hình 3 của bài báo: đường nghiệm trên dữ liệu prostate](<paper_enet_zh2005_tr310_hinh3_duong_prostate.png>)

**Hình 26 — Hình 3 của Zou & Hastie (2005), trang 311.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 3. (a) Lasso estimates as a function of s and (b) elastic net estimates
(λ = 1000) as a function of s: both estimates are piecewise linear, which is a key property of
our efficient algorithm; the solution paths also show that the elastic net is identical to
univariate soft thresholding in this example (····, final model selected)"* — *"Hình 3. (a) Ước
lượng lasso theo $s$ và (b) ước lượng elastic net ($\lambda = 1000$) theo $s$: cả hai đều tuyến
tính từng khúc, một tính chất then chốt của thuật toán hiệu quả của chúng tôi; các đường nghiệm
cũng cho thấy elastic net trùng với ngưỡng mềm một biến trong ví dụ này (···· là mô hình cuối
cùng được chọn)."*

**Cách đọc hình.** Trục ngang là $s$ chạy từ 0 (phạt tối đa, mọi hệ số bằng 0) tới 1 (hết phạt).
Mỗi đường là một biến; tên biến ghi ở mép phải. Vạch chấm dọc đánh dấu mô hình được chọn. Chỗ
một đường **rời khỏi trục 0** là chỗ biến đó được nhận vào mô hình.

**Điều hình KHÔNG nói ra.** Nó không cho biết **thứ tự** các biến vào là gì, và thứ tự đó khác
hẳn giữa hai bảng — dựng lại thì thấy rõ:

| | Thứ tự biến vào mô hình |
|---|---|
| (a) lasso | lcavol → lweight → svi → lbph → pgg45 → age → lcp → gleason |
| (b) elastic net $\lambda = 1000$ | lcavol → svi → lcp → lweight → pgg45 → gleason → lbph → age |

`lcp` vào thứ **ba** ở bảng (b) nhưng thứ **bảy** ở bảng (a). Đó chính là chỗ hai phương pháp
chọn hai mô hình khác nhau ở Bảng 1.

Tính tuyến tính từng khúc mà chú thích tuyên bố thì kiểm được trực tiếp: lấy trung điểm của mỗi
đoạn giữa hai nút rồi so nghiệm thật với đoạn thẳng nối hai đầu — lệch lớn nhất
$8{,}88\times10^{-16}$ cho lasso và $8{,}67\times10^{-19}$ cho elastic net.

![Dựng lại Hình 3 trên bản dữ liệu có lỗi](<enet_prostate_duong.png>)

**Hình 27 — Dựng lại Hình 3 của bài báo (hình tự vẽ).** Cùng 67 mẫu huấn luyện, cùng
$\lambda_2 = 1000$, trên **bản dữ liệu có lỗi** để so được với bản in.

**Cách đọc hình.** Giống Hình 26, nhưng mỗi nút của đường nghiệm được đánh dấu bằng một chấm, nên
đếm được: cả hai đường đều có **8 bước**. Nhãn biến nối bằng một đoạn mảnh tới đầu mút tương ứng,
để không đè lên nhau.

**Điều hình tự vẽ này bù cho hình gốc.** Hình gốc in ở khổ nhỏ và không ghi giá trị các nút; bản
dựng lại cho biết chính xác $s$ tại từng nút — với lasso là
$0{,}189$; $0{,}254$; $0{,}372$; $0{,}376$; $0{,}557$; $0{,}599$; $0{,}943$ — và các giá trị đó là
thứ cần để nói mô hình ở $s = 0{,}39$ nằm giữa hai nút nào.

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

h3_lasso_s = np.array([0. , 0.189317, 0.254388, 0.37237 , 0.376468, 0.557045, 0.599204, 0.943302, 1. ])
h3_lasso_b = np.array([[ 0. , 3.4767, 4.0742, 4.5577, 4.5672, 4.7094, 4.7642, 5.6818, 5.8201], [ 0. , 0. , 0.5975, 1.5255, 1.5353, 1.9954, 2.0898, 2.3641, 2.3774], [ 0. , 0. , 0. , 0. , 0. , 0. , -0.2607, -1.0865, -1.1581], [ 0. , 0. , 0. , 0. , 0.0291, 1.1658, 1.3312, 1.6754, 1.7224], [ 0. , 0. , 0. , 0.7552, 0.782 , 1.6272, 1.6917, 2.4397, 2.5154], [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , -2.0835, -2.3479], [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , -0.1699], [ 0. , 0. , 0. , 0. , 0. , 0.7318, 0.8664, 1.9921, 2.2532]])
h3_enet_s = np.array([0. , 0.049857, 0.088162, 0.091433, 0.133553, 0.282815, 0.41681 , 0.487238, 1. ])
h3_enet_b = np.array([[0. , 1.7306, 2.3954, 2.4333, 2.7987, 3.8346, 4.6094, 4.9584, 7.1816], [0. , 0. , 0. , 0. , 0.3658, 1.403 , 2.1793, 2.5288, 4.755 ], [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 2.2254], [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.3498, 2.5781], [0. , 0. , 0.6648, 0.7027, 1.0681, 2.1041, 2.8792, 3.2284, 5.453 ], [0. , 0. , 0. , 0.0378, 0.4033, 1.439 , 2.2138, 2.5628, 4.7862], [0. , 0. , 0. , 0. , 0. , 0. , 0.7753, 1.1245, 3.3487], [0. , 0. , 0. , 0. , 0. , 1.0362, 1.8112, 2.1602, 4.3837]])
h3_ten = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
mau = plt.cm.tab10(np.linspace(0, 1, 10))
for k, (ss, bb, ten) in enumerate([(h3_lasso_s, h3_lasso_b, "(a) lasso"),
                                   (h3_enet_s, h3_enet_b, "(b) elastic net, λ = 1000")]):
    for j in range(8):
        ax[k].plot(ss, bb[j], "o-", color=mau[j], lw=1.4, ms=3.0)
    cuoi = bb[:, -1]
    tt = np.argsort(cuoi)
    vt = cuoi[tt].astype(float).copy()
    for i in range(1, 8):
        if vt[i] - vt[i - 1] < 0.52:
            vt[i] = vt[i - 1] + 0.52
    for i, j in enumerate(tt):
        ax[k].plot([1.0, 1.045], [cuoi[j], vt[i]], color=mau[j], lw=0.6, alpha=0.6)
        ax[k].text(1.055, vt[i], h3_ten[j], fontsize=7.2, color=mau[j], va="center")
    ax[k].axhline(0, color="#111827", lw=0.6)
    ax[k].set_xlabel(r"$s = \|\beta\|_1 / \max\|\beta\|_1$")
    ax[k].set_title(ten, fontsize=9)
    ax[k].set_xlim(-0.02, 1.34)
ax[0].set_ylabel("hệ số đã chuẩn hoá")
fig.suptitle("Dựng lại Hình 3 của Zou & Hastie 2005 — 67 mẫu huấn luyện của bộ prostate, bản CÓ LỖI lweight",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_prostate_duong.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

![Sai số kiểm định chéo và sai số kiểm tra đi ngược nhau](<enet_cv_vs_test.png>)

**Hình 28 — Thủ tục §3.5 chọn $\lambda_2$ nào, và Bảng 1 báo cáo $\lambda_2$ nào (hình tự vẽ).**

**Cách đọc hình.** Trục ngang là bảy giá trị $\lambda_2$: sáu giá trị đầu là đúng lưới mà §3.5 đề
nghị, giá trị thứ bảy là $1000$ mà Bảng 1 dùng. Đường xanh là sai số kiểm định chéo 10 mảnh trên
67 mẫu huấn luyện, trung bình trên 20 lần chia mảnh; đường đỏ là sai số trên 30 mẫu kiểm tra.

| $\lambda_2$ | sai số CV trung bình | $s$ được chọn (trung vị) | sai số trên tập kiểm tra |
|---:|---:|---:|---:|
| 0 | 0,579405 | 0,870 | 0,5536 |
| 0,01 | **0,579039** | 0,880 | 0,5523 |
| 0,1 | 0,580810 | 0,660 | 0,5073 |
| 1 | 0,625034 | 0,620 | 0,4488 |
| 10 | 0,687300 | 0,320 | 0,3843 |
| 100 | 0,699193 | 0,260 | 0,3835 |
| 1000 | 0,700434 | 0,260 | **0,3805** |

**Hai đường đi ngược chiều nhau, đơn điệu, suốt cả bảng.** Sai số CV **tăng** đều theo $\lambda_2$;
sai số kiểm tra **giảm** đều. Thủ tục mà §3.5 mô tả — *"the chosen λ2 is the one giving the
smallest CV error"* — chọn $\lambda_2 = 0{,}01$, cho sai số kiểm tra $0{,}5523$, tức **tệ hơn cả
lasso**. Giá trị $1000$ mà Bảng 1 báo cáo là giá trị **tốt nhất trên tập kiểm tra**, và nó nằm
ngoài lưới mà chính bài báo đề nghị.

Một điều **khớp**: ở $\lambda_2 = 1000$, kiểm định chéo chọn $s$ có trung vị $0{,}260$ trên 20
lần chia, khoảng $[0{,}240\,;\,0{,}280]$ — đúng con số $0{,}26$ của Bảng 1. Vậy núm vặn thứ hai
dựng lại được; núm thứ nhất thì không.

Còn $s = 0{,}39$ mà Bảng 1 gán cho lasso thì **không dựng lại được**: 20 lần kiểm định chéo cho
trung vị $0{,}670$, khoảng $[0{,}500\,;\,1{,}000]$. Ghi ra đây như một chỗ chưa giải được, không
phải một lỗi đã chứng minh.

**Điều hình KHÔNG nói ra.** Nó không nói bên nào đúng. Tập kiểm tra chỉ có 30 mẫu, nên sai số
$0{,}3805$ đi kèm sai số chuẩn $0{,}1049$ — khoảng tin cậy rộng tới mức không tách được $0{,}38$
khỏi $0{,}55$ một cách chắc chắn. Hình này nói một điều hẹp hơn và chắc hơn: **thủ tục được mô tả
và con số được báo cáo không sinh ra từ nhau.**

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

cvte_l2 = np.array([ 0. , 0.01, 0.1 , 1. , 10. , 100. , 1000. ])
cvte_cv = np.array([0.579405, 0.579039, 0.58081 , 0.625034, 0.6873 , 0.699193, 0.700434])
cvte_te = np.array([0.553645, 0.552343, 0.507296, 0.448845, 0.38433 , 0.383545, 0.380516])
cvte_s = np.array([0.87, 0.88, 0.66, 0.62, 0.32, 0.26, 0.26])

x = np.arange(len(cvte_l2))
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(x, cvte_cv, "o-", color="#2563eb", lw=1.8, ms=5, label="sai số kiểm định chéo 10 mảnh (67 mẫu huấn luyện)")
ax.plot(x, cvte_te, "s-", color="#dc2626", lw=1.8, ms=5, label="sai số trên 30 mẫu KIỂM TRA")
for i in range(len(x)):
    ax.text(x[i], cvte_cv[i] + 0.013, "%.3f" % cvte_cv[i], fontsize=7, color="#2563eb", ha="center")
    ax.text(x[i], cvte_te[i] - 0.023, "%.3f" % cvte_te[i], fontsize=7, color="#dc2626", ha="center")
ax.axvline(int(np.argmin(cvte_cv)), color="#2563eb", ls="--", lw=1.0)
ax.axvline(int(np.argmin(cvte_te)), color="#dc2626", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(["0", "0,01", "0,1", "1", "10", "100", "1000"])
ax.set_xlabel(r"$\lambda_2$  (lưới §3.5 của bài báo là sáu giá trị đầu; 1000 nằm NGOÀI lưới đó)")
ax.set_ylabel("sai số bình phương trung bình")
ax.set_ylim(min(cvte_te) - 0.055, max(cvte_cv) + 0.055)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Hai đường đi NGƯỢC nhau: thủ tục §3.5 chọn λ₂ = 0,01 (vạch xanh),\n"
             "còn Bảng 1 báo cáo λ = 1000 (vạch đỏ) — điểm tốt nhất trên tập kiểm tra", fontsize=9)
fig.tight_layout()
fig.savefig("enet_cv_vs_test.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

### 3.12 §5, Bảng 2, Bảng 3 và Hình 4 — bốn ví dụ mô phỏng

![Mục 5 và bốn ví dụ mô phỏng](<paper_enet_zh2005_tr311_5_bon_vi_du.png>)

**Hình 29 — Zou & Hastie (2005), §5 và định nghĩa bốn ví dụ, trang 312–313.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 312: *"The purpose of this simulation is to show that the
elastic net not only dominates the lasso in terms of prediction accuracy but also is a better
variable selection procedure than the lasso."* Và về cách đo, nguyên văn cùng trang: *"Models
were fitted on training data only, and the validation data were used to select the tuning
parameters. We computed the test error (the mean-squared error) on the test data set."*

**Bốn ví dụ, chép lại đủ để dựng lại:**

| Ví dụ | Cỡ (huấn luyện / kiểm định / kiểm tra) | $p$ | $\beta$ | $\sigma$ | Tương quan |
|---|---|---:|---|---:|---|
| 1 | 20/20/200 | 8 | $(3;\,1{,}5;\,0;\,0;\,2;\,0;\,0;\,0)$ | 3 | $\operatorname{corr}(i,j) = 0{,}5^{\lvert i-j\rvert}$ |
| 2 | 20/20/200 | 8 | $\beta_j = 0{,}85$ với mọi $j$ | 3 | như ví dụ 1 |
| 3 | 100/100/400 | 40 | 10 số 0, 10 số 2, 10 số 0, 10 số 2 | 15 | $0{,}5$ với mọi cặp |
| 4 | 50/50/400 | 40 | 15 số 3, rồi 25 số 0 | 15 | ba nhóm 5 biến sinh từ ba nhân tố ẩn, cộng 25 biến nhiễu độc lập |

Ba ví dụ đầu lấy nguyên từ Tibshirani 1996; ví dụ 4 là ví dụ **mới** của bài báo này, dựng riêng
để có cấu trúc nhóm.

> **Một chỗ định nghĩa và số liệu không khớp nhau: "mean-squared error" là đại lượng nào.**
> Câu trích ở trên nói sai số được tính **trên tập kiểm tra**. Nhưng ví dụ 1 có $\sigma = 3$, tức
> phương sai nhiễu bằng 9, nên sai số dự đoán trên tập kiểm tra **không thể** nhỏ hơn 9 — mà Bảng
> 2 in $3{,}06$ cho lasso. Đo cả ba cách hiểu trên 50 lần lặp của ví dụ 1:
>
> | Cách hiểu | Trung vị đo được |
> |---|---:|
> | (a) so với $X\beta$ (phần không nhiễu) trên tập kiểm tra | 3,610 |
> | (b) so với $y$ có nhiễu trên tập kiểm tra | 12,323 |
> | **(c) sai số mô hình $(\hat\beta-\beta)^{\top}V(\hat\beta-\beta)$ của Tibshirani 1996** | **3,039** |
>
> Bài báo in $3{,}06$. Cách (c) khớp trong $0{,}7\ \%$; cách (b) — đúng theo chữ của câu trích —
> lệch **bốn lần**. Vậy Bảng 2 dùng **sai số mô hình** của bài lasso 1996, không dùng sai số dự
> đoán trên tập kiểm tra như câu văn nói. Mọi con số dựng lại dưới đây vì thế dùng định nghĩa (c).

![Bảng 2 của bài báo](<paper_enet_zh2005_tr312_bang2_mse.png>)

**Hình 30 — Bảng 2 của Zou & Hastie (2005), trang 313.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Table 2. Median mean-squared errors for the simulated examples and four
methods based on 50 replications†"*, với chú thích chân bảng: *"†The numbers in parentheses are
the corresponding standard errors (of the medians) estimated by using the bootstrap with B =500
resamplings on the 50 mean-squared errors."* — *"Bảng 2. Trung vị sai số bình phương trung bình
cho các ví dụ mô phỏng và bốn phương pháp, dựa trên 50 lần lặp. Các số trong ngoặc là sai số
chuẩn tương ứng (của trung vị), ước lượng bằng bootstrap với $B = 500$ lần lấy lại mẫu trên 50
giá trị sai số."*

![Đoạn bình luận Bảng 2](<paper_enet_zh2005_tr312_binh_luan_bang2.png>)

**Hình 31 — Zou & Hastie (2005), đoạn bình luận Bảng 2 và Bảng 3, trang 313.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 313: *"First we see that the naive elastic net either has a
very poor performance (in example 1) or behaves almost identically to either ridge regression (in
examples 2 and 3) or the lasso (in example 4). In all the examples, the elastic net is
significantly more accurate than the lasso, even when the lasso is doing much better than ridge
regression. The reductions in the prediction error in examples 1, 2, 3 and 4 are 18%, 18%, 13%
and 27% respectively. The simulation results indicate that the elastic net dominates the lasso
under collinearity."*

**Kiểm bốn tỉ lệ đó bằng chính Bảng 2 của bài báo:**

| Ví dụ | Lasso | Elastic net | Giảm thật | Bài báo in |
|---|---:|---:|---:|---:|
| 1 | 3,06 | 2,51 | 17,97 % | 18 % |
| 2 | 3,87 | 3,16 | 18,35 % | 18 % |
| 3 | 65,0 | 56,6 | 12,92 % | 13 % |
| **4** | **46,6** | **34,5** | **25,97 %** | **27 %** |

Ba dòng đầu làm tròn đúng. Dòng thứ tư thì không: $25{,}97$ làm tròn là **26 %**, không phải
27 %. Đây là một lỗi in, không đổi kết luận nào.

**Và bảng có hai cột dưới mà câu bình luận đi lướt qua.** Câu *"elastic net chính xác hơn hẳn
lasso ở mọi ví dụ"* đúng — nhưng chỉ so với lasso. Đọc cả bốn hàng của chính Bảng 2:

| Ví dụ | Phương pháp tốt nhất trong bốn | Giá trị | Elastic net |
|---|---|---:|---:|
| 1 | Elastic net | 2,51 | 2,51 |
| **2** | **Naive EN** | **2,73** | 3,16 |
| **3** | **Ridge** | **39,5** | 56,6 |
| 4 | Elastic net | 34,5 | 34,5 |

Ở ví dụ 3, ridge tốt hơn elastic net **43,3 %**. Và ở hai ví dụ, bản *naive* mà bài báo đặt tên
"ngây thơ" lại thắng bản đã sửa. Bài báo có nói điều đó — *"behaves almost identically to either
ridge regression"* — nhưng nói ở dạng mô tả hành vi, không ở dạng so thành tích.

> **Một chỗ dòng naive elastic net của ví dụ 1 tự mâu thuẫn với lưới tham số của chính bài báo.**
> Bảng 2 in naive EN $5{,}70$ so với lasso $3{,}06$ — tệ hơn **86 %**. Nhưng đặt $\lambda_2 = 0$
> trong công thức (3) thì nó **chính là** tiêu chí lasso (kiểm: lệch $3{,}6\times10^{-15}$), và
> $\lambda_2 = 0$ **nằm trong** lưới mà §3.5 đề nghị. Một thủ tục chọn tham số trên một lưới chứa
> cả lasso thì không thể cho kết quả trung vị tệ hơn lasso 86 %.
>
> Dựng lại 50 lần lặp của ví dụ 1: naive EN tệ hơn lasso ở **10/50** lần, lần tệ nhất tệ hơn
> $101{,}1\ \%$ — nhưng **trung vị tỉ số naive/lasso bằng $1{,}0000$**, và Bảng 2 báo cáo trung
> vị. Nên dòng đó không dựng lại được bằng thủ tục mà bài báo mô tả.

![Dựng lại Bảng 2](<enet_bang2.png>)

**Hình 32 — Dựng lại Bảng 2 (hình tự vẽ).** Cột nhạt là số in trong bài báo, cột đậm là bản dựng
lại trên 50 lần lặp, thanh sai số là SE bootstrap $B = 500$ như chú thích bảng yêu cầu.

| Phương pháp | Ví dụ 1 | Ví dụ 2 | Ví dụ 3 | Ví dụ 4 |
|---|---|---|---|---|
| Lasso | 3,16 (0,51) ‖ 3,06 | 3,34 (0,34) ‖ 3,87 | 46,22 (1,48) ‖ 65,0 | 59,64 (9,02) ‖ 46,6 |
| Elastic net | 2,95 (0,51) ‖ 2,51 | 3,22 (0,22) ‖ 3,16 | 34,83 (1,80) ‖ 56,6 | 32,13 (6,47) ‖ 34,5 |
| Ridge | 3,89 (0,60) ‖ 4,49 | 2,08 (0,19) ‖ 2,84 | 23,41 (1,13) ‖ 39,5 | 74,34 (6,20) ‖ 64,5 |
| Naive EN | 2,98 (0,52) ‖ 5,70 | 2,36 (0,20) ‖ 2,73 | 24,35 (1,52) ‖ 41,0 | 56,90 (7,60) ‖ 45,9 |

*(mỗi ô: bản dựng lại (SE) ‖ số in trong bài báo)*

**Cách đọc.** Ví dụ 1 dựng lại sát nhất — lasso $3{,}16$ so với $3{,}06$. Ba ví dụ còn lại lệch,
và lệch theo hai chiều khác nhau: ví dụ 3 bản dựng lại **thấp hơn** số in ở cả bốn phương pháp,
ví dụ 4 thì **cao hơn** ở ba trên bốn. Thứ **giữ nguyên** là các kết luận định tính: elastic net
thắng lasso ở cả bốn ví dụ (4/4), ridge thắng ở ví dụ 2 và 3 (2/2), và ví dụ 4 là ví dụ elastic
net thắng đậm nhất.

**Điều bản dựng lại KHÔNG chứng minh được.** Nó không nói bài báo sai — chênh lệch nằm ở phần
bài báo không mô tả đủ: lưới $s$ dùng bao nhiêu điểm, lưới ridge chạy từ đâu tới đâu, và các cột
được chuẩn hoá trên tập nào. Bài này chạy lưới $s$ 50 điểm, lưới $\lambda_2$ đúng sáu giá trị của
§3.5, và lưới ridge 60 điểm theo thang log.

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

b2_do = np.array([[ 3.1625, 3.3414, 46.2159, 59.6357], [ 2.9464, 3.2177, 34.8338, 32.1307], [ 3.894 , 2.0785, 23.414 , 74.3379], [ 2.9804, 2.3614, 24.3505, 56.9031]])
b2_se = np.array([[0.5145, 0.3354, 1.4841, 9.0214], [0.5069, 0.2171, 1.7975, 6.4702], [0.6044, 0.1935, 1.1252, 6.1998], [0.52 , 0.2022, 1.5188, 7.602 ]])
b2_in = np.array([[ 3.06, 3.87, 65. , 46.6 ], [ 2.51, 3.16, 56.6 , 34.5 ], [ 4.49, 2.84, 39.5 , 64.5 ], [ 5.7 , 2.73, 41. , 45.9 ]])
ten_pp = ["Lasso", "Elastic net", "Ridge", "Naive EN"]
mau = ["#dc2626", "#059669", "#2563eb", "#b45309"]

fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.8))
for v in range(4):
    x = np.arange(4)
    ax[v].bar(x - 0.2, b2_in[:, v], 0.38, color=mau, alpha=0.45, label="bài báo in")
    ax[v].bar(x + 0.2, b2_do[:, v], 0.38, color=mau,
              yerr=b2_se[:, v], capsize=3, label="dựng lại")
    ax[v].set_xticks(x)
    ax[v].set_xticklabels(ten_pp, rotation=30, ha="right", fontsize=7.6)
    ax[v].set_title("Ví dụ %d" % (v + 1), fontsize=9)
    if v == 0:
        ax[v].set_ylabel("trung vị sai số mô hình")
        ax[v].legend(fontsize=7.4)
fig.suptitle("Bảng 2 của Zou & Hastie 2005 — cột nhạt là số in, cột đậm là bản dựng lại "
             "(50 lần lặp, thanh sai số là SE bootstrap B = 500)", fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_bang2.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

![Hình 4 của bài báo: biểu đồ hộp](<paper_enet_zh2005_tr313_hinh4_hop_mse.png>)

**Hình 33 — Hình 4 của Zou & Hastie (2005), trang 314.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 4. Comparing the accuracy of prediction of the lasso, the elasic net
(Enet), ridge regression and the naive elastic net (NEnet) (the elastic net outperforms the lasso
in all four examples): (a) example 1; (b) example 2; (c) example 3; (d) example 4"* — *"Hình 4. So
sánh độ chính xác dự đoán của lasso, elastic net (Enet), hồi quy ridge và naive elastic net
(NEnet) (elastic net vượt lasso ở cả bốn ví dụ): (a) ví dụ 1; (b) ví dụ 2; (c) ví dụ 3; (d) ví dụ
4."*

**Cách đọc hình.** Bốn biểu đồ hộp, mỗi biểu đồ bốn cột tương ứng bốn phương pháp, mỗi hộp tóm
tắt phân bố 50 giá trị sai số. Hộp cho biết thứ mà Bảng 2 không cho biết: **độ phân tán**. Hộp
của ví dụ 4 trải rộng nhất, và điều đó khớp với bảng dựng lại ở trên, nơi sai số chuẩn của ví dụ
4 là $6$–$9$ trong khi ví dụ 1 chỉ $0{,}5$.

**Điều hình KHÔNG nói ra.** Chú thích khẳng định *"elastic net vượt lasso ở cả bốn ví dụ"* —
đúng, và bốn hộp cho thấy điều đó. Nhưng hình cũng chứa hai cột ridge và NEnet mà chú thích không
bình luận, và ở bảng (b), (c) hai cột đó thấp hơn cột Enet. Hình nói nhiều hơn chú thích của nó.

![Bảng 3 của bài báo](<paper_enet_zh2005_tr314_bang3_so_he_so.png>)

**Hình 34 — Bảng 3 của Zou & Hastie (2005), trang 315.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Table 3. Median number of non-zero coefficients"* — *"Bảng 3. Trung vị số hệ
số khác 0."*

**Tác giả nói gì.** Nguyên văn trang 313: *"Table 3 shows that the elastic net produces sparse
solutions. The elastic net tends to select more variables than the lasso does, owing to the
grouping effect. In example 4 where grouped selection is required, the elastic net behaves like
the 'oracle'."*

**Dựng lại bảng này thì khớp rất sát**, và hai ví dụ đầu khớp **từng số**:

| Phương pháp | Ví dụ 1 | Ví dụ 2 | Ví dụ 3 | Ví dụ 4 |
|---|---|---|---|---|
| Lasso | **5 ‖ 5** | **6 ‖ 6** | 20 ‖ 24 | 10 ‖ 11 |
| Elastic net | **6 ‖ 6** | **7 ‖ 7** | 24 ‖ 27 | 17 ‖ 16 |

*(dựng lại ‖ số in)*

Đây là kết quả đáng chú ý khi đặt cạnh Bảng 2: **bảng độ thưa dựng lại được, bảng sai số thì
không**. Số biến được chọn là một đại lượng bền — nó chỉ phụ thuộc vị trí nút trên đường nghiệm —
trong khi sai số bình phương phụ thuộc mọi chi tiết của lưới dò và cách chuẩn hoá.

Riêng ví dụ 4, giá đỡ thật có 15 biến; elastic net chọn trung vị 16–17, lasso chọn 10–11. Đó là
nghĩa cụ thể của chữ *"oracle"* trong câu trích: elastic net gần như tìm đúng cả nhóm, lasso bỏ
sót một phần ba.

![Ví dụ hai nhân tố ẩn](<paper_enet_zh2005_tr312_vi_du_hai_nhan_to.png>)

**Hình 35 — Zou & Hastie (2005), ví dụ dựng riêng cho Hình 5, trang 313.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 313: *"Here is an idealized example showing the important
differences between the elastic net and the lasso. Let Z1 and Z2 be two independent U(0, 20)
variables."* Rồi: *"The variables x1, x2 and x3 form a group whose underlying factor is Z1, and
x4, x5 and x6 form a second group whose underlying factor is Z2. The within-group correlations
are almost 1 and the between-group correlations are almost 0. An oracle would identify the
Z1-group as the important variates."*

Dựng lại đúng thiết lập đó — $y \sim N(Z_1 + 0{,}1Z_2,\,1)$, $x_i$ có nhiễu $N(0,1/16)$, 100 quan
sát:

* tương quan trong nhóm 1: $-0{,}9976$; $0{,}9979$; $-0{,}9979$ (dấu âm vì $x_2 = -Z_1+\varepsilon$);
* tương quan trong nhóm 2: $-0{,}9981$; $0{,}9979$; $-0{,}9977$;
* tương quan lớn nhất **giữa** hai nhóm: $0{,}2023$.

![Hình 5 của bài báo: đường nghiệm hai nhóm](<paper_enet_zh2005_tr314_hinh5_duong_hai_nhom.png>)

**Hình 36 — Hình 5 của Zou & Hastie (2005), trang 315.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 5. (a) Lasso and (b) elastic net (λ2 = 0.5) solution paths: the lasso
paths are unstable and (a) does not reveal any correction information by itself; in contrast, the
elastic net has much smoother solution paths, while clearly showing the 'grouped selection'—x1,
x2 and x3 are in one 'significant' group and x4, x5 and x6 are in the other 'trivial' group; the
decorrelation yields the grouping effect and stabilizes the lasso solution"* — *"Hình 5. Đường
nghiệm của (a) lasso và (b) elastic net ($\lambda_2 = 0{,}5$): các đường lasso không ổn định và
(a) tự nó không để lộ thông tin hiệu chỉnh nào; ngược lại, elastic net có đường nghiệm mượt hơn
hẳn, đồng thời cho thấy rõ 'chọn theo nhóm' — $x_1$, $x_2$, $x_3$ thuộc một nhóm 'quan trọng' và
$x_4$, $x_5$, $x_6$ thuộc nhóm 'tầm thường' còn lại; phép khử tương quan sinh ra hiệu ứng nhóm và
làm ổn định nghiệm lasso."*

**Cách đọc hình.** Sáu đường, sáu biến, số ghi ở mép phải. Ở bảng (a), sáu đường tách rời nhau và
không có cách nào đọc ra rằng chúng chia làm hai nhóm ba. Ở bảng (b), chúng chập thành **hai bó**:
một bó ba đường có biên độ lớn (nhóm $Z_1$), một bó ba đường sát 0 (nhóm $Z_2$).

![Dựng lại Hình 5](<enet_hai_nhom.png>)

**Hình 37 — Dựng lại Hình 5 của bài báo (hình tự vẽ).**

**Cách đọc hình và con số đi kèm.** Đo **độ trải trong nhóm** — khoảng cách giữa hệ số lớn nhất và
nhỏ nhất theo trị tuyệt đối trong cùng một nhóm, lấy trung bình dọc đường nghiệm:

| | Độ trải trong nhóm 1 | Độ trải trong nhóm 2 | Hệ số ở cuối đường |
|---|---:|---:|---|
| Lasso | **24,5243** | 5,5741 | $(41{,}5;\,-7{,}7;\,8{,}6;\,-10{,}2;\,0{,}6;\,17{,}3)$ |
| Elastic net $\lambda_2 = 0{,}5$ | **0,1483** | 0,0625 | $(24{,}9;\,-24{,}6;\,24{,}7;\,3{,}3;\,-3{,}4;\,3{,}5)$ |

Chênh **165 lần** ở nhóm 1. Và hàng hệ số cuối đường đọc thành lời được ngay: elastic net cho ba
biến của nhóm 1 độ lớn gần bằng nhau ($24{,}9$; $24{,}6$; $24{,}7$) và ba biến của nhóm 2 cũng
gần bằng nhau ($3{,}3$; $3{,}4$; $3{,}5$) — đúng cấu trúc đã dùng để sinh dữ liệu. Lasso cho
$41{,}5$ cho một biến của nhóm 1 và $17{,}3$ cho một biến của nhóm 2, tức nó **trộn lẫn hai
nhóm**.

**Điều hình này KHÔNG nói ra, và bài báo cũng không nói.** Cấu trúc hai bó ở đây khớp đúng hai
nhân tố đã dùng để sinh dữ liệu **vì đây là dữ liệu mô phỏng**. Với dữ liệu thật, một bó hệ số
chỉ phản ánh tương quan quan sát được; gọi nó là một nhóm cơ chế sinh học là một bước suy diễn
mà bản thân phương pháp không đỡ được.

<details>
<summary><b>Mã sinh hình này</b> — tự chứa, chạy trong vài giây</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

h5_lasso_s = np.array([0. , 0.162485, 0.484021, 0.598433, 0.747858, 0.967402, 1. ])
h5_lasso_b = np.array([[ 0. , 13.9606, 27.7738, 31.1637, 41.0485, 41.3486, 41.5049], [ 0. , 0. , -13.8132, -17.2663, -6.87 , -7.5914, -7.681 ], [ 0. , 0. , 0. , 2.9872, 9.8685, 8.8656, 8.6221], [ 0. , 0. , 0. , 0. , 0. , -9.4263, -10.2039], [ 0. , 0. , 0. , 0. , 0. , 0. , 0.6223], [ 0. , 0. , 0. , 0. , 6.4687, 15.8871, 17.2855]])
h5_enet_s = np.array([0. , 0.001181, 0.002702, 0.755778, 0.758222, 0.762535, 1. ])
h5_enet_b = np.array([[ 0. , 0.0998, 0.164 , 21.3797, 21.4197, 21.484 , 24.8849], [ 0. , 0. , -0.0643, -21.2821, -21.3208, -21.3825, -24.6347], [ 0. , 0. , 0. , 21.2053, 21.2451, 21.309 , 24.6894], [ 0. , 0. , 0. , 0. , 0. , 0. , 3.3459], [ 0. , 0. , 0. , 0. , 0. , -0.0873, -3.4307], [ 0. , 0. , 0. , 0. , 0.088 , 0.1754, 3.5196]])

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
mau = ["#dc2626", "#ea580c", "#b45309", "#2563eb", "#0891b2", "#0e7490"]
for k, (ss, bb, ten) in enumerate([(h5_lasso_s, h5_lasso_b, "(a) lasso"),
                                   (h5_enet_s, h5_enet_b, "(b) elastic net, λ₂ = 0,5")]):
    for j in range(6):
        ax[k].plot(ss, bb[j], "o-", color=mau[j], lw=1.5, ms=3.2, label="$x_%d$" % (j + 1))
    cuoi = bb[:, -1]
    tt = np.argsort(cuoi)
    vt = cuoi[tt].astype(float).copy()
    for i in range(1, 6):                       # đẩy nhãn ra xa nhau, không cho đè
        if vt[i] - vt[i - 1] < 2.6:
            vt[i] = vt[i - 1] + 2.6
    for i, j in enumerate(tt):
        ax[k].text(1.03, vt[i], "%d" % (j + 1), fontsize=7.6, color=mau[j], va="center")
    ax[k].axhline(0, color="#111827", lw=0.6)
    ax[k].set_xlabel(r"$s = \|\beta\|_1 / \max\|\beta\|_1$")
    ax[k].set_title(ten, fontsize=9)
    ax[k].set_xlim(-0.02, 1.10)
ax[0].set_ylabel("hệ số đã chuẩn hoá")
ax[0].legend(fontsize=7.4, ncol=2, loc="lower left")
fig.suptitle("Dựng lại Hình 5 của Zou & Hastie 2005 — $x_1,x_2,x_3$ sinh từ $Z_1$; $x_4,x_5,x_6$ từ $Z_2$",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("enet_hai_nhom.png", dpi=130, bbox_inches="tight", facecolor="white")
```

</details>

### 3.13 §6 — dữ liệu bạch cầu: phần không dựng lại được

![Hình 6 của bài báo: dừng sớm](<paper_enet_zh2005_tr315_hinh6_bach_cau.png>)

**Hình 38 — Hình 6 của Zou & Hastie (2005), trang 316.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 6. Leukaemia classification and gene selection by the elastic net
(λ = 0.01): (a) the early stopping strategy at 200 steps finds the optimal classifier with much
less computational cost than (b) the whole elastic net paths; with early stopping, the number of
steps is much more convenient than s, the fraction of L1-norm, since computing s depends on the
fit at the last step of algorithm LARS-EN; the actual values of s are not available in tenfold CV
(+) if the algorithm is stopped early; on the training set, 200 steps are equivalent to s = 0.50
(····)"*

**Cách đọc hình.** Bảng (a) vẽ sai số phân loại theo **số bước** $k$, bảng (b) theo $s$. Chú thích
nói rõ vì sao phải có bảng (a): $s$ **không tính được** nếu dừng sớm, vì nó là tỉ lệ so với đầu
cuối đường mà đầu cuối thì chưa chạy tới.

![Bảng 4 của bài báo](<paper_enet_zh2005_tr315_bang4_bach_cau.png>)

**Hình 39 — Bảng 4 của Zou & Hastie (2005), trang 316.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Table 4. Summary of the leukaemia classification results"* — *"Bảng 4. Tóm
tắt kết quả phân loại bạch cầu."*

**Phần kiểm được, và phần không.** Bộ dữ liệu Golub 1999 **không có trong repo này**, nên Bảng 4,
Hình 6 và Hình 7 chỉ kiểm được về tính nhất quán số học:

| Phương pháp | Sai số CV | Sai số kiểm tra | Số gen |
|---|---|---|---:|
| Golub | 3/38 = 0,0789 | 4/34 = 0,1176 | 50 |
| SVM–RFE | 2/38 = 0,0526 | 1/34 = 0,0294 | 31 |
| PLR–RFE | 2/38 = 0,0526 | 1/34 = 0,0294 | 26 |
| Nearest shrunken centroids | 2/38 = 0,0526 | 2/34 = 0,0588 | 21 |
| **Elastic net** | 3/38 = 0,0789 | **0/34 = 0,0000** | 45 |

Mọi mẫu số nhất quán: $38 + 34 = 72$, đúng số mẫu mà bài báo nêu. Và con số 45 gen là con số có ý
nghĩa nhất của cả mục: nguyên văn trang 317, *"note that the size of the training set is 38, so
the lasso can at most select 38 genes; in contrast, the elastic net selected more than 38 genes,
not limited by the sample size"*. Đây là kịch bản (a) của mục 2 xuất hiện trên dữ liệu thật.

> **Một chỗ bài báo tự lệch với chính nó, cách nhau một câu.** Trang 315 viết *"The leukaemia
> data consist of 7129 genes and 72 samples (Golub et al., 1999)"*, rồi ngay câu sau: *"The goal
> is to construct a diagnostic rule based on the expression level of those 7219 genes"*. Hai con
> số $7129$ và $7219$ là hai hoán vị chữ số của nhau; $7129$ là con số đúng của bộ Golub 1999.
> Lỗi in, không đổi kết luận nào.

![Hình 7 của bài báo: đường nghiệm trên dữ liệu bạch cầu](<paper_enet_zh2005_tr316_hinh7_duong_bach_cau.png>)

**Hình 40 — Hình 7 của Zou & Hastie (2005), trang 317.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Fig. 7. Leukaemia data—elastic net coefficients paths (up to k =100): the
numbers on the top indicate the number of non-zero coefficients (selected genes) at each step;
the optimal elastic net model is given by the fit at step 82 (····) with 45 selected genes; note
that the size of the training set is 38, so the lasso can at most select 38 genes; in contrast,
the elastic net selected more than 38 genes, not limited by the sample size; λ=0.01 is chosen by
tenfold CV; if a bigger λ is used, the grouping effect will be stronger"*

**Cách đọc hình.** Trục ngang là số bước $k$; dãy số ở mép trên đếm số gen đã vào mô hình tại mỗi
bước — $0, 5, 10, 14, 18, \dots, 52$. Vạch chấm ở bước 82 là mô hình được chọn, với 45 gen.

**Điều chú thích nói mà hình không cho thấy.** Câu cuối — *"nếu dùng $\lambda$ lớn hơn, hiệu ứng
nhóm sẽ mạnh hơn"* — là một phát biểu **không có hình nào trong bài báo đo**. Nó đúng theo Định
lý 1 (chặn tỉ lệ $1/\lambda_2$), nhưng bài báo không vẽ một bản $\lambda$ khác để so.

![Đoạn bình luận Hình 6 và Hình 7](<paper_enet_zh2005_tr316_binh_luan_hinh7.png>)

**Hình 41 — Zou & Hastie (2005), đoạn kết §6, trang 317.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 317: *"We stopped algorithm LARS-EN after 200 steps. As can
be seen from Fig. 6, using the number of steps k in the algorithm as the tuning parameter, the
elastic net classifier (λ=0.01 and k =82) gives a tenfold CV error of 3/38 and a test error of
0/34 with 45 genes selected."*

### 3.14 §7 — bài báo tự kết luận gì

![Mục 7: thảo luận](<paper_enet_zh2005_tr316_thao_luan.png>)

**Hình 42 — Zou & Hastie (2005), §7 Discussion, trang 317.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 317: *"We have proposed the elastic net, a novel shrinkage
and selection method. The elastic net produces a sparse model with good prediction accuracy, while
encouraging a grouping effect. The empirical results and simulations demonstrate the good
performance of the elastic net and its superiority over the lasso."*

**Cách đọc.** Ba tuyên bố, và mỗi tuyên bố có mức bằng chứng khác nhau trong chính bài báo:

| Tuyên bố | Bằng chứng trong bài báo | Bài này dựng lại được |
|---|---|---|
| cho mô hình **thưa** | Bảng 3 | **có** — 2/4 cột khớp từng số, 2/4 lệch 3–4 biến |
| **khuyến khích hiệu ứng nhóm** | Bổ đề 2, Định lý 1, Hình 5 | **có** — độ trải trong nhóm chênh 165 lần |
| **hơn hẳn lasso** | Bảng 1, Bảng 2, Hình 4 | **có về chiều**, không về giá trị; và ridge thắng elastic net ở 2/4 ví dụ |

Câu mà bài báo **không** viết ở §7, và đáng ghi ra: không chỗ nào trong mục thảo luận nhắc lại
rằng ridge thắng elastic net ở hai ví dụ mô phỏng, hay rằng bản *naive* thắng bản đã sửa ở hai ví
dụ. Đó là thông tin nằm trong Bảng 2 của chính bài báo.

### 3.15 ESL 2009 — cùng tác giả, cùng quyển sách, hai quy ước $\alpha$ ngược nhau

![Hình 3.13 của ESL: so phạt Lq với phạt elastic net](<paper_enet_esl2009_tr73_hinh3_13_mien_phat.png>)

**Hình 43 — Hình 3.13 của Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*,
2nd ed., trang 73.** Cắt từ
`AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf`.

**Chú thích gốc.** *"FIGURE 3.13. Contours of constant value of"* $\sum_j\lvert\beta_j\rvert^{q}$
*"for q = 1.2 (left plot), and the elastic-net penalty"* $\sum_j(\alpha\beta_j^2+(1-\alpha)\lvert\beta_j\rvert)$
*"for α = 0.2 (right plot). Although visually very similar, the elastic-net has sharp
(non-differentiable) corners, while the q = 1.2 penalty does not."* — *"Hình 3.13. Đường mức giá
trị không đổi của $\sum_j\lvert\beta_j\rvert^q$ với $q = 1{,}2$ (đồ thị trái), và của phạt
elastic-net $\sum_j(\alpha\beta_j^2+(1-\alpha)\lvert\beta_j\rvert)$ với $\alpha = 0{,}2$ (đồ thị
phải). Tuy nhìn rất giống nhau, elastic-net có các góc nhọn (không khả vi), còn phạt $q = 1{,}2$
thì không."*

**Cách đọc hình, và kiểm chú thích bằng số.** Hai tuyên bố của chú thích đều kiểm được. *"Nhìn
rất giống nhau"*: hai đường mức 1 đều đi qua $(\pm1,0)$ và $(0,\pm1)$, và lệch nhau theo trục dọc
nhiều nhất $0{,}027917$ tại $\beta_1 = 0{,}8476$ — dưới 3 % chiều cao hình. *"Elastic-net có góc
nhọn, $q=1{,}2$ thì không"*: đo hệ số góc một phía của biên khi tiến về đỉnh $(0,1)$:

| Khoảng cách tới đỉnh $\beta_1$ | Elastic net, $\alpha = 0{,}2$ | $L_q$, $q = 1{,}2$ |
|---:|---:|---:|
| $10^{-2}$ | $-0{,}669079$ | $-0{,}331866$ |
| $10^{-4}$ | $-0{,}666691$ | $-0{,}132075$ |
| $10^{-6}$ | $-0{,}666667$ | $-0{,}052580$ |
| $10^{-8}$ | $-0{,}666667$ | $-0{,}020932$ |

Elastic net **dừng lại** ở $-2/3$; $q = 1{,}2$ **tiếp tục tiến về 0**. Một số khác 0 nghĩa là
biên gãy tại đỉnh; một số tiến về 0 nghĩa là biên tiếp tuyến với trục, tức trơn. Hai đường trông
gần giống nhau ở khoảng cách 3 %, và khác nhau hoàn toàn về thứ quyết định tính thưa.

![Công thức (3.54) của ESL](<paper_enet_esl2009_tr73_ct3_54.png>)

**Hình 44 — ESL 2009, công thức (3.54), trang 73.** Cắt từ cùng PDF.

$$\lambda\sum_{j=1}^{p}\big(\alpha\beta_j^{2} + (1-\alpha)\lvert\beta_j\rvert\big) \tag{3.54}$$

**Tác giả nói gì.** Nguyên văn trang 73: *"Partly for this reason as well as for computational
tractability, Zou and Hastie (2005) introduced the elastic-net penalty"* (3.54) *", a different
compromise between ridge and lasso."* Và: *"The elastic-net selects variables like the lasso, and
shrinks together the coefficients of correlated predictors like ridge. It also has considerable
computational advantages over the Lq penalties."*

**Ý nghĩa, và chỗ phải cẩn thận.** Ở dạng này, $\alpha$ nhân vào $\beta_j^2$, tức **$\alpha = 1$
là ridge** — cùng quy ước với bài báo gốc. Nên $\alpha = 0{,}2$ của Hình 3.13 nghĩa là 20 % ridge
và 80 % lasso, tức một đường **gần lasso**, và đúng là hình vẽ ra một hình gần thoi.

![Bài tập 3.30 và công thức (3.91)](<paper_enet_esl2009_tr99_baitap3_30_ct3_91.png>)

**Hình 45 — ESL 2009, Bài tập 3.30 và công thức (3.91), trang 99.** Cắt từ cùng PDF.

$$\min_{\beta}\;\lVert y - X\beta\rVert^{2} + \lambda\big[\alpha\lVert\beta\rVert_2^{2} + (1-\alpha)\lVert\beta\rVert_1\big] \tag{3.91}$$

**Khung chứa gì.** Bài tập 3.30 yêu cầu chứng minh điều mà Bổ đề 1 của bài báo gốc làm: *"Show
how one can turn this into a lasso problem, using an augmented version of X and y."* Hai bài tập
liền trước cũng đáng đọc: Bài tập 3.28 hỏi chuyện gì xảy ra với lasso khi **nhân bản một cột**, và Bài tập 3.29
hỏi cùng câu đó cho ridge. Đó chính là Bổ đề 2 ở dạng bài tập.

**Ý nghĩa.** (3.91) dùng **cùng quy ước** với (3.54): $\alpha$ đứng trước chuẩn 2.

![Công thức (18.20) của ESL](<paper_enet_esl2009_tr662_ct18_20.png>)

**Hình 46 — ESL 2009, công thức (18.20), trang 662.** Cắt từ cùng PDF.

$$\sum_{j=1}^{p}\big(\alpha\lvert\beta_j\rvert + (1-\alpha)\beta_j^{2}\big) \tag{18.20}$$

**Tác giả nói gì.** Nguyên văn trang 662: *"In genomic applications, there are often strong
correlations among the variables; genes tend to operate in molecular pathways. The lasso penalty
is somewhat indifferent to the choice among a set of strong but correlated variables (Exercise
3.28). The ridge penalty, on the other hand, tends to shrink the coefficients of correlated
variables toward each other (Exercise 3.29 on page 99). The elastic net penalty (Zou and Hastie,
2005) is a compromise, and has the form"* (18.20) *". The second term encourages highly correlated
features to be averaged, while the first term encourages a sparse solution in the coefficients of
these averaged features."*

> **Đây là chỗ trùng tên đắt nhất của cả bài: $\alpha$ đã ĐỔI NGHĨA giữa hai chương của cùng một
> quyển sách.** Ở (3.54) trang 73, $\alpha$ nhân vào $\beta_j^2$; ở (18.20) trang 662, $\alpha$
> nhân vào $\lvert\beta_j\rvert$. Hai công thức cách nhau 589 trang, cùng dẫn cùng một bài báo, và
> cho hai giá trị khác nhau trên cùng một điểm. Đo tại $\beta = (0{,}6\,;\,-0{,}8)$:
>
> | $\alpha$ | (3.54) tr.73 | (18.20) tr.662 |
> |---:|---:|---:|
> | 0,0 | 1,4000 | 1,0000 |
> | 0,2 | 1,3200 | 1,0800 |
> | 0,8 | 1,0800 | 1,3200 |
> | 1,0 | 1,0000 | 1,4000 |
>
> Hai cột là **ảnh gương của nhau**, và chúng chỉ trùng đúng tại $\alpha = 0{,}5$.
>
> Hệ quả thực hành, và nó không nhỏ: Hình 3.13 vẽ $\alpha = 0{,}2$, Hình 18.5 vẽ $\alpha = 0{,}8$,
> và **cả hai đều là cấu hình gần lasso**. Ai chép con số $\alpha$ từ chương 3 sang chạy bằng thư
> viện theo quy ước chương 18 sẽ được đúng cấu hình **ngược lại** với ý định của mình.

![Hình 18.5 của ESL: đường nghiệm trên dữ liệu bạch cầu](<paper_enet_esl2009_tr662_hinh18_5_duong_bach_cau.png>)

**Hình 47 — Hình 18.5 của ESL 2009, trang 662.** Cắt từ cùng PDF.

**Chú thích gốc.** *"FIGURE 18.5. Regularized logistic regression paths for the leukemia data. The
left panel is the lasso path, the right panel the elastic-net path with α = 0.8. At the ends of
the path (extreme left), there are 19 nonzero coefficients for the lasso, and 39 for the elastic
net. The averaging effect of the elastic net results in more non-zero coefficients than the lasso,
but with smaller magnitudes."* — *"Hình 18.5. Đường nghiệm hồi quy logistic có phạt cho dữ liệu
bạch cầu. Bảng trái là đường lasso, bảng phải là đường elastic-net với $\alpha = 0{,}8$. Ở hai đầu
mút của đường (cực trái), có 19 hệ số khác 0 cho lasso và 39 cho elastic net. Hiệu ứng lấy trung
bình của elastic net cho ra nhiều hệ số khác 0 hơn lasso, nhưng với độ lớn nhỏ hơn."*

**Cách đọc hình.** Trục ngang là $\log\lambda$ chạy từ $-8$ tới $-1$, trục dọc là hệ số. Đi từ
phải sang trái là nới phạt dần. Hai bảng cùng dữ liệu, chỉ khác tỉ lệ trộn.

**Điều hình này bổ sung so với bài báo gốc.** Đây là **cùng bộ dữ liệu bạch cầu** của §6 bài báo
2005, nhưng với mất mát logistic thay vì bình phương, và với quy ước $\alpha$ ngược lại. Hai con
số 19 và 39 nói đúng điều mà bài báo gốc nói bằng 38 và 45: elastic net không bị trần $N$ chặn.

![Công thức (18.21) của ESL](<paper_enet_esl2009_tr663_ct18_21.png>)

**Hình 48 — ESL 2009, công thức (18.21), trang 663.** Cắt từ cùng PDF.

$$\max_{\{\beta_{0k},\beta_k\}_1^{K}}\Bigg[\sum_{i=1}^{N}\log\Pr(g_i\mid x_i) - \lambda\sum_{k=1}^{K}\sum_{j=1}^{p}\big(\alpha\lvert\beta_{kj}\rvert + (1-\alpha)\beta_{kj}^{2}\big)\Bigg] \tag{18.21}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $\Pr(g_i\mid x_i)$ | xác suất mô hình gán cho lớp đúng của mẫu $i$ | vô hướng trong $(0,1)$ |
| $K$ | số lớp của bài toán phân loại | số nguyên |
| $\beta_{kj}$ | hệ số của đặc trưng $j$ cho lớp $k$ | vô hướng |

**Ý nghĩa.** Đây là chỗ elastic net rời khỏi hồi quy tuyến tính: thay tổng bình phương phần dư
bằng log-likelihood của mô hình logistic đa lớp, giữ nguyên **y hệt** khoản phạt. Nguyên văn
trang 663: *"The elastic net penalty can be used with any linear model, in particular for
regression or classification."* Phạt không quan tâm phần mất mát là gì — đó là lý do cùng một
khoản phạt dùng được cho hồi quy, phân loại, và mô hình Cox.

![Đoạn ESL chọn α bằng kiểm định chéo hai chiều](<paper_enet_esl2009_tr663_chon_alpha_bang18_1.png>)

**Hình 49 — ESL 2009, đoạn về chọn $\alpha$ và dòng (8) của Bảng 18.1, trang 663.** Cắt từ cùng PDF.

**Tác giả nói gì.** Nguyên văn trang 663: *"The parameter α determines the mix of the penalties,
and is often pre-chosen on qualitative grounds. The elastic net can yield more that N non-zero
coefficients when p > N, a potential advantage over the lasso."* Và phần đo, nguyên văn cùng
trang: *"We used a sequence of 20 values of α between 0.05 and 1.0, and a 100 values of λ uniform
on the log scale covering the entire range. Values of α ∈ [0.75, 0.80] gave the minimum CV error,
with values of λ < 0.001 for all tied solutions. Although it has the lowest test error among all
methods, the margin is small and not significant. Interestingly, when CV is performed separately
for each value of α, a minimum test error of 8.8 is achieved at α = 0.10, but this is not the
value chosen in the two-dimensional CV."*

**Cách đọc, và vì sao đoạn này đáng dán nguyên văn.** Ba câu cuối là ba câu **tự phê bình** mà rất
ít tài liệu viết ra, và chúng khớp đúng chỗ mà mục 3.11 của bài này đo được trên dữ liệu prostate:

1. *"the margin is small and not significant"* — thắng nhưng biên thắng không có ý nghĩa thống kê;
2. giá trị $\alpha$ tốt nhất theo **kiểm định chéo hai chiều** ($0{,}75$–$0{,}80$) **khác** giá
   trị cho sai số kiểm tra nhỏ nhất ($0{,}10$);
3. và sách nói thẳng rằng $\alpha$ *"thường được chọn trước trên căn cứ định tính"*, tức không
   phải lúc nào cũng chọn bằng dữ liệu.

Chỗ (2) là đúng khuôn mà Hình 28 của bài này vẽ ra trên dữ liệu prostate: đường kiểm định chéo và
đường kiểm tra chỉ về hai hướng khác nhau.

### 3.16 glmnet 2010 — quy ước đang thống trị, và phép nhân lại bị bỏ

![Công thức (1), (2), (3) của glmnet](<paper_enet_glmnet2010_tr3_ct1_ct2_ct3.png>)

**Hình 50 — Friedman, Hastie & Tibshirani (2010), "Regularization Paths for Generalized Linear
Models via Coordinate Descent", *J. Statistical Software* 33(1), công thức (1)–(3), trang 3.** Cắt
từ `AI algorithms/_papers/2010 - Friedman, Hastie, Tibshirani - Regularization Paths for GLM via Coordinate Descent (glmnet).pdf`.

$$\min_{(\beta_0,\beta)\in\mathbb{R}^{p+1}} R_\lambda(\beta_0,\beta) = \min_{(\beta_0,\beta)}\left[\frac{1}{2N}\sum_{i=1}^{N}\big(y_i-\beta_0-x_i^{\top}\beta\big)^{2} + \lambda P_\alpha(\beta)\right] \tag{1}$$

$$P_\alpha(\beta) = (1-\alpha)\tfrac12\lVert\beta\rVert_{\ell_2}^{2} + \alpha\lVert\beta\rVert_{\ell_1} \tag{2}$$

$$= \sum_{j=1}^{p}\left[\tfrac12(1-\alpha)\beta_j^{2} + \alpha\lvert\beta_j\rvert\right] \tag{3}$$

| Ký hiệu | Nghĩa | Chiều |
|---|---|---|
| $N$ | số quan sát — glmnet viết hoa, bài báo 2005 viết thường $n$ | số nguyên |
| $\lambda$ | **tổng** mức phạt | vô hướng $\ge0$ |
| $\alpha$ | tỉ lệ trộn: $\alpha = 1$ lasso, $\alpha = 0$ ridge — **ngược** bài báo 2005 | $[0,1]$ |
| $1/(2N)$ | hệ số chia của phần mất mát; bài báo 2005 **không** có hệ số này | vô hướng |

**Tác giả nói gì.** Nguyên văn trang 3: *"Pα is the elastic-net penalty (Zou and Hastie 2005), and
is a compromise between the ridge-regression penalty (α = 0) and the lasso penalty (α = 1)."* Và
về vùng dùng được: *"The elastic net with α = 1−ε for some small ε > 0 performs much like the
lasso, but removes any degeneracies and wild behavior caused by extreme correlations."*

**Ý nghĩa của hai thay đổi.** Hệ số $1/(2N)$ làm mức phạt $\lambda$ **không phụ thuộc cỡ mẫu**:
thêm dữ liệu thì không phải chỉnh lại $\lambda$. Đổi tên $\alpha$ làm $\alpha = 1$ trở thành
lasso, tức mặc định quen thuộc nhất nằm ở đầu mút. Cả hai đều là quyết định về giao diện, không
về toán; nhưng chúng là lý do một con số $\lambda$ hay $\alpha$ chép từ bài báo 2005 sang glmnet
sẽ sai.

**Quy đổi giữa hai quy ước**, suy ra từ hai công thức và kiểm bằng số:

$$\lambda_1 = 2N\lambda\alpha, \qquad \lambda_2 = N\lambda(1-\alpha).$$

| glmnet | $\lambda_1$ | $\lambda_2$ | Lệch giữa hai bộ giải |
|---|---:|---:|---:|
| $\lambda = 0{,}02$, $\alpha = 1$ | 2,4000 | 0,0000 | $0$ |
| $\lambda = 0{,}02$, $\alpha = 0{,}5$ | 1,2000 | 0,6000 | $2{,}2\times10^{-16}$ |
| $\lambda = 0{,}05$, $\alpha = 0{,}2$ | 1,2000 | 2,4000 | $1{,}2\times10^{-16}$ |

Và với `scikit-learn`, nơi `alpha` đóng vai $\lambda$ còn `l1_ratio` đóng vai $\alpha$:
`ElasticNet(alpha=0.02, l1_ratio=0.5)` cho nghiệm lệch $1{,}7\times10^{-16}$ so với bộ giải theo
công thức (3) của bài báo 2005 với $(\lambda_1,\lambda_2) = (1{,}2\,;\,0{,}6)$.

![Chú thích 1 của glmnet: bỏ phân biệt naive](<paper_enet_glmnet2010_tr3_chu_thich1_bo_naive.png>)

**Hình 51 — Friedman, Hastie & Tibshirani (2010), chú thích chân trang 1, trang 3.** Cắt từ cùng PDF.

**Nguyên văn.** *"Zou and Hastie (2005) called this penalty the naive elastic net, and preferred a
rescaled version which they called elastic net. We drop this distinction here."* — *"Zou và Hastie
(2005) gọi khoản phạt này là naive elastic net, và thích hơn một bản đã được nhân lại mà họ gọi là
elastic net. Ở đây chúng tôi bỏ sự phân biệt đó."*

**Cách đọc, và vì sao chú thích hai dòng này đáng dán riêng một khung.** Toàn bộ §3.1 và §3.2 của
bài báo 2005 — bốn trang, ba lý lẽ, hai định lý — dành để lập luận rằng phải nhân nghiệm với
$1+\lambda_2$. Năm năm sau, thư viện chính thức của **chính nhóm tác giả đó** bỏ phép nhân ấy, và
giải thích bằng một câu trong chú thích chân trang.

Hệ quả thực hành, và nó là chỗ người dùng hay mắc: `glmnet` và `scikit-learn` cài đặt **bản
naive** theo tên gọi của bài báo 2005. Không được tự nhân $(1+\lambda_2)$ vào kết quả của chúng —
hàm mục tiêu của chúng là hàm mục tiêu đã công bố, và $\lambda$ đã được chọn cho chính hàm đó.

![Hình 1 của glmnet: ba phương pháp trên dữ liệu bạch cầu](<paper_enet_glmnet2010_tr4_hinh1_ba_phuong_phap.png>)

**Hình 52 — Hình 1 của Friedman, Hastie & Tibshirani (2010), trang 4.** Cắt từ cùng PDF.

**Chú thích gốc.** *"Figure 1: Leukemia data: profiles of estimated coefficients for three
methods, showing only first 10 steps (values for λ) in each case. For the elastic net, α = 0.2."*
— *"Hình 1: Dữ liệu bạch cầu: đường hệ số ước lượng cho ba phương pháp, chỉ vẽ 10 bước đầu (các
giá trị $\lambda$) trong mỗi trường hợp. Với elastic net, $\alpha = 0{,}2$."*

**Tác giả nói gì.** Nguyên văn trang 5: *"The lasso penalty admits at most N = 72 genes into the
model, while ridge regression gives all 3571 genes non-zero coefficients. The elastic-net penalty
provides a compromise between these two, and has the effect of averaging genes that are highly
correlated and then entering the averaged gene into the model."*

**Cách đọc hình.** Ba bảng cạnh nhau: lasso, elastic net $\alpha = 0{,}2$ (tức 20 % lasso theo quy
ước glmnet, gần ridge), và ridge. Bảng lasso có ít đường nhất; bảng ridge có tất cả; bảng giữa ở
giữa. Đây là kịch bản (a) của bài báo 2005, vẽ lại trên cùng loại dữ liệu bằng quy ước khác.

**Điều đáng ghi khi đặt ba hình bạch cầu cạnh nhau.** Hình 7 của bài báo 2005, Hình 18.5 của ESL
và Hình 1 của glmnet đều vẽ dữ liệu bạch cầu Golub 1999, và ba nguồn đưa ba con số gen khác nhau:
$7129$ (bài báo 2005), $7129$ (ESL) và $3571$ (glmnet). Chênh lệch đến từ bước tiền xử lý, không
từ dữ liệu; nhưng ai so ba hình mà không để ý sẽ tưởng chúng dùng ba bộ dữ liệu khác nhau.

---

## 4. Thuật toán từng bước

Bài báo gốc giải elastic net bằng **LARS-EN** (mục 3.10): dựng dữ liệu nhân tạo rồi chạy LARS.
Cách đang được dùng nhiều nhất hôm nay là **hạ theo toạ độ** của glmnet. Hai cách cho cùng
nghiệm; cách thứ hai ngắn hơn nhiều nên bài này trình bày nó, rồi kiểm chéo với cách thứ nhất.

**Ý tưởng một câu:** giữ nguyên mọi hệ số trừ một cái, thì bài toán còn lại **một chiều** và có
nghiệm đóng; quét qua từng hệ số như thế tới khi không còn gì đổi.

```
VÀO:  X (n × p, đã chuẩn hoá theo (2)), y (n, đã trừ trung bình), λ₁ ≥ 0, λ₂ ≥ 0
RA:   β̂ ∈ ℝᵖ

 1  β ← 0 ;  r ← y                       # r là phần dư, cập nhật dần cho rẻ
 2  lặp:
 3      lớn ← 0
 4      với j = 1 … p:
 5          z ← xⱼᵀ r + ‖xⱼ‖² βⱼ          # hệ số OLS một chiều, đã cộng lại phần của j
 6          mới ← soft(z, λ₁/2) / (‖xⱼ‖² + λ₂)
 7          nếu mới ≠ βⱼ:
 8              r ← r − xⱼ (mới − βⱼ)     # sửa phần dư, KHÔNG tính lại từ đầu
 9              lớn ← max(lớn, |mới − βⱼ|)
10              βⱼ ← mới
11      nếu lớn < ngưỡng: dừng
12  nếu muốn bản ĐÃ NHÂN LẠI của bài báo:  β̂ ← (1 + λ₂) · β
```

Ba dòng đáng dừng lại:

* **Dòng 6 là cả bài toán.** Tử số `soft(z, λ₁/2)` là phần $\ell_1$: nó cắt phẳng mọi hệ số nhỏ
  hơn ngưỡng về 0. Mẫu số `‖xⱼ‖² + λ₂` là phần $\ell_2$: nó chia nhỏ phần còn lại. Đổi $\lambda_2$
  về 0 thì dòng này là lasso; đổi $\lambda_1$ về 0 thì nó là ridge. **Một dòng, ba phương pháp.**
* **Dòng 8 là lý do thuật toán nhanh.** Tính lại $r = y - X\beta$ tốn $O(np)$; sửa nó tốn $O(n)$.
  Với $p$ lớn, chênh lệch đó là chênh lệch giữa chạy được và không chạy được.
* **Dòng 12 là chỗ hai thư viện khác bài báo.** `glmnet` và `scikit-learn` **dừng ở dòng 11** —
  xem Hình 51.

**Khởi động ấm.** Khi cần cả một đường nghiệm, quét $\lambda_1$ từ lớn xuống nhỏ và lấy nghiệm
trước làm điểm bắt đầu cho bước sau. Trên ví dụ $p > n$ ở khối mã dưới, khởi động ấm đưa 30 lần
giải từ **68 giây xuống 13 giây** cho cả chương trình.

### 4.1 Mã chạy được, và đầu ra thật

<details>
<summary><b>Elastic net từ đầu — một hàm cho cả ba phương pháp</b> (numpy thuần, chạy 13 giây)</summary>

```python
# -*- coding: utf-8 -*-
# Elastic net theo ĐÚNG quy ước (3) của Zou & Hastie 2005:
#     L(λ₁, λ₂, β) = |y − Xβ|² + λ₂|β|² + λ₁|β|₁
# Bản script rời, giống hệt khối này: tools/demo_enet.py
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def chuan_hoa(X, y):
    """Quy ước (2) tr.303: y trung bình 0, mỗi cột trung bình 0 và Σx² = 1."""
    Xc = X - X.mean(0)
    return Xc / np.sqrt((Xc ** 2).sum(0)), y - y.mean()


def soft(z, g):
    """Ngưỡng mềm: nghiệm của  min_b (b − z)² + 2g|b|."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def enet(X, y, l1, l2, b0=None, tol=1e-13, max_sweep=100000):
    """Hạ theo toạ độ cho (3). Trả (β của bản NAIVE, số lượt quét).

    Cập nhật một toạ độ: giữ nguyên các hệ số khác, bài toán còn lại một chiều và có
    nghiệm đóng — đó là lý do vòng lặp này không cần bất kỳ phép giải hệ nào.
    b0 là KHỞI ĐỘNG ẤM: quét một lưới λ₁ từ lớn xuống nhỏ thì mỗi bước chỉ còn vài lượt.
    """
    p = X.shape[1]
    b = np.zeros(p) if b0 is None else b0.copy()
    r = y - X @ b
    nn = (X ** 2).sum(0)
    for sweep in range(1, max_sweep + 1):
        lon = 0.0
        for j in range(p):
            z = X[:, j] @ r + nn[j] * b[j]          # hệ số OLS một chiều của phần dư
            moi = soft(z, l1 / 2.0) / (nn[j] + l2)  # so với (6): ngưỡng TRƯỚC, chia SAU
            if moi != b[j]:
                r -= X[:, j] * (moi - b[j])
                lon = max(lon, abs(moi - b[j]))
                b[j] = moi
        if lon < tol:
            return b, sweep
    raise RuntimeError("chưa hội tụ sau %d lượt quét (lệch %.2e)" % (max_sweep, lon))


def enet_sua(X, y, l1, l2):
    """Bản ĐÃ NHÂN LẠI của bài báo — công thức (12): β̂(EN) = (1+λ₂)·β̂(naive)."""
    b, _ = enet(X, y, l1, l2)
    return (1.0 + l2) * b


def qua_du_lieu_tang_cuong(X, y, l1, l2):
    """Bổ đề 1 tr.304: giải bằng cách dựng (y*, X*) rồi chạy MỘT bài toán lasso."""
    n, p = X.shape
    Xs = np.vstack([X, np.sqrt(l2) * np.eye(p)]) / np.sqrt(1.0 + l2)
    ys = np.r_[y, np.zeros(p)]
    gam = l1 / np.sqrt(1.0 + l2)
    bs, _ = enet(Xs, ys, gam, 0.0)                  # λ₂ = 0 ⇒ đúng bài toán lasso
    return bs / np.sqrt(1.0 + l2)                   # β̂(naive) = β̂*/√(1+λ₂)


def ridge_dong(X, y, l2):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(p), X.T @ y)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 60, 10
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.02 * rng.normal(size=n)   # cột 1 gần trùng cột 0
    X[:, 2] = X[:, 0] + 0.02 * rng.normal(size=n)   # cột 2 cũng vậy
    beta = np.zeros(p)
    beta[[0, 1, 2, 6]] = [1.0, 1.0, 1.0, -2.0]
    y = X @ beta + 0.5 * rng.normal(size=n)
    X, y = chuan_hoa(X, y)
    print("n = %d, p = %d ; ba cột 0,1,2 tương quan %.4f ; hệ số thật khác 0 ở cột 0,1,2,6\n"
          % (n, p, np.corrcoef(X[:, 0], X[:, 1])[0, 1]))

    # 1. λ₂ = 0 phải cho đúng lasso; λ₁ = 0 phải cho đúng ridge dạng đóng
    for l2 in (0.05, 0.5, 5.0):
        b_lap, _ = enet(X, y, 0.0, l2)
        print("   λ₁ = 0, λ₂ = %-5g: hạ theo toạ độ so với nghiệm ridge dạng đóng — lệch %.2e"
              % (l2, np.abs(b_lap - ridge_dong(X, y, l2)).max()))

    # 2. Bổ đề 1: hai đường tính hoàn toàn khác nhau phải ra cùng một nghiệm
    print()
    for l1, l2 in [(0.2, 0.5), (0.05, 2.0)]:
        a = enet(X, y, l1, l2)[0]
        b = qua_du_lieu_tang_cuong(X, y, l1, l2)
        print("   Bổ đề 1, λ₁ = %-5g λ₂ = %-4g: hạ toạ độ so với dữ liệu tăng cường — lệch %.2e"
              % (l1, l2, np.abs(a - b).max()))
        assert np.allclose(a, b, atol=1e-10)

    # 3. ba phương pháp trên cùng bộ dữ liệu
    print("\n   Cùng mức phạt, ba nghiệm khác hẳn nhau (in 4 chữ số):")
    print("      %-26s %s" % ("cột", "  ".join("%7d" % j for j in range(p))))
    print("      %-26s %s" % ("hệ số thật", "  ".join("%7.2f" % v for v in beta)))
    for ten, l1, l2 in [("Lasso   (λ₂ = 0)", 0.2, 0.0),
                        ("Naive EN (λ₁,λ₂ = 0,2;1)", 0.2, 1.0),
                        ("Elastic net, đã nhân lại", 0.2, 1.0),
                        ("Ridge   (λ₁ = 0)", 0.0, 1.0)]:
        b = enet_sua(X, y, l1, l2) if ten.startswith("Elastic") else enet(X, y, l1, l2)[0]
        print("      %-26s %s   khác 0: %2d/%d"
              % (ten, "  ".join("%7.4f" % v for v in b), int((np.abs(b) > 1e-9).sum()), p))

    # 4. hiệu ứng nhóm: độ trải của ba hệ số gần trùng nhau
    print("\n   Độ trải giữa ba hệ số của ba cột gần trùng nhau (càng nhỏ càng gom nhóm):")
    for l2 in (0.0, 0.01, 0.1, 1.0, 10.0):
        b = enet(X, y, 0.2, l2)[0][:3]
        print("      λ₂ = %-6g β̂ = [%8.4f %8.4f %8.4f]   độ trải = %.4f"
              % (l2, b[0], b[1], b[2], b.max() - b.min()))

    # 5. trần n biến: chỉ phần λ₂ mới vượt qua được
    print("\n   Khi p > n: số hệ số khác 0 lớn nhất mà mỗi phương pháp đạt được")
    rng2 = np.random.default_rng(7)
    n2, p2 = 20, 60
    X2 = rng2.normal(size=(n2, p2))
    y2 = X2 @ np.r_[np.full(8, 2.0), np.zeros(p2 - 8)] + 0.3 * rng2.normal(size=n2)
    X2, y2 = chuan_hoa(X2, y2)
    for ten, l2 in [("lasso      (λ₂ = 0)", 0.0), ("elastic net (λ₂ = 0,1)", 0.1),
                    ("elastic net (λ₂ = 1)", 1.0)]:
        tot, b_am, dung = 0, None, ""
        for l1 in np.logspace(1, -4, 30):           # từ lớn xuống nhỏ, khởi động ấm
            try:
                b_am, _ = enet(X2, y2, l1, l2, b0=b_am, tol=1e-9, max_sweep=20000)
            except RuntimeError:
                dung = "  (dừng ở λ₁ = %.4g: chưa hội tụ — đúng bệnh của lasso khi cột gần phụ thuộc)" % l1
                break
            tot = max(tot, int((np.abs(b_am) > 1e-9).sum()))
        print("      %-24s %3d  (n = %d, p = %d)%s" % (ten, tot, n2, p2, dung))
```

</details>

Đầu ra thật:

```
n = 60, p = 10 ; ba cột 0,1,2 tương quan 0.9998 ; hệ số thật khác 0 ở cột 0,1,2,6

   λ₁ = 0, λ₂ = 0.05 : hạ theo toạ độ so với nghiệm ridge dạng đóng — lệch 9.14e-13
   λ₁ = 0, λ₂ = 0.5  : hạ theo toạ độ so với nghiệm ridge dạng đóng — lệch 5.77e-14
   λ₁ = 0, λ₂ = 5    : hạ theo toạ độ so với nghiệm ridge dạng đóng — lệch 8.88e-16

   Bổ đề 1, λ₁ = 0.2   λ₂ = 0.5 : hạ toạ độ so với dữ liệu tăng cường — lệch 2.66e-15
   Bổ đề 1, λ₁ = 0.05  λ₂ = 2   : hạ toạ độ so với dữ liệu tăng cường — lệch 1.78e-15

   Cùng mức phạt, ba nghiệm khác hẳn nhau (in 4 chữ số):
      cột                              0        1        2        3        4        5        6        7        8        9
      hệ số thật                    1.00     1.00     1.00     0.00     0.00     0.00    -2.00     0.00     0.00     0.00
      Lasso   (λ₂ = 0)           10.0342   7.6608   3.9540  -0.3265  -0.0000   0.1530  -14.4430   0.4325   0.3996   0.5052   khác 0:  9/10
      Naive EN (λ₁,λ₂ = 0,2;1)    5.4918   5.4873   5.4855  -0.8286  -0.3347   0.0000  -7.1995  -0.2512  -0.1870  -0.0332   khác 0:  9/10
      Elastic net, đã nhân lại   10.9836  10.9745  10.9710  -1.6572  -0.6693   0.0000  -14.3990  -0.5024  -0.3740  -0.0665   khác 0:  9/10
      Ridge   (λ₁ = 0)            5.5141   5.5098   5.5076  -0.8766  -0.3751   0.0391  -7.2356  -0.2962  -0.2344  -0.0811   khác 0: 10/10

   Độ trải giữa ba hệ số của ba cột gần trùng nhau (càng nhỏ càng gom nhóm):
      λ₂ = 0      β̂ = [ 10.0342   7.6608   3.9540]   độ trải = 6.0802
      λ₂ = 0.01   β̂ = [  7.2388   7.2411   7.1075]   độ trải = 0.1336
      λ₂ = 0.1    β̂ = [  7.0180   7.0107   6.9949]   độ trải = 0.0230
      λ₂ = 1      β̂ = [  5.4918   5.4873   5.4855]   độ trải = 0.0063
      λ₂ = 10     β̂ = [  1.7187   1.7177   1.7178]   độ trải = 0.0010

   Khi p > n: số hệ số khác 0 lớn nhất mà mỗi phương pháp đạt được
      lasso      (λ₂ = 0)       19  (n = 20, p = 60)
      elastic net (λ₂ = 0,1)    60  (n = 20, p = 60)
      elastic net (λ₂ = 1)      60  (n = 20, p = 60)
```

Bốn chỗ đáng đọc trong đầu ra:

1. **Hai đường tính độc lập cho cùng một nghiệm.** Hạ theo toạ độ so với nghiệm ridge dạng đóng
   lệch $8{,}9\times10^{-16}$ tới $9{,}1\times10^{-13}$; hạ theo toạ độ so với đường qua **dữ liệu
   tăng cường** của Bổ đề 1 lệch $1{,}8\times10^{-15}$. Đây là phép tự kiểm, không cần thư viện
   ngoài.
2. **Ba cột gần trùng nhau ($\rho = 0{,}9998$) cho ba hành vi khác hẳn.** Lasso chia
   $(10{,}03\,;\,7{,}66\,;\,3{,}95)$ — ba số rất khác nhau cho ba cột gần như không phân biệt được.
   Elastic net chia $(10{,}98\,;\,10{,}97\,;\,10{,}97)$. Ridge cũng chia đều, nhưng giữ đủ 10/10
   biến.
3. **Độ trải giảm đơn điệu theo $\lambda_2$**: $6{,}0802 \to 0{,}1336 \to 0{,}0230 \to 0{,}0063
   \to 0{,}0010$. Đây là Định lý 1 nhìn bằng số: chặn tỉ lệ $1/\lambda_2$, và độ trải đo được
   giảm gần đúng theo nhịp đó.
4. **Trần $n$ biến**: với $n = 20$, $p = 60$, lasso dừng ở **19** biến còn cả hai cấu hình elastic
   net đi tới **60**.

Và một chi tiết của chính đoạn mã: hàm `enet()` **báo lỗi** khi hết ngân sách lặp thay vì lặng lẽ
trả về điểm đang đứng. Ở vòng quét cuối, nếu bỏ khởi động ấm thì lasso trên dữ liệu $p>n$ **không
hội tụ** sau 100 000 lượt — một hàm tối ưu im lặng trả về nghiệm dở nguy hiểm hơn hẳn một hàm báo
lỗi.

---

## 5. Cơ chế — vì sao thêm một chút $\ell_2$ lại ăn thua

Ba mục trước đã đo; mục này nói **vì sao**, và mỗi câu đều trỏ về một phép đo đã có.

**Câu trả lời ngắn: khoản $\ell_2$ làm hàm mục tiêu LỒI NGẶT, và mọi thứ còn lại là hệ quả.**

| Hệ quả | Cơ chế | Đo ở đâu |
|---|---|---|
| Nghiệm **duy nhất** | tổng của một hàm lồi và một hàm lồi ngặt thì lồi ngặt, mà hàm lồi ngặt có đúng một cực tiểu | mục 2.2: lasso cho **vô số** nghiệm với cùng giá trị hàm mục tiêu 116,7424224940 |
| **Hiệu ứng nhóm** | Bổ đề 2: với hai cột trùng khít, nghiệm duy nhất buộc phải đối xứng, tức hai hệ số bằng nhau | mục 2.2 và 3.5 |
| Vượt **trần $n$** | Bổ đề 1: bài toán tương đương có $n+p$ hàng và ma trận hạng $p$ | mục 2.1: 29 so với 200 |
| **Ổn định hơn** | Định lý 2: elastic net là lasso chạy trên ma trận tương quan đã kéo về phía $I$ | mục 3.8 |

Ba chỗ hay bị nói quá, và số liệu của chính bài báo nói gì:

> **"Elastic net gom các biến cùng nhóm lại."** Đúng — nhưng *nhóm* ở đây là nhóm **trong dữ
> liệu**, tức tập các cột có tương quan mẫu cao. Phương pháp không biết gì về sinh học, về nghiệp
> vụ, hay về nhân quả. Mục 3.12 đo được độ trải trong nhóm chênh 165 lần trên **dữ liệu mô phỏng
> nơi nhóm là thật vì chính người dựng đã đặt ra nó**.

> **"Chỉ cần thêm một chút $\ell_2$ là đủ."** Không có "một chút" nào là mặc định. Trên dữ liệu
> prostate, sai số kiểm định chéo **tăng đơn điệu** theo $\lambda_2$ trong khi sai số kiểm tra
> **giảm đơn điệu** — hai tín hiệu chỉ hai hướng ngược nhau trên cùng một bộ dữ liệu 97 mẫu
> (Hình 28).

> **"Elastic net tốt hơn lasso."** Theo Bảng 2 của bài báo thì đúng ở cả bốn ví dụ. Nhưng cũng
> theo chính bảng đó, **ridge** tốt hơn elastic net ở ví dụ 2 và ví dụ 3 — và ở ví dụ 3 tốt hơn
> tới 43,3 %. Câu đúng là *elastic net tốt hơn lasso*, không phải *elastic net tốt nhất*.

---

## 6. Chọn hai núm vặn

Elastic net là bài toán **hai tham số**, và đó là khác biệt lớn nhất về chi phí so với lasso hay
ridge. Bài báo mô tả thủ tục ở §3.5; mục này viết lại thành các bước thực hành, kèm chỗ đã đo được
là dễ sai.

1. **Chuẩn hoá trước, và chuẩn hoá bằng thống kê của TẬP HUẤN LUYỆN.** Phạt trừng phạt theo giá
   trị số của hệ số, nên đơn vị đo quyết định biến nào bị loại (mục 3.1). Trong kiểm định chéo,
   phải tính trung bình và độ dài cột **trên từng mảnh huấn luyện**, không trên cả bộ.
2. **Quét $\lambda_2$ trên một lưới thô, $\lambda_1$ (hoặc $s$, hoặc $k$) trên lưới dày.** Lưới
   $\lambda_2$ mà bài báo đề nghị là $(0;\,0{,}01;\,0{,}1;\,1;\,10;\,100)$. Với mỗi $\lambda_2$,
   một lần chạy đường nghiệm cho **toàn bộ** trục còn lại — đó là lý do lưới hai chiều không đắt
   gấp đôi mà chỉ đắt gấp số giá trị $\lambda_2$.
3. **Dùng khởi động ấm.** Quét từ phạt mạnh xuống phạt yếu và lấy nghiệm trước làm điểm xuất phát.
   Đo được ở mục 4: 68 giây xuống 13 giây.
4. **Kiểm xem điểm tối ưu có nằm sát biên lưới không.** Nếu có, nới lưới. Bảng 1 của bài báo chọn
   $\lambda_2 = 1000$ trong khi lưới đề nghị dừng ở 100 — tức điểm tối ưu nằm **ngoài** lưới, và
   đó là dấu hiệu lưới chưa đủ rộng chứ không phải một con số để chép lại.
5. **Đừng đọc một con số duy nhất từ kiểm định chéo.** Trên dữ liệu prostate, $s$ được chọn cho
   lasso có trung vị $0{,}670$ nhưng khoảng $[0{,}500\,;\,1{,}000]$ qua 20 lần chia mảnh — rộng
   tới mức con số trung vị không nói được gì chắc chắn. Chạy nhiều lần chia và **báo cáo khoảng**.
6. **Nếu mục tiêu là tập biến chứ không phải sai số dự đoán, phải đo riêng độ ổn định của tập
   biến.** Sai số dự đoán tốt không bảo đảm tập biến ổn định; đó là hai câu hỏi khác nhau.
7. **Kiểm trạng thái hội tụ.** Tăng `max_iter` mà không nhìn tiêu chí dừng là cách tạo ra một
   nghiệm dở trông như nghiệm thật — mục 4.1.

**Một ghi chú về $\alpha$ khi dùng thư viện.** Với `scikit-learn`:
`ElasticNet(alpha=λ, l1_ratio=α)` giải

$$\frac{1}{2N}\lVert y - X\beta\rVert^{2} + \lambda\Big(\alpha\lVert\beta\rVert_1 + \frac{1-\alpha}{2}\lVert\beta\rVert_2^{2}\Big),$$

tức **`alpha` là tổng mức phạt, `l1_ratio` là tỉ lệ trộn, và $\alpha = 1$ là lasso**. Quy đổi sang
ký hiệu của bài báo gốc: $\lambda_1 = 2N\lambda\alpha$ và $\lambda_2 = N\lambda(1-\alpha)$. Kiểm
bằng số ở mục 3.16: lệch $1{,}7\times10^{-16}$.

---

## 7. Giá phải trả, và khi nào đừng dùng

| Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
|---|---|---|---|
| Vượt trần $n$ biến khi $p>n$ | $29 \to 200$ biến trên bài toán $30\times200$; $19 \to 60$ trên $20\times60$ | mô hình to hơn, và "giữ cả nhóm" cũng có nghĩa giữ cả biến thừa trong nhóm | mục 2.1 · mục 4.1 |
| Hệ số trong nhóm hết chia tuỳ tiện | độ trải trong nhóm $24{,}52 \to 0{,}15$, nhỏ hơn **165 lần** | mất tính thưa triệt để: elastic net giữ nhiều biến hơn lasso ở mọi ví dụ của Bảng 3 | mục 3.12 |
| Nghiệm duy nhất | lasso cho **vô số** nghiệm với hai cột trùng khít; elastic net cho đúng một | — | mục 2.2 |
| **Hai** núm vặn thay vì một | — | lưới dò nhân lên theo số giá trị $\lambda_2$; với dữ liệu nhỏ, lưới hai chiều dễ chọn trúng một cặp may mắn | mục 6 |
| Hiệu năng dự báo | thắng lasso 4/4 ví dụ của Bảng 2 | **thua ridge 2/4 ví dụ** của chính bảng đó, có chỗ thua 43,3 % | mục 3.12 |
| Cài đặt | — | **bẫy hai phiên bản**: bài báo nhân $(1+\lambda_2)$, glmnet và scikit-learn thì không | mục 3.8 · 3.16 |
| Chọn tham số | — | kiểm định chéo và sai số kiểm tra có thể chỉ **hai hướng ngược nhau**; trên prostate chúng ngược nhau đơn điệu suốt cả lưới | mục 3.11 |

**Năm tình huống nên dùng thứ khác:**

1. **Khi $p \ll n$ và các cột gần trực giao.** Không có nhóm để gom, không có trần để vượt; lasso
   đơn giản hơn và chỉ có một núm vặn.
2. **Khi mục tiêu là một mô hình càng ít biến càng tốt.** Elastic net giữ nhiều biến hơn lasso ở
   **cả bốn** ví dụ của Bảng 3 — đó là thiết kế, không phải khuyết điểm, nhưng nó đi ngược mục
   tiêu này.
3. **Khi cấu trúc nhóm đã biết trước.** Nếu đã biết cột nào thuộc nhóm nào — các cột một-nóng của
   cùng một biến hạng mục, các phép đo lặp của cùng một cảm biến — thì *group lasso* chọn hoặc bỏ
   **cả nhóm** theo đúng định nghĩa nhóm của người dùng, thay vì suy ra nhóm từ tương quan.
4. **Khi cần suy diễn thống kê trên tập biến đã chọn.** Sai số chuẩn và p-value của OLS chạy trên
   giá đỡ đã chọn là **không hợp lệ**, vì chính dữ liệu đã dùng để chọn giá đỡ. Đây là hạn chế
   chung của mọi phương pháp chọn biến bằng phạt.
5. **Khi tập kiểm tra quá nhỏ để phân xử.** Trên prostate, 30 mẫu kiểm tra cho sai số chuẩn
   $0{,}1049$ quanh giá trị $0{,}3805$ — khoảng đó chồng lên cả giá trị của lasso. Một bảng xếp
   hạng dựng trên 30 mẫu không tách được các phương pháp.

---

## 8. So với cả họ

| | OLS | Ridge | Lasso | **Elastic net** | Group lasso |
|---|---|---|---|---|---|
| Có hệ số bằng đúng 0 | không | không | **có** | **có** | cả nhóm cùng lúc |
| Số biến tối đa khi $p>n$ | không xác định | $p$ | $n$ | **$p$** | tuỳ thiết kế |
| Nghiệm duy nhất | không khi $p>n$ | **có** | không bảo đảm | **có** khi $\lambda_2>0$ | tuỳ |
| Nhóm cột tương quan | hệ số bất ổn | chia gần đều | **chia tuỳ tiện** | **chia gần đều** | chọn/bỏ cả nhóm |
| Số núm vặn | 0 | 1 | 1 | **2** | 1 cộng cách chia nhóm |
| Cần biết trước nhóm | — | — | — | **không** | **có** |
| Nghiệm đóng | có | **có** | không | không | không |

Hai dòng cuối là hai dòng để chọn giữa elastic net và group lasso: elastic net **suy ra** nhóm từ
tương quan quan sát được, group lasso **được cho** nhóm. Khi nhóm đã biết chắc thì cái thứ hai
mạnh hơn; khi không biết thì cái thứ nhất là cách duy nhất trong hai cách.

---

## 9. Dùng ở đâu trong thực tế

* **Dữ liệu gen và vi mảng.** Bài toán gốc. $p$ hàng nghìn, $n$ vài chục; các gen cùng một đường
  trao đổi chất tương quan mạnh. Cả ba nguồn của bài này đều minh hoạ trên cùng bộ dữ liệu bạch
  cầu Golub 1999 (Hình 40, 47, 52).
* **Biến hạng mục mã hoá một-nóng.** Các cột con của cùng một biến hạng mục tương quan theo cấu
  tạo. Elastic net ổn định hơn lasso ở đây; nhưng nếu yêu cầu là *giữ hoặc bỏ cả biến hạng mục*
  thì group lasso hợp mục tiêu hơn.
* **Tín hiệu và ảnh với từ điển dư thừa.** Khi các nguyên tử của từ điển tương quan, lasso chọn
  tuỳ tiện giữa các nguyên tử gần giống nhau — đúng tình huống Bổ đề 2 mô tả.
* **Mô hình chấm điểm có nhiều đặc trưng phái sinh.** Nhiều đặc trưng cùng sinh ra từ một nguồn
  nên tương quan mạnh theo cấu tạo. Elastic net là một mốc so sánh tốt; nhưng tính thưa **không**
  tự bảo đảm mô hình giải thích được hay ổn định theo thời gian — hai thứ đó phải đo riêng.
* **Mô hình tuyến tính tổng quát và mô hình Cox.** Khoản phạt không quan tâm phần mất mát là gì
  (công thức 18.21), nên cùng một elastic net dùng được cho hồi quy logistic, đa thức và phân tích
  sống còn. Đây là dạng dùng nhiều nhất trên dữ liệu y sinh thật hôm nay.

---

## 10. Đối chiếu với nguồn gốc — đã lấy gì, bỏ gì, vì sao

### 10.1 Zou & Hastie 2005 — theo hình và bảng

Bài báo có **7 hình và 4 bảng**. Bảng dưới liệt kê **đủ cả 11**, không bỏ dòng nào.

| Hình/Bảng | Nội dung | Ở mục nào của bài này | Đã dẫn nhận xét của tác giả chưa | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|---|
| Figure 1, tr.304 | ba miền phạt ở mức 1, $\alpha = 0{,}5$ | mục 3.3, Hình 6 — đã cắt | có — chú thích tr.304 | — |
| Figure 2, tr.305 | nghiệm chính xác trong thiết kế trực giao | mục 3.4, Hình 9 — đã cắt | có — tr.305 | — |
| Figure 3, tr.311 | đường nghiệm lasso và elastic net trên prostate | mục 3.11, Hình 26 — đã cắt, kèm bản dựng lại ở Hình 27 | có — chú thích tr.311 | — |
| Figure 4, tr.314 | biểu đồ hộp sai số của bốn phương pháp | mục 3.12, Hình 33 — đã cắt | có — chú thích tr.314 | — |
| Figure 5, tr.315 | đường nghiệm của ví dụ hai nhân tố ẩn | mục 3.12, Hình 36 — đã cắt, kèm bản dựng lại ở Hình 37 | có — chú thích tr.315 | — |
| Figure 6, tr.316 | dừng sớm trên dữ liệu bạch cầu | mục 3.13, Hình 38 — đã cắt | có — chú thích tr.316 và tr.317 | — |
| Figure 7, tr.317 | đường hệ số elastic net trên dữ liệu bạch cầu | mục 3.13, Hình 40 — đã cắt | có — chú thích tr.317 | — |
| Table 1, tr.311 | so sánh năm phương pháp trên prostate | mục 3.11, Hình 24 — đã cắt, **dựng lại 5/5 dòng** | có — tr.311 và tr.312 | — |
| Table 2, tr.313 | trung vị sai số của bốn ví dụ mô phỏng | mục 3.12, Hình 30 — đã cắt, dựng lại ở Hình 32 | có — tr.313 | — |
| Table 3, tr.315 | trung vị số hệ số khác 0 | mục 3.12, Hình 34 — đã cắt, **dựng lại 4/8 ô khớp từng số** | có — tr.313 | — |
| Table 4, tr.316 | kết quả phân loại bạch cầu, năm phương pháp | mục 3.13, Hình 39 — đã cắt | có — tr.317 | — |

**11/11 hình và bảng đã cắt vào bài.**

### 10.2 Zou & Hastie 2005 — theo mục

Bài báo có **20 mục đánh số** (kể cả phụ lục). Bảng dưới liệt kê **đủ cả 20**.

| Mục của bài báo | Nói về gì | Vào bài ở đâu | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|
| 1 Introduction and motivation | ba kịch bản lasso tắc | mục 2, Hình 1 — trích nguyên văn | — |
| 2 Naive elastic net | tiêu đề nhóm | mục 3.2 | — |
| 2.1 Definition | công thức (2)–(5), chữ $\alpha$ | mục 3.1 và 3.2, Hình 3–5 | — |
| 2.2 Solution | Bổ đề 1, công thức (6), Hình 1–2 | mục 3.3 và 3.4, Hình 6–10 | — |
| 2.3 The grouping effect | công thức (7), Bổ đề 2, Định lý 1 | mục 3.5, Hình 11–14 | — |
| 2.4 Bayesian connections and the Lq-penalty | công thức (8)–(9), vì sao loại $1<q<2$ | mục 3.6, Hình 15 | — |
| 3 Elastic net | tiêu đề nhóm | mục 3.7 | — |
| 3.1 Deficiency of the naive elastic net | lập luận "co hai lần" | mục 3.7, Hình 16 | — |
| 3.2 The elastic net estimate | công thức (10)–(15), Định lý 2 | mục 3.8, Hình 17–19 | — |
| 3.3 Connections with univariate soft thresholding | công thức (16), UST | mục 3.9, Hình 20 | — |
| 3.4 Computation: the algorithm LARS-EN | thuật toán, cấu trúc thưa của $X^{*}$ | mục 3.10, Hình 21 | — |
| 3.5 Choice of tuning parameters | ba cách tham số hoá, lưới $\lambda_2$ | mục 3.10, Hình 22 | — |
| 4 Prostate cancer example | Bảng 1, Hình 3 | mục 3.11, Hình 24–28 | — |
| 5 A simulation study | bốn ví dụ, Bảng 2–3, Hình 4–5 | mục 3.12, Hình 29–37 | — |
| 6 Microarray classification and gene selection | Bảng 4, Hình 6–7 | mục 3.13, Hình 38–41 — **dữ liệu không có trong repo**, chỉ kiểm được tính nhất quán số học | — |
| 7 Discussion | ba tuyên bố kết luận | mục 3.14, Hình 42 | — |
| Appendix A Proofs | tiêu đề nhóm | mục 3.5, Hình 13 | — |
| A.1 Proof of lemma 2 | chứng minh hai phần của Bổ đề 2 | mục 3.5, Hình 13 | — |
| A.2 Proof of theorem 1 | công thức (17)–(20) | mục 3.5, Hình 13–14 — **chỗ chuẩn 2 thay chuẩn 1** | — |
| A.3 Proof of theorem 2 | công thức (21) | mục 3.8 và Hình 14 | — |

**20/20 mục đã vào bài.** Công cụ `pdfscan.py --muc` báo 19 mục — nó bỏ sót tiêu đề
*"2. Naive elastic net"*; con số đúng đếm tay là 20.

### 10.3 Zou & Hastie 2005 — theo công thức đánh số

Bài báo đánh số **21 công thức**. Bảng dưới liệt kê đủ cả 21.

| Công thức | Nội dung | Ở mục nào | Đã kiểm bằng số chưa |
|---|---|---|---|
| (1) tr.301 | mô hình tuyến tính $\hat y = \hat\beta_0 + \sum_j x_j\hat\beta_j$ | mục 1 | — (định nghĩa) |
| (2) tr.303 | quy ước chuẩn hoá | mục 3.1 | có — lệch $\le 3{,}6\times10^{-15}$ |
| (3) tr.303 | tiêu chí naive elastic net | mục 3.2 | có — nền của mọi phép kiểm khác |
| (4) tr.303 | $\hat\beta = \arg\min L$ | mục 3.2 | có |
| (5) tr.303 | dạng ràng buộc, $\alpha = \lambda_2/(\lambda_1+\lambda_2)$ | mục 3.2 | có — hệ số tỉ lệ $c = 1{,}000000$ |
| (6) tr.305 | nghiệm chính xác, thiết kế trực giao | mục 3.4 | có — lệch $2{,}2\times10^{-16}$ |
| (7) tr.306 | dạng phạt tổng quát $J(\beta)$ | mục 3.5 | có — dùng cho Bổ đề 2 |
| (8) tr.307 | tiên nghiệm của hồi quy cầu | mục 3.6 | có — $q = 1{,}2$ cho **0** hệ số bằng 0 |
| (9) tr.307 | tiên nghiệm của elastic net | mục 3.6 | có |
| (10) tr.307 | bài toán lasso trên dữ liệu tăng cường | mục 3.8 | có — lệch $8{,}9\times10^{-15}$ |
| (11) tr.307 | $\hat\beta(\text{EN}) = \sqrt{1+\lambda_2}\,\hat\beta^{*}$ | mục 3.8 | có |
| (12) tr.307 | $\hat\beta(\text{EN}) = (1+\lambda_2)\hat\beta(\text{naive})$ | mục 3.8 | có — lệch $2{,}8\times10^{-14}$ |
| (13) tr.308 | phân rã toán tử ridge, khử tương quan | mục 3.8 | có — lệch $2{,}8\times10^{-17}$ |
| (14) tr.309 | Định lý 2 | mục 3.8 | có — lệch $2{,}5\times10^{-14}$ |
| (15) tr.309 | lasso viết cùng dạng với (14) | mục 3.8 | có — lệch $1{,}1\times10^{-13}$ |
| (16) tr.309 | ngưỡng mềm một biến (UST) | mục 3.9 | có — hội tụ theo $1/\lambda_2$ |
| (17) tr.318 | điều kiện dừng cho hệ số khác 0 | mục 3.5 | có — dùng trong phép dựng lại câu $\lvert\cos\theta\rvert$ |
| (18) tr.318 | điều kiện dừng viết cho chỉ số $i$ | mục 3.5 | có |
| (19) tr.319 | điều kiện dừng viết cho chỉ số $j$ | mục 3.5 | có |
| (20) tr.319 | hiệu hai hệ số qua phần dư | mục 3.5 | có — khớp từng chữ số ở cả 5 dòng bảng |
| (21) tr.319 | bước đại số trong chứng minh Định lý 2 | mục 3.8 | có — qua (14) |

### 10.4 ESL 2009 — phạm vi §3.4.3 và §18.4

ESL là sách 764 trang; chỉ hai mục nói về elastic net, và bài này lấy **đủ** phần đó.

| Mục của ESL | Nội dung | Vào bài ở đâu | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|
| §3.4.3, công thức (3.54) tr.73 | định nghĩa phạt elastic-net, quy ước $\alpha = 1$ là ridge | mục 3.15, Hình 44 | — |
| §3.4.3, Hình 3.13 tr.73 | so đường mức $L_q$ ($q=1{,}2$) với elastic net ($\alpha = 0{,}2$) | mục 3.15, Hình 43 | — |
| Bài tập 3.28 tr.99 | lasso khi nhân bản một cột | mục 3.15 — nhắc trong phần đọc Hình 45 | — |
| Bài tập 3.29 tr.99 | ridge khi nhân bản một cột | mục 3.15 — như trên | — |
| Bài tập 3.30, công thức (3.91) tr.99 | đưa elastic net về lasso bằng dữ liệu tăng cường | mục 3.15, Hình 45 | — |
| §18.4, công thức (18.20) tr.662 | phạt elastic-net, quy ước $\alpha = 1$ là **lasso** | mục 3.15, Hình 46 | — |
| §18.4, công thức (18.21) tr.663 | elastic net cho hồi quy logistic đa lớp | mục 3.15, Hình 48 | — |
| §18.4, Hình 18.5 tr.662 | đường nghiệm lasso và elastic net trên dữ liệu bạch cầu | mục 3.15, Hình 47 | — |
| §18.4, Hình 18.6 tr.663 | đường sai số huấn luyện/kiểm tra/CV của lasso | — | về **lasso**, không về elastic net; sách nói rõ bản elastic net *"are qualitatively similar and are not shown"* |
| §18.4, Bảng 18.1 dòng (8) tr.655 | kết quả elastic net trong bảng so sánh 14 phương pháp | mục 3.15, Hình 49 — trích đoạn bình luận | bảng đầy đủ so 14 phương pháp trên một bài toán phân loại, ngoài phạm vi bài này |
| §3.9 tr.93 | hạ theo toạ độ cho lasso, và một câu nói nó dùng được cho elastic net | mục 4 | phần lasso đã nằm ở bài [Lasso](<../Lasso Regression/Lasso Regression.md>) |

### 10.5 glmnet 2010 — theo hình, bảng và mục

Bài báo có **2 hình và 5 bảng**; 20 mục đánh số. Bài này dùng nó cho **§2 và quy ước $\alpha$**;
phần thuật toán hạ theo toạ độ thuộc bài [Lasso](<../Lasso Regression/Lasso Regression.md>).

| Hình/Bảng/Mục | Nội dung | Vào bài ở đâu | Nếu KHÔNG lấy: vì sao |
|---|---|---|---|
| Figure 1, tr.4 | đường hệ số của ba phương pháp trên dữ liệu bạch cầu | mục 3.16, Hình 52 | — |
| Figure 2, tr.17 | kiểm định chéo 10 mảnh trên dữ liệu mô phỏng | — | về cách chọn $\lambda$ nói chung, không về elastic net |
| Table 1–5, tr.13–16 | bảng đo thời gian chạy | — | đo **lasso**, không đo elastic net; bài báo nói rõ tr.16: *"We do not perform comparisons on the elastic net versions of the penalties, since there is not much software available for elastic net"* |
| §1 Introduction | bối cảnh, danh sách các họ phạt | mục 3.16 | — |
| §2 Algorithms for the lasso, ridge regression and elastic net | công thức (1)–(3), chú thích 1 | mục 3.16, Hình 50–51 | — |
| §2.1–2.6 | cập nhật naive, hiệp phương sai, thưa, có trọng số, theo đường, chi tiết khác | — | thuật toán hạ theo toạ độ, đã nằm ở bài [Lasso](<../Lasso Regression/Lasso Regression.md>) |
| §3 Regularized logistic regression | mở rộng sang logistic | mục 9 — nhắc một câu | chi tiết thuật toán ngoài phạm vi |
| §4, §4.1, §4.2 | đa lớp, nhập nhằng tham số, phản hồi dạng ma trận | — | ngoài phạm vi bài hồi quy |
| §5, §5.1–5.4 | đo thời gian | — | như dòng Table 1–5 |
| §6 Selecting the tuning parameters | chọn $\lambda$ và $\alpha$ | mục 6 | — |
| §7 Discussion | bàn luận | — | không thêm gì về elastic net |
| Appendix A | chứng minh Định lý 1 của glmnet | — | về bài toán đa lớp |

---

## 11. Tài liệu tham khảo

```
[1] H. Zou, T. Hastie, "Regularization and Variable Selection via the Elastic Net",
    J. R. Statist. Soc. B 67(2):301–320, 2005.
    → BÀI GỐC. Toàn bộ mục 3.1–3.14 của bài này đi theo nó.
    AI algorithms/_papers/2005 - Zou, Hastie - Regularization and variable selection via the elastic net.pdf

[2] T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical Learning",
    2nd ed., Springer 2009 (bản in thứ 12).
    → (3.54) tr.73 · Hình 3.13 tr.73 · Bài tập 3.28–3.30 tr.99 · §18.4 (18.20)–(18.21)
      tr.662–664 · Hình 18.5 tr.662. Chú thích tr.3 là nguồn của chuyện lỗi dữ liệu prostate.
    AI algorithms/_papers/2009 - Hastie, Tibshirani, Friedman - The Elements of Statistical Learning (2nd ed, print 12).pdf

[3] J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized Linear
    Models via Coordinate Descent", J. Statistical Software 33(1), 2010.
    → (1)–(3) tr.3 và chú thích 1 tr.3 · Hình 1 tr.4. Quy ước α đang thống trị.
    AI algorithms/_papers/2010 - Friedman, Hastie, Tibshirani - Regularization Paths for GLM via Coordinate Descent (glmnet).pdf

[4] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso",
    J. R. Statist. Soc. B 58(1):267–288, 1996.
    → Ba ví dụ mô phỏng mà [1] §5 dùng lại, và định nghĩa sai số mô hình mà Bảng 2 dùng.
    Bản PDF là bản tải JSTOR, KHÔNG phải bản truy cập mở: nó nằm trong .gitignore và
    CHỈ có trên máy dựng bài. Mọi câu trích từ nó chỉ kiểm lại được trên máy có file đó.

[5] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression",
    The Annals of Statistics 32(2):407–451, 2004.
    → Thuật toán LARS, nền của LARS-EN ở [1] §3.4.
    AI algorithms/_papers/2004 - Efron et al - Least Angle Regression.pdf
```

**Dữ liệu.** `tools/prostate.csv` — bộ prostate của gói R `ElemStatLearn`, **bản đã sửa** lỗi
`lweight` của đối tượng 32. Mục 3.11 dựng lại Bảng 1 trên cả bản đã sửa lẫn bản có lỗi; bản có lỗi
được tạo ngay trong script bằng một dòng, không lưu thành file riêng.

**Mã.** Mọi con số trong bài đến từ [`tools/v_enet.py`](<../../../tools/v_enet.py>) — 20 mục, chạy
hết trong khoảng 25 giây. Sáu hình tự vẽ do sáu khối mã in ngay trong bài sinh ra; mỗi khối tự chứa, dán
sẵn mảng số, và chạy trong vài giây — bản rời của cả sáu khối nằm ở
[`tools/fig_enet.py`](<../../../tools/fig_enet.py>). Khối mã ở mục 4.1 có bản rời là
[`tools/demo_enet.py`](<../../../tools/demo_enet.py>).

**Phần không có trong repo.** Bộ dữ liệu bạch cầu của Golub và cộng sự 1999 không có ở đây, nên
Bảng 4, Hình 6 và Hình 7 của bài báo gốc chỉ kiểm được về tính nhất quán số học (mục 3.13).

---

## 12. Kiểm tra trước khi đóng bài

- [x] **Bảy khối đủ.** IO mục 0.1 · KN mục 0.2–0.3 · SS mục 2 · CT mục 3 · VS mục 5 · KQ mục
      3.11–3.13 · GIÁ mục 7
- [x] **Luật 6, nhóm ruột đủ 100 %.** 7/7 hình và 4/4 bảng của bài báo gốc đã cắt vào bài; 21/21
      công thức đánh số đã viết ra kèm bảng ký hiệu và đoạn ý nghĩa; 20/20 mục đã vào bài
- [x] **Mỗi hình ruột có đủ bốn phần**: khung đã cắt kèm số trang · chú thích gốc nguyên văn kèm
      bản dịch · nhận xét nguyên văn của chính tác giả kèm số trang · phần "cách đọc" và "điều
      hình KHÔNG nói ra"
- [x] **Hai bảng đối chiếu cho nguồn chính** (mục 10.1 theo hình/bảng, mục 10.2 theo mục), cộng
      một bảng theo **công thức** (mục 10.3), và bảng đối chiếu cho hai nguồn phụ (mục 10.4, 10.5)
- [x] **Mọi con số sinh từ script.** `tools/v_enet.py` in ra toàn bộ; không con số nào gõ tay
- [x] **Bảng ký hiệu có cột "Đọc là gì"**, kèm bảng năm quy ước $\alpha$ của năm nguồn — chỗ trùng
      tên đắt nhất của cả GĐ 1
- [x] **Ví dụ số chạy tay** ở mục 0.4, ba bước, bấm máy tính được
- [x] **Mã chạy được kèm đầu ra thật** ở mục 4.1, có `assert` nối hai đường tính độc lập
- [x] **Sáu hình tự vẽ có khối mã sinh ra chúng in ngay trong bài**, mỗi khối đúng một `savefig()`
- [x] **Mọi hình đã được mở ra xem** — sáu hình tự vẽ và 46 khung cắt đều được đọc lại để soát chữ
      bị đè, nhãn bị cắt và nội dung có đúng thứ caption nói không
- [x] **Báo cáo kết quả âm và chỗ nguồn tự mâu thuẫn**: 13 chỗ ở mục 0.5, mỗi chỗ kèm phép đo
- [x] **Nói rõ phần KHÔNG dựng lại được**: $s = 0{,}39$ của Bảng 1 · ba trong bốn cột của Bảng 2 ·
      toàn bộ §6 vì thiếu dữ liệu Golub 1999
- [x] Mọi liên kết cục bộ trỏ tới tệp đang tồn tại; mọi dẫn chiếu PDF khớp tên file trong `_papers/`

**Bốn chỗ còn để ngỏ, ghi ra để không ai đi tìm lại từ đầu:**

1. **$s = 0{,}39$ của dòng lasso trong Bảng 1.** Kiểm định chéo 10 mảnh trên 67 mẫu huấn luyện cho
   trung vị $0{,}670$, khoảng $[0{,}500\,;\,1{,}000]$ qua 20 lần chia. Không thiết lập nào thử
   được cho ra $0{,}39$.
2. **Ba trong bốn cột của Bảng 2.** Ví dụ 1 dựng lại sát ($3{,}16$ so với $3{,}06$); ví dụ 2, 3, 4
   lệch theo hai chiều khác nhau. Phần bài báo không mô tả — độ dày lưới $s$, khoảng lưới ridge,
   tập dùng để chuẩn hoá — đủ để giải thích, nhưng không đủ để dựng lại chính xác.
3. **Dòng naive elastic net của ví dụ 1 (5,70).** Không dựng lại được, vì $\lambda_2 = 0$ nằm
   trong lưới và ở đó naive elastic net **chính là** lasso.
4. **§6 dữ liệu bạch cầu.** Cần bộ Golub 1999 mới dựng lại được Bảng 4, Hình 6 và Hình 7.
