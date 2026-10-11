# Huber Regression

Branch: Machine Learning
Category: Supervised Learning
Subcategory: Regression
Giai đoạn: GĐ 1
Order: 420
Tier: S
Trạng thái: Đang viết
Level: Beginner
Author: Peter J. Huber
Date: March 1, 1964
url: https://doi.org/10.1214/aoms/1177703732

> **Huber Regression — hồi quy bền vững.** Thay hàm mất mát bình phương bằng một hàm
> **bình phương ở gần 0 và tuyến tính ở xa**. Điểm có phần dư rất lớn vì thế không mua thêm
> được ảnh hưởng, nhưng điểm nằm xa theo $X$ thì vẫn kéo được mô hình — và mục 6 đo chỗ đó
> bằng một con số.
>
> Bài này dựng lại **toàn bộ** bài báo gốc: cả bốn bảng số, cả bảy công thức đánh số, cả năm
> định lý và sáu bổ đề. Bốn bảng đánh số cộng hai bảng không đánh số ở tr.101 là **519 ô**;
> **517** trong đó dựng lại được từ chính công thức của bài báo, và **hai** không — mục 2.9.1
> và mục 2.14.5 chỉ ra từng ô.

---

## 0. Trước khi đọc

### 0.1 Đầu vào, đầu ra

Huber nhận **đúng những gì hồi quy tuyến tính nhận**, cộng một con số $k$, và trả ra **đúng
cùng hình dạng**. Không có gì mới về kiểu dữ liệu — toàn bộ thay đổi nằm ở hàm mất mát.

| | Cách hiểu thông thường | Ký hiệu | Ví dụ mục 0.4 |
|---|---|---|---|
| **Đầu vào** | một bảng gồm $n$ mẫu và $p$ cột số | $X\in\mathbb{R}^{n\times p}$ | $5\times1$ |
| **Đáp án khi huấn luyện** | một giá trị số cho mỗi mẫu | $y\in\mathbb{R}^{n}$ | $5$ số |
| **Siêu tham số** | "phần dư lớn tới đâu thì coi là điểm hỏng" | $k>0$ | $k=1{,}345$ |
| **Đại lượng phụ, ước lượng bên trong** | thang đo của phần dư | $S>0$ | đặt bằng $1$ cho gọn |
| **Đầu ra của việc học** | một hệ số cho mỗi cột | $\hat\beta\in\mathbb{R}^{p}$ | một số |
| **Dự đoán** | nhân từng dòng dữ liệu với các hệ số | $\hat y=X\hat\beta$ | $5$ số |

```
X   shape (n, p)      y   shape (n,)      k = 1.345   một số vô hướng

    ┌─ |e| ≤ kS  →  điểm này được xử lý Y HỆT bình phương tối thiểu
e = y − Xβ̂ ─┤
    └─ |e| > kS  →  ảnh hưởng của điểm này bị CHẶN, không tăng thêm nữa

β̂   shape (p,)        ŷ = X @ β̂
```

Bốn điều dễ hiểu sai, nói rõ ngay:

- **$k$ đo bằng đơn vị của $S$, không phải đơn vị của $y$.** Ngưỡng thật là tích $kS$. Đây là
  lý do Huber không cần người dùng biết trước $y$ lớn cỡ nào — nhưng cũng là lý do **cách ước
  lượng $S$** lại quan trọng tới mức mục 4.3 phải dành hẳn một mục cho nó.
- **$k=1{,}345$ không phải con số tuỳ tiện, và nó KHÔNG có trong bài báo 1964.** Bài báo gốc
  không chứa chuỗi "1.345" ở bất kỳ trang nào. Nó là nghiệm của một phương trình suy ra từ
  công thức (1) của bài báo — mục 5.3 giải phương trình đó ra $k = 1{,}3449975085$.
- **Huber chống ngoại lai theo $y$, KHÔNG chống điểm xa theo $X$.** Đây là giới hạn quan
  trọng nhất, và mục 6.2 đo nó: thêm **một** điểm vào 60 điểm sạch làm hệ số góc đi từ
  $1{,}50055665$ xuống $-0{,}01336733$.
- **Bài báo gốc nói về ước lượng MỘT SỐ, không nói về hồi quy.** Tên đầy đủ của nó là *Robust
  Estimation of a Location Parameter*. Phần hồi quy là phần bài viết này tự bắc cầu, ở mục 3,
  và bắc cầu chỗ nào thì nói rõ chỗ đó.

### 0.2 Bảng ký hiệu

Mục 0.3 giải nghĩa các **từ**; bảng này giải nghĩa các **ký hiệu**, kèm cách đọc thành tiếng.
Bài này dùng nhiều chữ Hy Lạp hơn hầu hết bài trong nhóm, và người mới thường kẹt ở chỗ không
đọc được ký hiệu chứ không phải ở phép toán.

| Ký hiệu | Đọc là gì | Nghĩa trong bài báo | Chiều / kiểu |
|---|---|---|---|
| $n$, $p$ | "en", "pê" | số mẫu; số cột của $X$ | vô hướng, số nguyên |
| $x_1,\dots,x_n$ | "ích một tới ích en" | $n$ quan sát MỘT CHIỀU của bài toán vị trí | $n$ số |
| $X$ | "ích lớn" | bảng dữ liệu đầu vào của bài toán hồi quy | $\mathbb{R}^{n\times p}$ |
| $T$, $T_n$ | "tê", "tê en" | **ước lượng vị trí** — một số, chưa phải một đường | vô hướng |
| $\xi$ | **"kxi"** | tham số vị trí THẬT cần ước lượng | vô hướng |
| $\hat\beta$ | "bê-ta mũ" | hệ số hồi quy ước lượng | $\mathbb{R}^{p}$ |
| $e$, $e_i$, $r$ | "e", "e chỉ số i", "rờ" | phần dư $y-X\hat\beta$; ba tên cho một thứ | $\mathbb{R}^{n}$ |
| $\rho$ | **"rô"** | hàm mất mát — biến phần dư thành mức phạt | hàm $\mathbb{R}\to\mathbb{R}$ |
| $\psi$ | **"pờ-xi"** | $\psi=\rho'$ — **hàm ảnh hưởng**, nói một điểm kéo nghiệm mạnh tới đâu | hàm $\mathbb{R}\to\mathbb{R}$ |
| $k$ | "ca" | ngưỡng của $\psi$. Bài viết cũ của database gọi là $\delta$; scikit-learn gọi là `epsilon` | vô hướng, $k\ge0$ |
| $S$, $S_n$, $S_\infty$ | "ét", "ét en", "ét vô cùng" | thang đo của phần dư; giá trị tiệm cận của nó dưới $\Phi$ | vô hướng dương |
| $\epsilon$ | **"ép-xi-lon"** | **tỉ lệ nhiễm bẩn** của $F=(1-\epsilon)\Phi+\epsilon H$ | $0\le\epsilon<1$ |
| $\Phi$, $\varphi$ | "phi lớn", "phi nhỏ" | hàm phân phối và mật độ của phân phối chuẩn chuẩn hoá | hàm |
| $F$, $G$, $H$ | "ép", "gờ", "hát" | phân phối thật · phân phối "mẫu gốc" · phân phối gây nhiễm bẩn | hàm phân phối |
| $F_0$, $f_0$ | "ép không", "ép nhỏ không" | phân phối **ít thuận lợi nhất** và mật độ của nó | hàm |
| $V(\psi,F)$ | "vê của pxi và ép" | **phương sai tiệm cận** của $\sqrt n\,T_n$ | vô hướng |
| $I(F)$ | "i của ép" | **thông tin Fisher** $\int (f'/f)^2 f\,dt$ | vô hướng |
| $K(\psi,F)$ | "ca của pxi và ép" | tiền trả cho nhà thống kê trong trò chơi mục 8; $K=V^{-1}$ | vô hướng |
| $\lambda(\xi)$ | "lam-đa của kxi" | $\int\psi(t-\xi)F(dt)$ — hàm mà nghiệm làm nó triệt tiêu | hàm |
| $c$, $c(F)$ | "xê" | **độ chệch tiệm cận** gây ra bởi nhiễm bẩn không đối xứng | vô hướng |
| $\beta$ | "bê-ta" | **hằng số vế phải** của hệ (6) tr.97 — KHÔNG phải hệ số hồi quy | vô hướng |
| $a$, $b$, $c$ | "a", "bê", "xê" | ba hằng số của $f_0$ ở mục 9 tr.92 | vô hướng |
| $q$, $q'$ | "cu", "cu phẩy" | ngưỡng của bài toán **thang đo** ở mục 10-11 | vô hướng |
| $m_1,m_2,m_3$ | "em một/hai/ba" | số quan sát dưới · trong · trên ngưỡng, trong thuật toán (6′) | số nguyên |

**Toán tử và tên viết tắt:**

| Ký hiệu | Đọc là gì | Nghĩa |
|---|---|---|
| $\lvert e\rvert$ | "trị tuyệt đối của e" | bỏ dấu. Huber chỉ quan tâm phần dư **lớn tới đâu**, không quan tâm dấu |
| $\operatorname{clip}(e,-a,+a)$ | "cắt e về khoảng từ trừ a tới a" | giữ nguyên nếu $e$ trong khoảng, còn không thì kéo về mép gần nhất. Đây là **toàn bộ** cơ chế bền vững, viết bằng một hàm |
| $\operatorname{sign}(t)$ | "dấu của t" | $+1$ nếu $t>0$, $-1$ nếu $t<0$ |
| $\sup$, $\inf$ | "xúp", "in-phờ" | giá trị lớn nhất / nhỏ nhất có thể, kể cả khi không đạt được |
| $E_\Phi$ | "kỳ vọng dưới phi" | trung bình theo phân phối chuẩn |
| **MAD** | "em-a-đê" | *median absolute deviation* — trung vị của $\lvert r_i-\operatorname{median}(r)\rvert$; chia thêm $0{,}6745$ để trùng độ lệch chuẩn khi dữ liệu chuẩn |
| **(M)**, **(6)**, **(6′)** | "em", "sáu", "sáu phẩy" | nhãn công thức của chính bài báo, giữ nguyên để tra ngược |

> **Năm chỗ trùng tên phải nhớ, vì cả năm đều gây nhầm thật:**
>
> 1. **$k$ của bài báo = `epsilon` của scikit-learn = $\delta$ của phần lớn sách giáo khoa.**
>    Một thứ, ba tên. Gọi `HuberRegressor(epsilon=1.345)` là đang đặt $k=1{,}345$.
> 2. **$\epsilon$ của bài báo KHÔNG phải `epsilon` của scikit-learn.** Của bài báo là **tỉ lệ
>    nhiễm bẩn**; của thư viện là **ngưỡng $k$**. Hai khái niệm khác hẳn, và chúng trùng tên
>    chỉ vì thư viện chọn nhầm chữ.
> 3. **$\beta$ ở mục 2.14 KHÔNG phải hệ số hồi quy.** Nó là vế phải của hệ (6) tr.97, và nó
>    chính là chỗ scikit-learn khác bài báo — mục 4.3.
> 4. **$c$ có hai nghĩa ngay trong bài báo:** ở mục 7 tr.82 nó là **độ chệch tiệm cận**; ở
>    mục 9 tr.92 nó là **một trong ba hằng số của $f_0$**. Hai mục cách nhau mười trang.
> 5. **$S$ không do người dùng đặt.** Nó được ước lượng bên trong, và **cách ước lượng khác
>    nhau giữa các thư viện** — đó chính là nguồn của chỗ lệch $0{,}62\ \%$ ở mục 4.3.

### 0.3 Mười bốn từ cần biết trước khi đọc tiếp

Định nghĩa tại chỗ; không cần mở bài nào khác trước.

| Từ | Trong bài này | Nghĩa |
|---|---|---|
| **Bình phương tối thiểu** (least squares, OLS) | thứ Huber thay thế | chọn hệ số làm $\sum e_i^2$ nhỏ nhất |
| **Tham số vị trí** (location parameter) | thứ bài báo 1964 ước lượng | một số nói "đám dữ liệu nằm ở đâu" — trung bình và trung vị đều là ước lượng của nó |
| **Phần dư** (residual) | $e=y-\hat y$ | giá trị thật trừ dự đoán |
| **Ngoại lai theo $y$** | mục 1, mục 6.1 | điểm có giá trị đích khác xa xu hướng tại vị trí $X$ của nó |
| **Điểm xa theo $X$** (leverage) | mục 6.2 | điểm có $X$ nằm xa đám dữ liệu; nó xoay được cả đường hồi quy dù phần dư nhỏ |
| **Hàm mất mát** (loss) | $\rho$ | công thức biến phần dư thành "mức phạt"; đổi nó là đổi thuật toán |
| **Hàm ảnh hưởng** | $\psi=\rho'$ | nói một điểm **kéo** nghiệm mạnh tới đâu |
| **Bền vững** (robust) | tính chất Huber có, theo một nghĩa HẸP | kết quả không đổi nhiều khi một phần nhỏ dữ liệu bị hỏng — nhưng "hỏng" ở đây chỉ nghĩa là hỏng theo $y$ |
| **Nhiễm bẩn** (contamination) | $F=(1-\epsilon)\Phi+\epsilon H$ | phần lớn dữ liệu theo phân phối chuẩn, một tỉ lệ $\epsilon$ đến từ một phân phối $H$ **tuỳ ý và không biết** |
| **Phương sai tiệm cận** | $V$ | khi $n$ lớn, $\sqrt n\,(T_n-\xi)$ xấp xỉ chuẩn với phương sai $V$; $V$ nhỏ là ước lượng chính xác |
| **Minimax** | cách đặt bài toán của cả bài báo | chọn phương pháp làm cho **trường hợp xấu nhất** đỡ xấu nhất; không tối ưu cho trường hợp tốt |
| **Điểm yên ngựa** (saddlepoint) | Định lý 1, 2, 5 | một cặp $(\psi_0,F_0)$ mà không bên nào đơn phương đổi được có lợi — nghiệm của một trò chơi hai người |
| **Ít thuận lợi nhất** (least favorable) | $F_0$ | phân phối khó ước lượng nhất trong lớp đang xét; Huber chứng minh nó tồn tại và duy nhất |
| **Winsor hoá** (Winsorizing) | mục 2.2, mục 2.6 | kéo các quan sát ở hai đầu về mép rồi lấy trung bình cộng — Huber chứng minh nghiệm của mình chính là thế |

### 0.4 Ví dụ chạy tay — năm quan sát, từng bước, có dạng đóng

Bài toán của Huber 1964 là bài toán một chiều: cho $n$ số, tìm **một số** mô tả vị trí của
chúng. Lấy năm số, đặt thang đo $S=1$ cho gọn, và $k=1{,}345$:

Lấy $x = (1,\;2,\;4,\;5,\;25)$. Bốn số đầu nằm sát nhau, số thứ năm cách chúng rất xa —
đúng hình dạng của một bộ dữ liệu có một quan sát hỏng. Ba phương pháp cho ba câu trả lời khác
nhau cho cùng một câu hỏi *"đám số này nằm ở đâu"*:

| Phương pháp | $\rho(t)$ | Giá trị |
|---|---|---|
| Trung bình mẫu | $t^2$ | $7{,}4000000000$ |
| Trung vị mẫu | $\lvert t\rvert$ | $4{,}0000000000$ |
| **Huber, $k = 1{,}345$** | bình phương ở gần, tuyến tính ở xa | $\mathbf{3{,}8275000000}$ |

Điểm $25$ kéo trung bình lên $7{,}4$ — một giá trị **lớn hơn mọi quan sát trừ chính nó**.
Trung vị bỏ qua nó hoàn toàn. Huber nằm giữa, và dưới đây là lý do bằng số.

Ở nghiệm $T = 3{,}8275$, ngưỡng là $[T-k,\;T+k] = [2{,}4825;\;5{,}1725]$:

| $x_i$ | $r = x_i - T$ | $\psi(r)$ | $\rho(r)$ | Trong ngưỡng? |
|---:|---:|---:|---:|---|
| $1$ | $-2{,}8275000000$ | $-1{,}3450000000$ | $2{,}8984750000$ | **không** |
| $2$ | $-1{,}8275000000$ | $-1{,}3450000000$ | $1{,}5534750000$ | **không** |
| $4$ | $+0{,}1725000000$ | $+0{,}1725000000$ | $0{,}0148781250$ | có |
| $5$ | $+1{,}1725000000$ | $+1{,}1725000000$ | $0{,}6873781250$ | có |
| $25$ | $+21{,}1725000000$ | $+1{,}3450000000$ | $27{,}5725000000$ | **không** |
| | | $\sum\psi = 0$ | | |

Đọc cột $\psi$ là hiểu toàn bộ cơ chế. Quan sát $25$ lệch $21{,}17$ đơn vị; quan sát $1$ lệch
$2{,}83$ đơn vị. Chúng **kéo nghiệm mạnh bằng nhau**, vì cả hai đều đã chạm trần $\pm1{,}345$.
Với trung bình mẫu thì điểm $25$ kéo mạnh gấp $21{,}17/2{,}83 = 7{,}5$ lần.

**Nghiệm này giải được trên giấy, không cần máy.** Gọi $m_1$ là số điểm dưới ngưỡng, $m_2$ số
điểm trong ngưỡng, $m_3$ số điểm trên ngưỡng. Điều kiện $\sum\psi=0$ thành

$$\sum_{\text{trong}} (x_i - T) \;+\; (m_3 - m_1)\,k \;=\; 0
\qquad\Longrightarrow\qquad
T = \frac{\sum_{\text{trong}} x_i + (m_3 - m_1)\,k}{m_2}$$

Ở đây $m_1 = 2$ (hai điểm $1$ và $2$), $m_2 = 2$ (hai điểm $4$ và $5$), $m_3 = 1$ (điểm $25$),
nên

$$T = \frac{9 + (1-2)\times 1{,}345}{2} = \frac{7{,}655}{2} = 3{,}8275$$

đúng tới chữ số cuối, lệch $4{,}4\times10^{-16}$ so với nghiệm lặp.

**Và đây chính là "Winsor hoá" mà bài báo nói tới.** Thay mọi quan sát ngoài ngưỡng bằng mép
gần nhất:

$$x^{\text{Winsor}} = (2{,}4825;\; 2{,}4825;\; 4;\; 5;\; 5{,}1725)$$

Trung bình cộng của năm số đó là $3{,}82750000000000$ — **bằng đúng $T$**, lệch $0$. Bài báo
phát biểu điều đó ở tr.79 mục (iii) và chứng minh nó bằng hai dòng; mục 2.6 dán nguyên văn.

Hai trường hợp giới hạn cũng kiểm được ngay trên năm số này:

| Đặt $k$ bằng | Nghiệm Huber | Trùng với | Lệch |
|---|---|---|---|
| $10\,000$ | $7{,}4000000000$ | trung bình mẫu $7{,}4$ | $0$ |
| $0{,}0001$ | $4{,}0000000000$ | trung vị mẫu $4$ | $0$ |

Đó là câu tr.75 của bài báo: *"Sample mean ($k = \infty$) and sample median ($k = 0$) are
limiting cases corresponding to $\epsilon = 0$ and $\epsilon = 1$, respectively"*.

### 0.5 Nguồn, và một hạn chế phải nói ngay

Toàn bộ phần ruột của bài này đến từ **một** bài báo:

> P. J. Huber, *Robust Estimation of a Location Parameter*, The Annals of Mathematical
> Statistics **35**(1):73–101, 1964.
> Bản PDF trong repo: `AI algorithms/_papers/1964 - Huber - Robust Estimation of a Location Parameter.pdf`

Bài báo dài **29 trang**, gồm **12 mục đánh số**, **5 định lý**, **6 bổ đề**, **7 công thức
đánh số** cộng ba nhãn `(M)`, `(i)`–`(iii)`, `(6′)`, **4 bảng số** và **0 hình vẽ**. Bốn bảng
đánh số có $403 + 18 + 15 + 63 = 499$ ô; cộng hai bảng không đánh số ở tr.101 là **519 ô**.

**Hạn chế phải nói trước, vì nó đổi cách đọc mọi đoạn trích bên dưới.** Bản PDF trong repo là
**bản quét JSTOR**: lớp text của nó **rỗng ở 28 trên 29 trang**, trung bình $4{,}9$ ký tự mỗi
trang. Hệ quả cụ thể:

| Việc | Tình trạng |
|---|---|
| Đọc nội dung bài báo | **được** — bằng ảnh render, `pdffig.py` ở 300 DPI |
| Cắt hình ruột vào bài | **được** — 31 khung trong mục 2 |
| Đối chiếu đoạn trích tự động bằng `kiemtrich.py` | **KHÔNG được** — không có gì để so |
| `pdfscan.py --muc` liệt kê mục | **KHÔNG được** — trả về 0 dòng |
| `sopaper.py` đếm hình/bảng | **KHÔNG được** — báo 0 hình 0 bảng, con số đó vô nghĩa |

Nên mọi đoạn tiếng Anh trong bài này được **chép tay từ ảnh render**, và không phép kiểm nào
trong `tools/` xác nhận được chúng. Đó là một chỗ yếu có thật, đã ghi vào checklist cuối bài.
`kiemtrich.py` nay **báo ra** tình trạng đó thay vì báo đỏ hàng loạt — xem mục 9.4.

Mọi con số trong bài sinh từ [`tools/v_huber.py`](<../../../tools/v_huber.py>), chạy hết trong
**5,6 giây**. Chạy lại: `cd tools && python v_huber.py`.

---

## 1. Trước Huber người ta làm thế nào, và tắc ở đâu

### 1.1 Cách cũ: bình phương tối thiểu, và lý do nó được chọn

Bài báo mở đầu bằng một đoạn lịch sử hiếm thấy trong một bài toán học, và đoạn đó là chỗ đặt
vấn đề rõ nhất của cả bài:

> *"It is interesting to look back to the very origin of the theory of estimation, namely to
> Gauss and his theory of least squares. Gauss was fully aware that his main reason for
> assuming an underlying normal distribution and a quadratic loss function was mathematical,
> i.e., computational, convenience."*
> — tr.73, đoạn cuối

Dịch: *"Nhìn lại tận gốc của lý thuyết ước lượng, tức là Gauss và lý thuyết bình phương tối
thiểu của ông, thì thấy một điều đáng chú ý. Gauss thừa biết rằng lý do chính khiến ông giả
định phân phối chuẩn và hàm mất mát bậc hai là sự tiện lợi về mặt toán học, tức là về mặt
tính toán."*

Rồi sang trang sau, Huber nói thẳng cái giá:

> *"this was often forgotten, partly because of the central limit theorem. However, if one
> wants to be honest, the central limit theorem can at most explain why many distributions
> occurring in practice are approximately normal. The stress is on the word
> “approximately.”"*
> — tr.74, dòng 1–5

Dịch: *"Điều đó về sau thường bị quên, một phần vì định lý giới hạn trung tâm. Nhưng nói cho
sòng phẳng thì định lý giới hạn trung tâm nhiều lắm cũng chỉ giải thích được vì sao nhiều
phân phối gặp trong thực tế XẤP XỈ chuẩn. Trọng âm nằm ở chữ 'xấp xỉ'."*

### 1.2 Bế tắc, nêu bằng số

Câu tiếp theo của bài báo là câu nêu bế tắc, và nó là câu bài viết này phải đo:

> *"What happens if the true distribution deviates slightly from the assumed normal one? As
> is now well known, the sample mean then may have a catastrophically bad performance:
> seemingly quite mild deviations may already explode its variance."*
> — tr.74, dòng 6–10

Dịch: *"Chuyện gì xảy ra nếu phân phối thật lệch một chút so với phân phối chuẩn được giả
định? Như nay đã rõ, trung bình mẫu khi ấy có thể cho kết quả tệ thảm hại: những sai lệch
trông rất nhẹ cũng đủ làm phương sai của nó bùng nổ."*

**Đo câu đó trên bài toán vị trí, bằng chính Bảng I của bài báo.** Cột $\epsilon$ của Bảng I
cho phương sai tiệm cận lớn nhất mà trung bình mẫu ($k\to\infty$) và trung vị ($k=0$) phải
chịu, khi có một tỉ lệ $\epsilon$ dữ liệu đến từ một phân phối **bất kỳ**:

| $\epsilon$ | Trung bình mẫu | Trung vị ($k=0$, Bảng I) | Huber ở $k$ tối ưu |
|---:|---:|---:|---:|
| $0$ | $1{,}000000$ | $1{,}571$ | $1{,}000402$ |
| $0{,}001$ | $\infty$ | $1{,}574$ | $1{,}009548$ |
| $0{,}01$ | $\infty$ | $1{,}603$ | $1{,}065241$ |
| $0{,}05$ | $\infty$ | $1{,}741$ | $1{,}256124$ |
| $0{,}10$ | $\infty$ | $1{,}939$ | $1{,}489854$ |

Cột thứ hai là **vô cùng** ở mọi $\epsilon>0$, và đó không phải cách nói cho mạnh: $H$ được
phép là một phân phối bất kỳ, nên nó có thể đặt khối lượng ở xa tuỳ ý, và phương sai của
trung bình mẫu tỉ lệ thẳng với khoảng cách đó. Một phần nghìn dữ liệu hỏng là đủ.

**Đo câu đó trên bài toán hồi quy.** Lấy $n=100$ điểm sinh từ $y = x + \text{nhiễu chuẩn}$,
hệ số thật bằng $1$, rồi đẩy lệch **đúng một điểm**:

| Mức đẩy lệch | Bình phương tối thiểu | Huber, $k=1{,}345$ |
|---:|---:|---:|
| $+0$ | $1{,}1762358737$ | $1{,}1095543344$ |
| $+10$ | $1{,}1187880208$ | $1{,}1010663787$ |
| $+100$ | $0{,}6017573443$ | $1{,}1010663787$ |
| $+1000$ | $-4{,}5685494199$ | $1{,}1010559667$ |
| $+100\,000$ | $-573{,}3022934887$ | $1{,}1010559667$ |

Bình phương tối thiểu **đổi dấu** ở mức $+1000$ và tiếp tục đi xa theo mức phá dữ liệu — hệ
số của nó tỉ lệ tuyến tính với giá trị của điểm hỏng, không có trần. Huber đứng yên ở bốn
chữ số sau khi phần dư vượt ngưỡng: $1{,}1010663787$ ở $+100$ và $1{,}1010559667$ ở
$+100\,000$, chênh $1{,}0\times10^{-5}$ trong khi dữ liệu bị phá gấp một nghìn lần.

**Nhưng không nên đọc bảng đó thành "Huber bất biến".** Ba chữ số đầu không đổi vì $\psi$ đã
chạm trần; vị trí $x$ của điểm hỏng, cách ước lượng thang đo, và 99 điểm còn lại vẫn quyết
định nghiệm. Mục 6.2 dựng đúng một ví dụ mà Huber **sụp hoàn toàn** — và nó chỉ cần một điểm.

### 1.3 Điều bài báo đặt ra mà các đề xuất trước đó không có

Huber ghi công những người đi trước ngay ở đoạn đó:

> *"Tukey and others proposed several more robust substitutes—trimmed means, Winsorized
> means, etc.—and explored their performance for a few typical violations of normality. A
> general theory of robust estimation is still lacking; it is hoped that the present paper
> will furnish the first few steps toward such a theory."*
> — tr.74, dòng 10–15

Dịch: *"Tukey và một số người khác đã đề xuất vài thứ thay thế bền vững hơn — trung bình cắt
bớt, trung bình Winsor hoá, v.v. — và khảo sát chúng trên một vài kiểu vi phạm giả định chuẩn
điển hình. Một lý thuyết tổng quát về ước lượng bền vững thì vẫn chưa có; hy vọng bài này sẽ
là vài bước đầu tiên hướng tới một lý thuyết như thế."*

Chỗ khác nhau nằm ở hai chữ **"a few typical"**. Trước 1964, người ta chọn một phương pháp
rồi thử nó trên vài phân phối cụ thể. Huber đảo ngược: ông cố định **lớp** phân phối có thể
xảy ra, rồi **chứng minh** phương pháp nào tốt nhất trong trường hợp xấu nhất của lớp đó. Đó
là lý do bài báo có năm định lý chứ không có một bảng mô phỏng nào.

---

## 2. Phần ruột — đi theo đúng thứ tự bài báo

### 2.0 Bản đồ: bài báo có gì, và mục nào của bài này lấy nó

Bài báo không có hình vẽ, nên phần ruột của nó gồm **bốn bảng số**, **bảy công thức đánh số**
cộng bốn nhãn khác, và **các phát biểu định lý**. Bảng dưới là bản đồ đầy đủ; mục 9 sẽ đối
chiếu lại theo hai chiều.

| Mục bài báo | Trang | Nội dung | Vào mục nào của bài này |
|---|---|---|---|
| 1. Introduction and summary | 73–75 | định nghĩa `(M)`, $\rho$ bền vững nhất, $f_0$ | 2.1 · 2.2 · 2.3 |
| 2. Asymptotic normality of (M)-estimators for convex $\rho$ | 76–78 | Bổ đề 1–4, công thức **(1)** | 2.4 |
| 3. The case of non-convex $\rho$ | 78 | Bổ đề 5 | 2.5 |
| 4. Examples | 78–80 | bốn ví dụ (i)–(iv) và phương sai của chúng | 2.6 |
| 5. Minimax questions; a special case | 80–82 | **Định lý 1** — điểm yên ngựa | 2.7 |
| 6. The contaminated normal distribution | 82 | **Hệ quả**, quan hệ $k\leftrightarrow\epsilon$ | 2.8 |
| — | 84–85 | **BẢNG I**, 403 ô | 2.9 |
| 7. The question of bias | 82–83 | công thức **(2)**, ba cận chệch | 2.10 |
| 8. Minimax theory | 86–90 | $K(\psi,F)$, Bổ đề 6, **Định lý 2, 3, 4** | 2.11 |
| 9. The minimax solution for $\epsilon$-normal distributions | 91–93 | **Định lý 5**, $f_0$, ba điều kiện `(i)(ii)(iii)`, **BẢNG II** | 2.12 |
| 10. Estimation of a scale parameter | 93–94 | công thức **(3)**, **(4)** | 2.13 |
| 11. Estimation of a location parameter, if scale and $\epsilon$ are unknown | 95–99 | công thức **(5)**, **BẢNG III**, ba Đề xuất, **(6)**, **(6′)**, **(7)**, **BẢNG IV** | 2.14 |
| 12. A generalization | 99–101 | $(M_r)$, Hodges–Lehmann, hai bảng không đánh số | 2.15 |
| References | 101 | 11 tài liệu | 10 |

### 2.1 Mục 1 tr.74 — định nghĩa (M), thứ mà mọi thứ còn lại dựng trên

![Định nghĩa (M) trong bản in gốc](<paper_huber1964_tr74_dinh_nghia_M.png>)

**Hình 1 — Định nghĩa (M), Huber 1964, trang 74.** Nguồn:
`AI algorithms/_papers/1964 - Huber - Robust Estimation of a Location Parameter.pdf`, trang 2
của PDF. Cắt bằng `python pdffig.py "<paper>.pdf" 2 --box 0.12,0.53,0.92,0.72 --dpi 300 --sat`.

**Nguyên văn của tác giả, chép từ ảnh:**

> *"It is quite natural to ask whether one can obtain more robustness by minimizing another
> function of the errors than the sum of their squares. We shall therefore concentrate our
> attention to estimators that can be defined by a minimum principle of the form (for a
> location parameter):"*
>
> $$T = T_n(x_1,\dots,x_n)\ \textit{minimizes}\ \sum_i\rho(x_i - T),
> \qquad\textit{where }\rho\textit{ is a non-constant function.}\tag{M}$$
>
> *"Of course, this definition generalizes at once to more general least squares type
> problems, where several parameters have to be determined."*
>
> *"This class of estimators contains in particular (i) the sample mean
> ($\rho(t) = t^2$), (ii) the sample median ($\rho(t) = \lvert t\rvert$), and more generally,
> (iii) all maximum likelihood estimators ($\rho(t) = -\log f(t)$, where $f$ is the assumed
> density of the untranslated distribution)."*

Dịch: *"Rất tự nhiên mà hỏi rằng liệu có thể đạt độ bền vững cao hơn bằng cách cực tiểu hoá
một hàm khác của sai số, thay vì tổng bình phương của chúng hay không. Vì vậy chúng tôi tập
trung vào các ước lượng định nghĩa được bằng một nguyên lý cực tiểu có dạng (cho một tham số
vị trí): $T$ cực tiểu hoá $\sum_i\rho(x_i-T)$, trong đó $\rho$ là một hàm không hằng. Tất
nhiên định nghĩa này mở rộng ngay lập tức sang các bài toán kiểu bình phương tối thiểu tổng
quát hơn, nơi phải xác định nhiều tham số. Lớp ước lượng này chứa riêng (i) trung bình mẫu
($\rho(t)=t^2$), (ii) trung vị mẫu ($\rho(t)=\lvert t\rvert$), và tổng quát hơn, (iii) mọi
ước lượng hợp lý cực đại ($\rho(t)=-\log f(t)$, với $f$ là mật độ giả định của phân phối chưa
dịch chuyển)."*

**Ký hiệu, giải nghĩa từng thứ:**

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $x_1,\dots,x_n$ | $n$ số thực | các quan sát, độc lập cùng phân phối $F(t-\xi)$ |
| $T_n$ | một số | ước lượng — một hàm của toàn bộ mẫu |
| $\rho$ | hàm $\mathbb{R}\to\mathbb{R}$ | hàm mất mát, **không hằng**, ngoài ra tự do |
| $\sum_i\rho(x_i-T)$ | một số | tổng mức phạt, hàm của $T$ |

**Ý nghĩa trừu tượng.** Công thức `(M)` **không phải một thuật toán** — nó là một **khuôn**.
Nó nói: mọi ước lượng vị trí mà ta quan tâm đều có thể viết thành "chọn $T$ làm nhỏ nhất một
tổng nào đó", và việc chọn phương pháp thu về việc chọn **một hàm** $\rho$. Đó là bước đổi
bài toán quan trọng nhất của cả bài báo: từ *"nên dùng trung bình hay trung vị?"* — một câu
hỏi có hai lựa chọn — sang *"$\rho$ nào là tốt nhất?"* — một câu hỏi tối ưu trên một **không
gian hàm**, và vì thế có thể có lời giải chứng minh được.

Câu *"generalizes at once to more general least squares type problems, where several
parameters have to be determined"* là **câu duy nhất** trong cả 29 trang nói tới hồi quy. Mục
3 của bài này nói rõ nó cho được gì và không cho được gì.

### 2.2 Mục 1 tr.75 — $\rho$ bền vững nhất, in thẳng ra

![Đoạn tr.75 nêu rho bền vững nhất](<paper_huber1964_tr75_rho_ben_vung_nhat.png>)

**Hình 2 — $\rho$ bền vững nhất, Huber 1964, trang 75.** Cùng nguồn, trang 3 của PDF.
`--box 0.12,0.31,0.92,0.45`.

**Nguyên văn:**

> *"If we adopt this measure of robustness, and if we restrict attention to $(M)$-estimators,
> then it will be shown that the most robust estimator is uniquely determined and corresponds
> to the following $\rho$: $\rho(t) = \tfrac12 t^2$ for $\lvert t\rvert < k$,
> $\rho(t) = k\lvert t\rvert - \tfrac12 k^2$ for $\lvert t\rvert \ge k$, with $k$ depending on
> $\epsilon$. This estimator is most robust even among all translation invariant estimators.
> Sample mean ($k = \infty$) and sample median ($k = 0$) are limiting cases corresponding to
> $\epsilon = 0$ and $\epsilon = 1$, respectively, and the estimator is closely related and
> asymptotically equivalent to Winsorizing."*

Dịch: *"Nếu nhận thước đo độ bền vững này, và nếu chỉ xét các ước lượng $(M)$, thì sẽ chứng
minh được rằng ước lượng bền vững nhất là duy nhất và ứng với $\rho$ sau: $\rho(t)=\frac12t^2$
khi $\lvert t\rvert<k$, $\rho(t)=k\lvert t\rvert-\frac12k^2$ khi $\lvert t\rvert\ge k$, với
$k$ phụ thuộc $\epsilon$. Ước lượng này bền vững nhất ngay cả trong số mọi ước lượng bất biến
với phép tịnh tiến. Trung bình mẫu ($k=\infty$) và trung vị mẫu ($k=0$) là hai trường hợp giới
hạn ứng với $\epsilon=0$ và $\epsilon=1$, và ước lượng này liên hệ chặt chẽ, đồng thời tương
đương tiệm cận, với phép Winsor hoá."*

Viết lại cho gọn, đây là **hàm mất mát Huber**:

$$\rho_k(t) = \begin{cases}
\tfrac12 t^2 & \lvert t\rvert \le k\\[4pt]
k\left(\lvert t\rvert - \tfrac12 k\right) & \lvert t\rvert > k
\end{cases}
\qquad\qquad
\psi_k(t) = \rho_k'(t) = \operatorname{clip}(t, -k, +k)$$

| Tính chất | Vì sao quan trọng |
|---|---|
| Liên tục và **khả vi một lần** tại $\pm k$ | tối ưu hoá trơn; khác hẳn $\lvert t\rvert$ của trung vị, vốn gãy tại 0 |
| **Lồi** | mọi cực tiểu cục bộ là cực tiểu toàn cục — đây là điều kiện của cả mục 2 bài báo |
| $\rho'$ **bị chặn** bởi $k$ | đây là nguồn duy nhất của tính bền vững |
| $\rho \to \tfrac12 t^2$ khi $k\to\infty$ | quay về bình phương tối thiểu |
| $\rho \to k\lvert t\rvert$ khi $k\to0$ | quay về trung vị |

**Ý nghĩa trừu tượng, và nó không phải là đọc lại công thức.** Hai nhánh của $\rho_k$ trả lời
hai câu hỏi khác nhau. Nhánh bình phương ở giữa nói: *khi phần dư nhỏ, hãy tin nó và dùng hết
thông tin trong nó* — đó chính là hành vi tối ưu dưới phân phối chuẩn. Nhánh tuyến tính ở xa
nói: *khi phần dư lớn, hãy ghi nhận dấu của nó và thôi* — vì một phần dư lớn không nói cho ta
biết nó lớn **vì dữ liệu thật hiếm** hay **vì quan sát đó hỏng**, và hai khả năng ấy đòi hai
phản ứng ngược nhau. Điểm nối $k$ là chỗ người dùng khai báo mình tin đến đâu.

Chữ *"uniquely determined"* là chữ đắt nhất của đoạn: Huber không nói "$\rho$ này tốt", ông
nói **không có $\rho$ nào khác**. Chứng minh nằm ở Định lý 1 và Định lý 2, mục 2.7 và 2.11.

### 2.3 Mục 1 tr.75 — $f_0$ có đuôi MŨ, và tác giả nói thẳng là mình bất ngờ

![Đoạn tr.75 về f_0](<paper_huber1964_tr75_f0_duoi_mu.png>)

**Hình 3 — Phân phối ít thuận lợi nhất $f_0$, Huber 1964, trang 75.** Cùng nguồn, trang 3.
`--box 0.12,0.55,0.92,0.67`.

**Nguyên văn:**

> *"In fact, this $(M)$-estimator is the maximum likelihood estimator corresponding to a
> unique least favorable distribution $F_0$ with density
> $f_0(t) = (1 - \epsilon)(2\pi)^{-1/2}e^{-\rho(t)}$. This $f_0$ behaves like a normal density
> for small $t$, like an exponential density for large $t$. At least for me, this was rather
> surprising—I would have expected an $f_0$ with much heavier tails."*

Dịch: *"Thật ra ước lượng $(M)$ này chính là ước lượng hợp lý cực đại ứng với một phân phối ít
thuận lợi nhất DUY NHẤT $F_0$, có mật độ $f_0(t)=(1-\epsilon)(2\pi)^{-1/2}e^{-\rho(t)}$. Mật
độ $f_0$ này hành xử như mật độ chuẩn khi $t$ nhỏ, và như mật độ mũ khi $t$ lớn. Ít nhất là
với tôi thì điều đó khá bất ngờ — tôi đã chờ đợi một $f_0$ có đuôi nặng hơn nhiều."*

Câu cuối là một câu hiếm trong văn bản toán, và nó đáng giữ nguyên: tác giả **ghi lại chỗ trực
giác của chính mình sai**. Trực giác nói rằng phân phối khó nhất phải là phân phối có đuôi rất
nặng, kiểu Cauchy. Phép chứng minh nói không: khó nhất là một phân phối chuẩn **bị gắn đuôi
mũ**, tức đuôi nặng hơn chuẩn nhưng nhẹ hơn Cauchy rất nhiều.

**Ba phép kiểm cho $f_0$, chạy trên năm cặp $(k,\epsilon)$ khớp nhau theo Hệ quả tr.82:**

| $k$ | $\epsilon_{\min}$ | $\int f_0\,dt$ | $\min h_0$ | $V(\psi_0,F_0)\cdot I(F_0)$ |
|---:|---:|---:|---:|---:|
| $0{,}5000$ | $0{,}441711$ | $1{,}000000000000$ | $0{,}000\times10^{0}$ | $1{,}000000000000$ |
| $1{,}0000$ | $0{,}142831$ | $1{,}000000000000$ | $0{,}000\times10^{0}$ | $1{,}000000000000$ |
| $1{,}3450$ | $0{,}057915$ | $1{,}000000000000$ | $0{,}000\times10^{0}$ | $1{,}000000000000$ |
| $2{,}0000$ | $0{,}008419$ | $1{,}000000000000$ | $0{,}000\times10^{0}$ | $1{,}000000000000$ |
| $2{,}6400$ | $0{,}001008$ | $1{,}000000000000$ | $0{,}000\times10^{0}$ | $1{,}000000000000$ |

Ba cột kiểm ba điều khác nhau, và cả ba đều là điều bài báo khẳng định:

1. **$\int f_0 = 1$** — $f_0$ thật sự là một mật độ, không phải một biểu thức hình thức.
2. **$\min h_0 \ge 0$** — phần nhiễm bẩn $h_0 = (f_0 - (1-\epsilon)\varphi)/\epsilon$ là một
   mật độ hợp lệ. Bài báo dành nguyên một đoạn ở tr.81 để chứng minh đúng điều này
   (*"it remains to check that its density $h_0$ is nonnegative"*), và nó là chỗ phép chứng
   minh có thể hỏng: nếu $h_0$ âm ở đâu đó thì $F_0$ không nằm trong lớp $C$ và cả định lý sụp.
3. **$V(\psi_0,F_0)\cdot I(F_0) = 1$** — đẳng thức $V = 1/I$ của tr.81, tức ước lượng đạt
   đúng cận Cramér–Rao **tại** $F_0$. Đó là nghĩa chính xác của câu "hợp lý cực đại cho $F_0$".

---

### 2.4 Mục 2 tr.76–78 — bốn bổ đề, và công thức (1)

Mục 2 của bài báo làm một việc kỹ thuật nhưng bắt buộc: chứng minh rằng ước lượng `(M)` với
$\rho$ **lồi** thì nhất quán và tiệm cận chuẩn, rồi viết ra phương sai tiệm cận của nó. Không
có bước này thì cả phần minimax phía sau không có đại lượng nào để tối ưu.

| Bổ đề | Trang | Phát biểu, rút gọn | Nó dùng để làm gì |
|---|---|---|---|
| 1 | 76 | $Q(\xi)=\sum\rho(x_i-\xi)$ lồi; tập cực tiểu $[T_n(x)]$ khác rỗng, lồi, compact; $\rho$ lồi ngặt thì nó là **một điểm** | bảo đảm "nghiệm" có nghĩa |
| 2 | 76 | $\lambda(\xi)=\int\psi(t-\xi)F(dt)$ tồn tại khắp nơi, **giảm đơn điệu**, dương ở $\xi$ rất âm và âm ở $\xi$ rất dương | bảo đảm $\lambda$ cắt trục hoành đúng một lần |
| 3 | 76–77 | *"Consistency of $T_n$"* — nếu $\lambda(\xi)>0$ với $\xi<c$ và $<0$ với $\xi>c$ thì $T_n\to c$ hầu chắc chắn | nghiệm hội tụ về đâu |
| 4 | 77 | *"Asymptotic normality"* — $\sqrt n\,(T_n-c)$ tiệm cận chuẩn, trung bình $0$, phương sai $V(\psi,F)=\int\psi^2(t-c)F(dt)/(\lambda'(c))^2$ | **đại lượng để tối ưu** |

Bổ đề 4 là bổ đề cả bài báo dựa vào. Nó nói: mọi ước lượng `(M)` với $\rho$ lồi đều có một con
số $V$ gắn với nó, và **tối ưu hoá $V$ là tối ưu hoá phương pháp**.

![Công thức (1) trang 78](<paper_huber1964_tr78_cong_thuc_1.png>)

**Hình 4 — Công thức (1), Huber 1964, trang 78.** Cùng nguồn, trang 6 của PDF.
`--box 0.12,0.33,0.92,0.45`.

**Nguyên văn và công thức:**

> *"The Schwarz inequality now yields the inequality"*
>
> $$V(\psi, F) = \frac{E\psi^2}{(E\psi')^2}
> = \frac{\int\psi^2 f\,dt}{\left[\int\psi\,(f'/f)\,f\,dt\right]^2}
> \;\ge\; \frac{1}{\int (f'/f)^2 f\,dt}\tag{1}$$
>
> *"We have strict equality unless $\psi = -pf'/f$, for some constant $p$, that is, unless
> $f(t) = \mathrm{const.}\exp(-\rho(t)/p)$ and then the $(M)$-estimator is the maximum
> likelihood estimator."*

Dịch: *"Bất đẳng thức Schwarz nay cho bất đẳng thức [công thức 1]. Ta có bất đẳng thức ngặt
trừ khi $\psi=-pf'/f$ với một hằng số $p$ nào đó, tức là trừ khi $f(t)=\text{hằng}\cdot
\exp(-\rho(t)/p)$, và khi đó ước lượng $(M)$ chính là ước lượng hợp lý cực đại."*

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $E\psi^2 = \int\psi^2 f\,dt$ | số dương | **phương sai** của $\psi$ tại nghiệm — mức dao động mỗi quan sát bơm vào |
| $E\psi' = \int\psi' f\,dt = -\lambda'(0)$ | số dương | **độ nhạy**: dịch tham số một đơn vị thì phương trình ước lượng đổi bao nhiêu |
| $\int(f'/f)^2 f\,dt$ | số dương | **thông tin Fisher** $I(F)$ |
| $p$ | số | hằng số tỉ lệ tuỳ ý — $\psi$ nhân với hằng số không đổi nghiệm |

**Ý nghĩa trừu tượng.** Công thức (1) là một tỉ số **nhiễu trên tín hiệu**, viết cho ước lượng
thay vì cho một kênh truyền. Tử số là nhiễu: mỗi quan sát đẩy phương trình ước lượng đi một
lượng ngẫu nhiên $\psi(x_i-T)$, và tổng các cú đẩy ấy dao động quanh 0 với phương sai
$E\psi^2$. Mẫu số là tín hiệu: nếu tham số thật dịch đi một chút thì toàn bộ phương trình dịch
theo, và tốc độ dịch là $E\psi'$. Ước lượng chính xác khi cú đẩy ít dao động **và** phương
trình nhạy.

Vế phải của (1) nói rằng tỉ số ấy có **trần dưới** không phụ thuộc $\psi$: thông tin Fisher.
Không $\psi$ nào vượt được. Và đẳng thức chỉ xảy ra khi $\psi$ tỉ lệ với $-f'/f$ — tức khi ước
lượng $(M)$ tình cờ **chính là** hợp lý cực đại cho phân phối thật. Đó là lý do toàn bộ phần
còn lại của bài báo đi tìm một phân phối $F_0$ rồi lấy hợp lý cực đại của nó: đó là cách duy
nhất đạt đẳng thức, và mục tiêu là đạt đẳng thức **tại chỗ xấu nhất**.

**Kiểm bằng số dưới $F=\Phi$, với $\psi=\operatorname{clip}(t,-k,k)$:**

| $k$ | $E_\Phi\psi^2$ | $E_\Phi\psi'$ | $V$ | $V \ge 1$? |
|---:|---:|---:|---:|---|
| $0{,}500$ | $0{,}1851283651$ | $0{,}3829249225$ | $1{,}2625416081$ | có |
| $1{,}000$ | $0{,}5160585510$ | $0{,}6826894921$ | $1{,}1072674722$ | có |
| $1{,}345$ | $0{,}7101645483$ | $0{,}8213747654$ | $1{,}0526312912$ | có |
| $2{,}000$ | $0{,}9205369256$ | $0{,}9544997361$ | $1{,}0103912784$ | có |
| $5{,}000$ | $0{,}9999988921$ | $0{,}9999994267$ | $1{,}0000000387$ | có |
| $20{,}000$ | $1{,}0000000000$ | $1{,}0000000000$ | $1{,}0000000000$ | có |

Thông tin Fisher của $\Phi$ bằng $1$, nên cận dưới là $1$, và cột $V$ chạm nó **chỉ khi
$k\to\infty$**. Đó đúng là câu *"unless $\psi=-pf'/f$"* của bài báo, vì với mật độ chuẩn thì
$-\varphi'/\varphi = t$, tức $\psi$ không bị cắt.

### 2.5 Mục 3 tr.78 — khi $\rho$ KHÔNG lồi, Bổ đề 5

Mục 3 dài nửa trang và nó nói một điều tiêu cực:

> *"If $\rho$ is not convex, the estimators $T_n$ will, in general, no longer converge toward
> some constant $c$. Apparently, one has to impose not only local but also some global
> conditions on $\rho$ in order to have consistency."*
> — tr.78

Dịch: *"Nếu $\rho$ không lồi thì nói chung các ước lượng $T_n$ sẽ không còn hội tụ về một hằng
số $c$ nào. Rõ ràng là phải áp không chỉ điều kiện địa phương mà cả điều kiện toàn cục lên
$\rho$ thì mới có tính nhất quán."*

Bổ đề 5 vớt lại phần tiệm cận chuẩn dưới **giả thiết mạnh hơn**: nếu đã biết $T_n\to0$ theo
xác suất, $\psi$ liên tục và có đạo hàm liên tục đều, $E\psi^2<\infty$ và $0<E\psi'<\infty$,
thì $\sqrt n\,T_n$ tiệm cận chuẩn với phương sai $E\psi^2/(E\psi')^2$ — **cùng công thức** như
Bổ đề 4, nhưng nay phải giả định tính nhất quán thay vì chứng minh nó.

**Vì sao mục này đáng có trong bài viết dù nó không dùng tới về sau.** Nó là chỗ bài báo tự
khoanh vùng: mọi kết quả mạnh của Huber đòi $\rho$ **lồi**. Các hàm mất mát bền vững hơn ra
đời sau — Tukey biweight, Hampel — đều **không lồi**, và chúng đổi lấy độ bền cao hơn bằng
đúng thứ mục 3 cảnh báo: mất bảo đảm hội tụ, và nghiệm phụ thuộc điểm khởi đầu.

### 2.6 Mục 4 tr.78–80 — bốn ví dụ, và phương sai của từng cái

![Ví dụ (iii) trang 79](<paper_huber1964_tr79_vi_du_iii.png>)

**Hình 5 — Ví dụ (iii) và (iv), Huber 1964, trang 79.** Cùng nguồn, trang 7 của PDF.
`--box 0.12,0.17,0.92,0.47`.

**Nguyên văn của ví dụ (iii) — đây là chỗ bài báo chứng minh tính chất Winsor hoá:**

> *"(iii) $\rho(t) = \tfrac12 t^2$ for $\lvert t\rvert \le k$,
> $\rho(t) = k\lvert t\rvert - \tfrac12 k^2$ for $\lvert t\rvert > k$. The corresponding
> estimator is closely related to Winsorizing, since it has the following property: if we put
> for short $t_0 = T_n(x)$, then all observations $x_i$ such that
> $\lvert x_i - t_0\rvert > k$ may be replaced by $t_0 \pm k$, whichever is nearer, without
> changing the estimate $T_n(x)$, and $t_0$ equals the arithmetic mean of the modified set of
> observations."*

Dịch: *"(iii) [công thức $\rho$]. Ước lượng tương ứng liên hệ chặt chẽ với phép Winsor hoá, vì
nó có tính chất sau: đặt cho gọn $t_0=T_n(x)$, thì mọi quan sát $x_i$ thoả
$\lvert x_i-t_0\rvert>k$ đều có thể thay bằng $t_0\pm k$, cái nào gần hơn, mà KHÔNG làm đổi
ước lượng $T_n(x)$; và $t_0$ bằng đúng trung bình cộng của tập quan sát đã sửa."*

Mục 0.4 của bài này đã kiểm đúng câu đó trên năm số, lệch $0$.

**Bốn ví dụ và phương sai tiệm cận của chúng dưới $F=\Phi$:**

| | $\rho(t)$ | Ước lượng | $V$ dưới $\Phi$, ở $k=1{,}345$ |
|---|---|---|---:|
| (i) | $t^2$ | trung bình mẫu | $1{,}0000000000$ |
| (ii) | $\lvert t\rvert$ | trung vị mẫu | $1{,}5707963268\;(=\pi/2)$ |
| (iii) | Huber | Winsor hoá | $\mathbf{1{,}0526312912}$ |
| (iv) | $\tfrac12t^2$ rồi **phẳng** | trung bình cắt bớt | $2{,}5837988408$ |

Bài báo viết công thức $V$ cho (iii) ở dạng tổng quát theo $F$:

$$V = \frac{\displaystyle\int_{-k}^{+k} t^2\,F(dt) + k^2F(-k) + k^2\bigl(1-F(k)\bigr)}
{\left(\displaystyle\int_{-k}^{+k} F(dt)\right)^{\!2}}$$

Đây là công thức sinh ra **toàn bộ Bảng I**, nên nó là công thức quan trọng nhất của mục 4.
Tử số cộng hai phần: phần trong ngưỡng đóng góp bình phương phần dư thật, phần ngoài ngưỡng
đóng góp $k^2$ **bất kể xa bao nhiêu**. Mẫu số là xác suất rơi vào trong ngưỡng, bình phương.

Với (iv), bài báo tự gắn một cảnh báo ngay dưới công thức:

> *"(However, one should be reminded that we did not prove the formula in this case, $\rho$
> not being convex. I conjecture that it is valid if and only if the expression inside the
> parentheses of the denominator is strictly positive.)"*

Dịch: *"(Tuy nhiên cần nhắc rằng chúng tôi KHÔNG chứng minh công thức trong trường hợp này, vì
$\rho$ không lồi. Tôi phỏng đoán rằng nó đúng khi và chỉ khi biểu thức trong ngoặc ở mẫu số là
dương ngặt.)"* — một lời tự khai giới hạn, ở ngay cạnh con số.

**Bốn nhận xét của chính tác giả ở tr.79–80, đánh số (i)–(iv), chép nguyên văn:**

> (i) *"The sample mean is very sensitive to the tails of $F$."*
>
> (ii) *"The sample median is very sensitive to the behavior of $F$ at its median, and
> neglects its behavior elsewhere."*
>
> (iii) *"“Winsorizing” avoids these shortcomings and seems to be practically foolproof.
> Apparently, this is connected with the fact that the corresponding $\psi$ is monotone,
> bounded and absolutely continuous."*
>
> (iv) *"The “trimming procedure” is rather sensitive to the behavior of $F$ at the
> rejection points $\pm k$; a high density at these points will play havoc with the estimate."*

Bốn câu này là **chẩn đoán**, không phải kết quả: chúng nói *vì sao* mỗi phương pháp hỏng, chứ
không chỉ nói nó hỏng bao nhiêu. Câu (iii) nêu ba tính chất — $\psi$ **đơn điệu, bị chặn, liên
tục tuyệt đối** — và ba tính chất đó chính là ba điều kiện kỹ thuật mà các định lý ở mục 5, 8,
9 đòi hỏi. Câu (iv) giải thích vì sao cắt bớt tệ hơn Winsor hoá dù nghe giống nhau: cắt bớt
làm $\psi$ **nhảy** xuống $0$ tại $\pm k$, nên số quan sát rơi sát hai điểm đó quyết định
nghiệm.

### 2.7 Mục 5 tr.80–82 — Định lý 1, điểm yên ngựa

![Định lý 1 trang 80](<paper_huber1964_tr80_dinh_ly_1.png>)

**Hình 6 — Định lý 1, Huber 1964, trang 80.** Cùng nguồn, trang 8 của PDF.
`--box 0.12,0.72,0.92,0.93`.

Trước khi phát biểu định lý, bài báo giải thích vì sao đi hỏi câu minimax:

> *"At first glance, it might seem absurd to look for asymptotic minimax solutions, since,
> asymptotically, one could do better by estimating the true underlying distribution... However,
> this seems to require an exorbitant number of observations."*
> — tr.80

Dịch: *"Thoạt nhìn thì việc đi tìm nghiệm minimax tiệm cận có vẻ vô lý, vì về mặt tiệm cận,
người ta có thể làm tốt hơn bằng cách ước lượng chính phân phối thật... Nhưng cách đó có vẻ
đòi hỏi một số lượng quan sát khổng lồ."*

**Phát biểu Định lý 1, nguyên văn:**

> *"THEOREM 1. The asymptotic variance $V(\psi, F)$ has a saddlepoint: there is an
> $F_0 = (1 - \epsilon)G + \epsilon H_0$ and a $\psi_0$ such that*
>
> $$\sup_F V(\psi_0, F) = V(\psi_0, F_0) = \inf_\psi V(\psi, F_0)$$
>
> *where $F$ ranges over those distributions in $C$ for which $E_F\psi_0 = 0$."*

Dịch: *"ĐỊNH LÝ 1. Phương sai tiệm cận $V(\psi,F)$ có một điểm yên ngựa: tồn tại một
$F_0=(1-\epsilon)G+\epsilon H_0$ và một $\psi_0$ sao cho [đẳng thức trên], trong đó $F$ chạy
trên những phân phối thuộc $C$ mà $E_F\psi_0=0$."*

![Mật độ f_0 tường minh, trang 81](<paper_huber1964_tr81_f0_tuong_minh.png>)

**Hình 7 — $f_0$ tường minh và quan hệ $k\leftrightarrow\epsilon$, Huber 1964, trang 81.**
Cùng nguồn, trang 9 của PDF. `--box 0.12,0.10,0.92,0.36`.

Phần sau của định lý cho $f_0$ **viết ra được**, với $t_0<t_1$ là hai đầu mút của khoảng mà
$\lvert g'/g\rvert\le k$:

$$(1-\epsilon)^{-1} = \int_{t_0}^{t_1} g(t)\,dt + \frac{g(t_0)+g(t_1)}{k}$$

$$f_0(t) = \begin{cases}
(1-\epsilon)\,g(t_0)\,e^{k(t-t_0)} & t \le t_0\\[2pt]
(1-\epsilon)\,g(t) & t_0 < t < t_1\\[2pt]
(1-\epsilon)\,g(t_1)\,e^{-k(t-t_1)} & t \ge t_1
\end{cases}$$

Đọc ba nhánh ấy như một mệnh lệnh cắt dán: **giữ nguyên phần giữa của $g$, thay hai đuôi bằng
hai hàm mũ**, và chỗ nối đặt đúng tại điểm mà độ dốc log của $g$ chạm $k$. Phân phối khó ước
lượng nhất không phải một phân phối kỳ dị nào — nó là chính phân phối quen thuộc, bị nối thêm
hai cái đuôi nặng vừa đủ mức mà lớp $C$ cho phép.

> *"$\psi_0 = -f_0'/f_0$ is monotone and bounded and corresponds to the maximum likelihood
> estimator of the location parameter when $F_0$ is the underlying distribution."*

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $G$, $g$ | phân phối, mật độ | phần "sạch" của dữ liệu, cố định; $-\log g$ phải lồi |
| $H_0$, $h_0$ | phân phối, mật độ | phần nhiễm bẩn **xấu nhất**, do định lý sinh ra |
| $t_0$, $t_1$ | số | hai đầu mút của vùng "tin được"; với $G=\Phi$ thì $t_0=-k$, $t_1=+k$ |
| $k$ | số dương | ngưỡng, xác định qua $\epsilon$ bằng phương trình trên |

**Ý nghĩa trừu tượng.** Định lý 1 là chỗ bài toán *"chọn phương pháp"* biến thành bài toán
*"tìm điểm cân bằng của một trò chơi"*. Hai người chơi: nhà thống kê chọn $\psi$, thiên nhiên
chọn $F$ trong lớp nhiễm bẩn. Đẳng thức ba vế nói rằng có một cặp $(\psi_0,F_0)$ mà **không
bên nào đơn phương đổi được có lợi** — thiên nhiên đã chơi nước xấu nhất, nhà thống kê đã trả
lời tối ưu. Giá trị $V(\psi_0,F_0)$ vì thế là một **bảo đảm**: dùng $\psi_0$ thì phương sai
không bao giờ vượt con số đó, dù thiên nhiên làm gì.

Dạng của $f_0$ nói thêm một điều trực quan: nó **giữ nguyên phần giữa** của $g$ và **thay hai
đuôi bằng đuôi mũ**. Chỗ nối là đúng chỗ $\lvert g'/g\rvert$ chạm $k$. Nói cách khác, phân
phối khó nhất là phân phối *"giống chuẩn ở giữa, nhưng nặng hơn đúng mức cho phép ở hai đuôi"*.

### 2.8 Mục 6 tr.82 — Hệ quả cho phân phối chuẩn bị nhiễm bẩn

![Hệ quả trang 82](<paper_huber1964_tr82_he_qua.png>)

**Hình 8 — Hệ quả, Huber 1964, trang 82.** Cùng nguồn, trang 10 của PDF.
`--box 0.12,0.34,0.92,0.63`.

**Nguyên văn:**

> *"COROLLARY. Define the estimator $T_n = T_n(x_1,\dots,x_n)$ by the property that it
> minimizes $\sum_i\rho(x_i - T_n)$, where $\rho(t) = \tfrac12 t^2$ for $\lvert t\rvert < k$,
> $\rho(t) = k\lvert t\rvert - \tfrac12 k^2$ for $\lvert t\rvert \ge k$. Let $\psi = \rho'$.
> For symmetric $H$, $n^{1/2}T_n$ is asymptotically normal with asymptotic mean 0, and attains
> the maximal asymptotic variance*
>
> $$\sup_F V(\psi, F) = \frac{(1-\epsilon)E_\Phi\psi^2 + \epsilon k^2}
> {\bigl[(1-\epsilon)E_\Phi\psi'\bigr]^2}$$
>
> *whenever $H$ puts all its mass outside the interval $[-k, +k]$. If $\epsilon$ and $k$ are
> related by*
>
> $$(1-\epsilon)^{-1} = \int_{-k}^{+k}\varphi(t)\,dt + \frac{2\varphi(k)}{k}$$
>
> *then $T_n$ minimizes the maximal asymptotic variance among all translation invariant
> estimators, the maximum being taken over the set of all symmetric $\epsilon$-contaminated
> distributions $F = (1-\epsilon)\Phi + \epsilon H$. There is a unique asymptotically least
> favorable $F_0$ having the density $f_0(t) = (1-\epsilon)(2\pi)^{-1/2}e^{-\rho(t)}$."*

Và ngay dưới, bài báo giải nghĩa hai cột của Bảng I:

> *"Table I gives the value of $\sup_F V(\psi,F)$ in dependence of $k$ and $\epsilon$. Notice
> that $E_\Phi\psi' = P_\Phi[\lvert x\rvert < k]$ is the probability that an observation falls
> inside the interval $(-k, +k)$ if $\Phi$ is the true underlying distribution.
> $\epsilon_{\min}$ is the value of $\epsilon$ for which the corresponding $k$ yields the
> asymptotic minimax solution."*

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $E_\Phi\psi^2$ | số trong $(0,1]$ | $\int_{-k}^{k}t^2\varphi\,dt + 2k^2(1-\Phi(k))$ |
| $E_\Phi\psi'$ | số trong $(0,1]$ | $=P_\Phi[\lvert x\rvert<k]=2\Phi(k)-1$ |
| $\epsilon k^2$ | số | phần đóng góp **xấu nhất** của nhiễm bẩn: mọi điểm hỏng đóng góp đúng $k^2$ |
| $\epsilon_{\min}$ | số trong $[0,1]$ | mức nhiễm bẩn mà $k$ đang xét là nghiệm minimax |

**Ý nghĩa trừu tượng, và đây là chỗ đáng dừng lại.** Tử số của $\sup V$ tách làm hai phần cộng
nhau, và phép tách ấy là toàn bộ nội dung của Hệ quả. Phần $(1-\epsilon)E_\Phi\psi^2$ là đóng
góp của dữ liệu sạch, tính được vì ta biết $\Phi$. Phần $\epsilon k^2$ là đóng góp của dữ liệu
hỏng — và nó là **một hằng số**, không phụ thuộc $H$ nữa. Đó chính là lý do việc chặn $\psi$
ăn thua: nếu $\psi$ không bị chặn thì số hạng thứ hai là $\epsilon\int\psi^2 dH$, và thiên
nhiên đẩy nó lên vô cùng bằng cách đặt $H$ ra xa. Chặn $\psi$ biến một số hạng **không có
trần** thành một số hạng **tính được**.

Điều kiện *"whenever $H$ puts all its mass outside the interval $[-k,+k]$"* nói rõ trường hợp
xấu nhất trông thế nào: kẻ phá hoại tối ưu không cần tinh vi, chỉ cần đặt mọi điểm hỏng ra
ngoài ngưỡng. Xa hơn nữa cũng không thêm được gì.

**Quan hệ $k\leftrightarrow\epsilon$**, viết lại cho dễ dùng:

$$\frac{1}{1-\epsilon} = \bigl(2\Phi(k)-1\bigr) + \frac{2\varphi(k)}{k}
\qquad\Longleftrightarrow\qquad
\epsilon_{\min}(k) = 1 - \frac{1}{\bigl(2\Phi(k)-1\bigr) + 2\varphi(k)/k}$$

Kiểm rằng $\epsilon_{\min}$ đúng là "mức nhiễm bẩn mà $k$ này tối ưu": giải ngược
$\epsilon_{\min}(k)=\epsilon$ rồi so với $k$ cực tiểu hoá $\sup_F V$ trên lưới $0{,}01$:

| $\epsilon$ | $k$ giải từ $\epsilon_{\min}$ | $k$ cực tiểu $V$ trên lưới |
|---:|---:|---:|
| $0{,}001$ | $2{,}632880$ | $2{,}63$ |
| $0{,}005$ | $2{,}162534$ | $2{,}16$ |
| $0{,}01$ | $1{,}945111$ | $1{,}95$ |
| $0{,}02$ | $1{,}717437$ | $1{,}72$ |
| $0{,}05$ | $1{,}398377$ | $1{,}40$ |
| $0{,}10$ | $1{,}140171$ | $1{,}14$ |
| $0{,}20$ | $0{,}861592$ | $0{,}86$ |

Bảy trên bảy dòng khớp tới độ phân giải của lưới. Cột $\epsilon_{\min}$ của Bảng I vì thế đọc
được theo hai chiều: *"dùng $k$ này thì tôi đang chống mức nhiễm bẩn nào"* và *"tôi sợ mức
nhiễm bẩn này thì phải đặt $k$ bằng bao nhiêu"*.

---

### 2.9 BẢNG I tr.84–85 — dựng lại toàn bộ 403 ô

Bảng I là bảng trung tâm của bài báo: nó cho **phương sai tiệm cận lớn nhất** mà ước lượng
Huber phải chịu, theo hai biến $k$ và $\epsilon$. Nó chiếm trọn hai trang và được in **XOAY
90 độ**, nên hai hình dưới đã được xoay lại cho đọc được.

![Bảng I phần 1, trang 84](<paper_huber1964_tr84_bang_I_phan_1.png>)

**Hình 9 — BẢNG I, phần $k=0{,}0$ đến $1{,}9$, Huber 1964, trang 84.** Cùng nguồn, trang 12
của PDF. `--box 0.12,0.06,0.90,0.86 --dpi 240 --sat --xoay 270`.

![Bảng I phần 2, trang 85](<paper_huber1964_tr85_bang_I_phan_2.png>)

**Hình 10 — BẢNG I, phần $k=2{,}0$ đến $3{,}0$ và chú thích, Huber 1964, trang 85.** Cùng
nguồn, trang 13 của PDF. `--box 0.14,0.04,0.45,0.80 --dpi 300 --sat --xoay 270`.

**Tiêu đề và chú thích, nguyên văn:**

> **TABLE I** — *"Upper bounds for the asymptotic variance of $n^{1/2}T_n$ for
> $F = (1-\epsilon)\Phi + \epsilon H$, $H$ symmetric"*
>
> *"$T_n$ is defined by $\Sigma\psi(x_i - T_n) = 0$, where $\psi(t) = t$ for
> $\lvert t\rvert < k$, $\psi(t) = k\,\mathrm{sign}(t)$ for $\lvert t\rvert \ge k$."*

Dịch tiêu đề: *"Cận trên cho phương sai tiệm cận của $n^{1/2}T_n$ khi
$F=(1-\epsilon)\Phi+\epsilon H$, với $H$ đối xứng."* Dịch chú thích: *"$T_n$ được định nghĩa
bởi $\Sigma\psi(x_i-T_n)=0$, trong đó $\psi(t)=t$ khi $\lvert t\rvert<k$, và
$\psi(t)=k\,\mathrm{sign}(t)$ khi $\lvert t\rvert\ge k$."*

**Cách đọc bảng.** Bốn cột đầu là **tính chất của $\psi$** dưới phân phối chuẩn, không phụ
thuộc $\epsilon$; mười cột sau là **cận trên của phương sai** ở mười mức nhiễm bẩn. Đọc một
hàng là biết: đặt ngưỡng này thì bao nhiêu phần trăm dữ liệu sạch nằm trong ngưỡng
($E_\Phi\psi'$), ngưỡng này tối ưu cho mức nhiễm bẩn nào ($\epsilon_{\min}$), và phải trả bao
nhiêu phương sai ở từng mức nhiễm bẩn.

**Điều bảng KHÔNG nói ra.** Cột $\epsilon=0$ không phải "phương sai khi dữ liệu sạch" theo
nghĩa thông thường — nó là **giới hạn** của cận trên khi $\epsilon\to0$, và vì $\Phi$ có thông
tin Fisher bằng 1 nên nó chính là **nghịch đảo hiệu suất** của ước lượng tại phân phối chuẩn.
Mục 5.3 dùng đúng cột đó để suy ra con số $1{,}345$.

**Bảng dựng lại, toàn bộ 403 ô, sinh từ công thức của Hệ quả tr.82:**

| $k$ | $E_\Phi\psi'$ | $E_\Phi\psi^2$ | $\epsilon_{\min}$ | $\epsilon=0{,}000$ | $\epsilon=0{,}001$ | $\epsilon=0{,}002$ | $\epsilon=0{,}005$ | $\epsilon=0{,}010$ | $\epsilon=0{,}020$ | $\epsilon=0{,}050$ | $\epsilon=0{,}100$ | $\epsilon=0{,}200$ | $\epsilon=0{,}500$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $0{,}0$ | $0{,}0000$ | $0{,}0000$ | $1{,}0000$ | $1{,}571$ | $1{,}574$ | $1{,}577$ | $1{,}587$ | $1{,}603$ | $1{,}636$ | $1{,}740$ | $1{,}939$ | $2{,}454$ | $6{,}283$ |
| $0{,}1$ | $0{,}0797$ | $0{,}0095$ | $0{,}8753$ | $1{,}492$ | $1{,}495$ | $1{,}498$ | $1{,}508$ | $1{,}523$ | $1{,}556$ | $1{,}658$ | $1{,}853$ | $2{,}358$ | $6{,}137$ |
| $0{,}2$ | $0{,}1585$ | $0{,}0358$ | $0{,}7542$ | $1{,}423$ | $1{,}426$ | $1{,}429$ | $1{,}438$ | $1{,}454$ | $1{,}485$ | $1{,}586$ | $1{,}778$ | $2{,}276$ | $6{,}030$ |
| $0{,}3$ | $0{,}2358$ | $0{,}0758$ | $0{,}6401$ | $1{,}362$ | $1{,}365$ | $1{,}368$ | $1{,}377$ | $1{,}393$ | $1{,}424$ | $1{,}524$ | $1{,}714$ | $2{,}209$ | $5{,}961$ |
| $0{,}4$ | $0{,}3108$ | $0{,}1265$ | $0{,}5354$ | $1{,}309$ | $1{,}312$ | $1{,}315$ | $1{,}324$ | $1{,}339$ | $1{,}370$ | $1{,}470$ | $1{,}659$ | $2{,}154$ | $5{,}930$ |
| $0{,}5$ | $0{,}3829$ | $0{,}1851$ | $0{,}4417$ | $1{,}263$ | $1{,}266$ | $1{,}268$ | $1{,}277$ | $1{,}293$ | $1{,}324$ | $1{,}423$ | $1{,}613$ | $2{,}111$ | $5{,}935$ |
| $0{,}6$ | $0{,}4515$ | $0{,}2491$ | $0{,}3599$ | $1{,}222$ | $1{,}225$ | $1{,}228$ | $1{,}237$ | $1{,}252$ | $1{,}284$ | $1{,}384$ | $1{,}576$ | $2{,}079$ | $5{,}976$ |
| $0{,}7$ | $0{,}5161$ | $0{,}3160$ | $0{,}2899$ | $1{,}187$ | $1{,}190$ | $1{,}193$ | $1{,}202$ | $1{,}217$ | $1{,}249$ | $1{,}351$ | $1{,}546$ | $2{,}058$ | $6{,}053$ |
| $0{,}8$ | $0{,}5763$ | $0{,}3840$ | $0{,}2311$ | $1{,}156$ | $1{,}159$ | $1{,}162$ | $1{,}172$ | $1{,}187$ | $1{,}220$ | $1{,}324$ | $1{,}522$ | $2{,}047$ | $6{,}166$ |
| $0{,}9$ | $0{,}6319$ | $0{,}4511$ | $0{,}1825$ | $1{,}130$ | $1{,}133$ | $1{,}136$ | $1{,}146$ | $1{,}162$ | $1{,}195$ | $1{,}302$ | $1{,}506$ | $2{,}046$ | $6{,}317$ |
| $1{,}0$ | $0{,}6827$ | $0{,}5161$ | $0{,}1428$ | $1{,}107$ | $1{,}111$ | $1{,}114$ | $1{,}124$ | $1{,}140$ | $1{,}175$ | $1{,}284$ | $1{,}495$ | $2{,}055$ | $6{,}506$ |
| $1{,}1$ | $0{,}7287$ | $0{,}5777$ | $0{,}1109$ | $1{,}088$ | $1{,}091$ | $1{,}095$ | $1{,}105$ | $1{,}122$ | $1{,}158$ | $1{,}272$ | $1{,}490$ | $2{,}072$ | $6{,}734$ |
| $1{,}2$ | $0{,}7699$ | $0{,}6352$ | $0{,}0855$ | $1{,}072$ | $1{,}075$ | $1{,}079$ | $1{,}089$ | $1{,}107$ | $1{,}144$ | $1{,}263$ | $1{,}491$ | $2{,}099$ | $7{,}003$ |
| $1{,}3$ | $0{,}8064$ | $0{,}6880$ | $0{,}0655$ | $1{,}058$ | $1{,}062$ | $1{,}065$ | $1{,}076$ | $1{,}095$ | $1{,}134$ | $1{,}258$ | $1{,}496$ | $2{,}135$ | $7{,}314$ |
| $1{,}4$ | $0{,}8385$ | $0{,}7358$ | $0{,}0498$ | $1{,}047$ | $1{,}050$ | $1{,}054$ | $1{,}066$ | $1{,}086$ | $1{,}126$ | $1{,}256$ | $1{,}507$ | $2{,}179$ | $7{,}669$ |
| $1{,}5$ | $0{,}8664$ | $0{,}7785$ | $0{,}0376$ | $1{,}037$ | $1{,}041$ | $1{,}045$ | $1{,}057$ | $1{,}078$ | $1{,}121$ | $1{,}258$ | $1{,}522$ | $2{,}233$ | $8{,}069$ |
| $1{,}6$ | $0{,}8904$ | $0{,}8160$ | $0{,}0282$ | $1{,}029$ | $1{,}034$ | $1{,}038$ | $1{,}051$ | $1{,}073$ | $1{,}118$ | $1{,}262$ | $1{,}542$ | $2{,}296$ | $8{,}517$ |
| $1{,}7$ | $0{,}9109$ | $0{,}8487$ | $0{,}0211$ | $1{,}023$ | $1{,}027$ | $1{,}032$ | $1{,}046$ | $1{,}069$ | $1{,}116$ | $1{,}270$ | $1{,}567$ | $2{,}367$ | $9{,}012$ |
| $1{,}8$ | $0{,}9281$ | $0{,}8767$ | $0{,}0156$ | $1{,}018$ | $1{,}023$ | $1{,}027$ | $1{,}042$ | $1{,}066$ | $1{,}117$ | $1{,}280$ | $1{,}595$ | $2{,}448$ | $9{,}558$ |
| $1{,}9$ | $0{,}9426$ | $0{,}9006$ | $0{,}0115$ | $1{,}014$ | $1{,}019$ | $1{,}024$ | $1{,}039$ | $1{,}065$ | $1{,}119$ | $1{,}292$ | $1{,}628$ | $2{,}537$ | $10{,}154$ |
| $2{,}0$ | $0{,}9545$ | $0{,}9205$ | $0{,}0084$ | $1{,}010$ | $1{,}016$ | $1{,}021$ | $1{,}038$ | $1{,}065$ | $1{,}122$ | $1{,}307$ | $1{,}665$ | $2{,}635$ | $10{,}802$ |
| $2{,}1$ | $0{,}9643$ | $0{,}9371$ | $0{,}0061$ | $1{,}008$ | $1{,}014$ | $1{,}019$ | $1{,}037$ | $1{,}066$ | $1{,}127$ | $1{,}324$ | $1{,}705$ | $2{,}742$ | $11{,}501$ |
| $2{,}2$ | $0{,}9722$ | $0{,}9507$ | $0{,}0044$ | $1{,}006$ | $1{,}012$ | $1{,}018$ | $1{,}037$ | $1{,}068$ | $1{,}133$ | $1{,}342$ | $1{,}750$ | $2{,}858$ | $12{,}253$ |
| $2{,}3$ | $0{,}9786$ | $0{,}9617$ | $0{,}0032$ | $1{,}004$ | $1{,}011$ | $1{,}017$ | $1{,}037$ | $1{,}071$ | $1{,}140$ | $1{,}363$ | $1{,}798$ | $2{,}982$ | $13{,}058$ |
| $2{,}4$ | $0{,}9836$ | $0{,}9705$ | $0{,}0023$ | $1{,}003$ | $1{,}010$ | $1{,}017$ | $1{,}038$ | $1{,}074$ | $1{,}148$ | $1{,}386$ | $1{,}850$ | $3{,}114$ | $13{,}914$ |
| $2{,}5$ | $0{,}9876$ | $0{,}9776$ | $0{,}0016$ | $1{,}002$ | $1{,}010$ | $1{,}017$ | $1{,}040$ | $1{,}078$ | $1{,}156$ | $1{,}410$ | $1{,}905$ | $3{,}255$ | $14{,}821$ |
| $2{,}6$ | $0{,}9907$ | $0{,}9831$ | $0{,}0011$ | $1{,}002$ | $1{,}010$ | $1{,}017$ | $1{,}041$ | $1{,}082$ | $1{,}166$ | $1{,}436$ | $1{,}963$ | $3{,}405$ | $15{,}779$ |
| $2{,}7$ | $0{,}9931$ | $0{,}9873$ | $0{,}0008$ | $1{,}001$ | $1{,}010$ | $1{,}018$ | $1{,}044$ | $1{,}087$ | $1{,}176$ | $1{,}463$ | $2{,}025$ | $3{,}562$ | $16{,}787$ |
| $2{,}8$ | $0{,}9949$ | $0{,}9906$ | $0{,}0005$ | $1{,}001$ | $1{,}010$ | $1{,}019$ | $1{,}046$ | $1{,}092$ | $1{,}186$ | $1{,}492$ | $2{,}090$ | $3{,}726$ | $17{,}843$ |
| $2{,}9$ | $0{,}9963$ | $0{,}9931$ | $0{,}0004$ | $1{,}001$ | $1{,}010$ | $1{,}020$ | $1{,}048$ | $1{,}097$ | $1{,}197$ | $1{,}523$ | $2{,}158$ | $3{,}899$ | $18{,}947$ |
| $3{,}0$ | $0{,}9973$ | $0{,}9950$ | $0{,}0003$ | $1{,}000$ | $1{,}010$ | $1{,}021$ | $1{,}051$ | $1{,}103$ | $1{,}209$ | $1{,}554$ | $2{,}229$ | $4{,}078$ | $20{,}098$ |

**Kết quả đối chiếu: 402 trên 403 ô khớp; một ô không.**

| Phép so | Kết quả |
|---|---|
| Số ô đã đối chiếu | $403$ |
| Lệch quá **nửa đơn vị** chữ số cuối | $\mathbf{1}$ |
| Phép so **chặt** (làm tròn về đúng số chữ số in rồi so bằng nhau) | $16$ ô khác |
| Biên sai số của $309$ ô $V$ còn lại | $0{,}5700$ đơn vị chữ số cuối |
| Lệch trung bình của $309$ ô đó | $-0{,}0447$ đơn vị |
| Ô ngoại lệ | $9{,}8283$ đơn vị — **gấp 17 lần** biên trên |

Mười lăm trong mười sáu ô "khác" theo phép so chặt đều lệch trong khoảng
$[-0{,}57;\ +0{,}45]$ đơn vị của chữ số cuối, tức **nằm trọn trong phần nửa đơn vị mà một phép
làm tròn cho phép**. Đó là mức của việc tra bảng $\Phi$ năm chữ số bằng tay năm 1963, không
phải mức của sai sót.

#### 2.9.1 Ô thứ mười sáu, và nó TỰ LỘ

Ô duy nhất nằm ngoài khoảng đó là $k = 2{,}1$, $\epsilon = 0{,}020$:

| | Giá trị |
|---|---|
| Bài báo in | $1{,}137$ |
| Dựng lại từ công thức của chính bài báo | $\mathbf{1{,}127172}$ |
| Chênh | $-0{,}0098$, tức $9{,}83$ đơn vị chữ số cuối |

**Không cần công thức để thấy nó sai — chỉ cần đọc cột.** Năm ô liền nhau của cột
$\epsilon=0{,}020$, đúng như bản in:

| $k$ | $2{,}0$ | $2{,}1$ | $2{,}2$ | $2{,}3$ | $2{,}4$ |
|---|---:|---:|---:|---:|---:|
| **Bài báo in** | $1{,}122$ | $\mathbf{1{,}137}$ | $1{,}133$ | $1{,}140$ | $1{,}148$ |
| Sai phân | | $+0{,}015$ | $\mathbf{-0{,}004}$ | $+0{,}007$ | $+0{,}008$ |
| **Dựng lại** | $1{,}122441$ | $\mathbf{1{,}127172}$ | $1{,}133019$ | $1{,}139870$ | $1{,}147626$ |
| Sai phân | | $+0{,}005$ | $+0{,}006$ | $+0{,}007$ | $+0{,}008$ |

Dãy in **đi lên rồi đi xuống rồi lại đi lên**. Dãy dựng lại thì tăng đều.

**Và "phải tăng" ở đây là một phát biểu kiểm được, không phải cảm giác.** Hàm $V(k,\epsilon)$
theo $k$ **không** đơn điệu trên toàn trục — nó có đúng một cực tiểu, và cực tiểu đó nằm ở
chính $k$ thoả $\epsilon_{\min}(k)=\epsilon$:

| Đại lượng | Giá trị |
|---|---|
| Cực tiểu của cột $\epsilon=0{,}020$, quét lưới $0{,}001$ | $k = 1{,}717$, $V = 1{,}116297$ |
| $k$ giải từ phương trình $\epsilon_{\min}(k) = 0{,}020$ | $k = 1{,}717437$ |
| Số chỗ **giảm** trên lưới $0{,}001$ trong $[1{,}717;\ 6{,}0]$ | $\mathbf{0}$ trên $4283$ |

Năm ô $k=2{,}0\ldots2{,}4$ nằm **trọn bên phải cực tiểu**, nên chúng bắt buộc phải tăng dần.
Thay $1{,}137$ bằng $1{,}127$ thì sai phân thành $+0{,}005\;+0{,}006\;+0{,}007\;+0{,}008$ —
đơn điệu trở lại, và khớp đúng dãy dựng lại.

Kết luận: **một lỗi in một chữ số**, không phải lỗi tính. Mọi ô khác của bảng khớp, kể cả hai
ô bên cạnh nó trong cùng cột.

#### 2.9.2 Kiểm câu nhận định của chính tác giả về bảng này

Ngay dưới bảng, bài báo viết:

> *"It can be seen from the table that the performance of $T_n$ is not very sensitive to the
> choice of $k$—any value between 1.0 and 2.0 will give acceptable results for all
> $\epsilon \le 0.2$. The limiting case $k = 0$ is the sample median."*
> — tr.82

Dịch: *"Từ bảng có thể thấy rằng hiệu năng của $T_n$ không nhạy lắm với việc chọn $k$ — bất kỳ
giá trị nào giữa $1{,}0$ và $2{,}0$ cũng cho kết quả chấp nhận được với mọi
$\epsilon\le0{,}2$. Trường hợp giới hạn $k=0$ là trung vị mẫu."*

Câu đó đo được: với mỗi $\epsilon$, lấy $V$ nhỏ nhất trên toàn trục $k$, rồi xem $k$ **tệ
nhất** trong khoảng $[1{,}0;\ 2{,}0]$ tệ hơn bao nhiêu phần trăm.

| $\epsilon$ | $k$ tối ưu | $V$ tối ưu | $V$ tệ nhất trong $[1;2]$ | Tệ hơn |
|---:|---:|---:|---:|---:|
| $0$ | $3{,}00$ | $1{,}000402$ | $1{,}107267$ | $+10{,}68\ \%$ |
| $0{,}001$ | $2{,}63$ | $1{,}009548$ | $1{,}110526$ | $+10{,}00\ \%$ |
| $0{,}002$ | $2{,}44$ | $1{,}017116$ | $1{,}113795$ | $+9{,}51\ \%$ |
| $0{,}005$ | $2{,}16$ | $1{,}036725$ | $1{,}123668$ | $+8{,}39\ \%$ |
| $0{,}01$ | $1{,}95$ | $1{,}065241$ | $1{,}140344$ | $+7{,}05\ \%$ |
| $0{,}02$ | $1{,}72$ | $1{,}116298$ | $1{,}174547$ | $+5{,}22\ \%$ |
| $0{,}05$ | $1{,}40$ | $1{,}256124$ | $1{,}306808$ | $+4{,}03\ \%$ |
| $0{,}10$ | $1{,}14$ | $1{,}489854$ | $1{,}664687$ | $+11{,}73\ \%$ |
| $0{,}20$ | $0{,}86$ | $2{,}045533$ | $2{,}635003$ | $\mathbf{+28{,}82\ \%}$ |
| $0{,}50$ *(ngoài phạm vi câu trên)* | $0{,}44$ | $5{,}927703$ | $10{,}801669$ | $+82{,}22\ \%$ |

**Câu của tác giả đúng, với một điều kiện ông không viết ra.** Trong khoảng
$\epsilon\in[0{,}005;\ 0{,}05]$ thì giá phải trả cho việc chọn $k$ sai trong $[1;2]$ chỉ là
$4$–$8\ \%$ phương sai — rất rẻ, đúng như "not very sensitive". Nhưng ở hai đầu thì không:
ở $\epsilon = 0{,}20$ nó là $\mathbf{28{,}8\ \%}$, và $0{,}20$ chính là biên mà câu ấy tự
tuyên bố phủ. Chữ *"acceptable"* đang gánh phần nặng nhất của câu, và bài báo không định
nghĩa nó.

Dòng $\epsilon=0{,}5$ nằm ngoài phạm vi câu trên, in ra để thấy hàm đi tiếp thế nào.

---

### 2.10 Mục 7 tr.82–83 — câu hỏi về độ chệch, công thức (2)

Sáu mục đầu giả định $H$ **đối xứng**. Mục 7 bỏ giả định đó, và đó là chỗ xuất hiện một loại
sai khác hẳn: không phải phương sai lớn, mà **chệch** — ước lượng hội tụ về sai chỗ.

![Cận chệch trang 83](<paper_huber1964_tr83_can_chech.png>)

**Hình 11 — Công thức (2) và ba cận chệch, Huber 1964, trang 83.** Cùng nguồn, trang 11 của
PDF. `--box 0.12,0.12,0.92,0.52`.

**Nguyên văn phần đặt vấn đề, tr.82:**

> *"Assume that we have the situation of Section 5, the prototype distribution being
> $F = (1-\epsilon)G + \epsilon H$, with a symmetric $G$. What happens if the contaminating
> distribution $H$ is not symmetric? (Suggested interpretation: most observations are
> distributed according to $G(x - \vartheta)$, but a small fraction $\epsilon$ of them is
> distributed according to $H(x)$, and is unrelated to the parameter $\vartheta$ to be
> estimated. This might happen if the observations are made with an instrument that jams with
> probability $\epsilon$.)"*

Dịch: *"Giả sử ta ở tình huống của Mục 5, phân phối mẫu gốc là $F=(1-\epsilon)G+\epsilon H$
với $G$ đối xứng. Chuyện gì xảy ra nếu phân phối gây nhiễm bẩn $H$ KHÔNG đối xứng? (Cách hiểu
gợi ý: phần lớn quan sát phân phối theo $G(x-\vartheta)$, nhưng một tỉ lệ nhỏ $\epsilon$ trong
số đó phân phối theo $H(x)$ và không liên quan gì tới tham số $\vartheta$ cần ước lượng. Điều
này có thể xảy ra nếu phép đo được thực hiện bằng một thiết bị bị kẹt với xác suất
$\epsilon$.)"*

Đoạn trong ngoặc là đoạn quý nhất của mục: nó cho một **mô hình vật lý** của nhiễm bẩn không
đối xứng — cảm biến kẹt, và khi kẹt thì nó trả một giá trị không liên quan gì tới thứ cần đo.

**Công thức (2):**

$$\lvert c(F)\rvert \;\le\; \frac{\epsilon}{1-\epsilon}\cdot
\frac{\sup\lvert\psi\rvert}{-\lambda_G'(\vartheta c)}\tag{2}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $c(F)$ | số | **độ chệch tiệm cận**: $T_n\to c$ thay vì $\to 0$ |
| $\sup\lvert\psi\rvert$ | số dương | trần của hàm ảnh hưởng, bằng $k$ |
| $-\lambda_G'$ | số dương | độ nhạy của phương trình ước lượng, $=E_\Phi\psi'$ khi thay $\vartheta c$ bởi $0$ |
| $\epsilon/(1-\epsilon)$ | số dương | tỉ lệ nhiễm bẩn, quy về "trên mỗi quan sát sạch" |

**Ý nghĩa trừu tượng, và nó là mặt trái của công thức (1).** Công thức (1) nói chặn $\psi$ làm
phương sai có trần. Công thức (2) nói **cùng cái trần ấy** cũng làm độ chệch có trần — và hai
lần "trần" đó cùng một chữ $k$. Nếu $\psi$ không bị chặn thì $\sup\lvert\psi\rvert=\infty$ và
(2) trở thành vô nghĩa: một quan sát hỏng có thể kéo giới hạn đi bao xa tuỳ nó muốn. Đây là lý
do sâu hơn khiến việc chặn ảnh hưởng ăn thua, và nó không xuất hiện trong phần minimax.

**Ba cận mà bài báo in, và dựng lại:**

| $\epsilon$ | $k$ bài báo in | $k$ đo lại từ $\epsilon_{\min}$ | $\lvert c\rvert$ bài báo in | $\lvert c\rvert$ đo lại |
|---:|---:|---:|---:|---:|
| $0{,}1$ | $1{,}14$ | $1{,}140171$ | $0{,}17$ | $0{,}16986891$ |
| $0{,}01$ | $1{,}95$ | $1{,}945111$ | $0{,}021$ | $0{,}02075925$ |
| $0{,}001$ | $2{,}64$ | $2{,}632880$ | $0{,}0027$ | $0{,}00266471$ |

Ba trên ba khớp tới đúng số chữ số bài báo in.

**Câu về cỡ mẫu, và nó có HAI cách đọc.** Bài báo viết:

> *"As a consequence of these considerations, if somebody wants to control the asymptotic bias
> and keep it below $\tfrac12 n^{-1/2}$, that is, below one half of the asymptotic standard
> deviation of $T_n$, he should not increase the sample size above $n = 9$, 600 and 35,000,
> respectively, for the above-mentioned values of $\epsilon$ and $k$."*

Dịch: *"Từ những cân nhắc trên, nếu ai muốn khống chế độ chệch tiệm cận và giữ nó dưới
$\frac12 n^{-1/2}$, tức là dưới một nửa độ lệch chuẩn tiệm cận của $T_n$, thì người đó không
nên tăng cỡ mẫu lên quá $n = 9$, $600$ và $35\,000$, tương ứng với các giá trị $\epsilon$ và
$k$ nói trên."*

Hai vế của câu nói hai thứ khác nhau. Vế đầu viết ngưỡng là $\frac12 n^{-1/2}$; vế sau gọi nó
là *"một nửa độ lệch chuẩn tiệm cận"*, mà độ lệch chuẩn tiệm cận là $\sqrt{V/n}$ chứ không
phải $n^{-1/2}$. Thử cả hai trên cả ba dòng rồi để tổng sai lệch chọn:

| $\epsilon$ | $n$ bài báo | (A) $1/(4c^2)$ | (B) $V/(4c^2)$ | Sai của (A) | Sai của (B) |
|---:|---:|---:|---:|---:|---:|
| $0{,}1$ | $9$ | $8{,}66$ | $12{,}91$ | $-3{,}72\ \%$ | $+43{,}44\ \%$ |
| $0{,}01$ | $600$ | $580{,}11$ | $617{,}96$ | $-3{,}31\ \%$ | $+2{,}99\ \%$ |
| $0{,}001$ | $35\,000$ | $35\,207{,}24$ | $35\,543{,}42$ | $+0{,}59\ \%$ | $+1{,}55\ \%$ |
| | **tổng sai lệch** | $\mathbf{0{,}0763}$ | $0{,}4798$ | | |

Cách đọc **(A)**, tức đọc theo chữ, khớp hơn **6,3 lần**. Nghĩa là bài báo lấy độ lệch chuẩn
tiệm cận bằng $n^{-1/2}$, tức bỏ qua thừa số $\sqrt V$ — chấp nhận được vì $V$ chỉ trong
khoảng $1{,}0$–$1{,}5$ ở ba dòng này, và cả ba con số đều được làm tròn về số chẵn.

**Ý nghĩa thực hành của ba con số đó, và nó đáng nhớ hơn chính chúng.** Câu này nói một điều
phản trực giác: với nhiễm bẩn không đối xứng, **thêm dữ liệu có lúc là hại**. Phương sai giảm
theo $1/n$ nhưng độ chệch **không giảm**, nên qua một cỡ mẫu nào đó thì sai số tổng bị chệch
chi phối, và khoảng tin cậy hẹp lại quanh một giá trị sai. Bài báo nói thẳng hệ quả ở tr.86:

> *"less than $\epsilon n = 1$, 6 or 35 observations, respectively, would come from $H$, and
> one would not even know which ones. This may be taken as a justification for using minimax
> procedures, in the sense that sample sizes reasonable for a given amount of unknown
> contamination will not allow to determine the nature of this contamination, except in rather
> extreme cases."*

Dịch: *"chưa tới $\epsilon n = 1$, $6$ hoặc $35$ quan sát đến từ $H$, và người ta thậm chí
không biết đó là những quan sát nào. Điều này có thể coi là một biện minh cho việc dùng thủ
tục minimax, theo nghĩa là những cỡ mẫu hợp lý với một mức nhiễm bẩn chưa biết cho trước sẽ
không cho phép xác định được bản chất của nhiễm bẩn ấy, trừ những trường hợp khá cực đoan."*

Kiểm: $0{,}1\times9 = 0{,}9$, $0{,}01\times600 = 6{,}0$, $0{,}001\times35\,000 = 35{,}0$ —
khớp cả ba.

**Cận của trung vị, để so.** Bài báo viết: *"The sample median gives the smallest bias, namely
$\lvert c\rvert \le \epsilon(1-\epsilon)^{-1}(2\varphi(0))^{-1} \approx 1.25\epsilon(1-\epsilon)^{-1}$"*.
Kiểm: $(2\varphi(0))^{-1} = 1{,}2533141373$, đúng là $1{,}25$ sau khi làm tròn.

| $\epsilon$ | Cận của trung vị | Cận của Huber ở $k$ tối ưu | Huber chệch hơn |
|---:|---:|---:|---:|
| $0{,}001$ | $0{,}00125457$ | $0{,}00265802$ | $2{,}12$ lần |
| $0{,}01$ | $0{,}01265974$ | $0{,}02072009$ | $1{,}64$ lần |
| $0{,}1$ | $0{,}13925713$ | $0{,}16986891$ | $1{,}22$ lần |

Đây là một **kết quả âm** cho Huber mà bài báo báo cáo trung thực: về độ chệch trước nhiễm
bẩn không đối xứng, **trung vị tốt hơn**, và Huber trả giá từ $1{,}2$ tới $2{,}1$ lần. Bài báo
bình luận ngay: *"It does not seem that one can make the bias much smaller without doing
serious damage to the variance of the estimate."*

---

### 2.11 Mục 8 tr.86–90 — lý thuyết minimax, dạng trừu tượng

Mục 5 giải một trường hợp riêng bằng cách kiểm trực tiếp. Mục 8 làm lại toàn bộ ở mức trừu
tượng, và câu mở đầu nói rõ vì sao:

> *"The most robust estimator constructed in Section 5 coincided with the maximum likelihood
> estimator for some least favorable distribution. The present section shows that this is
> quite a general feature of most robust $(M)$-estimators."*
> — tr.86

![Trò chơi với Thiên nhiên, trang 86](<paper_huber1964_tr86_tro_choi_K.png>)

**Hình 12 — Trò chơi với Thiên nhiên và hàm trả $K(\psi,F)$, Huber 1964, trang 86.** Cùng
nguồn, trang 14 của PDF. `--box 0.12,0.28,0.92,0.60`.

**Nguyên văn phần luật chơi:**

> *"Consider the following game against Nature:*
> *Nature chooses a distribution $F$ from some set $C$ of distributions on the real line.*
> *The statistician chooses a function $\psi$ from some set $\Psi$ of functions.*
> *The payoff to the statistician is*
>
> $$K(\psi, F) = \left(\int\psi'\,dF\right)^{\!2}\Big/\int\psi^2\,dF$$
>
> *where $\psi'$ denotes the derivative (in measure) of $\psi$."*

> *"The reason for preferring the utility function $K(\psi, F)$ over the loss function
> $V(\psi, F) = K(\psi, F)^{-1}$ used in earlier sections is that $K$ has more convenient
> convexity properties."*

Dịch câu cuối: *"Lý do thích hàm lợi ích $K(\psi,F)$ hơn hàm mất mát $V(\psi,F)=K(\psi,F)^{-1}$
dùng ở các mục trước là $K$ có tính chất lồi tiện hơn."*

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $C$ | tập phân phối | nước đi của Thiên nhiên; phải **lồi** |
| $\Psi$ | tập hàm | nước đi của nhà thống kê |
| $K(\psi,F)$ | số dương | $=1/V$ — càng lớn càng tốt cho nhà thống kê |

![Bổ đề 6, trang 86](<paper_huber1964_tr86_bo_de_6.png>)

**Hình 13 — Bổ đề 6, Huber 1964, trang 86.** Cùng nguồn, trang 14.
`--box 0.12,0.77,0.92,0.93`.

**Bổ đề 6** là bất đẳng thức lồi mà ba định lý sau đều dùng:

$$\frac{\bigl(\alpha u_1 + (1-\alpha)u_2\bigr)^2}{\alpha v_1 + (1-\alpha)v_2}
\;\le\; \alpha\frac{u_1^2}{v_1} + (1-\alpha)\frac{u_2^2}{v_2},
\qquad v_1, v_2 > 0,\ 0\le\alpha\le1$$

Đặt $u_i=\int\psi'dF_i$ và $v_i=\int\psi^2dF_i$ thì vế trái là $K(\psi,\alpha F_1+(1-\alpha)F_2)$,
nên Bổ đề 6 nói gọn: **$K(\psi,\cdot)$ là hàm lồi của $F$.** Hệ quả ngay: mọi chiến lược trộn
của Thiên nhiên đều bị một chiến lược thuần trội, nên không cần xét chiến lược ngẫu nhiên.

![Định lý 2, trang 87](<paper_huber1964_tr87_dinh_ly_2.png>)

**Hình 14 — Định lý 2, Huber 1964, trang 87.** Cùng nguồn, trang 15.
`--box 0.12,0.35,0.92,0.53`.

**Định lý 2 — đây là định lý trung tâm của cả bài báo:**

> *"THEOREM 2. Let $C$ be a convex set of distribution functions such that every $F \in C$ has
> an absolutely continuous density $f$ satisfying $I(F) = \int(f'/f)^2 f\,dt < \infty$. (i) If
> there is an $F_0 \in C$ such that $I(F_0) \le I(F)$ for all $F \in C$, and if
> $\psi_0 = -f_0'/f_0$ is contained in $\Psi$, then $(\psi_0, F_0)$ is a saddlepoint of the
> game, that is,*
>
$$K(\psi, F_0) \le K(\psi_0, F_0) = I(F_0) \le K(\psi_0, F)$$

Đọc chuỗi ấy từ hai đầu vào giữa. Vế trái nói: **giữ nguyên nước đi xấu nhất của thiên nhiên
thì không nhà thống kê nào làm tốt hơn $\psi_0$.** Vế phải nói: **giữ nguyên $\psi_0$ thì
không nước đi nào của thiên nhiên làm hại hơn $F_0$.** Ở giữa là một con số duy nhất — thông
tin Fisher của $F_0$ — và nó chính là **giá của trò chơi**, tức mức phương sai mà người dùng
được bảo đảm dù dữ liệu hỏng thế nào.

> *for all $\psi \in \Psi$ and all $F \in C$."*

**Ý nghĩa trừu tượng, và đây là câu đáng nhớ nhất của bài báo.** Định lý 2 nói rằng bài toán
*"chọn ước lượng bền vững nhất"* **tương đương** bài toán *"tìm phân phối có thông tin Fisher
nhỏ nhất trong lớp đang xét"*. Hai bài toán trông không liên quan: một cái tối ưu trên không
gian **hàm $\psi$**, cái kia tối ưu trên không gian **phân phối $F$**. Định lý 2 nối chúng,
và phép nối ấy đổi một bài toán khó thành một bài toán giải được — vì $I(F)$ là một phiếm hàm
lồi, và cực tiểu hoá một phiếm hàm lồi trên một tập lồi compact là chuyện có sẵn công cụ.

Nói bằng lời thường: *phân phối khó nhất là phân phối chứa ít thông tin nhất, và cách tốt nhất
để đối phó với nó là giả vờ rằng nó đúng là phân phối thật.*

![Định nghĩa I(F) và Định lý 3, trang 89](<paper_huber1964_tr89_dinh_nghia_I_F.png>)

**Hình 15 — Định nghĩa mở rộng của $I(F)$ và Định lý 3, Huber 1964, trang 89.** Cùng nguồn,
trang 17. `--box 0.12,0.28,0.92,0.48`.

Định lý 2 giả định mọi $F\in C$ có mật độ. Định lý 3 và 4 gỡ giả định đó:

> **DEFINITION.** $I(F) = \sup_\psi\left(\int\psi'\,dF\right)^{\!2}\Big/\int\psi^2\,dF$
> *where $\psi$ ranges over the set of continuous and continuously differentiable functions
> with compact support satisfying $\int\psi^2\,dF \ne 0$.*
>
> **THEOREM 3.** *$I(F) < \infty$ if and only if $F$ has an absolutely continuous density $f$
> such that $\int(f'/f)^2f\,dt < \infty$, $f'$ being the derivative in measure of $f$, and
> then $I(F) = \int(f'/f)^2 f\,dt$.*

![Định lý 4, trang 90](<paper_huber1964_tr90_dinh_ly_4.png>)

**Hình 16 — Định lý 4, Huber 1964, trang 90.** Cùng nguồn, trang 18.
`--box 0.12,0.72,0.92,0.85`.

> **THEOREM 4.** *Assume that $C$ is vaguely compact and convex. If $\inf_{F\in C}I(F) = a < \infty$,
> then there exists a unique $F_0 \in C$ such that $I(F_0) = a$.*

**Vì sao ba định lý kỹ thuật này đáng có trong bài viết.** Định lý 2 nói *nếu* $F_0$ tồn tại
thì nó là lời giải. Định lý 4 nói $F_0$ **tồn tại và duy nhất**. Định lý 3 là cầu nối: nó cho
phép định nghĩa $I(F)$ cho **mọi** phân phối, kể cả phân phối không có mật độ, bằng một
$\sup$ — rồi chứng minh định nghĩa mới trùng định nghĩa cũ ở chỗ định nghĩa cũ có nghĩa. Thiếu
bước đó thì tập $C$ không compact và Định lý 4 không áp được.

Chính vì Định lý 4 mà chữ *"uniquely determined"* ở tr.75 có căn cứ. Đó là chỗ phân biệt bài
báo này với một đề xuất phương pháp: nó không nói *"hàm này tốt"*, nó nói *"không có hàm nào
khác"*.

---

### 2.12 Mục 9 tr.91–93 — Định lý 5, và BẢNG II

Mục 9 đổi lớp phân phối: thay vì "chuẩn bị nhiễm bẩn" $(1-\epsilon)\Phi+\epsilon H$, nó xét
lớp **$\epsilon$-chuẩn** — mọi $F$ đối xứng nằm cách $\Phi$ không quá $\epsilon$ theo khoảng
cách Kolmogorov.

![Định lý 5, trang 91](<paper_huber1964_tr91_dinh_ly_5.png>)

**Hình 17 — Định lý 5, Huber 1964, trang 91.** Cùng nguồn, trang 19 của PDF.
`--box 0.12,0.11,0.92,0.25`.

> **THEOREM 5.** *Assume that $C$ is vaguely compact and convex, and that the set
> $C_1 = \{F \in C \mid I(F) < \infty\}$ is dense in $C$. Let $F_0$ be the unique distribution
> minimizing $I(F)$. If $F_0$ happens to be such that $\psi_0 = -f_0'/f_0$ is in $\Psi$, is
> differentiable in measure and $\psi_0'$ is either continuous or nonnegative upper
> semi-continuous, and vanishes at infinity, then $(\psi_0, F_0)$ is not only a saddlepoint
> with respect to $C_1$, but also with respect to $C$.*

![Mục 9 trang 91](<paper_huber1964_tr91_muc_9.png>)

**Hình 18 — Mở đầu mục 9 và lớp $C$, Huber 1964, trang 91.** Cùng nguồn, trang 19.
`--box 0.12,0.62,0.92,0.78`.

**Nguyên văn định nghĩa lớp và một lời tự khai:**

> *"We say that a distribution $F$ is $\epsilon$-normal with approximate mean $\xi$ and
> approximate variance $\sigma^2$ if it satisfies
> $\sup_t\lvert F(t) - \Phi((t-\xi)/\sigma)\rvert \le \epsilon$."*
>
> $$C = \{F \mid \sup_t\lvert F(t) - \Phi(t)\rvert \le \epsilon,\; F(t) + F(-t) = 1\}$$
>
> *"The saddlepoint $(\psi_0, F_0)$ was found by some kind of enlightened guesswork, using
> variational techniques; instead of going through the cumbersome heuristic arguments, I shall
> state the solution explicitly and shall then verify it."*
>
> *"The solution is valid only for values of $\epsilon$ between 0 and (approximately) 0.03,
> but it seems that this range is just the range important in practice."*

Dịch hai câu cuối: *"Điểm yên ngựa $(\psi_0,F_0)$ được tìm ra bằng một kiểu phỏng đoán có hiểu
biết, dùng kỹ thuật biến phân; thay vì đi qua những lập luận trực giác rườm rà, tôi sẽ phát
biểu lời giải một cách tường minh rồi kiểm chứng nó."* và *"Lời giải chỉ đúng với các giá trị
$\epsilon$ từ 0 tới (xấp xỉ) 0,03, nhưng có vẻ khoảng đó đúng là khoảng quan trọng trong thực
hành."*

Câu *"enlightened guesswork"* đáng giữ: tác giả nói thẳng rằng lời giải không được **tìm ra**
bằng suy luận có hệ thống, nó được **đoán ra rồi kiểm lại**. Đó là một câu trung thực hiếm gặp
trong một bài báo toán.

![Mật độ f_0 và ba điều kiện, trang 92](<paper_huber1964_tr92_f0_va_ba_phuong_trinh.png>)

**Hình 19 — $f_0$, $\psi_0$ và ba điều kiện (i)(ii)(iii), Huber 1964, trang 92.** Cùng nguồn,
trang 20 của PDF. `--box 0.12,0.11,0.92,0.38`.

$$f_0(t) = f_0(-t) = \begin{cases}
\varphi(a)\bigl(\cos\tfrac{ca}{2}\bigr)^{-2}\bigl(\cos\tfrac{ct}{2}\bigr)^2 & 0 \le t \le a\\[3pt]
\varphi(t) & a < t < b\\[3pt]
\varphi(b)\,e^{-b(t-b)} & t \ge b
\end{cases}$$

$$\psi_0(t) = -\psi_0(-t) = \begin{cases}
c\tan\tfrac{ct}{2} & 0 \le t \le a\\[3pt]
t & a < t < b\\[3pt]
b & t \ge b
\end{cases}$$

với ba hằng số $a$, $b$, $c$ xác định bởi ba điều kiện được bài báo đánh nhãn:

$$c\tan\left(\tfrac{1}{2}ca\right) = a \qquad (0 \le ca < \pi)\tag{i}$$

$$\int_0^a f_0\,dt = \int_0^a \varphi\,dt - \epsilon\tag{ii}$$

$$\int_b^\infty f_0\,dt = \int_b^\infty \varphi\,dt + \epsilon\tag{iii}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $a$ | số dương | ranh giới trong: dưới nó $f_0$ là một hàm **cô-sin bình phương**, không phải chuông |
| $b$ | số dương | ranh giới ngoài: trên nó $f_0$ có **đuôi mũ** |
| $c$ | số dương | tham số của nhánh cô-sin; $\psi_0$ ở giữa là $c\tan(ct/2)$, **không phải** đường thẳng |
| $(a,b)$ | khoảng | vùng $f_0$ trùng **đúng** $\varphi$ |

**Ý nghĩa trừu tượng, và nó khác hẳn mục 5.** Ở mục 5 (chuẩn bị nhiễm bẩn), $f_0$ giữ nguyên
$\varphi$ ở **giữa** và đổi hai **đuôi**. Ở đây thì ngược lại một nửa: $f_0$ đổi cả phần
**lõi** $[0,a]$ lẫn phần **đuôi** $[b,\infty)$, và chỉ giữ nguyên **vành giữa** $(a,b)$. Lý do
nằm ở lớp: điều kiện $\sup_t\lvert F-\Phi\rvert\le\epsilon$ cho phép thiên nhiên **rút bớt**
khối lượng ở lõi (điều kiện ii, dấu trừ) và **thêm** vào đuôi (điều kiện iii, dấu cộng), miễn
tổng chênh lệch tích luỹ không quá $\epsilon$. Rút bớt ở lõi làm $\psi_0$ **cong lên** thành
$c\tan(ct/2)$ thay vì thẳng — tức ở lớp này, ước lượng tối ưu còn **khuếch đại** các phần dư
rất nhỏ.

![Bảng II, trang 92](<paper_huber1964_tr92_bang_II.png>)

**Hình 20 — BẢNG II, Huber 1964, trang 92.** Cùng nguồn, trang 20.
`--box 0.10,0.70,0.92,0.90 --sat`.

**Dựng lại BẢNG II** bằng cách giải hệ ba phương trình phi tuyến (i)(ii)(iii). Tích phân ở (ii)
có dạng đóng: $\int_0^a\cos^2(ct/2)\,dt = a/2 + \sin(ca)/(2c)$; tích phân ở (iii) bằng
$\varphi(b)/b$.

| $\epsilon$ | $a$ in | $a$ đo lại | $b$ in | $b$ đo lại | $c$ in | $c$ đo lại | Sai số hệ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $0{,}001$ | $0{,}65$ | $0{,}653368$ | $2{,}44$ | $2{,}436408$ | $1{,}37$ | $1{,}365825$ | $2{,}7\times10^{-17}$ |
| $0{,}002$ | $0{,}75$ | $0{,}753408$ | $2{,}23$ | $2{,}231719$ | $1{,}35$ | $1{,}350687$ | $1{,}1\times10^{-16}$ |
| $0{,}005$ | $0{,}91$ | $0{,}911771$ | $1{,}95$ | $1{,}948302$ | $1{,}32$ | $1{,}323350$ | $2{,}2\times10^{-16}$ |
| $0{,}01$ | $1{,}06$ | $1{,}056368$ | $1{,}72$ | $1{,}724094$ | $1{,}29$ | $1{,}295287$ | $5{,}3\times10^{-16}$ |
| $0{,}02$ | $1{,}24$ | $\mathbf{1{,}228800}$ | $1{,}49$ | $1{,}492062$ | $1{,}26$ | $1{,}258733$ | $2{,}2\times10^{-16}$ |
| $0{,}03$ | $1{,}34$ | $1{,}346175$ | $1{,}36$ | $1{,}353362$ | $1{,}23$ | $1{,}232381$ | $2{,}8\times10^{-17}$ |

**Mười bốn trên mười tám ô làm tròn đúng.** Bốn ô lệch, và cả bốn đều nhỏ:

| Ô | Bài báo in | Đo lại | Chênh, đơn vị chữ số cuối |
|---|---:|---:|---:|
| $\epsilon=0{,}01$, cột $c$ | $1{,}29$ | $1{,}295287$ | $+0{,}5$ |
| $\epsilon=0{,}02$, cột $a$ | $1{,}24$ | $1{,}228800$ | $\mathbf{-1{,}1}$ |
| $\epsilon=0{,}03$, cột $a$ | $1{,}34$ | $1{,}346175$ | $+0{,}6$ |
| $\epsilon=0{,}03$, cột $b$ | $1{,}36$ | $1{,}353362$ | $-0{,}7$ |

Không ô nào lệch quá $1{,}2$ đơn vị của chữ số cuối được in. Bảng II chỉ in **hai** chữ số
thập phân và được giải bằng tay năm 1963; sai số của hệ tại nghiệm dựng lại là $\sim10^{-16}$,
tức hệ được thoả tuyệt đối. Nên đây là mức của phép giải tay, không phải lỗi công thức.

**Kiểm câu "chỉ đúng tới $\epsilon\approx0{,}03$".** Điều kiện hợp lệ của lời giải là
$0\le a\le b$; giải hệ ở các $\epsilon$ lớn hơn:

| $\epsilon$ | $a$ | $b$ | $c$ | $a \le b$? |
|---:|---:|---:|---:|---|
| $0{,}03$ | $1{,}34617$ | $1{,}35336$ | $1{,}23238$ | **có** |
| $0{,}035$ | $1{,}39470$ | $1{,}30021$ | $1{,}22122$ | **KHÔNG** |
| $0{,}04$ | $1{,}43871$ | $1{,}25404$ | $1{,}21099$ | **KHÔNG** |
| $0{,}05$ | $1{,}51679$ | $1{,}17671$ | $1{,}19262$ | **KHÔNG** |

$a$ vượt $b$ **ngay sau $0{,}03$** — phép đo xác nhận đúng con số tác giả nêu, và cho biết
thêm rằng ranh giới rất sắc: ở $0{,}03$ thì $a$ và $b$ chỉ còn cách nhau $0{,}0072$.

**Kiểm bất đẳng thức chốt của chứng minh.** Chứng minh ở tr.93 phải dùng $c^2 + a^2 - 2 \ge 0$:

| $\epsilon$ | $0{,}001$ | $0{,}002$ | $0{,}005$ | $0{,}01$ | $0{,}02$ | $0{,}03$ |
|---|---:|---:|---:|---:|---:|---:|
| $c^2+a^2-2$ | $+0{,}29236864$ | $+0{,}39197873$ | $+0{,}58258279$ | $+0{,}79368026$ | $+1{,}09435826$ | $+1{,}33095029$ |

Sáu trên sáu dương, và biên nhỏ nhất là $0{,}29236864$ — chứng minh không chạy sát ranh giới.

---

### 2.13 Mục 10 tr.93–94 — ước lượng thang đo, công thức (3) và (4)

Mục 10 mở đầu bằng một lời thừa nhận:

> *"The theory of estimating a scale parameter is less satisfactory than that of estimating a
> location parameter. Perhaps the main source of trouble is that there is no natural
> “canonical” parameter to be estimated."*
> — tr.93

Mẹo của bài báo là đổi biến: ước lượng thang đo của $X$ quy về ước lượng **vị trí** của
$Y = \log X^2$, vì nhân $X$ với một hằng số là **tịnh tiến** $Y$.

![Công thức (3) và (4), trang 94](<paper_huber1964_tr94_cong_thuc_3_va_4.png>)

**Hình 21 — Công thức (3) và (4), Huber 1964, trang 94.** Cùng nguồn, trang 22 của PDF.
`--box 0.12,0.62,0.92,0.84 --sat`.

$$(1-\epsilon)^{-1} = \int_{-q}^{+q}\varphi(t)\,dt + 2q(q^2-1)^{-1}\varphi(q)\tag{3}$$

$$\psi_0(q, t) = \begin{cases} t & \lvert t\rvert < q\\ q\,\mathrm{sign}(t) & \lvert t\rvert \ge q\end{cases}\tag{4}$$

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $q$ | số dương | ngưỡng của bài toán thang đo, đóng vai trò của $k$ |
| $q^2 - 1$ | số | **mẫu số khác** với mục 6, nơi nó là $k$ |
| $\psi_0(q,t)$ | hàm | chính là $\operatorname{clip}(t,-q,q)$ — cùng dạng, khác tên |

**Ý nghĩa trừu tượng, và chỗ khác nhau đáng chú ý.** Công thức (3) là bản tương ứng của quan
hệ $k\leftrightarrow\epsilon$ ở mục 6, nhưng mẫu số là $(q^2-1)$ thay vì $q$. Lý do nằm ở phép
đổi biến: với $Y=\log X^2$, hàm $-g'/g$ không phải $u$ mà là $\tfrac12(e^u-1)$, nên chỗ nối của
$\psi$ rơi vào một điểm khác. Hệ quả cụ thể: cùng một giá trị số, đọc như $k$ hay như $q$ cho
hai mức nhiễm bẩn khác nhau.

| Giá trị | $\epsilon$ từ **(3)** (thang đo) | $\epsilon_{\min}$ từ mục 6 (vị trí) | Tỉ số |
|---:|---:|---:|---:|
| $1{,}10$ | $0{,}66787657$ | $0{,}11092361$ | $6{,}0210$ |
| $1{,}45$ | $0{,}18012466$ | $0{,}04329978$ | $4{,}1599$ |
| $1{,}70$ | $0{,}07412370$ | $0{,}02106190$ | $3{,}5193$ |
| $2{,}00$ | $0{,}02580420$ | $0{,}00841922$ | $3{,}0649$ |
| $2{,}08$ | $0{,}01944287$ | $0{,}00652875$ | $2{,}9780$ |
| $2{,}53$ | $0{,}00380768$ | $0{,}00144104$ | $2{,}6423$ |
| $3{,}01$ | $0{,}00059927$ | $0{,}00024504$ | $2{,}4456$ |

Hai cột chênh nhau từ $2{,}4456$ tới $6{,}0210$ lần, và tỉ số **giảm đơn điệu** theo ngưỡng. Đây là một **chỗ trùng tên trong chính bài
báo** mà bài báo có ý thức tránh: ở tr.94 nó viết *"To avoid confusion later on, we replace the
letters $\psi$, $k$ of Theorem 1 by $\chi$, $c$"* — tức tác giả đổi tên ký hiệu đúng vì lý do
này.

---

### 2.14 Mục 11 tr.95–99 — ba Đề xuất, hệ (6), thuật toán (6′), và hai bảng

Đây là mục dài nhất và là mục **duy nhất** của bài báo nói về việc thực sự tính ra con số. Nó
cũng là mục mà mọi thư viện hồi quy bền vững ngày nay cài đặt, dù không phải thư viện nào cũng
cài đúng phiên bản nào — mục 4.3 đo chỗ đó.

#### 2.14.1 Công thức (5) và BẢNG III

![Công thức (5), trang 95](<paper_huber1964_tr95_cong_thuc_5.png>)

**Hình 22 — Công thức (5), Huber 1964, trang 95.** Cùng nguồn, trang 23 của PDF.
`--box 0.12,0.52,0.92,0.66 --sat`.

Ước lượng thang đo $S_n$ ở mục 10 có một khuyết điểm mà bài báo nói thẳng:

> *"The flaw is that $S_n$ may have quite a considerable bias for distributions not in $C$,
> for instance for the normal distribution $\Phi$ itself."*
> — tr.95

Cách chữa: chia cho giá trị tiệm cận của nó dưới $\Phi$. Gọi $S_\infty$ là giá trị đó,
$S_n' = S_n/S_\infty$, và $q' = qS_\infty$, thì

$$n^{-1}\sum_i\psi_0^2(q', x_i/S_n') = E_\Phi\psi_0^2(q', t)\tag{5}$$

**Suy luận của bài báo, nguyên văn:** *"$1 = E_\Phi\psi_0^2(q, t/S_\infty) =
S_\infty^{-2}E_\Phi\psi_0^2(qS_\infty, t)$"*. Thay $q' = qS_\infty$ vào vế phải thì nó thành
$S_\infty^{-2}\,E_\Phi\psi^2(q')$, nên

$$S_\infty = \sqrt{E_\Phi\psi^2(q')}$$

Nghĩa là cột $S_\infty$ của Bảng III **không cần giải phương trình nào** — nó là căn bậc hai
của chính cột $E_\Phi\psi^2$ của Bảng I, đọc ở mức $q'$.

![Bảng III, trang 95](<paper_huber1964_tr95_bang_III.png>)

**Hình 23 — BẢNG III, Huber 1964, trang 95.** Cùng nguồn, trang 23.
`--box 0.10,0.72,0.92,0.92 --sat`.

| $q'$ | $S_\infty$ in | $S_\infty$ đo lại | $q=q'/S_\infty$ in | $q$ đo lại | $\epsilon_{\min}$ in | $\epsilon_{\min}$ đo lại |
|---:|---:|---:|---:|---:|---:|---:|
| $1{,}1$ | $0{,}760$ | $0{,}760069047$ | $1{,}45$ | $1{,}447236938$ | $0{,}182$ | $0{,}181925750$ |
| $1{,}5$ | $0{,}882$ | $0{,}882306759$ | $1{,}70$ | $1{,}700088983$ | $0{,}074$ | $0{,}074100490$ |
| $2{,}0$ | $0{,}960$ | $0{,}959446156$ | $2{,}08$ | $2{,}084535946$ | $0{,}019$ | $0{,}019132500$ |
| $2{,}5$ | $0{,}989$ | $0{,}988716331$ | $2{,}53$ | $2{,}528531108$ | $0{,}0038$ | $0{,}003828530$ |
| $3{,}0$ | $0{,}997$ | $0{,}997500515$ | $3{,}01$ | $3{,}007517243$ | $0{,}0006$ | $0{,}000605251$ |

**Mười ba trên mười lăm ô làm tròn đúng.** Hai ô còn lại lệch đúng **một đơn vị** của chữ số
cuối: $q'=2{,}0$ cho $0{,}959446$ (bài báo in $0{,}960$) và $q'=3{,}0$ cho $0{,}997501$ (bài
báo in $0{,}997$). Cả hai nằm sát ranh giới làm tròn.

#### 2.14.2 Ba Đề xuất, và hệ (6)

![Đề xuất 1 và 2, trang 96](<paper_huber1964_tr96_de_xuat_1_va_2.png>)

**Hình 24 — Đề xuất 1 và Đề xuất 2, Huber 1964, trang 96.** Cùng nguồn, trang 24 của PDF.
`--box 0.12,0.35,0.92,0.72 --sat`.

**Nguyên văn phần đặt vấn đề:**

> *"The following are three heuristic proposals on how one can apply the theory of
> $(M)$-estimators to this problem. I conjecture that the third proposal is asymptotically
> minimax... The first two proposals are asymptotically minimax if $k$ is chosen in relation
> to the true $\epsilon$."*
>
> *"All three proposals are asymptotically equivalent to Winsorizing; they differ among each
> other in the determination of the number of observations to be Winsorized."*

**PROPOSAL 1** — chọn $k$ trước, rồi giải

$$n^{-1}\sum\psi\bigl(k, (x_i-T)/S\bigr) = 0,
\qquad n^{-1}\sum\psi'\bigl(k, (x_i-T)/S\bigr) = E_\Phi\psi'(k, x)$$

Phương trình thứ hai nói bằng lời: **hãy chọn thang đo sao cho SỐ QUAN SÁT lọt vào ngưỡng đúng
bằng số mà dữ liệu chuẩn sẽ cho.** Vì vế trái đếm quan sát nên nó là một số nguyên chia cho
$n$, còn vế phải là một số thực — hai vế chỉ khớp được gần đúng, và chính bài báo nói ra điều
đó ngay dưới đây.

> *"The second equation can only be fulfilled approximately, since $\sum\psi'(k,(x_i-T)/S)$ is
> an integer, namely the number of observations contained in the interval $[T - kS, T + kS]$.
> Hence this proposal is similar (and asymptotically equivalent) to Winsorizing a fixed,
> predetermined number of observations... $k$ might be compared to a kind of insurance against
> ill effects from contamination: if one insures against high contamination by choosing a small
> $k$, one has to pay by losing some efficiency if the actual contamination is low, and vice
> versa."*

Câu so với **bảo hiểm** là câu giải thích ý nghĩa của $k$ rõ nhất trong cả bài báo: phí bảo
hiểm là hiệu suất mất đi khi không có sự cố, và mức bảo hiểm là phương sai không bùng nổ khi
có sự cố.

**PROPOSAL 2** — chọn $k$ trước, rồi giải

$$n^{-1}\sum\psi\bigl(k, (x_i-T)/S\bigr) = 0,
\qquad n^{-1}\sum\psi^2\bigl(k, (x_i-T)/S\bigr) = E_\Phi\psi^2(k, x)$$

Chỗ khác Đề xuất 1 nằm ở **bình phương**: thay vì đếm xem bao nhiêu quan sát lọt vào ngưỡng,
nó cân xem tổng ảnh hưởng của chúng nặng bao nhiêu. Vế trái vì thế là một số **thực**, đổi
liên tục theo $S$, nên hai vế khớp được **đúng** chứ không chỉ gần đúng — và đó là lý do kỹ
thuật khiến Đề xuất 2 ít nhạy với việc chọn sai $k$ hơn Đề xuất 1.

> *"It corresponds to Winsorizing a variable number of observations: slightly more if $F$ has
> heavier tails, slightly less if $F$ has lighter tails... Certainly for large $n$, but probably
> also for small $n$, this proposal is better than the preceding one, since it is less
> sensitive to a “wrong” choice of $k$."*

![Đề xuất 3, trang 97](<paper_huber1964_tr97_de_xuat_3.png>)

**Hình 25 — Đề xuất 3 và ước lượng phương sai, Huber 1964, trang 97.** Cùng nguồn, trang 25.
`--box 0.12,0.12,0.92,0.42 --sat`.

**PROPOSAL 3** — chọn cả $k$ để cực tiểu hoá phương sai tiệm cận ước lượng được:

$$\text{cực tiểu hoá}\quad \sum\psi^2(k, x_i - T)\Big/\Bigl[\sum\psi'(k, x_i - T)\Bigr]^2
\quad\text{với ràng buộc}\quad \sum\psi(k, x_i - T) = 0$$

Hai Đề xuất đầu bắt người dùng **khai trước** mình sợ mức nhiễm bẩn nào, rồi mới ước lượng.
Đề xuất 3 bỏ luôn bước khai đó: nó lấy chính ước lượng phương sai làm hàm mục tiêu và để dữ
liệu chọn $k$. Cái giá là bài toán không còn lồi theo $(k,T)$, nên không có bảo đảm hội tụ —
và bài báo nói thẳng điều đó ở câu ngay dưới.

> *"This proposal seems to give the best asymptotic properties of all three, but numerical
> computation of $k$ and $T$ apparently is a very difficult task."*
>
> *"For practical applications, I personally would favor Proposal 2, since it seems to fit best
> into the framework of conventional least squares techniques. For nonlinear least squares
> problems, which have to be solved by iterative methods anyway, this proposal probably could
> be incorporated into existing computer programs with an only marginal increase of the amount
> of computation."*

Đoạn này là chỗ duy nhất trong bài báo nói **nên dùng cái nào**, và nó chọn Đề xuất 2 — không
phải vì nó tốt nhất về lý thuyết (Đề xuất 3 mới thế), mà vì nó **ghép được vào phần mềm sẵn
có**. Sáu mươi năm sau, `scikit-learn` làm gần đúng như vậy; mục 4.3 nói "gần" nghĩa là gì.

Trong cùng mục, bài báo cho công thức ước lượng phương sai — và đây chính là thứ tài liệu hiện
đại gọi là **ước lượng bánh kẹp** (sandwich):

$$\frac{n}{n-1}\cdot
\frac{\sum\psi^2\bigl(k, (x_i-T)/S\bigr)}{\Bigl[\sum\psi'\bigl(k, (x_i-T)/S\bigr)\Bigr]^2}\cdot S^2$$

Đây chính là công thức (1) của mục 2.4, thay kỳ vọng lý thuyết bằng tổng trên mẫu: tử số là
**nhiễu đo được**, mẫu số là **độ nhạy đếm được** — số quan sát lọt vào ngưỡng. Ý nghĩa thực
hành: sai số chuẩn của Huber **không** tính được bằng công thức của bình phương tối thiểu, vì
mẫu số ở đây là số quan sát *được tin*, không phải $n$.

> *"The correction term $n/(n-1)$ is suggested by the classical theory ($k = \infty$); one will
> have to revise it in the light of later knowledge, but it seems to agree quite well with
> exploratory Monte Carlo computations for sample sizes 5 and 10."*

![Hệ (6), trang 97](<paper_huber1964_tr97_he_6.png>)

**Hình 26 — Hệ (6), Huber 1964, trang 97.** Cùng nguồn, trang 25.
`--box 0.12,0.58,0.92,0.72 --sat`.

$$\sum_{i=1}^n \psi\bigl(k, (x_i-T)/S\bigr) = 0,
\qquad \sum_{i=1}^n \psi^2\bigl(k, (x_i-T)/S\bigr) = n\beta\tag{6}$$

Hệ này là Đề xuất 2 viết lại với vế phải để **ngỏ**: thay vì cố định $E_\Phi\psi^2(k)$, bài
báo đặt tên nó là $\beta$ và coi như một lựa chọn. Chỗ để ngỏ ấy nhỏ bằng một chữ cái, và
mục 4.3 cho thấy sáu mươi năm sau nó đúng là chỗ các thư viện tách nhau.

> *"Proposal 2 corresponds to $\beta = E_\Phi\psi^2(k, x)$, and for this choice we have,
> obviously, $\beta < k^2$ whenever $k > 0$. I do not want to exclude the possibility that
> small sample size considerations might lead to a different choice of $\beta$."*

| Ký hiệu | Chiều | Nghĩa |
|---|---|---|
| $T$ | số | ước lượng vị trí, ẩn thứ nhất |
| $S$ | số dương | ước lượng thang đo, ẩn thứ hai |
| $\beta$ | số dương | **hằng số vế phải**; Đề xuất 2 đặt $\beta = E_\Phi\psi^2(k)$ |
| $n\beta$ | số dương | tổng "năng lượng" mà $\psi$ được phép có ở nghiệm |

**Ý nghĩa trừu tượng, và đây là chỗ đắt nhất của cả mục.** Phương trình thứ nhất định vị; phương
trình thứ hai **định thang**. Đọc riêng phương trình thứ hai: nó nói *hãy chọn $S$ sao cho tổng
bình phương của các $\psi$ đã chuẩn hoá bằng đúng cái tổng mà dữ liệu chuẩn sẽ cho*. Tức là:
**định nghĩa "thang đo" là con số làm cho dữ liệu trông giống dữ liệu chuẩn nhất, theo đúng
thước đo mà thuật toán đang dùng.** Đó là một định nghĩa tự quy chiếu nhưng hoàn toàn chặt chẽ,
và mệnh đề ở tr.97–98 chứng minh nó có nghiệm duy nhất.

Câu *"I do not want to exclude the possibility that... a different choice of $\beta$"* là câu
để ngỏ, và sáu mươi năm sau `scikit-learn` chọn đúng một $\beta$ khác — $\beta = 1$.

#### 2.14.3 Thuật toán (6′), và tốc độ hội tụ

![Thuật toán (6'), trang 98](<paper_huber1964_tr98_thuat_toan_6_phay.png>)

**Hình 27 — Hệ (6′) và thuật toán lặp, Huber 1964, trang 98.** Cùng nguồn, trang 26 của PDF.
`--box 0.12,0.62,0.92,0.80 --sat`.

Gọi $m_1$, $m_2$, $m_3$ là số quan sát thoả $x_i \le T-kS$, $T-kS < x_i < T+kS$, và
$T+kS \le x_i$; gọi $\bar x' = m_2^{-1}\sum' x_i$ là **trung bình cắt bớt** trên $m_2$ quan sát
ở giữa. Khi đó hệ (6) viết lại thành

$$T = \bar x' + kS\,\frac{m_3 - m_1}{m_2}$$
$$S^2 = \frac{\sum'(x_i - \bar x')^2}{n\beta - \bigl(m_1 + m_3 + (m_3-m_1)^2/m_2\bigr)k^2}\tag{6'}$$

và bài báo nói ngay cách dùng nó: *"This may be used to compute $T$ and $S$ by iterations:
start with some initial values $T_0$, $S_0$."*

> *"For automatic computation, it might be wise to safeguard against too small values of $m_2$
> and of the denominator in the expression defining $S_1^2$."*
>
> *"Since, for a fixed sample, the values of $T_1$ and $S_1$ are uniquely determined by the pair
> of integers $m_1$, $m_3$, the process must either stop after a finite number of steps, or it
> will repeat itself periodically. I do not know whether and when the latter case occurs."*

Câu cuối là một **giới hạn tự khai**: tác giả không biết thuật toán của mình có thể lặp vòng
hay không. Nó vẫn chưa được trả lời trong bài báo.

**Chạy (6′) trên một mẫu 60 điểm** — 57 điểm chuẩn cộng ba điểm đặt ở $12$, $15$, $20$:

| $k$ | $T$ | $S$ | Số lượt áp (6′) |
|---:|---:|---:|---:|
| $1{,}000$ | $+0{,}1670307171$ | $0{,}9822393449$ | $4$ |
| $1{,}345$ | $+0{,}1886219230$ | $1{,}0225069318$ | $5$ |
| $1{,}500$ | $+0{,}1933288002$ | $1{,}0148478676$ | $3$ |
| $2{,}000$ | $+0{,}2170546885$ | $1{,}0106253416$ | $3$ |

Để so, trên cùng mẫu đó: trung bình $+0{,}8797921736$, trung vị $+0{,}1701946718$, độ lệch
chuẩn $3{,}6100553865$, MAD$/0{,}6745 = 1{,}0528670864$. Ba điểm trên sáu mươi đủ để kéo trung
bình đi **gấp năm lần** so với trung vị, và làm độ lệch chuẩn phồng lên **gấp ba lần rưỡi** so
với thang đo mà (6′) tìm được.

**Phép thử ngược — hệ (6) có thật sự được thoả ở nghiệm không:**

| $k$ | $\sum\psi$ | $\sum\psi^2 - n\beta$ |
|---:|---:|---:|
| $1{,}000$ | $+0{,}000\times10^{0}$ | $+3{,}553\times10^{-15}$ |
| $1{,}345$ | $+2{,}220\times10^{-15}$ | $+0{,}000\times10^{0}$ |
| $1{,}500$ | $-8{,}882\times10^{-16}$ | $+0{,}000\times10^{0}$ |
| $2{,}000$ | $+0{,}000\times10^{0}$ | $+0{,}000\times10^{0}$ |

Cả hai phương trình được thoả tới mức sai số máy — tức thuật toán (6′) thật sự giải hệ (6),
không phải giải một thứ gần giống.

**Kiểm câu của tác giả về tốc độ:**

> *"Ordinarily, the procedure converges rather fast, as an exploratory study with Monte Carlo
> methods shows. For $k = 1.5$, sample sizes up to 100 and distributions that do not have too
> heavy tails, the stationary value is on the average reached after 1-2 steps, starting from
> sample mean and sample variance. For distributions with heavier tails (Cauchy), somewhat more
> steps are needed (about 2 for sample size 10, 4 for sample size 100)."*

Chạy lại đúng thiết lập đó — khởi đầu bằng trung bình mẫu và phương sai mẫu, $k = 1{,}5$:

| Phân phối | $n$ | Số mẫu | Bước **đổi** trung bình | Trung vị | Tối đa | Bài báo nói |
|---|---:|---:|---:|---:|---:|---|
| Chuẩn | $100$ | $2000$ | $\mathbf{1{,}6960}$ | $2{,}0$ | $3$ | *"1-2 steps"* |
| Cauchy | $10$ | $2000$ | $\mathbf{1{,}5560}$ | $1{,}0$ | $3$ | *"about 2"* |
| Cauchy | $100$ | $2000$ | $\mathbf{4{,}2730}$ | $4{,}0$ | $7$ | *"4"* |

Ba trên ba khớp. Lưu ý cách đếm: mỗi lượt áp (6′) cuối cùng chỉ **xác nhận** $T$ và $S$ không
đổi nữa, nên nó không phải một bước đi; bảng trên đã trừ nó ra. Không trừ thì ba con số là
$2{,}6960$, $2{,}5560$ và $5{,}2730$.

#### 2.14.4 Công thức (7) và BẢNG IV

![Công thức (7), trang 99](<paper_huber1964_tr99_cong_thuc_7.png>)

**Hình 28 — Công thức (7) và cách dựng Bảng IV, Huber 1964, trang 99.** Cùng nguồn, trang 27
của PDF. `--box 0.12,0.42,0.92,0.60 --sat`.

> *"Table IV gives sharp upper bounds for the asymptotic variance of $n^{1/2}T$, if $T$ and $S$
> are computed according to Proposal 2, and $F = (1-\epsilon)\Phi + \epsilon H$, $H$ symmetric.
> These upper bounds can be determined as follows. The asymptotic variance of $n^{1/2}T$ is
> $E_F\psi^2(kS, x)/(E_F\psi'(kS, x))^2$, where $S$ satisfies $E_F\psi^2(kS, x) = S^2\beta(k)$,
> with $\beta(k) = E_\Phi\psi^2(k, x)$. Hence, we have, with $q = kS$,"*
>
> $$q^2\beta(k)/k^2 = (1-\epsilon)\beta(q) + \epsilon E_H\psi^2(q, x)
> \le (1-\epsilon)\beta(q) + \epsilon q^2$$
>
> *"or"*
>
> $$\beta(q)/q^2 \ge \bigl(\beta(k)/k^2 - \epsilon\bigr)/(1-\epsilon)\tag{7}$$
>
> *"Since $\psi^2(q,x)/q^2$ is monotone decreasing for increasing $q$, $\beta(q)/q^2$ is also
> monotone decreasing, and equality in (7) determines a sharp upper bound $q_{\max}$ for $q$."*

Và ở tr.100, bài báo nói rõ cách dựng bảng:

> *"one may determine the upper bound $q_{\max}$ from (7) and enter Table 1 with $k = q_{\max}$
> and $\epsilon$ to determine the upper bound for the variance. This was done, using linear
> interpolation, to obtain Table 4."*

![Bảng IV, trang 99](<paper_huber1964_tr99_bang_IV.png>)

**Hình 29 — BẢNG IV, Huber 1964, trang 99.** Cùng nguồn, trang 27.
`--box 0.10,0.62,0.92,0.90 --sat`.

**Ý nghĩa trừu tượng của (7).** Khi $S$ được ước lượng **cùng lúc** với $T$, nhiễm bẩn không
chỉ làm hỏng $T$ trực tiếp mà còn **thổi phồng $S$**, và $S$ phồng thì ngưỡng thật $kS$ nới
ra, và ngưỡng nới ra thì nhiều điểm hỏng lọt vào trong. Công thức (7) khoá vòng lặp đó lại:
nó cho một **trần $q_{\max}$** cho ngưỡng hiệu dụng, đạt được khi $H$ đặt toàn bộ khối lượng
ra ngoài $[-q_{\max}, q_{\max}]$.

**Dựng lại BẢNG IV theo cả hai đường** — bằng công thức chính xác, và bằng nội suy tuyến tính
trên Bảng I đúng như bài báo làm:

| $k$ | $\epsilon=0{,}000$ | $0{,}010$ | $0{,}020$ | $0{,}050$ | $0{,}100$ | $0{,}200$ |
|---:|---:|---:|---:|---:|---:|---:|
| $1{,}0$ | $1{,}1070$ | $1{,}1378$ | $1{,}1707$ | $1{,}2762$ | $1{,}4904$ | $2{,}1517$ |
| $1{,}1$ | $1{,}0880$ | $1{,}1199$ | $1{,}1540$ | $1{,}2652$ | $1{,}4948$ | $2{,}2315$ |
| $1{,}2$ | $1{,}0720$ | $1{,}1051$ | $1{,}1407$ | $1{,}2587$ | $1{,}5058$ | $2{,}3445$ |
| $1{,}3$ | $1{,}0580$ | $1{,}0933$ | $1{,}1310$ | $1{,}2560$ | $1{,}5254$ | $2{,}4963$ |
| $1{,}4$ | $1{,}0470$ | $1{,}0843$ | $1{,}1239$ | $1{,}2585$ | $1{,}5544$ | $2{,}6997$ |
| $1{,}5$ | $1{,}0370$ | $1{,}0768$ | $1{,}1196$ | $1{,}2642$ | $1{,}5916$ | $2{,}9759$ |
| $1{,}6$ | $1{,}0290$ | $1{,}0719$ | $1{,}1169$ | $1{,}2746$ | $1{,}6399$ | $3{,}3622$ |
| $1{,}7$ | $1{,}0230$ | $1{,}0681$ | $1{,}1166$ | $1{,}2879$ | $1{,}6987$ | $3{,}9220$ |
| $1{,}8$ | $1{,}0180$ | $1{,}0657$ | $1{,}1184$ | $1{,}3053$ | $1{,}7720$ | $q_{\max}=3{,}36$ |
| $1{,}9$ | $1{,}0140$ | $1{,}0650$ | $1{,}1214$ | $1{,}3269$ | $1{,}8606$ | $q_{\max}=4{,}02$ |
| $2{,}0$ | $1{,}0100$ | $1{,}0654$ | $1{,}1353$ | $1{,}3521$ | $\mathbf{1{,}9679}$ | $q_{\max}=5{,}15$ |

Ba ô cuối cột $\epsilon=0{,}200$ được bài báo in là `>4`, và phép dựng lại giải thích vì sao:
$q_{\max}$ ở đó vượt $3{,}0$, tức vượt biên của Bảng I, nên không nội suy được.

**Kết quả đối chiếu: 62 trên 63 ô khớp.**

| Đo | Kết quả |
|---|---|
| Số ô đối chiếu | $63$ |
| Lệch quá $1{,}5$ đơn vị chữ số cuối, theo đường **nội suy** | $7$ |
| Biên phân tán của $60$ ô, bỏ ô lớn nhất ra | $8{,}3$ đơn vị chữ số cuối |
| Lệch trung bình của $60$ ô đó | $0{,}9$ đơn vị |
| Ô ngoại lệ: $k=2{,}0$, $\epsilon=0{,}100$ | $\mathbf{61{,}1}$ đơn vị |

#### 2.14.5 Ô thứ hai của bài báo không dựng lại được

| | Giá trị |
|---|---|
| Bài báo in | $2{,}029$ |
| Dựng lại bằng công thức (7) + Bảng I, nội suy tuyến tính | $\mathbf{1{,}9679}$ |
| Dựng lại bằng công thức chính xác | $1{,}9681$ |
| Chênh | $-0{,}0611$, tức $61{,}1$ đơn vị chữ số cuối |

Ô này lệch **gấp $7{,}4$ lần** ô lệch nhiều thứ hai và **gấp 66 lần** mức trung bình của 60 ô
còn lại — nó không phải phần đuôi của cùng một phân bố sai số.

**Và nó cũng tự lộ, y như ô của Bảng I.** Đọc cột $\epsilon=0{,}100$ theo sai phân:

| $k$ | $1{,}0$ | $1{,}1$ | $1{,}2$ | $1{,}3$ | $1{,}4$ | $1{,}5$ | $1{,}6$ | $1{,}7$ | $1{,}8$ | $1{,}9$ | $2{,}0$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bài báo in** | $1{,}490$ | $1{,}495$ | $1{,}506$ | $1{,}526$ | $1{,}557$ | $1{,}592$ | $1{,}639$ | $1{,}697$ | $1{,}769$ | $1{,}859$ | $\mathbf{2{,}029}$ |
| Sai phân | | $+0{,}005$ | $+0{,}011$ | $+0{,}020$ | $+0{,}031$ | $+0{,}035$ | $+0{,}047$ | $+0{,}058$ | $+0{,}072$ | $+0{,}090$ | $\mathbf{+0{,}170}$ |
| **Dựng lại** | $1{,}490$ | $1{,}494$ | $1{,}506$ | $1{,}525$ | $1{,}554$ | $1{,}591$ | $1{,}639$ | $1{,}699$ | $1{,}771$ | $1{,}860$ | $\mathbf{1{,}968}$ |
| Sai phân | | $+0{,}004$ | $+0{,}012$ | $+0{,}020$ | $+0{,}028$ | $+0{,}038$ | $+0{,}048$ | $+0{,}059$ | $+0{,}073$ | $+0{,}089$ | $\mathbf{+0{,}108}$ |

Bước cuối của cột đã in là $+0{,}170$, **gần gấp đôi** bước liền trước ($+0{,}090$), trong khi
mọi bước khác tăng đều và khớp dãy dựng lại tới đơn vị chữ số cuối.

**Và không cách đọc nào của (7) cho ra $2{,}029$.** Muốn ra con số đó thì phải vào Bảng I ở
$k = 2{,}7063$, trong khi công thức (7) cho $q_{\max} = 2{,}6080$ — chênh $0{,}0983$. Đây là
một ô không dựng lại được, và bài viết này ghi nó ra thay vì bỏ qua.

---

### 2.15 Mục 12 tr.99–101 — mở rộng $(M_r)$, và Hodges–Lehmann

![Mục 12, trang 100](<paper_huber1964_tr100_muc_12.png>)

**Hình 30 — Mục 12, định nghĩa $(M_r)$, Huber 1964, trang 100.** Cùng nguồn, trang 28 của PDF.
`--box 0.12,0.42,0.92,0.60 --sat`.

> *"$(M)$-estimators $T_n$ have been defined by the property that they minimize an expression
> of the form $\sum_{i=1}^n\rho(x_i - T_n)$. More generally, one may define
> $(M_r)$-estimators by the property that they minimize an expression of the form
> $\sum_I\rho_r(x_I - T_n)$ where $\rho_r$ is a symmetric function of $r$ arguments, the
> summation is extended over the $\binom{n}{r}$ subsets $I = \{i_1,\dots,i_r\}$..."*
>
> *"Since the amount of computation needed for determining the value of an $(M_r)$-estimator
> increases with the $r$th power of the sample size, they will hardly be of practical use for
> $r > 2$."*

**Ví dụ $r=2$.** Lấy $\rho_2(t_1,t_2) = \lvert t_1 + t_2\rvert$ thì
$\sum_I\rho_2(x_I - T)$ được cực tiểu hoá bởi **trung vị của mọi $(x_i+x_j)/2$** — tức đúng ước
lượng Hodges–Lehmann (1963). Phương sai tiệm cận của nó là

$$V(\psi_2, F) = \frac{1}{12\left(\int f^2\,dt\right)^{2}}$$

và trong trường hợp chuẩn bị nhiễm bẩn có cận trên sắc $V(\psi_2,F) < (1-\epsilon)^{-4}\pi/3$.

![Hai bảng trang 101](<paper_huber1964_tr101_hai_bang_so.png>)

**Hình 31 — Hai bảng số không đánh số, Huber 1964, trang 101.** Cùng nguồn, trang 29 của PDF.
`--box 0.12,0.11,0.92,0.36 --sat`.

**Bảng thứ nhất — cận Hodges–Lehmann, dựng lại từ $(1-\epsilon)^{-4}\pi/3$:**

| $\epsilon$ | $0{,}000$ | $0{,}001$ | $0{,}002$ | $0{,}005$ | $0{,}010$ | $0{,}020$ | $0{,}050$ | $0{,}100$ | $0{,}200$ | $0{,}500$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bài báo in | $1{,}047$ | $1{,}051$ | $1{,}056$ | $1{,}068$ | $1{,}090$ | $1{,}135$ | $1{,}286$ | $1{,}596$ | $2{,}557$ | $16{,}76$ |
| Đo lại | $1{,}047198$ | $1{,}051397$ | $1{,}055617$ | $1{,}068406$ | $1{,}090154$ | $1{,}135336$ | $1{,}285684$ | $1{,}596094$ | $2{,}556635$ | $16{,}755161$ |

**Mười trên mười ô khớp**, và $\pi/3 = 1{,}0471975512$ đúng là giá trị ở $\epsilon = 0$.

**Bảng thứ hai — mô hình nhiễm bẩn "quy ước" $F = (1-\epsilon)\Phi(t) + \epsilon\Phi(t/3)$:**

| $\epsilon$ | $0{,}0$ | $0{,}01$ | $0{,}02$ | $0{,}05$ | $0{,}10$ |
|---|---:|---:|---:|---:|---:|
| Hodges–Lehmann, bài báo in | $1{,}047$ | $1{,}07$ | $1{,}09$ | $1{,}17$ | $1{,}31$ |
| Hodges–Lehmann, đo lại | $1{,}04719755$ | $1{,}07064731$ | $1{,}09469712$ | $1{,}17061179$ | $1{,}31074145$ |
| Đề xuất 2, $k=1{,}5$, bài báo in | $1{,}037$ | $1{,}06$ | $1{,}08$ | $1{,}16$ | $1{,}30$ |
| Đề xuất 2, $k=1{,}5$, đo lại | $1{,}03709076$ | $1{,}05964650$ | $1{,}08293194$ | $1{,}15746262$ | $1{,}29922548$ |

**Mười trên mười ô khớp.**

Bài báo kết luận: *"apparently, the Hodges-Lehmann estimator and the estimators proposed in the
preceding sections are close competitors."* Đọc bảng kỹ hơn một chút thì thấy Đề xuất 2 thắng
ở **cả năm cột**, với biên lớn nhất $0{,}0131$ điểm phương sai. Câu "close competitors" của
tác giả vì thế đúng theo nghĩa "sát nhau", nhưng **có chiều**: Đề xuất 2 nhỉnh hơn ở mọi cột
được in ra.

---

## 3. Từ ước lượng VỊ TRÍ sang HỒI QUY — chỗ bài báo KHÔNG đi

Bài báo 1964 nói về ước lượng **một số**. Hồi quy Huber là phần đời sau, và mục này nói rõ chỗ
nào là của bài báo, chỗ nào là bắc cầu.

### 3.1 Bài báo cho phép bắc cầu, bằng đúng một câu

Câu duy nhất trong 29 trang chạm tới hồi quy nằm ngay dưới định nghĩa `(M)` ở tr.74:

> *"Of course, this definition generalizes at once to more general least squares type problems,
> where several parameters have to be determined."*

Phép bắc cầu là thay $x_i - T$ bằng phần dư của mô hình tuyến tính:

$$\hat\beta = \arg\min_\beta \sum_{i=1}^n \rho_k\!\left(\frac{y_i - x_i^\top\beta}{S}\right)$$

Tất cả những gì bài báo chứng minh về **tính lồi** chuyển sang ngay: $\rho_k$ lồi, và
$\beta\mapsto y_i - x_i^\top\beta$ tuyến tính, nên hàm mục tiêu vẫn lồi theo $\beta$.

### 3.2 Nhưng điều kiện dừng đổi hình dạng, và đó là chỗ mọi thứ khác đi

Cho đạo hàm theo $\beta$ bằng $0$:

$$\sum_{i=1}^n \psi_k\!\left(\frac{e_i}{S}\right) x_i = \mathbf 0,
\qquad e_i = y_i - x_i^\top\beta$$

So với bài toán vị trí, nơi điều kiện là $\sum_i\psi(x_i - T) = 0$, có **một thừa số mới**:
$x_i$. Và đó là toàn bộ nội dung của mục 6.2 dưới đây.

| | Điều kiện dừng | Ảnh hưởng của một quan sát |
|---|---|---|
| **Bài toán vị trí** (bài báo) | $\sum_i\psi(x_i-T)=0$ | $\psi(x_i-T)$ — **bị chặn** bởi $k$ |
| **Hồi quy** (bắc cầu) | $\sum_i\psi(e_i/S)\,x_i=\mathbf 0$ | $\psi(e_i/S)\,x_i$ — **KHÔNG bị chặn**, vì $x_i$ tự do |

Huber chặn được thừa số thứ nhất; **không ai chặn thừa số thứ hai**. Bài báo 1964 không gặp
vấn đề này vì trong bài toán vị trí mọi quan sát đóng góp với trọng số **bằng nhau** — không
có "leverage" nào để mà cao.

Đây là lý do câu *"generalizes at once"* đúng về **hình thức** và thiếu về **tính chất**: phép
mở rộng giữ được tính lồi và giữ được thuật toán, nhưng **không giữ được bảo đảm bền vững**.
Mục 6.2 đo phần mất đi bằng một con số.

### 3.3 Ba bảng vẽ ra ba tầng của cơ chế

Bài báo không có hình vẽ nào trong cả 29 trang, nên mọi hình dạng dưới đây là hình tự vẽ, và
mỗi bảng bù một chỗ mà bản in chỉ nói bằng chữ.

![Hàm rho, psi và mật độ f_0](<huber_rho_psi_f0.png>)

**Hình 32 — $\rho$, $\psi$ và mật độ ít thuận lợi nhất $f_0$ (hình tự vẽ).**

**Cách đọc hình.**

- **Bảng trái — $\rho$.** Ba đường: bình phương ($\rho=\frac12t^2$), Huber $k=1{,}345$, và trị
  tuyệt đối. Hai vạch đứng ở $\pm1{,}345$ là chỗ Huber **đổi nhánh**. Điều cần nhìn: bên trong
  hai vạch, đường Huber **nằm chồng** lên đường bình phương; bên ngoài, nó **thẳng** trong khi
  đường bình phương vọt lên. Tại $t=4$ thì $\rho$ bình phương là $8$ còn Huber là $4{,}47548750$, tức **rẻ hơn $1{,}7875$ lần**.
- **Bảng giữa — $\psi=\rho'$.** Đây là hình quan trọng nhất của bài. Đường bình phương là
  $\psi=t$, **không có trần**. Đường Huber phẳng ở $\pm1{,}345$ từ hai vạch trở ra. Đường trị
  tuyệt đối là bậc thang $\pm1$, **gãy** tại $0$. Ba hình dạng ấy là ba câu trả lời cho câu
  hỏi *"một điểm sai nhiều thì nên kéo mạnh tới đâu"*: vô hạn · có trần · luôn kéo hết sức.
- **Bảng phải — $f_0$ so với $\varphi$, trục dọc thang log.** Đường liền là $f_0$ ứng với
  $k=1{,}345$ (tức $\epsilon = 0{,}0579148930$), đường đứt là mật độ chuẩn. Điều cần nhìn: hai
  đường **trùng hình dạng** ở giữa nhưng $f_0$ thấp hơn theo đúng thừa số $(1-\epsilon)$; ở hai
  đuôi, $\varphi$ **cong xuống** (parabol trên thang log) còn $f_0$ **thẳng** — đó chính là
  "đuôi mũ" mà tr.75 nói, và trên thang log thì hàm mũ là một đường thẳng.

<details>
<summary><b>Mã sinh Hình 32</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

K = 1.345
EPS_F0 = 0.0579148930          # eps_min(1.345), in ra bởi v_huber.py mục 17
t = np.linspace(-4.0, 4.0, 1601)


def rho(t, k):
    a = np.abs(t)
    return np.where(a <= k, 0.5 * t * t, k * a - 0.5 * k * k)


fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))

ax[0].plot(t, 0.5 * t ** 2, color="#6b7280", lw=1.5, label=r"bình phương $\frac{1}{2} t^2$")
ax[0].plot(t, rho(t, K), color="#d97706", lw=2.0, label=r"Huber $k=1{,}345$")
ax[0].plot(t, np.abs(t), color="#be185d", lw=1.5, ls=":", label=r"trị tuyệt đối $|t|$")
for v in (-K, K):
    ax[0].axvline(v, color="#111827", lw=0.8, ls="--", alpha=0.6)
ax[0].set_ylim(0, 8.4)
ax[0].set_xlabel("phần dư $t$"); ax[0].set_ylabel(r"$\rho(t)$")
ax[0].set_title("HÀM MẤT MÁT\nHuber chồng lên bình phương ở giữa,\nthẳng ở ngoài", fontsize=9)
ax[0].legend(fontsize=7.6, loc="upper center")

ax[1].plot(t, t, color="#6b7280", lw=1.5, label=r"bình phương: $\psi=t$")
ax[1].plot(t, np.clip(t, -K, K), color="#d97706", lw=2.2, label=r"Huber: $\psi=\mathrm{clip}$")
ax[1].plot(t, np.sign(t), color="#be185d", lw=1.5, ls=":", label=r"trị tuyệt đối: $\psi=\mathrm{sign}$")
for v in (-K, K):
    ax[1].axvline(v, color="#111827", lw=0.8, ls="--", alpha=0.6)
    ax[1].axhline(v, color="#d97706", lw=0.7, ls="--", alpha=0.5)
ax[1].set_ylim(-4.2, 4.2)
ax[1].set_xlabel("phần dư $t$"); ax[1].set_ylabel(r"$\psi(t)=\rho'(t)$")
ax[1].set_title("HÀM ẢNH HƯỞNG\nchỉ Huber vừa CÓ TRẦN vừa LIÊN TỤC", fontsize=9)
ax[1].legend(fontsize=7.6, loc="upper left")

f0 = (1.0 - EPS_F0) * norm.pdf(0.0) * np.exp(-rho(t, K))
ax[2].semilogy(t, norm.pdf(t), color="#2563eb", lw=1.5, ls="--", label=r"chuẩn $\varphi(t)$")
ax[2].semilogy(t, f0, color="#dc2626", lw=2.0, label=r"$f_0$ ở $k=1{,}345$")
for v in (-K, K):
    ax[2].axvline(v, color="#111827", lw=0.8, ls="--", alpha=0.6)
ax[2].set_xlabel("$t$"); ax[2].set_ylabel("mật độ, thang log")
ax[2].set_title("PHÂN PHỐI ÍT THUẬN LỢI NHẤT\nchuông ở giữa, ĐUÔI MŨ ở ngoài\n"
                "(trên thang log: đuôi mũ là ĐƯỜNG THẲNG)", fontsize=9)
ax[2].legend(fontsize=7.6, loc="lower center")

fig.tight_layout()
fig.savefig("huber_rho_psi_f0.png", bbox_inches="tight")
```

</details>

---

## 4. Thuật toán, viết đủ để dựng lại

### 4.1 IRLS — cách cả thế giới cài đặt, và vì sao nó không cần bộ tối ưu mới

```
   X (n×p), y (n), k = 1.345
        │
        ▼
   β̂ ← nghiệm bình phương tối thiểu       khởi đầu; bài toán lồi nên điểm nào cũng được
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  LẶP tới khi β̂ thôi đổi (|Δβ̂| < tol)                │
   │                                                     │
   │   1. r  ← y − X β̂            phần dư hiện tại        │
   │   2. S  ← ước lượng thang đo  ← MỤC 4.3 nói chỗ này  │
   │           │                     khác nhau giữa các   │
   │           │                     thư viện             │
   │   3. w  ← min(1, kS / |r|)   TRỌNG SỐ                │
   │           │                  · |r| ≤ kS  →  w = 1    │
   │           │                  · |r| > kS  →  w < 1    │
   │   4. β̂  ← bình phương tối thiểu CÓ TRỌNG SỐ           │
   │           trên (√w · X, √w · y)                      │
   └─────────────────────────────────────────────────────┘
        │
        ▼
   β̂ cuối cùng      (không hội tụ ⟹ BÁO LỖI, không trả vòng cuối)
```

**Bỏ từng phần đi thì hỏng chuyện gì:**

| Thành phần | Nhận vào | Trả ra | Bỏ nó đi thì hỏng gì |
|---|---|---|---|
| **Bước 3 — trọng số** | $r$, $k$, $S$ | $w\in(0,1]$ | Đặt $w\equiv1$ là quay về đúng bình phương tối thiểu. Đây là **toàn bộ** Huber, gói trong một dòng. |
| **Bước 2 — ước lượng $S$** | $r$ | một số dương | Ngưỡng $kS$ mất đơn vị: đổi $y$ từ mét sang milimét là kết quả đổi theo. Dùng độ lệch chuẩn thay ước lượng bền vững cũng hỏng — chính ngoại lai làm phồng độ lệch chuẩn, nên ngưỡng nở ra và bao luôn ngoại lai. |
| **Bước 4 — bình phương tối thiểu có trọng số** | $w$, $X$, $y$ | $\hat\beta$ mới | Không có gì mới ở đây: đó là bộ giải sẵn có. Điểm hay của IRLS chính là **không cần viết bộ tối ưu mới** — đúng câu tr.97 của bài báo về *"existing computer programs"*. |
| **Vòng lặp** | $\hat\beta$ hiện tại | $\hat\beta$ đã ổn định | Chạy một vòng là chưa đủ: trọng số phụ thuộc phần dư, mà phần dư phụ thuộc $\hat\beta$. Hai thứ phải cùng ổn định. |
| **Câu lệnh báo lỗi khi chưa hội tụ** | — | — | Trả về vòng lặp cuối một cách im lặng là cách chắc chắn để có một con số sai mà không ai biết. Khi $S$ được ước lượng lại mỗi vòng thì đây là một thủ tục **ghép**, và hội tụ không được bảo đảm sẵn — đúng chỗ tr.98 nói *"I do not know whether and when the latter case occurs"*. |

```python
import numpy as np


def irls_huber(X, y, k=1.345, thang=None, so_vong=500, tol=1e-14):
    """IRLS cho hồi quy Huber. thang=None -> ước lượng lại bằng MAD mỗi vòng."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]          # khởi đầu bằng bình phương tối thiểu
    for _ in range(so_vong):
        r = y - X @ b
        s = thang if thang is not None else max(
            np.median(np.abs(r - np.median(r))) / 0.6745, 1e-12)
        w = np.clip(k * s / np.maximum(np.abs(r), 1e-300), None, 1.0)
        sw = np.sqrt(w)
        moi = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.abs(moi - b).max() < tol:
            return moi
        b = moi
    raise RuntimeError("IRLS chưa hội tụ; tăng so_vong hoặc kiểm tra dữ liệu/thang đo")
```

Trọng số $w_i = \min(1,\;kS/\lvert r_i\rvert)$ có nghĩa rất trực tiếp: điểm trong ngưỡng có
trọng số $1$ (như bình phương tối thiểu), điểm ngoài ngưỡng có trọng số giảm theo
$1/\lvert r_i\rvert$ — nên đóng góp của nó vào phương trình chuẩn là $w_i r_i = kS$, **hằng
số**. Đó là cách viết lại $\psi$ bị chặn dưới dạng trọng số.

### 4.2 Thuật toán (6′) của chính bài báo, cài đặt nguyên văn

IRLS là cách đời sau; đây là cách bài báo dặn. Nó giải **đồng thời** $T$ và $S$ cho bài toán
vị trí, và mỗi vòng chỉ cần đếm ba con số nguyên.

```python
import numpy as np
from scipy.stats import norm


def E_psi_binh(k):
    """E_Phi psi^2(k) — vế phải của hệ (6) theo Đề xuất 2."""
    return (2 * norm.cdf(k) - 1) - 2 * k * norm.pdf(k) + 2 * k * k * (1 - norm.cdf(k))


def huber_6_phay(x, k, beta=None, T0=None, S0=None, so_vong=200, tol=1e-14):
    """Thuật toán (6') tr.98. m1, m2, m3 và trung bình cắt x' theo đúng bài báo."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if beta is None:
        beta = E_psi_binh(k)                       # Đề xuất 2
    T = float(np.mean(x)) if T0 is None else float(T0)
    S = float(np.std(x, ddof=1)) if S0 is None else float(S0)
    for vong in range(so_vong):
        lo, hi = T - k * S, T + k * S
        m1 = int(np.sum(x <= lo))
        m3 = int(np.sum(x >= hi))
        giua = (x > lo) & (x < hi)
        m2 = int(giua.sum())
        if m2 == 0:
            raise RuntimeError("m2 = 0, hệ (6') không xác định")
        xp = float(x[giua].mean())                 # trung bình cắt bớt
        mau = n * beta - (m1 + m3 + (m3 - m1) ** 2 / m2) * k * k
        if mau <= 0:
            raise RuntimeError("mẫu số của S^2 không dương — bài báo dặn phải chặn")
        S1 = float(np.sqrt(((x[giua] - xp) ** 2).sum() / mau))
        T1 = xp + k * S1 * (m3 - m1) / m2
        if abs(T1 - T) < tol and abs(S1 - S) < tol:
            return T1, S1, vong + 1
        T, S = T1, S1
    raise RuntimeError("(6') chưa hội tụ")
```

Hai câu bảo vệ `m2 == 0` và `mau <= 0` không phải trang trí — chúng là đúng hai chỗ bài báo
dặn ở tr.99: *"it might be wise to safeguard against too small values of $m_2$ and of the
denominator in the expression defining $S_1^2$."*

### 4.3 scikit-learn cài hệ (6) với $\beta = 1$, không phải $\beta = E_\Phi\psi^2$

Đây là phát hiện chính của mục 4, và nó giải thích một chỗ lệch mà bản trước của bài này chỉ
mô tả được chứ không truy được về nguồn.

`HuberRegressor` cực tiểu hoá theo $(w,\sigma)$:

$$\sum_i\Bigl(\sigma + H_\varepsilon\bigl(r_i/\sigma\bigr)\,\sigma\Bigr) + \alpha\lVert w\rVert^2$$

Lấy đạo hàm theo $\sigma$ rồi cho bằng $0$ thì ra **đúng phương trình thứ hai của hệ (6)**
tr.97, với một $\beta$ cụ thể:

$$n^{-1}\sum_i\psi^2\bigl(k, r_i/S\bigr) = \mathbf{1}$$

Đề xuất 2 của bài báo thì đặt $\beta = E_\Phi\psi^2(k,x)$, và chính bài báo ghi
$\beta < k^2$; ở $k = 1{,}345$ thì

$$\beta_{\text{Đề xuất 2}} = E_\Phi\psi^2(1{,}345) = 0{,}7101645483,
\qquad \beta_{\text{sklearn}} = 1,
\qquad \text{tỉ số } 1{,}4081243600$$

**Hệ quả đoán trước được:** vế phải lớn hơn đòi $S$ **nhỏ hơn**, nên ngưỡng $kS$ chặt hơn, nên
nhiều điểm bị giảm trọng số hơn.

**Đo, trên cùng dữ liệu $n=100$, cùng $k=1{,}345$:**

| Mức đẩy lệch | sklearn: hệ số | `scale_` | Hệ (6) với $\beta=1$: hệ số | $S$ | Đề xuất 2: hệ số | $S$ |
|---:|---:|---:|---:|---:|---:|---:|
| $0$ | $1{,}1095543344$ | $0{,}568033$ | $1{,}1095525312$ | $0{,}568032$ | $1{,}1199984817$ | $0{,}889447$ |
| $100$ | $1{,}1010663787$ | $0{,}583348$ | $1{,}1010652955$ | $0{,}583348$ | $1{,}1099811489$ | $0{,}910258$ |
| $100\,000$ | $1{,}1010559667$ | $0{,}583350$ | $1{,}1010652955$ | $0{,}583348$ | $1{,}1099811489$ | $0{,}910258$ |

**Cột 2 và cột 4 trùng nhau tới dung sai của bộ tối ưu** — đó là bằng chứng:

| Mức đẩy lệch | $\lvert$hệ số$\rvert$ chênh | $\lvert$thang đo$\rvert$ chênh |
|---:|---:|---:|
| $0$ | $1{,}803\times10^{-6}$ | $3{,}395\times10^{-7}$ |
| $100$ | $1{,}083\times10^{-6}$ | $5{,}411\times10^{-7}$ |
| $100\,000$ | $9{,}329\times10^{-6}$ | $2{,}430\times10^{-6}$ |

Ba trên ba dòng khớp tới $10^{-6}$, tức **`scikit-learn` = hệ (6) của Huber 1964 với
$\beta = 1$**.

**Và đây là bảng ba cách, để thấy mức chênh giữa chúng:**

| Mức đẩy lệch | IRLS + MAD: hệ số · thang | sklearn: hệ số · thang | Đề xuất 2: hệ số · thang | MAD lệch sklearn | ĐX 2 lệch sklearn |
|---:|---|---|---|---:|---:|
| $0$ | $1{,}116426$ · $0{,}806835$ | $1{,}109554$ · $0{,}568033$ | $1{,}119998$ · $0{,}889447$ | $0{,}6193\ \%$ | $0{,}9413\ \%$ |
| $100$ | $1{,}107912$ · $0{,}859078$ | $1{,}101066$ · $0{,}583348$ | $1{,}109981$ · $0{,}910258$ | $0{,}6217\ \%$ | $0{,}8096\ \%$ |
| $100\,000$ | $1{,}107912$ · $0{,}859078$ | $1{,}101056$ · $0{,}583350$ | $1{,}109981$ · $0{,}910258$ | $0{,}6227\ \%$ | $0{,}8106\ \%$ |

> **Điều đáng nhớ nhất của cả mục 4.** "Hồi quy Huber" **không phải một thuật toán duy nhất**.
> Nó là một họ, và các thư viện khác nhau ở đúng một chỗ: **hằng số $\beta$ ở vế phải của
> phương trình thang đo**. Bài báo để ngỏ chỗ đó bằng một câu — *"I do not want to exclude the
> possibility that small sample size considerations might lead to a different choice of
> $\beta$"* — và sáu mươi năm sau, đó đúng là chỗ các bản cài đặt tách nhau.
>
> Khi hai thư viện cho hai con số, **kiểm $\beta$ trước**, đừng kiểm thuật toán.

### 4.4 Sai số chuẩn: công thức "bánh kẹp" đã có từ 1964

Công thức ở tr.97, áp cho bộ $n=100$ của mục 1.2:

| Đại lượng | Giá trị |
|---|---|
| $n$ | $100$ |
| $\sum\psi^2\bigl(k,(x_i-T)/S\bigr)$ | $99{,}99994132$ |
| $\sum\psi'$, tức số quan sát trong ngưỡng | $61$ |
| Ước lượng phương sai của $T$ | $0{,}0087589426$ |
| **Sai số chuẩn** | $\mathbf{0{,}0935892226}$ |
| Sai số chuẩn của bình phương tối thiểu, để so | $0{,}0963067742$ |

Chú ý rằng $\sum\psi^2 \approx n$ — đó chính là phương trình $\beta = 1$ của `scikit-learn`,
hiện ra trong một phép tính hoàn toàn khác. Và $\sum\psi' = 61$ nói rằng **39 trên 100** quan
sát đang bị giảm trọng số ở ngưỡng này, dù dữ liệu sinh ra từ phân phối chuẩn thuần tuý — một
con số đáng biết trước khi dùng `scale_` của thư viện để suy ra gì đó.

---

## 5. Vì sao chặn ảnh hưởng lại ăn thua

Câu trả lời có ba tầng, và cả ba đều ở trong bài báo.

### 5.1 Tầng một — cơ chế: gradient bị chặn thì một điểm không mua thêm được ảnh hưởng

Điều kiện dừng là $\sum_i\psi(e_i)\,x_i = 0$. Hàm $\psi$ nói một điểm **kéo** nghiệm mạnh tới
đâu. Toàn bộ khác biệt nằm ở một dòng:

| | $\psi(e)$ | Điểm có $e=100$ kéo mạnh gấp mấy điểm có $e=1$? |
|---|---|---|
| **Bình phương tối thiểu** | $e$ | **100 lần** |
| **Huber** | $\operatorname{clip}(e,-kS,+kS)$ | $kS/1$ lần, rồi **thôi** |

Ví dụ ở mục 0.4 cho con số: quan sát $25$ lệch $21{,}17$ đơn vị và quan sát $1$ lệch $2{,}83$
đơn vị, nhưng cả hai kéo bằng nhau vì cùng chạm trần. Với trung bình mẫu thì tỉ số là $7{,}5$
lần.

### 5.2 Tầng hai — vì sao TRẦN chứ không phải LOẠI BỎ

Loại bỏ hẳn các điểm ngoài ngưỡng là "trung bình cắt bớt" — ví dụ (iv) của bài báo. Nó nghe
hợp lý hơn, và nó **tệ hơn**:

| | $V$ dưới $\Phi$ ở $k=1{,}345$ | Vì sao |
|---|---:|---|
| Huber (Winsor hoá) | $1{,}0526312912$ | $\psi$ liên tục |
| Cắt bớt | $2{,}5837988408$ | $\psi$ **nhảy** xuống $0$ tại $\pm k$ |

Gấp **$2{,}4546$ lần**. Bài báo chẩn đoán ở tr.80: *"The “trimming procedure” is rather sensitive
to the behavior of $F$ at the rejection points $\pm k$; a high density at these points will
play havoc with the estimate."* Một quan sát nhích qua ngưỡng làm nó đi từ đóng góp đầy đủ
sang đóng góp **không gì cả**, nên nghiệm nhảy. Huber thì chỉ chuyển từ "đóng góp $e$" sang
"đóng góp $k$" — liên tục.

Đây cũng là chỗ trả lời câu hỏi thực hành hay gặp: *"sao không cứ loại ngoại lai ra rồi chạy
bình phương tối thiểu?"* Vì loại bỏ là một quyết định **rời rạc**, và mọi quyết định rời rạc
đều tạo ra một chỗ nghiệm nhảy.

### 5.3 Tầng ba — con số 1,345, và nó đến từ đâu

Bài báo **không chứa** con số $1{,}345$ và **không chứa** chuỗi "95 %". Cả hai là hệ quả của
công thức (1), và suy ra được bằng hai dòng.

Hiệu suất tiệm cận tại $\Phi$ là tỉ số giữa cận Cramér–Rao và phương sai thực tế. Thông tin
Fisher của $\Phi$ bằng $1$, nên hiệu suất $= 1/V(k)$. Giải $1/V(k) = 0{,}95$:

| Đại lượng | Giá trị |
|---|---|
| $k$ cho hiệu suất đúng $95\ \%$ | $\mathbf{1{,}3449975085}$ |
| Hiệu suất tại $k = 1{,}345$ | $0{,}9500002597$ |
| Hiệu suất tại $k = 1{,}35$ | $0{,}9505193431$ |

Con số $1{,}345$ là $1{,}3449975085$ làm tròn về bốn chữ số.

**Và nó có nghĩa thứ hai, đọc từ cột $\epsilon_{\min}$ của Bảng I:**

$$\epsilon_{\min}(1{,}345) = 0{,}0579148930$$

Tức $k = 1{,}345$ đồng thời là **nghiệm minimax cho mức nhiễm bẩn $5{,}79\ \%$**. Hai cách đọc
khác nhau của cùng một con số, và cách thứ hai mới là cách của bài báo:

| Đặt $k$ vì | Câu hỏi nó trả lời | Ở $k=1{,}345$ |
|---|---|---|
| Hiệu suất | "chấp nhận mất bao nhiêu khi dữ liệu sạch?" | mất $0{,}0499997403$, tức $5{,}0000\ \%$ |
| $\epsilon_{\min}$ | "đang chống mức nhiễm bẩn nào?" | $5{,}79\ \%$ |

**Bảng hiệu suất đầy đủ:**

| $k$ | Hiệu suất tại $\Phi$ | $V$ tại $\Phi$ | $\epsilon_{\min}$ |
|---:|---:|---:|---:|
| $0{,}500$ | $0{,}7920531043$ | $1{,}2625416081$ | $0{,}44171076$ |
| $1{,}000$ | $0{,}9031241548$ | $1{,}1072674722$ | $0{,}14283089$ |
| $\mathbf{1{,}345}$ | $\mathbf{0{,}9500002597}$ | $1{,}0526312912$ | $0{,}05791489$ |
| $1{,}500$ | $0{,}9642357653$ | $1{,}0370907572$ | $0{,}03760623$ |
| $2{,}000$ | $0{,}9897155897$ | $1{,}0103912784$ | $0{,}00841922$ |
| $3{,}000$ | $0{,}9995984137$ | $1{,}0004017476$ | $0{,}00025470$ |
| $5{,}000$ | $0{,}9999999613$ | $1{,}0000000387$ | $0{,}00000002$ |

**Đối chiếu ngược với Bảng I** để chắc rằng con số này thật sự nằm trong bảng của bài báo:
hàng $k=1{,}3$ cột $\epsilon=0$ in $1{,}058$, đo lại $1{,}058047$; hàng $k=1{,}4$ in $1{,}047$,
đo lại $1{,}046591$. Nội suy tuyến tính tại $1{,}345$ cho $1{,}053050$, công thức cho
$1{,}052631$ — chênh $0{,}0004$, tức nội suy đủ tốt.

![Hiệu suất và mức nhiễm bẩn tối ưu theo k](<huber_hieu_suat.png>)

**Hình 33 — Hai cách đọc cùng một $k$ (hình tự vẽ).**

**Cách đọc hình.**

- **Trục dọc trái, đường cam** — hiệu suất tiệm cận tại $\Phi$, tức $1/V(k)$. Nó tăng đơn điệu
  từ $0$ (ở $k=0$, trung vị) tới $1$ (ở $k\to\infty$, trung bình). Vạch ngang ở $0{,}95$ và
  vạch đứng ở $1{,}345$ cắt nhau **đúng trên đường** — đó là định nghĩa của con số.
- **Trục dọc phải, đường xanh, thang log** — $\epsilon_{\min}(k)$, mức nhiễm bẩn mà $k$ đó là
  nghiệm minimax. Nó **giảm** khi $k$ tăng: ngưỡng rộng hơn nghĩa là đang phòng ít hơn.
- Điều cần nhìn: hai đường đi **ngược chiều**, và điểm $k=1{,}345$ là một chỗ chọn trên đường
  đánh đổi ấy, không phải một điểm tối ưu của bất kỳ hàm nào. Chọn $k=2$ thì hiệu suất lên
  $98{,}97\ \%$ nhưng mức phòng tụt còn $0{,}84\ \%$.

<details>
<summary><b>Mã sinh Hình 33</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

PHI, phi = norm.cdf, norm.pdf


def E_psi_pham(k):
    return 2 * PHI(k) - 1


def E_psi_binh(k):
    return (2 * PHI(k) - 1) - 2 * k * phi(k) + 2 * k * k * (1 - PHI(k))


def eps_min(k):
    return 1.0 - 1.0 / ((2 * PHI(k) - 1) + 2 * phi(k) / k)


ks = np.linspace(0.05, 3.0, 591)
hieu_suat = E_psi_pham(ks) ** 2 / E_psi_binh(ks)
em = eps_min(ks)

fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(ks, hieu_suat, color="#d97706", lw=2.2, label="hiệu suất tại $\\Phi$ (trục trái)")
ax.axhline(0.95, color="#111827", lw=0.9, ls="--", alpha=0.7)
ax.axvline(1.3449975085, color="#111827", lw=0.9, ls="--", alpha=0.7)
ax.plot([1.3449975085], [0.95], "o", color="#dc2626", ms=7, zorder=5)
ax.annotate("$k = 1{,}3449975085$\nhiệu suất $= 0{,}95$",
            xy=(1.3449975085, 0.95), xytext=(1.62, 0.70), fontsize=8.4,
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.9))
ax.set_xlabel("ngưỡng $k$")
ax.set_ylabel("hiệu suất tiệm cận tại phân phối chuẩn")
ax.set_ylim(0.0, 1.03)

ax2 = ax.twinx()
ax2.semilogy(ks, em, color="#2563eb", lw=2.0, ls="-.",
             label="$\\epsilon_{\\min}(k)$ (trục phải, thang log)")
ax2.set_ylabel("$\\epsilon_{\\min}$ — mức nhiễm bẩn mà $k$ là nghiệm minimax")
ax2.grid(False)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="center right")
ax.set_title("Cùng một $k$, hai cách đọc — và hai đường đi NGƯỢC CHIỀU", fontsize=10)

fig.savefig("huber_hieu_suat.png", bbox_inches="tight")
```

</details>

---

## 6. Giá phải trả, và khi nào đừng dùng

### 6.1 Bảng được — mất

> | Mặt được | Đo được bao nhiêu | Trả giá bằng gì | Số này ở đâu ra |
> |---|---|---|---|
> | Một điểm hỏng theo $y$ không kéo được nghiệm | hệ số góc lệch $-64{,}1422\ \%$ $\to$ $-0{,}0633\ \%$ với $1/60$ điểm bị đẩy $+100$ | **mất nghiệm đóng**: bình phương tối thiểu giải một hệ, Huber phải lặp tới hội tụ | mục 6.2 · mục 4.1 |
> | Vẫn trơn, vẫn lồi | khả vi một lần; cắt bớt thì không, và tệ hơn $2{,}4546$ lần | thêm siêu tham số $k$, và một đại lượng phụ $S$ phải ước lượng | mục 5.2 · mục 4.3 |
> | Gần như không mất gì khi dữ liệu sạch | giữ $95{,}00\ \%$ hiệu suất ở $k=1{,}345$ | $5{,}0000\ \%$ đó là mất **thật**: nếu chắc chắn dữ liệu sạch và chuẩn thì bình phương tối thiểu tốt hơn | mục 5.3 |
> | Có bảo đảm minimax, chứng minh được | Định lý 1, 2, 4, 5; $V$ có trần ở mọi $\epsilon$ | bảo đảm đó chỉ cho **bài toán vị trí** và chỉ cho $H$ **đối xứng** | mục 2.7 · 2.11 |
> | Chống được ngoại lai theo $y$ | bảng mục 1.2 | **KHÔNG chống được điểm xa theo $X$** — điểm sụp đổ vẫn là $1/n$ | **mục 6.3** |
> | — | — | **chệch** trước nhiễm bẩn KHÔNG đối xứng: tệ hơn trung vị từ $1{,}22$ tới $2{,}12$ lần | mục 2.10 |
> | — | — | thêm dữ liệu có lúc là **hại**: quá $n=9$, $600$, $35\,000$ thì chệch chi phối | mục 2.10 |
> | Kết quả tái lập giữa các thư viện | — | **không đảm bảo**: $\beta$ khác nhau cho hệ số lệch $0{,}62$–$0{,}94\ \%$ | mục 4.3 |

Năm dòng cuối là năm dòng thua, và **dòng điểm sụp đổ là dòng quan trọng nhất của cả bài**.

### 6.2 Ngoại lai theo $y$: chỗ Huber làm đúng việc của nó

$n = 60$ điểm sinh từ $y = 2 + 1{,}5x$, đẩy lệch **đúng một điểm**, tức $1{,}67\ \%$ dữ liệu:

| Mức đẩy | Bình phương tối thiểu: chặn · góc | Huber: chặn · góc | BPTT lệch góc | Huber lệch góc |
|---|---|---|---:|---:|
| không đẩy | $+2{,}022501$ · $+1{,}518192$ | $+2{,}024748$ · $+1{,}500557$ | — | — |
| $+20$ | $+2{,}339040$ · $+1{,}322127$ | $+2{,}028906$ · $+1{,}499050$ | $-11{,}8582\ \%$ | $-0{,}0633\ \%$ |
| $+50$ | $+2{,}813849$ · $+1{,}028030$ | $+2{,}028906$ · $+1{,}499050$ | $-31{,}4647\ \%$ | $-0{,}0633\ \%$ |
| $+100$ | $+3{,}605196$ · $+0{,}537867$ | $+2{,}028906$ · $+1{,}499050$ | $\mathbf{-64{,}1422\ \%}$ | $\mathbf{-0{,}0633\ \%}$ |

Hệ số góc thật là $1{,}5$. Một điểm hỏng trong sáu mươi làm bình phương tối thiểu mất **gần
hai phần ba** hệ số; Huber lệch $0{,}0633\ \%$.

Và điều đáng chú ý hơn: **hệ số Huber giống hệt nhau ở cả ba mức** $+20$, $+50$, $+100$ — tới
mười chữ số. Khi một điểm đã bị xếp là "ngoài ngưỡng", việc nó ngoài xa bao nhiêu không còn
quan trọng nữa. Đó chính là cơ chế của mục 5.1, nhìn thấy bằng số.

![Ngoại lai theo y và điểm xa theo X](<huber_diem_sup_do.png>)

**Hình 34 — Hai loại dữ liệu hỏng, và Huber chỉ chống được một loại (hình tự vẽ).**

**Cách đọc hình.**

- **Bảng trái — ngoại lai theo $y$.** 60 điểm, một điểm khoanh đỏ bị đẩy lên $+100$ (nằm ngoài
  khung, mũi tên chỉ lên). Ba đường: đen đứt là quan hệ thật, xám là bình phương tối thiểu, cam
  là Huber. Điều cần nhìn: đường **đen đứt vẽ ĐÈ lên đường cam** và hai đường trùng nhau tới
  mức chỉ thấy các khoảng hở của nét đứt, còn đường xám xoay hẳn đi. Đây là trường hợp Huber
  được thiết kế để xử lý.
- **Bảng phải — điểm xa theo $X$, trục hoành thang log.** Trục hoành là **vị trí $x$ của điểm
  hỏng**, từ $1$ tới $5000$; trục dọc là hệ số góc mà hai phương pháp trả về. Điều cần nhìn:
  hai đường **hội tụ vào nhau** khi $x$ tăng — ở $x = 5000$ chúng chênh nhau
  $5{,}98\times10^{-7}$. Nghĩa là phần bền vững của Huber **biến mất hẳn**.
- **Đường chấm đỏ ở bảng phải, đọc theo trục dọc BÊN PHẢI, thang log** — phần dư của chính
  điểm hỏng dưới nghiệm Huber. Nó **co về 0**: ở $x=1$ là $39{,}9530469762$, ở $x=5000$ là
  $0{,}0000443698$, tức nhỏ đi gần **chín trăm nghìn lần**. Đó là cơ chế: đường khớp bị kéo đi
  **qua** điểm hỏng, nên phần dư của nó nhỏ, nên $\psi$ không bao giờ chạm ngưỡng, nên Huber
  **không nhìn thấy nó**. Chú giải đặt dưới khung để không đè lên đường nào.

<details>
<summary><b>Mã sinh Hình 34</b> — khối tự chứa, copy và chạy trực tiếp</summary>

```python
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

# Mọi mảng dưới đây do tools/v_huber.py mục 17 in ra; dán thẳng, không tính lại.
rng = np.random.default_rng(4)
X60 = rng.normal(size=(60, 1))[:, 0]
y60 = 2.0 + 1.5 * X60 + rng.normal(size=60)
DAY_100 = [3.6051961277, 0.5378673900, 2.0289057327, 1.4990501567]   # chặn·góc BPTT, chặn·góc Huber
X_LEV = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
GOC_OLS_LEV = [0.7923637083, 0.1933227051, -0.8938765520, -1.0092735930, -0.2370683869,
               0.7196398293, 1.1025691233, 1.3003004997, 1.4199948781, 1.4599859019,
               1.4799909463, 1.4919960101]
GOC_HUB_LEV = [1.4637532213, 1.4348924112, 1.3423120135, 1.1974526464, 0.9141054639,
               0.7122301531, 1.1015054751, 1.3001685402, 1.4199938060, 1.4599854058,
               1.4799898892, 1.4919954121]
DU_LEV = [39.9530469762, 39.8533473548, 39.1814484055, 36.9325837786, 28.2096128914,
          0.5462625192, 0.1219155016, 0.0282883411, 0.0047250478, 0.0011895083,
          0.0002851542, 0.0000443698]
GOC_THAT = 1.5

fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.3))

# --- bảng trái: ngoại lai theo y ---
yb = y60.copy()
yb[0] += 100.0
xs = np.linspace(X60.min() - 0.2, X60.max() + 0.2, 100)
ax[0].scatter(X60, yb, s=16, color="#6b7280", zorder=3)
ax[0].scatter([X60[0]], [yb[0]], s=70, facecolor="none", edgecolor="#dc2626", lw=1.6, zorder=4)
ax[0].annotate("điểm bị đẩy $+100$\n(ngoài khung)", xy=(X60[0], 8.6), xytext=(X60[0] - 0.1, 6.2),
               fontsize=8.2, ha="center", color="#dc2626",
               arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
ax[0].plot(xs, DAY_100[0] + DAY_100[1] * xs, color="#6b7280", lw=1.8,
           label="bình phương tối thiểu (góc $0{,}5379$)")
ax[0].plot(xs, DAY_100[2] + DAY_100[3] * xs, color="#d97706", lw=3.0,
           label="Huber $k=1{,}345$ (góc $1{,}4991$)")
# Vẽ quan hệ thật SAU CÙNG và bằng nét đứt mảnh: nó gần trùng đường Huber, nên nếu
# vẽ trước thì bị che kín và hình mất một đường mà chú giải vẫn liệt kê.
ax[0].plot(xs, 2.0 + GOC_THAT * xs, "--", color="#111827", lw=1.3, zorder=6,
           label="quan hệ thật (góc $1{,}5$)")
ax[0].set_ylim(-3.0, 9.0)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
ax[0].set_title("NGOẠI LAI THEO $y$ — 1/60 điểm\nHuber nằm chồng lên quan hệ thật", fontsize=9.6)
ax[0].legend(fontsize=7.8, loc="lower right")

# --- bảng phải: điểm xa theo X ---
ax[1].semilogx(X_LEV, GOC_OLS_LEV, "o-", color="#6b7280", lw=1.6, ms=4,
               label="bình phương tối thiểu")
ax[1].semilogx(X_LEV, GOC_HUB_LEV, "s-", color="#d97706", lw=2.0, ms=4.5,
               label="Huber $k=1{,}345$")
ax[1].axhline(GOC_THAT, color="#111827", lw=1.1, ls="--", alpha=0.8, label="góc thật $1{,}5$")
ax[1].set_xlabel("vị trí $x$ của điểm hỏng (thang log)")
ax[1].set_ylabel("hệ số góc ước lượng được")
ax[1].set_title("ĐIỂM XA THEO $X$ — cũng 1 điểm\nhai đường HỘI TỤ: phần bền vững biến mất",
                fontsize=9.6)

ax2 = ax[1].twinx()
ax2.loglog(X_LEV, DU_LEV, ":", color="#dc2626", lw=1.8, marker="^", ms=4,
           label="phần dư của chính điểm hỏng (trục phải)")
ax2.set_ylabel("phần dư của chính điểm hỏng (thang log)", color="#dc2626")
ax2.tick_params(axis="y", colors="#dc2626")
ax2.grid(False)

# Chú giải đặt DƯỚI khung: mọi vị trí bên trong đều bị một trong bốn đường đi qua.
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=7.8, ncol=2,
             loc="upper center", bbox_to_anchor=(0.5, -0.17))

fig.tight_layout()
fig.savefig("huber_diem_sup_do.png", bbox_inches="tight")
```

</details>

### 6.3 Điểm sụp đổ: một điểm đủ, và đây là con số

**Điểm sụp đổ** (breakdown point) là tỉ lệ dữ liệu hỏng tối đa mà ước lượng còn chịu được.

| Ước lượng | Điểm sụp đổ | Ghi chú |
|---|---|---|
| Bình phương tối thiểu | $1/n$, tiến về **0 %** | một điểm thích hợp làm ước lượng lớn tuỳ ý |
| **Huber, bài toán VỊ TRÍ** | **gần 50 %** | ở đây không có leverage; đây là trường hợp bài báo 1964 xét |
| **Huber, HỒI QUY** | $1/n$, tiến về **0 %** | giảm ảnh hưởng phần dư lớn, **không** bảo vệ trước $X$ tuỳ ý |
| Trung vị (vị trí) | gần $50\ \%$ | không tự chuyển thành $50\ \%$ cho hồi quy |
| Hồi quy phân vị | phụ thuộc thiết kế | vẫn dễ tổn thương trước điểm xa theo $X$ |
| LTS, LMS | tới **50 %** | không lồi, tốn kém, cần thuật toán ngẫu nhiên |
| RANSAC | phụ thuộc tỉ lệ inlier | thủ tục ngẫu nhiên; có thể thất bại nếu giả định inlier sai |

**Hai dòng đầu về Huber nói ngược nhau, và cả hai đều đúng** — đó là chỗ hay bị đọc nhầm nhất
về phương pháp này. Bài báo 1964 chứng minh tính bền vững cho **bài toán vị trí**, nơi mọi
quan sát có trọng số bằng nhau. Hồi quy thêm thừa số $x_i$ vào gradient, và thừa số đó không
bị chặn.

**Đo trực tiếp.** Lấy 60 điểm sạch sinh từ $y=2+1{,}5x$, rồi thêm **một** điểm ở $(x_{\text{xa}}, 0)$:

| Dữ liệu | Hệ số góc Huber |
|---|---:|
| 60 điểm sạch | $\mathbf{1{,}50055665}$ |
| thêm 1 điểm tại $x = 100$, $y = 0$ | $\mathbf{-0{,}01336733}$ |
| thêm 1 điểm tại $x = 1000$, $y = 0$ | $-0{,}00186955$ |
| thêm 1 điểm tại $x = 10\,000$, $y = 0$ | $-0{,}00019221$ |

**Một điểm trên sáu mươi mốt — $1{,}64\ \%$ dữ liệu — kéo hệ số từ $1{,}50$ về $-0{,}013$**,
tức gần như bằng $0$. Đó là sụp đổ hoàn toàn, và nó xảy ra ở mức nhiễm bẩn nhỏ hơn mọi
$\epsilon$ trong Bảng I.

**Cơ chế, đọc từ cột phần dư của Hình 34.** Điểm ở xa theo $X$ có đòn bẩy lớn tới mức nó kéo
được đường khớp đi **qua chính nó**. Khi đường đã đi qua nó thì phần dư của nó nhỏ — ở
$x=5000$ chỉ còn $0{,}0000443698$ — nên $\psi$ không chạm ngưỡng, nên nó **không bị giảm trọng
số**. Huber chỉ giảm trọng số những điểm nó **nhìn thấy là lệch**, và một điểm leverage cao
thì làm cho mình trông không lệch.

Bảng đầy đủ, kèm cột quyết định:

| $x$ của điểm hỏng | BPTT góc | Huber góc | Phần dư điểm đó | $\lvert$Huber $-$ BPTT$\rvert$ |
|---:|---:|---:|---:|---:|
| $1$ | $0{,}79236371$ | $1{,}46375322$ | $39{,}953047$ | $6{,}714\times10^{-1}$ |
| $3$ | $-0{,}29417579$ | $1{,}40600945$ | $39{,}696079$ | $1{,}700\times10^{0}$ |
| $10$ | $-1{,}00927359$ | $1{,}19745265$ | $36{,}932584$ | $2{,}207\times10^{0}$ |
| $50$ | $0{,}71963983$ | $0{,}71223015$ | $0{,}546263$ | $7{,}410\times10^{-3}$ |
| $500$ | $1{,}41999488$ | $1{,}41999381$ | $0{,}004725$ | $1{,}072\times10^{-6}$ |
| $5000$ | $1{,}49199601$ | $1{,}49199541$ | $0{,}000044$ | $\mathbf{5{,}980\times10^{-7}}$ |

**Đọc theo cột cuối, không theo cột "Huber góc".** Cột "Huber góc" **không đơn điệu** vì hai
hiệu ứng ngược chiều chồng nhau; chỉ cột cuối mới đọc được thành một kết luận, và kết luận đó
là: **leverage càng lớn thì Huber càng trùng bình phương tối thiểu.**

> Nói gọn: Huber bảo vệ khỏi $y$ sai, không bảo vệ khỏi $X$ sai. Muốn cả hai thì cần
> MM-estimator, LTS hoặc RANSAC — và cả ba đều trả giá bằng tính lồi, đúng thứ mục 3 của bài
> báo cảnh báo.

### 6.4 Năm kiểu hỏng, tóm lại

| Kiểu hỏng | Huber chịu được? | Bằng chứng |
|---|---|---|
| Ngoại lai theo $y$, tỉ lệ nhỏ | **có** | mục 6.2: lệch $0{,}06\ \%$ với $1/60$ điểm |
| Ngoại lai theo $y$, tỉ lệ lớn ($\epsilon$ tới $0{,}5$) | có, nhưng phương sai tăng tới $5{,}93$ | Bảng I, cột $\epsilon=0{,}500$ |
| Điểm xa theo $X$ | **KHÔNG** | mục 6.3: một điểm đủ để sụp |
| Nhiễm bẩn **không đối xứng** | chệch, tệ hơn trung vị $1{,}22$–$2{,}12$ lần | mục 2.10 |
| Đơn vị của $y$ đổi | có, **nếu** $S$ được ước lượng lại | mục 4.1 bảng thành phần |

---

## 7. So với họ hàng gần

| Phương pháp | $\rho$ hoặc cơ chế | Lồi? | Điểm sụp đổ (hồi quy) | Khi nào hơn Huber |
|---|---|---|---|---|
| [Linear Regression](<../Linear Regression/Linear Regression.md>) | $t^2$ | có | $0\ \%$ | dữ liệu sạch và chuẩn: hơn đúng $5{,}0000\ \%$ hiệu suất |
| [Quantile Regression](<../Quantile Regression/Quantile Regression.md>), $\tau=0{,}5$ | $\lvert t\rvert$ | có | $0\ \%$ | cần **phân vị**, không cần trung bình; chệch nhỏ hơn trước nhiễm bẩn lệch |
| Trung bình cắt bớt | $\frac12t^2$ rồi phẳng | **không** | — | không bao giờ: tệ hơn $2{,}4546$ lần ở cùng $k$ |
| [Support Vector Regression (SVR)](<../Support Vector Regression (SVR)/Support Vector Regression (SVR).md>) | $\varepsilon$-nhạy: **bằng 0** ở gần, tuyến tính ở xa | có | $0\ \%$ | cần nghiệm thưa theo mẫu |
| [Ridge Regression](<../Ridge Regression (Tikhonov Regularization)/Ridge Regression (Tikhonov Regularization).md>) | $t^2$ + phạt $\lVert\beta\rVert^2$ | có | $0\ \%$ | vấn đề là **đa cộng tuyến**, không phải ngoại lai |
| LTS, LMS, MM-estimator | cắt hẳn / trung vị bình phương | **không** | tới $50\ \%$ | có điểm xa theo $X$ — xem mục 6.3 |
| [Gradient Boosting (GBM)](<../../Ensemble/Gradient Boosting (GBM)/Gradient Boosting (GBM).md>) `loss='huber'` | $\rho_k$ trên phần dư của mô hình cộng | — | — | quan hệ **phi tuyến** và có ngoại lai |

**Chỗ dễ nhầm giữa Huber và SVR.** Cả hai đều tuyến tính ở xa, nhưng ở gần thì ngược nhau:
Huber là **bình phương** (mọi phần dư nhỏ đều đóng góp), SVR là **bằng 0** (phần dư nhỏ hơn
$\varepsilon$ không đóng góp gì). Hệ quả: SVR cho nghiệm **thưa theo mẫu**, Huber thì không.

**Và một biến thể quen thuộc trong học sâu:** `smooth L1 loss` chính là $\rho_k$ với
$k=1$, thường chia thêm cho $k$. Tên khác, tham số hoá khác, nên phải kiểm định nghĩa của
từng framework trước khi so con số — cùng loại nhầm với chuyện $\beta$ ở mục 4.3.

---

## 8. Dùng vào việc gì

**Dùng khi:**

- Dữ liệu đo có lỗi cảm biến, lỗi nhập liệu, sự kiện hiếm — mà **không nên** loại bằng tay
  (loại bằng tay là một quyết định chủ quan khó biện minh, và nó tạo ra chỗ nghiệm nhảy đúng
  như mục 5.2 đo).
- Cần một mô hình chạy tự động, định kỳ, không có người soát dữ liệu mỗi lần. Đây đúng là
  tình huống *"an instrument that jams with probability $\epsilon$"* mà bài báo mô tả ở tr.82.
- Muốn kết quả gần bình phương tối thiểu khi dữ liệu sạch nhưng không sập khi có vài điểm
  hỏng — và chấp nhận trả $5{,}0000\ \%$ hiệu suất cho việc đó.

**Không dùng khi:**

- **Ngoại lai nằm ở $X$ chứ không ở $y$** — mục 6.3. Đây là lý do số một.
- Ngoại lai chính là thứ cần tìm (phát hiện gian lận, phát hiện dị thường). Ở đó mô hình phải
  **nhìn thấy** chúng, không phải giảm trọng số chúng.
- Nhiễm bẩn **không đối xứng** và cần ước lượng không chệch — mục 2.10 cho thấy trung vị tốt
  hơn ở đúng chỗ này.
- Cần suy luận thống kê chuẩn mà không muốn dùng công thức riêng — sai số chuẩn phải tính bằng
  công thức bánh kẹp ở mục 4.4, không dùng công thức của bình phương tối thiểu được.

### 8.1 Quy trình dùng thật

1. **Vẽ dữ liệu và kiểm lỗi đơn vị/nhập liệu trước.** Hồi quy bền vững không phải lý do để giữ
   dữ liệu sai mà ta sửa được.
2. **Kiểm leverage theo $X$ trước khi kiểm phần dư theo $y$.** Tính $h_{ii}$ của ma trận mũ;
   điểm nào có $h_{ii}$ lớn thì Huber **không** bảo vệ được — mục 6.3.
3. **Chuẩn hoá đặc trưng, và hiểu quy ước thang đo của thư viện đang dùng.** `epsilon` của
   `scikit-learn` là $k$; `scale_` của nó là $S$ giải từ hệ (6) với $\beta=1$ — mục 4.3.
4. **So bình phương tối thiểu với Huber trên cùng một lần chia dữ liệu.** Ghi lại những điểm
   bị giảm trọng số; kiểm chúng bằng kiến thức miền thay vì tự động xoá.
5. **Chọn $k$ theo mức nhiễm bẩn mình tin, không theo mặc định.** Cột $\epsilon_{\min}$ của
   Bảng I dịch $k$ sang mức nhiễm bẩn: $k=1{,}345$ ứng $5{,}79\ \%$, $k=2$ ứng $0{,}84\ \%$,
   $k=1$ ứng $14{,}28\ \%$.
6. **Nếu hai thư viện cho hai con số, kiểm $\beta$ trước.** Mục 4.3 cho thấy chênh
   $0{,}62$–$0{,}94\ \%$ đến từ đó, không từ thuật toán.
7. **Đánh giá trên dữ liệu chưa thấy.** "Hệ số ít đổi khi thêm một ngoại lai giả" là một phép
   thử chịu tải hữu ích, không thay thế được việc kiểm định.

---

## 9. Đối chiếu với bài báo gốc

Luật 6 đòi hai bảng đối chiếu: một theo **hình/bảng**, một theo **mục**. Bài báo này không có
hình vẽ nào, nên bảng thứ nhất là bảng theo **bảng số**; và vì phần ruột của một bài báo toán
nằm ở công thức chứ không ở hình, bài viết thêm một bảng thứ ba theo **công thức đánh số**.

### 9.1 Đối chiếu theo BẢNG SỐ

Bài báo có **0 hình vẽ**, **4 bảng đánh số** và **2 bảng không đánh số** ở tr.101.

| # | Đối tượng | Trang | Số ô | Đã vào bài? | Dựng lại được? | Ở mục nào |
|---|---|---|---:|---|---|---|
| 1 | **TABLE I** | 84–85 | $403$ | **có**, Hình 9 + Hình 10, và bảng dựng lại đầy đủ | $402/403$ | 2.9 |
| 2 | **TABLE II** | 92 | $18$ | **có**, Hình 20 | $18/18$, lệch tối đa $1{,}2$ đơn vị chữ số cuối | 2.12 |
| 3 | **TABLE III** | 95 | $15$ | **có**, Hình 23 | $13/15$ làm tròn đúng, $2$ lệch $1$ đơn vị | 2.14.1 |
| 4 | **TABLE IV** | 99 | $63$ | **có**, Hình 29, và bảng dựng lại đầy đủ | $62/63$ | 2.14.4 |
| 5 | Bảng không số: cận Hodges–Lehmann | 101 | $10$ | **có**, Hình 31 | $10/10$ | 2.15 |
| 6 | Bảng không số: so với $\Phi(t/3)$ | 101 | $10$ | **có**, Hình 31 | $10/10$ | 2.15 |
| | **Tổng** | | $\mathbf{519}$ | $\mathbf{6/6}$ | $\mathbf{517/519}$ | |

Hai ô không dựng lại được, ghi rõ:

| Ô | Bài báo in | Dựng lại | Chênh | Bằng chứng nó sai |
|---|---:|---:|---:|---|
| BẢNG I, $k=2{,}1$, $\epsilon=0{,}020$ | $1{,}137$ | $1{,}127172$ | $9{,}83$ đơn vị | phá đơn điệu của chính cột nó; $17$ lần biên của $309$ ô kia |
| BẢNG IV, $k=2{,}0$, $\epsilon=0{,}100$ | $2{,}029$ | $1{,}9681$ | $61{,}1$ đơn vị | bước cuối của cột gấp đôi bước liền trước; $7{,}4$ lần ô lệch nhì |

### 9.2 Đối chiếu theo MỤC

Bài báo có **12 mục đánh số** cộng phần tài liệu tham khảo.

| Mục | Trang | Tên nguyên văn | Đã vào bài? | Ở đâu | Vì sao không lấy |
|---|---|---|---|---|---|
| 1 | 73–75 | *Introduction and summary* | **đủ** | 1.1 · 1.3 · 2.1 · 2.2 · 2.3 | — |
| 2 | 76–78 | *Asymptotic normality of (M)-estimators for convex $\rho$* | **đủ** | 2.4 | — |
| 3 | 78 | *The case of non-convex $\rho$* | **đủ** | 2.5 | — |
| 4 | 78–80 | *Examples* | **đủ** | 2.6 · 5.2 | — |
| 5 | 80–82 | *Minimax questions; a special case* | **đủ** | 2.7 | — |
| 6 | 82 | *The contaminated normal distribution* | **đủ** | 2.8 · 2.9 | — |
| 7 | 82–83 | *The question of bias* | **đủ** | 2.10 | — |
| 8 | 86–90 | *Minimax theory* | **phát biểu đủ, KHÔNG chép chứng minh** | 2.11 | chứng minh Định lý 3 dài 3 trang, thuần lý thuyết độ đo; bài viết chép phát biểu và nói rõ mỗi định lý dùng để làm gì |
| 9 | 91–93 | *The minimax solution for $\epsilon$-normal distributions* | **đủ** | 2.12 | — |
| 10 | 93–94 | *Estimation of a scale parameter* | **đủ** | 2.13 | — |
| 11 | 95–99 | *Estimation of a location parameter, if scale and $\epsilon$ are unknown* | **đủ** | 2.14 · 4.2 · 4.3 · 4.4 | — |
| 12 | 99–101 | *A generalization* | **đủ phần có số**; phần lý thuyết $U$-statistics chỉ nêu | 2.15 | mở rộng $(M_r)$ với $r>2$ được chính bài báo gọi là *"hardly of practical use"* |
| — | 101 | *References* (11 tài liệu) | **đủ** | 10 | — |

**Ba chỗ bài viết KHÔNG chép, và lý do:**

1. **Chứng minh Bổ đề 1–5** (tr.76–78) — kỹ thuật thuần tuý về hội tụ. Bài viết chép **phát
   biểu** của cả năm và nói rõ mỗi bổ đề dùng để làm gì; chứng minh không đổi cách dùng.
2. **Chứng minh Định lý 3** (tr.89–90, sáu bước i–vi) — lý thuyết độ đo. Bài viết chép phát
   biểu và giải thích vì sao nó cần thiết (mục 2.11).
3. **Chứng minh Mệnh đề tồn tại–duy nhất của hệ (6)** (tr.97–98) — bài viết chép **kết quả** và
   hai điều kiện bảo vệ mà nó sinh ra (`m2 == 0`, `mau <= 0` ở mục 4.2).

### 9.3 Đối chiếu theo CÔNG THỨC ĐÁNH SỐ và PHÁT BIỂU

| Nhãn | Trang | Nội dung | Đã vào bài? | Ở mục |
|---|---|---|---|---|
| **(M)** | 74 | định nghĩa ước lượng $(M)$ | **có**, kèm ba phần | 2.1 |
| **(1)** | 78 | $V=E\psi^2/(E\psi')^2 \ge 1/I(F)$ | **có**, kèm ba phần | 2.4 |
| **(2)** | 83 | cận trên của độ chệch $\lvert c(F)\rvert$ | **có**, kèm ba phần | 2.10 |
| **(i)(ii)(iii)** | 92 | ba điều kiện xác định $a$, $b$, $c$ | **có**, và **giải ra** | 2.12 |
| **(3)** | 94 | quan hệ $q\leftrightarrow\epsilon$ cho thang đo | **có**, kèm ba phần | 2.13 |
| **(4)** | 94 | $\psi_0(q,t)$ | **có** | 2.13 |
| **(5)** | 95 | định nghĩa $S_n'$ | **có**, và suy ra $S_\infty=\sqrt{E_\Phi\psi^2(q')}$ | 2.14.1 |
| **(6)** | 97 | hệ hai phương trình cho $(T,S)$ với hằng số $\beta$ | **có**, kèm ba phần | 2.14.2 |
| **(6′)** | 98 | thuật toán lặp | **có**, và **cài đặt chạy được** | 2.14.3 · 4.2 |
| **(7)** | 99 | bất đẳng thức xác định $q_{\max}$ | **có**, kèm ba phần | 2.14.4 |
| LEMMA 1–4 | 76–77 | tính lồi, tồn tại $\lambda$, nhất quán, chuẩn tiệm cận | **phát biểu đủ** | 2.4 |
| LEMMA 5 | 78 | chuẩn tiệm cận khi $\rho$ không lồi | **phát biểu đủ** | 2.5 |
| LEMMA 6 | 86 | bất đẳng thức lồi | **có, viết ra** | 2.11 |
| THEOREM 1 | 80–81 | điểm yên ngựa, $f_0$ tường minh | **có, nguyên văn** | 2.7 |
| COROLLARY | 82 | trường hợp chuẩn bị nhiễm bẩn | **có, nguyên văn** | 2.8 |
| THEOREM 2 | 87 | saddlepoint $\Leftrightarrow$ cực tiểu $I(F)$ | **có, nguyên văn** | 2.11 |
| DEFINITION | 89 | $I(F)$ mở rộng | **có, nguyên văn** | 2.11 |
| THEOREM 3 | 89 | $I(F)<\infty \Leftrightarrow$ có mật độ liên tục tuyệt đối | **có, nguyên văn** | 2.11 |
| THEOREM 4 | 90 | tồn tại và **duy nhất** $F_0$ | **có, nguyên văn** | 2.11 |
| THEOREM 5 | 91 | saddlepoint mở rộng sang cả $C$ | **có, nguyên văn** | 2.12 |
| PROPOSAL 1, 2, 3 | 96–97 | ba cách ước lượng đồng thời $(T,S)$ | **có, nguyên văn cả ba** | 2.14.2 |
| PROPOSITION | 97 | tồn tại–duy nhất nghiệm của (6) | **kết quả có**, chứng minh không chép | 2.14.2 · 9.2 |

**Phủ $21/21$ nhãn.** Mọi công thức được bài báo đánh số đều có mặt trong bài, viết ra, giải
nghĩa từng ký hiệu kèm chiều, và kèm một đoạn nói **ý nghĩa trừu tượng** của nó.

### 9.4 Bốn phép kiểm mà bài này KHÔNG chạy được, và vì sao

Đây là phần phải nói ra, vì im lặng ở đây trông giống hệt sạch sẽ.

| Công cụ | Bình thường kiểm gì | Ở bài này | Lý do |
|---|---|---|---|
| `kiemtrich.py` | đối chiếu đoạn trích tiếng Anh với lớp text PDF | **không kiểm được** | bản quét, $4{,}9$ ký tự/trang trên 29 trang |
| `pdfscan.py --muc` | liệt kê mục của bài báo | **trả 0 dòng** | cùng lý do |
| `sopaper.py` | đếm hình/bảng của bài báo | **báo 0 hình 0 bảng** — con số đó vô nghĩa | cùng lý do |
| `congthuc.py` | dò công thức "thả trần" | chạy được, vì nó đọc **bài viết** chứ không đọc PDF | — |

`kiemtrich.py` nay **báo ra** tình trạng này thay vì báo đỏ hàng loạt: nó đo mật độ ký tự của
lớp text, và nếu dưới $100$ ký tự mỗi trang thì in rõ *"LỚP TEXT CỦA PDF RỖNG HOẶC VỠ… script
KHÔNG kiểm được đoạn trích nào"* cùng số đoạn **tìm thấy**. Trước bản vá đó, nó sẽ báo mọi
đoạn trích là `KHÔNG KHỚP`, và một danh sách sai dài đúng bằng một danh sách thiếu thật.

Hệ quả cho người đọc: **mọi đoạn tiếng Anh trong bài này được chép tay từ ảnh render ở 300
DPI, và không phép kiểm tự động nào xác nhận chúng.** Ai cần chắc chắn thì mở PDF ở trang
tương ứng — mỗi caption đều ghi số trang của bản in và số trang của PDF.

---

## 10. Tài liệu tham khảo

```
[1] P. J. Huber, "Robust Estimation of a Location Parameter",
    The Annals of Mathematical Statistics 35(1):73–101, 1964.
    → BÀI GỐC. Toàn bộ mục 2 của bài này dựng lại từ đây.
    → BẢN QUÉT JSTOR: lớp text rỗng ở 28/29 trang. Xem mục 0.5 và 9.4.
    AI algorithms/_papers/1964 - Huber - Robust Estimation of a Location Parameter.pdf

[2] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
    The Annals of Statistics 29(5):1189–1232, 2001, mục 4.2.
    → Hàm mất mát Huber trong boosting; xem mục 7.
    AI algorithms/_papers/2001 - Friedman - Greedy Function Approximation A Gradient Boosting Machine.pdf

[3] R. Koenker, "Quantile Regression" (bài tổng quan).
    → So sánh với cách tiếp cận phân vị; xem mục 7.
    AI algorithms/_papers/2000 - Koenker - Quantile Regression (bai tong quan).pdf
```

**Mười một tài liệu mà bài báo gốc dẫn**, chép từ tr.101 để người đọc biết bối cảnh 1964:
Anscombe (1960) · Cramér (1946) · Hájek (1962) · Hodges & Lehmann (1963) · Hoeffding (1948) ·
LeCam (1953) · LeCam (1958) · Loève (1960) · Stein (1956) · Tukey (1960) · Wald (1949).
Trong đó Tukey (1960) là tài liệu mà mục 1 của bài báo dựa vào để đặt vấn đề, và Hodges &
Lehmann (1963) là đối thủ được so ở mục 12.

**Mã và số liệu.** Mọi con số trong bài sinh từ
[`tools/v_huber.py`](<../../../tools/v_huber.py>) — 17 mục, chạy hết trong $5{,}6$ giây, không
cần dữ liệu ngoài. Chạy lại: `cd tools && python v_huber.py`.

**Hình gốc.** 31 khung cắt từ bản in bằng `tools/pdffig.py` ở 300 DPI, riêng Hình 9 (Bảng I
phần 1) ở 240 DPI vì khung đó rộng gần cả trang,
mỗi khung ghi rõ tham số `--box` trong caption. Hai khung của Bảng I dùng thêm cờ `--xoay 270`
vì bảng được in xoay 90 độ trên trang; mọi khung dùng cờ `--sat` để không cắt ngang thân chữ.
Hai cờ đó được thêm vào `pdffig.py` trong đợt viết bài này.

**Hình tự vẽ.** Ba hình — 32, 33, 34 — do ba khối mã tự chứa in ngay trong bài sinh ra. Mỗi
khối có đủ dữ liệu, đủ thư viện và đúng một lệnh `savefig`.

---

## Kiểm tra trước khi đóng bài

- [x] **Bài đọc được một mình** — mọi khái niệm nền được nhắc lại đủ dùng ngay trong bài; mục
      0.3 giải nghĩa 14 từ, mục 0.2 giải nghĩa 30 ký hiệu kèm cách đọc
- [x] **Bảng ký hiệu có cột "Đọc là gì"** và **năm chỗ trùng tên**, trong đó hai chỗ trùng ngay
      trong chính bài báo ($c$ ở mục 7 và mục 9; $q$ ở mục 6 và mục 10) — mục 0.2
- [x] **Ví dụ chạy tay có DẠNG ĐÓNG** — mục 0.4, $T = (9 - 1{,}345)/2 = 3{,}8275$, lệch
      $4{,}4\times10^{-16}$ so với nghiệm lặp, và tính chất Winsor hoá kiểm lại lệch $0$
- [x] **Khối SS nêu bế tắc bằng SỐ** — mục 1.2: bình phương tối thiểu đi từ $1{,}1762$ xuống
      $-573{,}3023$; và cột "trung bình mẫu" của Bảng I là $\infty$ ở mọi $\epsilon>0$
- [x] **PHẦN RUỘT THEO BÀI BÁO GỐC (Luật 6)** — mục 2, 15 tiểu mục theo đúng thứ tự 12 mục của
      bài báo; **31 khung cắt** từ bản in; **21/21 nhãn công thức và phát biểu** đã vào bài
- [x] **Mọi công thức đánh số có đủ ba phần** — công thức kèm số hiệu · mọi ký hiệu giải nghĩa
      kèm chiều · và một đoạn **ý nghĩa trừu tượng** riêng. Mục 9.3 liệt kê từng nhãn
- [x] **Mọi đoạn trích có bản dịch tiếng Việt đi kèm**
- [x] **HAI bảng đối chiếu, cộng một bảng thứ ba** — mục 9.1 theo bảng số, 9.2 theo mục, 9.3
      theo công thức. Ô "vì sao không lấy" được điền cho cả ba chỗ không chép
- [x] **Từng ô của bốn bảng số đã đối chiếu** — $403 + 18 + 15 + 63 = 499$ ô của bốn bảng
      đánh số, cộng $20$ ô của hai bảng không đánh số ở tr.101 là **$519$ ô**; **$517/519$
      dựng lại được**
- [x] **Hai ô không dựng lại được đã ghi ra, kèm bằng chứng chúng sai** — mục 2.9.1 và 2.14.5;
      cả hai tự lộ bằng cách phá đơn điệu của chính cột chúng
- [x] **Phép so đúng cho bảng in ít chữ số** — làm tròn giá trị đo lại về đúng số chữ số bài
      báo in rồi so bằng nhau, thay vì dùng một ngưỡng tự nghĩ ra
- [x] **Phép thử ngược cho thuật toán (6′)** — hệ (6) được thoả tới $3{,}6\times10^{-15}$ ở cả
      bốn mức $k$, tức thuật toán thật sự giải hệ chứ không giải một thứ gần giống
- [x] **Ba câu về tốc độ hội tụ của tác giả được kiểm lại** — $1{,}6960$ · $1{,}5560$ ·
      $4{,}2730$ bước đổi, so với *"1-2"* · *"about 2"* · *"4"* của bài báo
- [x] **Kết quả âm được báo cáo** — mục 2.10: về độ chệch trước nhiễm bẩn không đối xứng,
      **trung vị tốt hơn Huber** $1{,}22$–$2{,}12$ lần, và bài báo tự nói điều đó
- [x] **Câu nhận định của tác giả được đo, không chỉ chép** — mục 2.9.2: câu *"not very
      sensitive to the choice of $k$"* đúng ở giữa dải nhưng phải trả $28{,}82\ \%$ ở
      $\epsilon=0{,}20$, tức đúng biên mà chính câu ấy tuyên bố phủ
- [x] **Câu có HAI cách đọc được thử cả hai rồi để tổng sai lệch chọn** — mục 2.10, cỡ mẫu
      $9$/$600$/$35\,000$: cách đọc theo chữ khớp hơn $6{,}3$ lần
- [x] **Truy được chỗ lệch với thư viện về một dòng của bài báo** — mục 4.3: `scikit-learn`
      là hệ (6) với $\beta=1$, khớp tới $10^{-6}$ ở cả hệ số lẫn thang đo, trong khi Đề xuất 2
      của bài báo đặt $\beta = E_\Phi\psi^2 = 0{,}7101645483$
- [x] **Con số $1{,}345$ được suy ra, không chép** — mục 5.3: $k = 1{,}3449975085$, và bài báo
      1964 **không chứa** con số đó lẫn chuỗi "95 %"
- [x] **Nói rõ hỏng khi nào, bằng số** — mục 6.3: một điểm trên $61$ kéo hệ số từ
      $1{,}50055665$ về $-0{,}01336733$
- [x] **Phân biệt bài toán VỊ TRÍ với HỒI QUY** — mục 3: điểm sụp đổ của Huber là gần $50\ \%$
      cho vị trí và $1/n$ cho hồi quy, và mục 3.2 chỉ ra thừa số $x_i$ là nguyên nhân
- [x] **Ba hình tự vẽ, mỗi hình một khối mã tự chứa với đúng một `savefig`** — Hình 32, 33, 34
- [x] **Mỗi hình có caption và mục "Cách đọc hình"** — cả 34 hình
- [x] **Không xưng hô trong thân bài** (Luật 3)
- [ ] **CHƯA LÀM ĐƯỢC: đối chiếu đoạn trích tự động.** Bản PDF là bản quét, lớp text rỗng ở
      28/29 trang, nên `kiemtrich.py` không kiểm được đoạn nào. Mọi đoạn tiếng Anh trong bài
      được chép tay từ ảnh render 300 DPI — xem mục 9.4. Nếu sau này có bản PDF có lớp text
      thì chạy `python kiemtrich.py --bai "Huber Regression"` là kiểm được ngay
- [ ] **CHƯA LÀM: dựng lại bảng của Huber 1973** (*Robust regression: asymptotics, conjectures
      and Monte Carlo*, Ann. Statist. 1(5):799–821). Đó mới là bài báo về **hồi quy** Huber;
      repo hiện chưa có bản PDF của nó. Mục 3 của bài này bắc cầu bằng lập luận, không bằng
      bài báo — và chỗ đó đã nói rõ
