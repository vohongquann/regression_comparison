# Chính tắc

**"Chính tắc"** (tiếng Anh: *non-singular* hoặc *full rank*) trong ngữ cảnh toán học và học máy, đặc biệt với ma trận $\mathbf{X}$, là một khái niệm quan trọng liên quan đến khả năng **nghịch đảo được** và **không có dư thừa thông tin tuyến tính**.

### Định nghĩa:

Ma trận $\mathbf{X}^T \mathbf{X}$ được gọi là **chính tắc** nếu:

- Nó **có hạng đầy đủ** (*full rank*): các cột của $\mathbf{X}$ là **tuyến tính độc lập**, không cột nào là tổ hợp tuyến tính của các cột khác.
- Tồn tại **nghịch đảo** của $\mathbf{X}^T \mathbf{X}$, tức là: $det(\mathbf{X}^T \mathbf{X}) \neq 0$

Ngược lại, nếu $\mathbf{X}^T \mathbf{X}$ **không chính tắc** (tức là *singular*, hay *bị suy biến*), thì không thể tính nghiệm bằng công thức giải tích:

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} 
$$

### Nguyên nhân khiến $\mathbf{X}$ không chính tắc:

1. **Tập huấn luyện có quá ít mẫu** so với số đặc trưng $(n<p)$ → hệ phương trình không đủ điều kiện để giải duy nhất.
2. **Tồn tại hai hoặc nhiều đặc trưng tương quan tuyến tính mạnh** → phương sai bằng 0 theo một số chiều → không thể đảo.
3. **Tái sử dụng hoặc sao chép cột đặc trưng** không cẩn thận khi xử lý dữ liệu → làm mất tính độc lập tuyến tính.

### Hệ quả:

- Không thể sử dụng công thức giải tích để tính nghiệm.
- Phải dùng các kỹ thuật khác như:
    - **Gradient Descent**
    - **Regularization (Ridge, Lasso)** để khắc phục vấn đề chính tắc bằng cách thêm điều kiện ràng buộc.

## Ví dụ:

Giả sử bạn có một bài toán hồi quy tuyến tính với dữ liệu đầu vào:

$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}, \quad
\mathbf{y} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}

$$

## **2. Hạn chế: X\mathbf{X}X không chính tắc**

Tính XTX\mathbf{X}^T \mathbf{X}XTX:

XTX=[667890789310890108126]\mathbf{X}^T \mathbf{X} = 
\begin{bmatrix}
66 & 78 & 90 \\
78 & 93 & 108 \\
90 & 108 & 126
\end{bmatrix}

XTX=

667890789310890108126

Giá trị **định thức** của XTX\mathbf{X}^T \mathbf{X}XTX là:

det⁡(XTX)=0\det(\mathbf{X}^T \mathbf{X}) = 0

det(XTX)=0

⇒ **Ma trận này không khả nghịch (không chính tắc)**. Tức là không thể áp dụng công thức:

β^=(XTX)−1XTy\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}

β^=(XTX)−1XTy

---

## **3. Nguyên nhân trực quan**

Quan sát hàng hoặc cột của X\mathbf{X}X, bạn sẽ thấy:

- Mỗi hàng là tổ hợp tuyến tính của các hàng khác.
- Các cột **không độc lập tuyến tính**. Ví dụ:

Col3=Col1+Col2\text{Col}_3 = \text{Col}_1 + \text{Col}_2

Col3=Col1+Col2

⇒ Điều này dẫn đến **đa cộng tuyến hoàn hảo**.

---

## **4. Cách xử lý**

### **A. Dùng Gradient Descent**

Không cần nghịch đảo, chỉ cần tính gradient và cập nhật:

β←β−α∇L(β)\beta \leftarrow \beta - \alpha \nabla L(\beta)

β←β−α∇L(β)

Dù XTX\mathbf{X}^T \mathbf{X}XTX không khả nghịch, Gradient Descent vẫn hoạt động được (từ từ cập nhật đến điểm hội tụ).

---

### **B. Dùng Regularization (Ridge Regression)**

Thêm một hằng số nhỏ λI\lambda IλI để làm cho ma trận khả nghịch:

β^ridge=(XTX+λI)−1XTy\hat{\beta}_{ridge} = (\mathbf{X}^T \mathbf{X} + \lambda I)^{-1} \mathbf{X}^T \mathbf{y}

β^ridge=(XTX+λI)−1XTy

Giúp tránh được vấn đề không chính tắc.

---

## **5. Kết luận**

| Đặc điểm | Ma trận chính tắc | Ma trận không chính tắc |
| --- | --- | --- |
| Có thể nghịch đảo XTX\mathbf{X}^T \mathbf{X}XTX | ✔ | ✘ |
| Có thể dùng công thức giải tích | ✔ | ✘ |
| Cần dùng Gradient Descent | Không bắt buộc | Bắt buộc |
| Có thể gây ra bất ổn trong huấn luyện | Ít | Cao |

Nếu bạn cần code Python minh họa trường hợp này với `numpy`, mình có thể cung cấp.