# Giả định Gaussian noise

### **Mô hình giả định rằng:**

Dữ liệu đầu ra $y$ được tạo ra bởi mô hình tuyến tính cộng với một **nhiễu ngẫu nhiên phân phối chuẩn**:

$$
y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

**Giải thích:**

- $\varepsilon$: là thành phần sai số (noise).
- $\mathcal{N}(0, \sigma^2)$: phân phối chuẩn với kỳ vọng bằng 0 và phương sai $\sigma^2$.
- Điều này ngụ ý rằng **mọi sai số giữa dự đoán và thực tế đều là ngẫu nhiên, không có thiên lệch, và phân bố đều xung quanh 0**.

### **Tại sao cần giả định này?**

1. **Để sử dụng được công thức đạo hàm và nghiệm giải tích**:
    - Khi $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, thì việc **tối thiểu hóa hàm bình phương lỗi** (least squares) là tương đương với **tối đa hóa xác suất (MLE)**.
    - Do đó → $\hat{\beta} = \arg \min_\beta \| y - X\beta \|^2$ là **ước lượng MLE** dưới giả định Gaussian noise.
2. **Giúp đưa ra các kết luận thống kê**:
    - Ước lượng khoảng tin cậy.
    - Kiểm định giả thuyết.
    - Phân tích phương sai (ANOVA).
    

| Tính chất | Ý nghĩa |
| --- | --- |
| Sai số trung bình bằng 0 | Mô hình không thiên lệch |
| Phân phối chuẩn | Cho phép áp dụng lý thuyết xác suất và thống kê mạnh mẽ |
| Phương sai không đổi (homoscedasticity) | Đảm bảo ước lượng không thiên lệch, chính xác |
| Không tự tương quan | Sai số của mẫu này không ảnh hưởng mẫu khác |