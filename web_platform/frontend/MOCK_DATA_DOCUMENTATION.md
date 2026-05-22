# Tài Liệu Hướng Dẫn Cấu Trúc Dữ Liệu (Mock Data Contract)

Tài liệu này mô tả chi tiết cấu trúc dữ liệu mà **Frontend** hiện đang mong đợi nhận được từ **AI Engine / Backend**. Hiện tại, quá trình phân tích URL (bao gồm cả phân tích chính và đề xuất sản phẩm thay thế) đang được thực hiện thông qua dữ liệu giả lập (mock data) ở phía AI Engine (`ai_engine/main.py`) thông qua Socket.IO và Webhook.

Khi team AI Engine và Backend tiến hành tích hợp mô hình AI và cào dữ liệu thực tế (Real Data), vui lòng **đảm bảo cấu trúc JSON trả về tuân thủ đúng định dạng** dưới đây để Frontend có thể hiển thị chính xác trên Dashboard Đa Phương Thức.

## 1. Flow Nhận Dữ Liệu

- Người dùng nhập URL tại Frontend.
- Frontend gọi API `POST /api/analyze` trên Node.js Backend.
- Node.js đẩy job vào Redis/Bull Queue và trả về `productId`.
- Frontend kết nối Socket.IO tới Room `room-{productId}`.
- Python AI Engine nhận job, cào dữ liệu và chạy AI model.
- AI Engine liên tục báo cáo tiến trình (Progress) qua webhook.
- Khi hoàn tất, AI Engine gọi Webhook Finished kèm theo toàn bộ **Payload** bên dưới.
- Node.js lưu Payload vào PostgreSQL và bắn sự kiện Socket `finished` xuống Frontend.

## 2. Cấu Trúc JSON Payload Cuối Cùng (Real Data Requirement)

Đây là cục data lớn mà AI Engine phải gửi qua webhook cho Node.js khi hoàn thành toàn bộ quá trình phân tích.

```json
{
  "productId": 123,
  "url": "https://shopee.vn/product/...",
  
  // 1. Thông tin chung về Sản Phẩm
  "productData": {
    "name": "Tên sản phẩm được trích xuất (vd: Áo thun levent)",
    "thumbnail": "URL hình ảnh đại diện của sản phẩm"
  },
  
  // 2. Nhận định tóm tắt từ LLM (Generative AI)
  "summary": "Tóm tắt ngắn gọn từ LLM (khoảng 2-3 câu). Vd: Sản phẩm được đánh giá khá tốt, tuy nhiên có nhiều phản ánh về khâu đóng gói...",
  
  // 3. Danh sách Đánh giá chi tiết (Reviews)
  // Data này được dùng để hiển thị bảng data table và phân bố biểu đồ
  "reviews": [
    {
      "review_text": "Sản phẩm xài rất thích, đáng tiền.",
      "rating": 5, // Số sao từ 1 đến 5
      "image_path": "URL hình ảnh đính kèm trong review (nếu không có thì trả về null hoặc chuỗi rỗng)",
      "label": "intact", // Phân loại hình ảnh. Các giá trị hợp lệ: "intact", "damaged", "wrong_item", "irrelevant"
      "sentiment": "positive" // Cảm xúc văn bản. Các giá trị hợp lệ: "positive", "neutral", "negative"
    },
    // ... Danh sách các review khác
  ],
  
  // 4. Metadata Phân tích Chuyên Sâu (AI Metadata)
  "metadata": {
    // Chỉ số rủi ro / an toàn chung
    "spamPercentage": 15, // Tỷ lệ đánh giá spam/seeding (0 - 100)
    "trustScore": 85, // Điểm tin cậy của sản phẩm (0 - 100)
    
    // Phân tích khía cạnh (Aspect-Based Sentiment Analysis - ABSA)
    // Team AI tự trích xuất các tiêu chí quan trọng dựa trên tập dữ liệu
    "aspects": {
      "Sản phẩm (Product)": 4.5,
      "Đóng gói (Packaging)": 2.5,
      "Giao hàng (Shipping)": 4.0
    },
    
    // Từ khóa nổi bật phân theo cảm xúc
    "keywords": {
      "positive": [
        {"text": "tuyệt vời", "value": 60},
        {"text": "chất lượng", "value": 45}
        // value là độ phổ biến/trọng số của từ khóa (dùng cho Word Cloud)
      ],
      "negative": [
        {"text": "móp méo", "value": 50},
        {"text": "sai hàng", "value": 60}
      ]
    },
    
    // Gợi ý hành động thông minh từ LLM (dựa trên phân tích chéo)
    "smartAdvice": "💡 Gợi ý mua hàng: Khuyến nghị bạn NHẮN TIN CHO SHOP yêu cầu bọc thêm màng xốp nổ trước khi đặt hàng.",
    
    // Danh sách Sản Phẩm Thay Thế (Nếu Trust Score thấp)
    // Hệ thống khuyến nghị hiển thị 5 sản phẩm.
    "alternativeProducts": [
      {
        "name": "Sản phẩm tương tự 1",
        "thumbnail": "URL hình ảnh sản phẩm thay thế",
        "url": "https://shopee.vn/link-den-san-pham-thay-the",
        "trustScore": 95
      }
      // Đảm bảo trả về mảng gồm 5 items
    ],
    
    // Biểu đồ chuỗi thời gian (Sentiment Trend)
    "sentimentTimeSeries": [
      {
        "date": "10/05",
        "positive": 45,
        "neutral": 10,
        "negative": 5
      }
      // Dữ liệu cho 14 ngày hoặc 30 ngày gần nhất
    ]
  }
}
```
**Lưu ý**: Mock data này có thể thay đổi cấu trúc tùy thuộc vào team AI Agent, frontend sẽ điều chỉnh để update theo tương ứng.

## 3. Các Lưu Ý Quan Trọng
1. **Alternative Products URL**: Mỗi sản phẩm thay thế bắt buộc phải có trường `url` dẫn tới trang mua hàng. Nút "Truy cập" và "Phân tích" trên Frontend phụ thuộc trực tiếp vào trường URL này.
2. **Labels & Sentiments**: Frontend sử dụng các enum cố định cho việc render màu sắc và icon, đảm bảo trả đúng các giá trị tiếng Anh:
   - Sentiment: `positive`, `neutral`, `negative`
   - Label Ảnh: `intact`, `damaged`, `wrong_item`, `irrelevant`

Tài liệu này có thể được cập nhật trong quá trình triển khai thực tế khi phát sinh các nhu cầu mới về dữ liệu!
