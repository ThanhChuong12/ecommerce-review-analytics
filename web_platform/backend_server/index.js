/**
 * index.js — Express API Gateway
 * Điều phối traffic giữa:
 *   - Scraping Agent (Python subprocess hoặc REST)
 *   - AI Pipeline (Python FastAPI service)
 *   - React Frontend
 *
 * TODO:
 *   [ ] POST /api/scrape    — Kích hoạt scraping_agent với URL sản phẩm
 *   [ ] GET  /api/reviews   — Trả về danh sách review đã xử lý
 *   [ ] GET  /api/insights  — Trả về kết quả phân tích từ backend_ai
 *   [ ] GET  /api/report    — Tải báo cáo tổng hợp (PDF/markdown)
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// --- Routes ---
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "multimodal-review-api" });
});

// TODO: import và mount route handlers
// const scrapeRouter = require("./routes/scrape");
// const reviewsRouter = require("./routes/reviews");
// app.use("/api/scrape", scrapeRouter);
// app.use("/api/reviews", reviewsRouter);

app.listen(PORT, () => {
  console.log(`API Gateway running on http://localhost:${PORT}`);
});
