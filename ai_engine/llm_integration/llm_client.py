"""
llm_client.py
-------------
Kết nối API Gemini / GPT để:
  - Tổng hợp insight từ kết quả phân tích
  - Sinh báo cáo tự động bằng ngôn ngữ tự nhiên

Cấu hình qua file .env (xem .env.example):
  GEMINI_API_KEY=your_key_here
  OPENAI_API_KEY=your_key_here   # optional

TODO:
  - [ ] Load API key từ .env
  - [ ] Implement summarize_reviews() gọi Gemini API
  - [ ] Implement generate_report() tạo báo cáo markdown/PDF
"""

import os

def summarize_reviews(reviews: list, provider: str = "gemini") -> str:
    """
    Tổng hợp danh sách review thành insight ngắn gọn.
    provider: 'gemini' hoặc 'openai'
    """
    raise NotImplementedError("TODO: Implement LLM summarization")


def generate_report(analysis_results: dict) -> str:
    """Sinh báo cáo phân tích đầy đủ dưới dạng markdown."""
    raise NotImplementedError("TODO: Implement report generation")
