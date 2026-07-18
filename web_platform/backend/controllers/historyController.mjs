import { Product, Review, Report } from '../models/index.mjs';

export const getHistoryList = async (req, res) => {
    try {
        const userId = req.user.id;

        const products = await Product.findAll({
            where: { userId },
            order: [['createdAt', 'DESC']]
        });

        return res.status(200).json({ success: true, data: products });
    } catch (error) {
        return res.status(500).json({ error: 'Lỗi server' });
    }
};

export const getHistoryDetail = async (req, res) => {
    try {
        const { productId } = req.params;
        const userId = req.user.id;

        const product = await Product.findOne({
            where: { id: productId, userId: userId },
            include: [
                { model: Review, as: 'reviews' },
                { model: Report, as: 'report' }
            ]
        });

        if (!product) return res.status(404).json({ error: 'Không tìm thấy hoặc bạn không có quyền xem' });
        return res.status(200).json({ success: true, data: product });
    } catch (error) {
        return res.status(500).json({ error: 'Lỗi server' });
    }
};

export const deleteHistory = async (req, res) => {
    try {
        const { productId } = req.params;
        const userId = req.user.id;

        const deletedCount = await Product.destroy({
            where: { id: productId, userId: userId }
        });

        if (deletedCount === 0) return res.status(404).json({ error: 'Không có quyền xoá' });

        return res.status(200).json({ success: true, message: 'Đã xoá lịch sử' });
    } catch (error) {
        return res.status(500).json({ error: 'Lỗi server' });
    }
};

import puppeteer from 'puppeteer';

export const exportPDF = async (req, res) => {
    try {
        const { productId } = req.params;
        const userId = req.user.id;

        const product = await Product.findOne({
            where: { id: productId, userId: userId },
            include: [
                { model: Review, as: 'reviews' },
                { model: Report, as: 'report' }
            ]
        });

        if (!product) return res.status(404).json({ error: 'Không tìm thấy' });

        const metadata = product.report?.metadata || {};
        const reviews = product.reviews || [];

        let s = { positive: 0, neutral: 0, negative: 0 };
        let l = { intact: 0, damaged: 0, wrong_item: 0, irrelevant: 0 };
        let imagesHtml = '';
        let reviewsHtml = '';

        let imageCount = 0;
        let reviewDetailCount = 0;

        reviews.forEach((r) => {
            if (s[r.sentiment] !== undefined) s[r.sentiment]++;
            if (l[r.label] !== undefined) l[r.label]++;

            // Only include reviews with actual text content (skip rating-only reviews)
            if (reviewDetailCount < 20 && r.review_text && r.review_text.trim().length > 0) {
                reviewDetailCount++;
                const sentClass = r.sentiment === 'positive' ? 'badge-pos' : r.sentiment === 'negative' ? 'badge-neg' : 'badge-neu';
                const sentText = r.sentiment === 'positive' ? 'Tích cực' : r.sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập';

                reviewsHtml += `
                    <tr>
                        <td><div class="stars">${'★'.repeat(r.rating)}${'☆'.repeat(5 - r.rating)}</div><div class="review-text">${r.review_text}</div></td>
                        <td><span class="badge ${sentClass}">${sentText}</span></td>
                    </tr>
                `;
            }

            if (r.image_path && imageCount < 20) {
                imageCount++;
                const lblText = r.label === 'intact' ? 'Nguyên vẹn' : r.label === 'damaged' ? 'Hỏng / Móp' : 'Không l.quan';
                const lblColor = r.label === 'intact' ? '#10b981' : r.label === 'damaged' ? '#ef4444' : '#94a3b8';
                const lblBg   = r.label === 'intact' ? 'rgba(16,185,129,0.85)' : r.label === 'damaged' ? 'rgba(239,68,68,0.85)' : 'rgba(100,116,139,0.85)';
                imagesHtml += `
                    <div class="image-card">
                        <img src="${r.image_path}" alt="Review image" loading="lazy">
                        <div class="lbl" style="background:${lblBg};color:#fff;">${lblText}</div>
                    </div>
                `;
            }
        });

        const totalReviews = reviews.length || 1;
        const totalImages  = reviews.filter(r => r.image_path).length || 1;

        const aspectNames = { 'Product': 'Sản phẩm', 'Shipping': 'Vận chuyển', 'Service': 'Chất lượng dịch vụ', 'Price': 'Giá cả' };
        const aspectsHtml = Object.entries(metadata.aspectSentiment || {}).map(([key, val]) => {
            const displayName = aspectNames[key] || key;
            const pct = Math.round((val / 5) * 100);
            const barColor = pct >= 70 ? '#10b981' : pct >= 45 ? '#f59e0b' : '#ef4444';
            return `
                <div class="stat-row">
                    <div style="flex:1;color:#cbd5e1;">${displayName}</div>
                    <div style="width:140px;margin:0 12px;"><div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${barColor};"></div></div></div>
                    <div style="width:36px;text-align:right;font-weight:700;color:#f8fafc;">${val}/5</div>
                </div>
            `;
        }).join('');

        const keywordsHtmlPos = (metadata.keywords?.positive || []).map(k => `<span class="kw pos">${k.text || k}</span>`).join('');
        const keywordsHtmlNeg = (metadata.keywords?.negative || []).map(k => `<span class="kw neg">${k.text || k}</span>`).join('');

        const altProductsHtml = (metadata.alternativeProducts || []).map(alt => {
            const reason = alt.reason || '';
            let badgeBg = '#312e81'; let badgeColor = '#a5b4fc';
            if (reason.includes('tương tự') || reason.includes('Tương tự')) { badgeBg = '#1e1b4b'; badgeColor = '#c4b5fd'; }
            else if (reason.includes('Đánh giá') || reason.includes('Uy tín')) { badgeBg = '#022c22'; badgeColor = '#6ee7b7'; }
            else if (reason.includes('Giá')) { badgeBg = '#1e1b4b'; badgeColor = '#d8b4fe'; }
            const rerankPct = alt.rerank_score ? Math.round(alt.rerank_score * 100) : null;
            return `
            <div class="box alt-product">
                <img src="${alt.thumbnail}" alt="product">
                ${reason ? `<span style="background:${badgeBg};color:${badgeColor};font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;margin:4px 0;display:inline-block;">${reason}</span>` : ''}
                <h4 style="font-size:11px;margin:4px 0;color:#f1f5f9;text-align:center;overflow:hidden;max-height:32px;">${alt.name}</h4>
                <div style="color:#10b981;font-size:10px;margin-top:2px;font-weight:700;">Trust: ${alt.trustScore || 0}/100</div>
                ${rerankPct !== null ? `<div style="width:100%;margin-top:6px;"><div style="display:flex;justify-content:space-between;font-size:8px;color:#64748b;margin-bottom:2px;"><span>AI Score</span><span style="color:#818cf8;font-weight:700;">${rerankPct}%</span></div><div style="background:#1e293b;border-radius:3px;height:3px;overflow:hidden;"><div style="background:linear-gradient(to right,#6366f1,#60a5fa);height:100%;width:${rerankPct}%;"></div></div></div>` : ''}
            </div>
        `}).join('');

        const spamPct   = metadata.spamPercentage || 0;
        const trustScore = metadata.trustScore || 0;
        const trustColor = trustScore >= 70 ? '#10b981' : trustScore >= 45 ? '#f59e0b' : '#ef4444';

        const htmlContent = `
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <title>Báo Cáo Phân Tích · ${product.name}</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
                <style>
                    @page { size: A4; margin: 12mm 14mm; }
                    *, *::before, *::after { box-sizing: border-box; }
                    body {
                        font-family: 'Inter', sans-serif;
                        color: #1e293b;
                        background: #f8fafc;
                        font-size: 12px;
                        line-height: 1.6;
                        margin: 0;
                        -webkit-print-color-adjust: exact;
                        print-color-adjust: exact;
                    }

                    /* ── Header ── */
                    .header {
                        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
                        border-radius: 0;
                        padding: 28px 32px 22px;
                        margin-bottom: 20px;
                        position: relative;
                        overflow: hidden;
                    }
                    .header::before {
                        content: '';
                        position: absolute; top: -40px; right: -40px;
                        width: 180px; height: 180px;
                        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                        border-radius: 0;
                    }
                    .header-badge {
                        display: inline-block;
                        background: rgba(255,255,255,0.15);
                        border: 1px solid rgba(255,255,255,0.25);
                        color: #ffffff;
                        font-size: 9px;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 1.5px;
                        padding: 3px 10px;
                        border-radius: 0;
                        margin-bottom: 10px;
                    }
                    .title {
                        font-size: 22px;
                        color: #ffffff;
                        margin: 0 0 6px;
                        font-weight: 800;
                        letter-spacing: -0.3px;
                    }
                    .product-name {
                        font-size: 13px;
                        color: #93c5fd;
                        font-weight: 600;
                        margin: 0 0 4px;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        max-width: 85%;
                    }
                    .subtitle { color: #cbd5e1; font-size: 10px; margin: 0; }

                    /* ── Stat Cards ── */
                    .cards { display: flex; gap: 14px; margin-bottom: 20px; }
                    .card {
                        flex: 1;
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
                        border-radius: 0;
                        padding: 18px 16px;
                        text-align: center;
                    }
                    .card-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 8px; }
                    .card-value { font-size: 30px; font-weight: 800; line-height: 1; color: #0f172a; }

                    /* ── Alert Box ── */
                    .alert-box {
                        background: #fdf2f2;
                        border: 1px solid #ef4444;
                        border-radius: 0;
                        padding: 16px 20px;
                        margin-bottom: 20px;
                    }
                    .alert-box h2 { color: #991b1b; margin: 0 0 8px; font-size: 13px; font-weight: 700; }
                    .alert-box p { color: #b91c1c; margin: 0; font-size: 12px; }

                    /* ── Section ── */
                    .section {
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
                        border-radius: 0;
                        padding: 20px 22px;
                        margin-bottom: 18px;
                        page-break-inside: avoid;
                    }
                    .section-title {
                        font-size: 13px;
                        font-weight: 700;
                        color: #0f172a;
                        margin: 0 0 16px;
                        padding-bottom: 12px;
                        border-bottom: 1px solid #e2e8f0;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .section-title span.num {
                        background: rgba(30,58,138,0.08);
                        color: #1e3a8a;
                        border-radius: 0;
                        padding: 1px 7px;
                        font-size: 11px;
                        font-weight: 700;
                    }

                    /* ── Two-column layout ── */
                    .two-col { display: flex; gap: 14px; margin-bottom: 18px; }
                    .two-col .section { flex: 1; margin-bottom: 0; }

                    /* ── Summary quote ── */
                    .summary-quote {
                        font-size: 12.5px;
                        font-style: normal;
                        color: #334155;
                        border: 1px solid #1e3a8a;
                        padding: 16px 20px;
                        margin: 0;
                        background: rgba(30,58,138,0.03);
                        border-radius: 0;
                        white-space: pre-wrap;
                    }

                    /* ── Progress bars ── */
                    .progress-bar { background: #f1f5f9; border-radius: 0; overflow: hidden; height: 8px; }
                    .progress-fill { height: 100%; border-radius: 0; }
                    .stat-row { display: flex; align-items: center; margin-bottom: 10px; font-size: 12px; font-weight: 500; }

                    /* ── Keywords ── */
                    .keywords { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
                    .kw { padding: 3px 10px; border-radius: 0; font-size: 11px; font-weight: 600; border: 1px solid; }
                    .kw.pos { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
                    .kw.neg { background: #fdf2f2; border-color: #fecaca; color: #991b1b; }
                    .kw-label { font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }

                    /* ── Image grid ── */
                    .image-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
                    .image-card { border-radius: 0; overflow: hidden; border: 1px solid #e2e8f0; position: relative; }
                    .image-card img { width: 100%; height: 100px; object-fit: cover; display: block; border-radius: 0; }
                    .image-card .lbl { text-align: center; font-size: 9px; padding: 4px 6px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; border-radius: 0; }
                    .img-caption { color: #64748b; font-size: 10px; margin-bottom: 14px; font-style: italic; }

                    /* ── Table ── */
                    table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
                    th { background: #f8fafc; color: #475569; font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
                    td { padding: 10px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: top; color: #334155; }
                    tr:last-child td { border-bottom: none; }
                    .stars { color: #f59e0b; font-size: 11px; margin-bottom: 3px; }
                    .review-text { color: #1e293b; line-height: 1.5; }

                    /* ── Badges ── */
                    .badge { padding: 3px 8px; border-radius: 0; font-size: 10px; font-weight: 700; display: inline-block; white-space: nowrap; border: 1px solid; }
                    .badge-pos { background: #e6fbf1; color: #047857; border-color: #a7f3d0; }
                    .badge-neg { background: #fdf2f2; color: #b91c1c; border-color: #fecaca; }
                    .badge-neu { background: #f5f3ff; color: #6d28d9; border-color: #ddd6fe; }

                    /* ── Alt products ── */
                    .alt-products-grid { display: flex; gap: 12px; flex-wrap: wrap; }
                    .box.alt-product { flex: 1; min-width: 120px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0; padding: 10px; text-align: center; display: flex; flex-direction: column; align-items: center; }
                    .alt-product img { width: 100%; height: 90px; object-fit: cover; border-radius: 0; border: 1px solid #e2e8f0; }

                    /* ── Footer ── */
                    .footer { text-align: center; margin-top: 24px; color: #94a3b8; font-size: 10px; padding-top: 14px; border-top: 1px solid #e2e8f0; }
                </style>
            </head>
            <body>
                <!-- Header -->
                <div class="header">
                    <div class="header-badge">AI Analytics Report</div>
                    <h1 class="title">Báo Cáo Phân Tích Chuyên Sâu</h1>
                    <div class="product-name">${product.name}</div>
                    <div class="subtitle">Trích xuất lúc ${new Date().toLocaleString('vi-VN')} · Powered by Multi-modal AI</div>
                </div>

                <!-- Stat cards -->
                <div class="cards">
                    <div class="card">
                        <div class="card-label">Tổng Đánh Giá</div>
                        <div class="card-value" style="color:#059669;">${reviews.length}</div>
                    </div>
                    <div class="card">
                        <div class="card-label">Trust Score</div>
                        <div class="card-value" style="color:${trustColor === '#10b981' ? '#059669' : trustColor};">${trustScore}<span style="font-size:14px;font-weight:500;color:#94a3b8;">/100</span></div>
                    </div>
                    <div class="card">
                        <div class="card-label">Tỷ lệ Spam</div>
                        <div class="card-value" style="color:${spamPct > 30 ? '#dc2626' : '#d97706'};">${spamPct}<span style="font-size:14px;font-weight:500;color:#94a3b8;">%</span></div>
                    </div>
                    <div class="card">
                        <div class="card-label">Tích Cực</div>
                        <div class="card-value" style="color:#4f46e5;">${Math.round((s.positive / totalReviews) * 100)}<span style="font-size:14px;font-weight:500;color:#94a3b8;">%</span></div>
                    </div>
                </div>

                ${metadata.smartAdvice ? `
                <div class="alert-box">
                    <h2>⚠️ Cảnh Báo & Gợi Ý AI</h2>
                    <p>${metadata.smartAdvice}</p>
                </div>
                ` : ''}

                <!-- AI Summary -->
                <div class="section">
                    <div class="section-title"><span class="num">1</span> Tổng Quan Cảm Xúc — AI Summary</div>
                    <div class="summary-quote">${product.report?.summary_text || 'Không có dữ liệu'}</div>
                </div>

                <!-- Sentiment + ABSA -->
                <div class="two-col">
                    <div class="section">
                        <div class="section-title"><span class="num">2</span> Phân Bố Cảm Xúc & Nhãn Ảnh</div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Tích cực</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(s.positive/totalReviews)*100}%;background:#10b981;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${s.positive}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Trung lập</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(s.neutral/totalReviews)*100}%;background:#8b5cf6;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${s.neutral}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Tiêu cực</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(s.negative/totalReviews)*100}%;background:#ef4444;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${s.negative}</div>
                        </div>
                        <div style="border-top:1px solid #e2e8f0;margin:12px 0;"></div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Nguyên vẹn</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(l.intact/totalImages)*100}%;background:#10b981;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${l.intact}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Hỏng/Móp</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(l.damaged/totalImages)*100}%;background:#ef4444;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${l.damaged}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width:80px;color:#64748b;">Không l.quan</div>
                            <div style="flex:1;margin:0 10px;"><div class="progress-bar"><div class="progress-fill" style="width:${(l.irrelevant/totalImages)*100}%;background:#64748b;"></div></div></div>
                            <div style="width:28px;text-align:right;color:#0f172a;font-weight:700;">${l.irrelevant}</div>
                        </div>
                    </div>

                    <div class="section">
                        <div class="section-title"><span class="num">3</span> Phân Tích Khía Cạnh (ABSA)</div>
                        ${aspectsHtml || '<p style="color:#94a3b8;font-style:italic;font-size:12px;">Không có dữ liệu khía cạnh</p>'}
                    </div>
                </div>

                <!-- Keywords -->
                <div class="section">
                    <div class="section-title"><span class="num">4</span> Từ Khóa Nổi Bật</div>
                    <div style="margin-bottom:12px;">
                        <div class="kw-label">Điểm mạnh</div>
                        <div class="keywords">${keywordsHtmlPos || '<span style="color:#94a3b8;">Không có</span>'}</div>
                    </div>
                    <div>
                        <div class="kw-label">Điểm yếu</div>
                        <div class="keywords">${keywordsHtmlNeg || '<span style="color:#94a3b8;">Không có</span>'}</div>
                    </div>
                </div>

                ${imagesHtml ? `
                <div class="section">
                    <div class="section-title"><span class="num">5</span> Một Số Hình Ảnh Đính Kèm</div>
                    <p class="img-caption">Hiển thị tối đa 20 hình ảnh từ các đánh giá, được phân loại tự động bởi AI.</p>
                    <div class="image-grid">${imagesHtml}</div>
                </div>
                ` : ''}

                <!-- Review detail table -->
                <div class="section">
                    <div class="section-title"><span class="num">6</span> Chi Tiết Đánh Giá <span style="font-size:10px;font-weight:400;color:#64748b;">(tối đa 20 review có nội dung)</span></div>
                    <table>
                        <thead>
                            <tr>
                                <th>Nội dung đánh giá</th>
                                <th style="width:80px;">Cảm xúc</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${reviewsHtml || '<tr><td colspan="2" style="text-align:center;color:#64748b;">Không có dữ liệu</td></tr>'}
                        </tbody>
                    </table>
                </div>

                ${altProductsHtml ? `
                <div class="section">
                    <div class="section-title"><span class="num">7</span> Sản Phẩm Thay Thế Tương Tự</div>
                    <div class="alt-products-grid">${altProductsHtml}</div>
                </div>
                ` : ''}

                <div class="footer">
                    Báo cáo được tạo tự động bởi Hệ thống Phân tích Đánh giá Đa phương thức AI &copy; ${new Date().getFullYear()}
                </div>
            </body>
            </html>
        `;
        res.status(200).send(htmlContent);
    } catch (error) {
        console.error("PDF Export Error:", error);
        return res.status(500).json({ error: 'Lỗi xuất file PDF' });
    }
};




