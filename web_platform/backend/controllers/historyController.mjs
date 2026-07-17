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
        reviews.forEach((r, idx) => {
            if (s[r.sentiment] !== undefined) s[r.sentiment]++;
            if (l[r.label] !== undefined) l[r.label]++;

            if (idx < 20) {
                const sentClass = r.sentiment === 'positive' ? 'badge-pos' : r.sentiment === 'negative' ? 'badge-neg' : 'badge-neu';
                const sentText = r.sentiment === 'positive' ? 'Tích cực' : r.sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập';
                const lblText = r.label === 'intact' ? 'Nguyên vẹn' : r.label === 'damaged' ? 'Hỏng/Móp' : r.label === 'wrong_item' ? 'Sai hàng' : 'Không l.quan';

                let imgLblBg = '#e2e8f0';
                let imgLblColor = '#334';
                if (r.label === 'intact') {
                    imgLblBg = '#d1fae5'; // green-100
                    imgLblColor = '#065f46'; // green-800
                } else if (r.label === 'damaged') {
                    imgLblBg = '#ffe4e6'; // red-100
                    imgLblColor = '#be123c'; // red-800
                } else if (r.label === 'wrong_item') {
                    imgLblBg = '#ffedd5'; // orange-100
                    imgLblColor = '#9a3412'; // orange-800
                } else if (r.label === 'irrelevant') {
                    imgLblBg = '#fee2e2'; // warning red-100 (irrelevant)
                    imgLblColor = '#991b1b'; // warning red-800
                }

                reviewsHtml += `
                    <tr>
                        <td><div style="color:#f59e0b;font-size:11px;">${'★'.repeat(r.rating)}${'☆'.repeat(5 - r.rating)}</div>${r.review_text}</td>
                        <td><span class="badge ${sentClass}">${sentText}</span></td>
                        <td><span class="badge" style="background:${imgLblBg};color:${imgLblColor}">${lblText}</span></td>
                    </tr>
                `;
            }

            if (r.image_path && imageCount < 20) {
                imageCount++;
                const lblText = r.label === 'intact' ? 'Nguyên vẹn' : r.label === 'damaged' ? 'Hỏng/Móp' : r.label === 'wrong_item' ? 'Sai hàng' : 'Không l.quan';
                // Professional solid background colors for image labels
                const lblColor = r.label === 'intact' ? '#10b981' : r.label === 'damaged' ? '#be123c' : r.label === 'wrong_item' ? '#f59e0b' : '#dc2626';
                imagesHtml += `
                    <div class="image-card">
                        <img src="${r.image_path}" alt="Review image" loading="lazy">
                        <div class="lbl" style="background: ${lblColor}">${lblText}</div>
                    </div>
                `;
            }
        });

        const totalReviews = reviews.length || 1;
        const totalImages = reviews.filter(r => r.image_path).length || 1;

        const aspectNames = {
            'Product': 'Sản phẩm',
            'Packaging': 'Đóng gói',
            'Shipping': 'Vận chuyển'
        };
        const aspectsHtml = Object.entries(metadata.aspectSentiment || {}).map(([key, val]) => {
            const displayName = aspectNames[key] || key;
            return `
                <div class="stat-row">
                    <div style="flex:1;">${displayName}</div>
                    <div style="width: 150px;">
                        <div class="progress-bar"><div class="progress-fill" style="width: ${(val / 5) * 100}%; background: #d946ef;"></div></div>
                    </div>
                    <div style="width: 40px; text-align:right; font-weight:bold;">${val}/5</div>
                </div>
            `;
        }).join('');

        const keywordsHtmlPos = (metadata.keywords?.positive || []).map(k => `<span class="kw pos">${k.text || k}</span>`).join('');
        const keywordsHtmlNeg = (metadata.keywords?.negative || []).map(k => `<span class="kw neg">${k.text || k}</span>`).join('');

        const altProductsHtml = (metadata.alternativeProducts || []).map(alt => {
            const reason = alt.reason || '';
            let badgeBg = '#dbeafe'; let badgeColor = '#1d4ed8';
            if (reason.includes('tương tự') || reason.includes('Tương tự')) { badgeBg = '#ede9fe'; badgeColor = '#6d28d9'; }
            else if (reason.includes('Đánh giá') || reason.includes('Uy tín') || reason.includes('Mua nhiều')) { badgeBg = '#d1fae5'; badgeColor = '#065f46'; }
            else if (reason.includes('Giá')) { badgeBg = '#f3e8ff'; badgeColor = '#7c3aed'; }
            const rerankPct = alt.rerank_score ? Math.round(alt.rerank_score * 100) : null;
            return `
            <div class="box alt-product" style="position: relative; display:flex; flex-direction:column; align-items:center;">
                <img src="${alt.thumbnail}" alt="product">
                ${reason ? `<span style="background:${badgeBg};color:${badgeColor};font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;margin-bottom:4px;display:inline-block;">${reason}</span>` : ''}
                <h4 style="font-size:11px; margin:6px 0 4px; text-align:center; overflow:hidden; max-height:34px;">${alt.name}</h4>
                <div style="color:#10b981; font-size:10px; margin-top:2px; font-weight:bold;">Trust: ${alt.trustScore || 0}/100</div>
                ${rerankPct !== null ? `
                <div style="width:100%;margin-top:4px;">
                    <div style="display:flex;justify-content:space-between;font-size:8px;color:#94a3b8;margin-bottom:1px;">
                        <span>AI Score</span><span style="color:#6366f1;font-weight:bold;">${rerankPct}%</span>
                    </div>
                    <div style="background:#e2e8f0;border-radius:3px;height:3px;overflow:hidden;">
                        <div style="background:linear-gradient(to right,#6366f1,#60a5fa);height:100%;width:${rerankPct}%;"></div>
                    </div>
                </div>` : ''}
            </div>
        `}).join('');


        const htmlContent = `
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <title>BÁO CÁO PHÂN TÍCH CHUYÊN SÂU</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
                <style>
                    @page {
                        size: A4;
                        margin: 15mm;
                    }
                    body { 
                        font-family: 'Inter', sans-serif; 
                        color: #1e293b; 
                        line-height: 1.6; 
                        background: #ffffff; 
                        font-size: 13px; 
                        margin: 0;
                        -webkit-print-color-adjust: exact;
                        print-color-adjust: exact;
                    }
                    .header { 
                        text-align: center; 
                        border-bottom: 4px solid #2563eb; 
                        padding-bottom: 20px; 
                        margin-bottom: 30px; 
                    }
                    .title { 
                        font-size: 26px; 
                        color: #1e3a8a; 
                        margin: 0; 
                        font-weight: 800; 
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    .subtitle { 
                        color: #64748b; 
                        font-size: 13px; 
                        margin-top: 8px; 
                        font-weight: 500;
                    }
                    .section { 
                        margin-bottom: 30px; 
                        background: #ffffff; 
                        padding: 24px; 
                        border-radius: 12px; 
                        border: 1px solid #e2e8f0; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                        page-break-inside: avoid; 
                    }
                    .section h2 { 
                        color: #0f172a; 
                        margin-top: 0; 
                        font-size: 17px; 
                        border-bottom: 2px solid #e2e8f0; 
                        padding-bottom: 12px; 
                        margin-bottom: 20px; 
                        font-weight: 700;
                    }
                    .flex { display: flex; gap: 20px; flex-wrap: wrap; }
                    .box { 
                        flex: 1; 
                        min-width: 150px; 
                        background: #f8fafc; 
                        padding: 20px; 
                        border-radius: 12px; 
                        border: 1px solid #e2e8f0; 
                        text-align: center; 
                    }
                    .box h3 { 
                        margin: 0 0 8px 0; 
                        color: #475569; 
                        font-size: 12px; 
                        text-transform: uppercase; 
                        font-weight: 600;
                        letter-spacing: 0.5px;
                    }
                    .box .value { font-size: 28px; font-weight: 800; }
                    .danger-box { 
                        background: #fff1f2; 
                        border: 1px solid #fecdd3; 
                        padding: 20px; 
                        border-radius: 12px; 
                        margin-bottom: 30px; 
                    }
                    .danger-box h2 { 
                        color: #be123c; 
                        margin-top: 0; 
                        font-size: 17px; 
                        border-bottom: 1px solid #fda4af; 
                        padding-bottom: 12px; 
                        margin-bottom: 12px;
                    }
                    
                    .progress-bar { background: #e2e8f0; border-radius: 8px; overflow: hidden; height: 10px; margin-top: 5px; }
                    .progress-fill { height: 100%; border-radius: 8px; transition: width 0.3s ease; }
                    .stat-row { display: flex; justify-content: space-between; margin-bottom: 12px; align-items: center; font-weight: 500;}
                    
                    table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 12px; }
                    th, td { border-bottom: 1px solid #e2e8f0; padding: 12px 14px; text-align: left; vertical-align: top; }
                    th { background: #f8fafc; color: #475569; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;}
                    tr:nth-child(even) { background: #fdfdfd; }
                    
                    .badge { padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; display: inline-block; white-space: nowrap; }
                    .badge-pos { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
                    .badge-neg { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
                    .badge-neu { background: #f3e8ff; color: #5b21b6; border: 1px solid #e9d5ff; }
                    
                    .keywords { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
                    .kw { background: #ffffff; border: 1px solid #cbd5e1; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500;}
                    .kw.pos { border-color: #34d399; color: #059669; background: #ecfdf5;}
                    .kw.neg { border-color: #f87171; color: #dc2626; background: #fef2f2;}
 
                    .image-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
                    .image-card { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; position: relative; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
                    .image-card img { width: 100%; height: 110px; object-fit: cover; display: block; }
                    .image-card .lbl { color: white; text-align: center; font-size: 10px; padding: 4px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;}
                    
                    .alt-product { padding: 12px; background: #ffffff; }
                    .alt-product img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #f1f5f9; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1 class="title">BÁO CÁO PHÂN TÍCH CHUYÊN SÂU</h1>
                    <div class="subtitle">Sản phẩm: <strong>${product.name}</strong><br>Trích xuất lúc: ${new Date().toLocaleString('vi-VN')}</div>
                </div>
                
                <div class="flex" style="margin-bottom: 25px;">
                    <div class="box">
                        <h3>Tổng Đánh Giá</h3>
                        <div class="value" style="color: #10b981;">${reviews.length}</div>
                    </div>
                    <div class="box">
                        <h3>Trust Score</h3>
                        <div class="value" style="color: #3b82f6;">${metadata.trustScore || 0}/100</div>
                    </div>
                    <div class="box">
                        <h3>Tỷ lệ Spam</h3>
                        <div class="value" style="color: #ef4444;">${metadata.spamPercentage || 0}%</div>
                    </div>
                </div>

                ${metadata.smartAdvice ? `
                <div class="danger-box">
                    <h2>⚠️ Cảnh Báo & Gợi Ý</h2>
                    <p style="color: #9f1239; font-size: 15px; margin: 0; font-weight: 500;">${metadata.smartAdvice}</p>
                </div>
                ` : ''}

                <div class="section">
                    <h2>1. Tổng Quan Cảm Xúc (AI Summary)</h2>
                    <p style="font-size: 15px; font-style: italic; color: #334155; border-left: 4px solid #8b5cf6; padding-left: 15px; margin: 0;">"${product.report?.summary_text || 'Không có dữ liệu'}"</p>
                </div>

                <div class="flex" style="margin-bottom: 25px;">
                    <div class="section" style="flex: 1; margin-bottom: 0;">
                        <h2>2. Phân Bố Cảm Xúc & Nhãn Ảnh</h2>
                        <div class="stat-row">
                            <div style="width: 80px;">Tích cực</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(s.positive / totalReviews) * 100}%; background: #10b981;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${s.positive}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width: 80px;">Trung lập</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(s.neutral / totalReviews) * 100}%; background: #8b5cf6;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${s.neutral}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width: 80px;">Tiêu cực</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(s.negative / totalReviews) * 100}%; background: #ef4444;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${s.negative}</div>
                        </div>
                        <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 15px 0;">
                        <div class="stat-row">
                            <div style="width: 80px;">Nguyên vẹn</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(l.intact / totalImages) * 100}%; background: #10b981;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${l.intact}</div>
                        </div>
                        <div class="stat-row">
                            <div style="width: 80px;">Hỏng/Móp</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(l.damaged / totalImages) * 100}%; background: #ef4444;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${l.damaged}</div>
                        </div>

                        <div class="stat-row">
                            <div style="width: 80px;">Không l.quan</div>
                            <div style="flex:1; margin: 0 10px;"><div class="progress-bar"><div class="progress-fill" style="width: ${(l.irrelevant / totalImages) * 100}%; background: #dc2626;"></div></div></div>
                            <div style="width: 30px; text-align:right;">${l.irrelevant}</div>
                        </div>
                    </div>
                    
                    <div class="section" style="flex: 1; margin-bottom: 0;">
                        <h2>3. Phân Tích Khía Cạnh (ABSA)</h2>
                        ${aspectsHtml || '<p style="color:#64748b; font-style:italic;">Không có dữ liệu khía cạnh</p>'}
                    </div>
                </div>

                <div class="section">
                    <h2>4. Từ Khóa Nổi Bật</h2>
                    <div style="margin-bottom: 10px;">
                        <strong>Điểm mạnh:</strong> 
                        <div class="keywords" style="margin-top: 5px;">${keywordsHtmlPos || '<span style="color:#94a3b8">Không có</span>'}</div>
                    </div>
                    <div>
                        <strong>Điểm yếu:</strong> 
                        <div class="keywords" style="margin-top: 5px;">${keywordsHtmlNeg || '<span style="color:#94a3b8">Không có</span>'}</div>
                    </div>
                </div>

                ${imagesHtml ? `
                <div class="section">
                    <h2>5. Hình Ảnh Đính Kèm</h2>
                    <div class="image-grid">
                        ${imagesHtml}
                    </div>
                </div>
                ` : ''}

                <div class="section">
                    <h2>6. Chi Tiết Đánh Giá (Tối đa 20 review đầu)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Nội dung</th>
                                <th style="width:80px;">Cảm xúc</th>
                                <th style="width:80px;">Nhãn ảnh</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${reviewsHtml || '<tr><td colspan="3" style="text-align:center;">Không có dữ liệu</td></tr>'}
                        </tbody>
                    </table>
                </div>

                ${altProductsHtml ? `
                <div class="section">
                    <h2>7. Sản Phẩm Thay Thế Tương Tự</h2>
                    <div class="flex">
                        ${altProductsHtml}
                    </div>
                </div>
                ` : ''}

                <div style="text-align: center; margin-top: 30px; color: #94a3b8; font-size: 12px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
                    Báo cáo được tạo tự động bởi Hệ thống Phân tích Đa phương thức AI &copy; ${new Date().getFullYear()}
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
