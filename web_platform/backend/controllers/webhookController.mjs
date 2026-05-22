import { Product, Review, Report } from '../models/index.mjs';
import { getIo } from '../socket.mjs';

export const finishedWebhook = async (req, res) => {
    try {
        const { productId, productData, reviews, summary, metadata } = req.body;

        // Update Product info
        await Product.update({
            name: productData.name,
            thumbnail: productData.thumbnail,
            status: 'COMPLETED'
        }, { where: { id: productId } });

        // BulkCreate review vào DB
        if (reviews && reviews.length > 0) {
            const reviewRecords = reviews.map(r => ({
                product_id: productId,
                review_text: r.review_text,
                rating: r.rating,
                image_path: r.image_path,
                label: r.label,
                sentiment: r.sentiment
            }));
            await Review.bulkCreate(reviewRecords);
        }

        // Lưu Report đánh giá rủi ro
        await Report.create({
            product_id: productId,
            summary_text: summary,
            risk_level: 'Tính toán bên python', // Hoặc có thể thêm từ payload
            metadata: metadata || {}
        });

        // Báo Frontend render UI
        const io = getIo();
        io.to(`room-${productId}`).emit('finished', {
            productId, productData, summary, reviews, metadata
        });

        return res.status(200).json({ success: true, message: 'Node.js đã lưu DB xong.' });
    } catch (error) {
        console.error('Webhook Error:', error);
        if (req.body.productId) {
            await Product.update({ status: 'FAILED' }, { where: { id: req.body.productId } });
        }
        return res.status(500).json({ error: 'Lỗi khi lưu DB' });
    }
};

export const updateProgressWebhook = async (req, res) => {
    try {
        // API để Python báo cáo % tiến độ liên tục (vd: Scraping 20%, ResNet 50%)
        const { productId, progress, message } = req.body;

        const io = getIo();
        io.to(`room-${productId}`).emit('progress', { progress, message });

        return res.status(200).json({ success: true });
    } catch (error) {
        return res.status(500).json({ error: 'Lỗi server' });
    }
};
