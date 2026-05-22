import { Product } from '../models/index.mjs';
import analysisQueue from '../queue/analysisQueue.mjs';

export const analyzeUrl = async (req, res) => {
    try {
        const { url, userId } = req.body;
        if (!url) return res.status(400).json({ error: 'Thiếu URL sản phẩm' });

        const product = await Product.create({
            url,
            status: 'PENDING',
            userId: userId || null
        });

        const job = await analysisQueue.add('multimodal-task', {
            productId: product.id,
            url: url
        });

        return res.status(200).json({
            success: true,
            productId: product.id,
            jobId: job.id,
            message: 'Đã nhận URL, đang chuẩn bị cào dữ liệu.'
        });
    } catch (error) {
        console.error('Error analyzeUrl:', error);
        return res.status(500).json({ error: 'Lỗi server' });
    }
};
