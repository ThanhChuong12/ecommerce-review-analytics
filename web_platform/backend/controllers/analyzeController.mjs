import { Product } from '../models/index.mjs';
import { User } from '../models/index.mjs'
import analysisQueue from '../queue/analysisQueue.mjs';

export const analyzeUrl = async (req, res) => {
    try {
        const { url, userId } = req.body;
        if (!url) return res.status(400).json({ error: 'Thiếu URL sản phẩm' });

        let productId;
        if (userId) {
            const userExists = await User.findByPk(userId);
            if (!userExists) {
                return res.status(400).json({ error: 'Tài khoản chưa được đồng bộ với hệ thống. Vui lòng tải lại trang (F5) và thử lại.' });
            }

            const product = await Product.create({
                url,
                status: 'PENDING',
                userId: userId
            });
            productId = product.id;
        } else {
            productId = `temp-${Date.now()}`;
        }

        const job = await analysisQueue.add('multimodal-task', {
            productId: productId,
            url: url
        });

        return res.status(200).json({
            success: true,
            productId: productId,
            jobId: job.id,
            message: 'Đã nhận URL, đang chuẩn bị cào dữ liệu.'
        });
    } catch (error) {
        console.error('Error analyzeUrl:', error);
        return res.status(500).json({ error: 'Lỗi server' });
    }
};
