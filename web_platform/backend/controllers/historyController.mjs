import { Product, Review, Report } from '../models/index.mjs';

export const getHistoryList = async (req, res) => {
    try {
        // Lấy ID chuẩn xác từ JWT Token (Hacker không thể làm giả được)
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
            // Check chéo bảo mật: Sản phẩm này phải khớp ID với chủ nhân của Token mới cho xem
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

        // Chỉ cho phép xóa nếu sản phẩm đó thuộc về Token đang giữ
        const deletedCount = await Product.destroy({
            where: { id: productId, userId: userId }
        });

        if (deletedCount === 0) return res.status(404).json({ error: 'Không có quyền xoá' });

        return res.status(200).json({ success: true, message: 'Đã xoá lịch sử' });
    } catch (error) {
        return res.status(500).json({ error: 'Lỗi server' });
    }
};
