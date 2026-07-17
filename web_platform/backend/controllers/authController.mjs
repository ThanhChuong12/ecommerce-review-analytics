import { User } from '../models/index.mjs';

export const syncUser = async (req, res) => {
    try {
        const { id, email, name, avatar } = req.body;
        if (!id || !email) return res.status(400).json({ error: 'Thiếu dữ liệu user' });

        // Upsert user profile
        const [user, created] = await User.upsert({
            id, email, name, avatar
        });

        return res.status(200).json({ success: true, user });
    } catch (error) {
        console.error('Error syncUser:', error);
        return res.status(500).json({ error: 'Lỗi server' });
    }
};
