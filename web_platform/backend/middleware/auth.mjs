import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);

export const verifyToken = async (req, res, next) => {
    // 1. Lấy token từ header Authorization
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Từ chối truy cập: Không tìm thấy JWT Token' });
    }

    const token = authHeader.split(' ')[1];

    // 2. Nhờ Supabase giải mã và verify độ uy tín của JWT
    const { data: { user }, error } = await supabase.auth.getUser(token);

    if (error || !user) {
        return res.status(401).json({ error: 'Từ chối truy cập: JWT Token không hợp lệ hoặc đã hết hạn' });
    }

    // 3. Gắn thông tin user (đã được verify 100%) vào request để các hàm sau xài
    req.user = user;
    next();
};
