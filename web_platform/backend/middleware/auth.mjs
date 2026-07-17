import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);

export const verifyToken = async (req, res, next) => {
    // 1. Extract token from authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Từ chối truy cập: Không tìm thấy JWT Token' });
    }

    const token = authHeader.split(' ')[1];

    // 2. Verify token using Supabase auth
    const { data: { user }, error } = await supabase.auth.getUser(token);

    if (error || !user) {
        return res.status(401).json({ error: 'Từ chối truy cập: JWT Token không hợp lệ hoặc đã hết hạn' });
    }

    // 3. Attach verified user to request
    req.user = user;
    next();
};
