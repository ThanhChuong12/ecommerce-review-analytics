import express from 'express';
import http from 'http';
import cors from 'cors';
import dotenv from 'dotenv';
import { sequelize } from './models/index.mjs';
import { initSocket } from './socket.mjs';
import analyzeRoutes from './routes/analyzeRoutes.mjs';
import webhookRoutes from './routes/webhookRoutes.mjs';
import authRoutes from './routes/authRoutes.mjs';
import historyRoutes from './routes/historyRoutes.mjs';

// Kích hoạt worker chạy ngầm bằng cách import nó vào server
import './queue/worker.mjs';

dotenv.config();

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json({ limit: '50mb' })); // Tăng limit vì webhook gửi data mảng 2000 reviews có thể khá nặng

initSocket(server);

app.use('/api', analyzeRoutes);
app.use('/api/webhook', webhookRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/history', historyRoutes);

const PORT = process.env.PORT || 5000;

const startServer = async () => {
    try {
        // alter: true tự động map code model sinh ra bảng Postgres, đỡ phải viết file Migration
        await sequelize.sync({ alter: true });
        console.log('Đã đồng bộ Database (Postgres) thành công.');

        server.listen(PORT, () => {
            console.log(`🔥 Node.js Server đang chạy trên port ${PORT}`);
        });
    } catch (error) {
        console.error('Lỗi khi khởi động server:', error);
    }
};

startServer();
