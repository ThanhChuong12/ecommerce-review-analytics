import { Worker } from 'bullmq';
import axios from 'axios';
import redisConnection from '../config/redis.mjs';
import { Product } from '../models/index.mjs';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

const worker = new Worker('AnalysisQueue', async (job) => {
    const { productId, url } = job.data;
    console.log(`[Worker] Bốc Job ${job.id} - Bắt đầu điều phối Product ID: ${productId}`);

    try {
        // 1. Chuyển trạng thái sang PROCESSING
        await Product.update({ status: 'PROCESSING' }, { where: { id: productId } });

        // 2. Giao việc cho FastAPI Python chạy nền
        console.log(`[Worker] Đang gửi lệnh sang Python FastAPI cho URL: ${url}`);
        await axios.post(`${PYTHON_API_URL}/process-job`, {
            productId: productId,
            url: url
        });

        console.log(`[Worker] Giao thành công. Giờ chờ Python gọi Webhook trả kết quả.`);
    } catch (error) {
        console.error(`[Worker] Lỗi kết nối Python:`, error.message);
        await Product.update({ status: 'FAILED' }, { where: { id: productId } });
        throw error;
    }
}, { connection: redisConnection });

worker.on('failed', (job, err) => {
    console.log(`[Worker] Job ${job.id} thất bại: ${err.message}`);
});

export default worker;
