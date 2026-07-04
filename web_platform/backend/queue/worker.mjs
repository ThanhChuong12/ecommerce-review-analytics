import { Worker } from 'bullmq';
import axios from 'axios';
import redisConnection from '../config/redis.mjs';
import { Product } from '../models/index.mjs';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

const worker = new Worker('Queue', async (job) => {
    const { productId, url } = job.data;

    try {
        if (!String(productId).startsWith('temp-')) {
            await Product.update({ status: 'PROCESSING' }, { where: { id: productId } });
        }

        await axios.post(`${PYTHON_API_URL}/process-job`, {
            productId: productId,
            url: url
        });
    } catch (error) {
        if (!String(productId).startsWith('temp-')) {
            await Product.update({ status: 'FAILED' }, { where: { id: productId } });
        }
        throw error;
    }
}, { connection: redisConnection });

export default worker;
