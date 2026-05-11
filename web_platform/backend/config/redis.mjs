import Redis from 'ioredis';
import dotenv from 'dotenv';
dotenv.config();

// Upstash cung cấp REST URL, nhưng ioredis cần chuẩn kết nối Native Redis qua TLS
// Ta sẽ tách hostname từ REST URL ra để dùng
const restUrl = process.env.UPSTASH_REDIS_REST_URL || '';
const host = restUrl.replace('https://', '').replace('http://', '');
const token = process.env.UPSTASH_REDIS_REST_TOKEN;

const redisConnection = new Redis({
    host: host,
    port: 6379,
    password: token,
    tls: {}, // Bắt buộc bật TLS khi dùng Upstash Native Redis
    maxRetriesPerRequest: null,
});

redisConnection.on('error', (err) => {
    console.error('Redis connection error:', err);
});

export default redisConnection;
