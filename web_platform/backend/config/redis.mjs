import Redis from 'ioredis';
import dotenv from 'dotenv';
dotenv.config();

// dùng bullMQ
const restUrl = process.env.UPSTASH_REDIS_REST_URL || '';
const host = restUrl.replace('https://', '');
const token = process.env.UPSTASH_REDIS_REST_TOKEN;

const redisConnection = new Redis({
    host: host,
    port: 6379,
    password: token,
    tls: {}, // bắt buộc cho upstash
    maxRetriesPerRequest: null, // bắt buộc cho bullMQ
    // keepAlive: 10000,
    // family: 0
});


// chỉ dùng cho cache
// const redisConnection = new Redis(process.env.UPSTASH_REDIS_URL);

redisConnection.on('error', (err) => {
    console.error('Redis connection error:', err);
});

export default redisConnection;
