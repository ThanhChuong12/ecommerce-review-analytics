import Redis from 'ioredis';
import dotenv from 'dotenv';
dotenv.config();

// Connection for BullMQ
const restUrl = process.env.UPSTASH_REDIS_REST_URL || '';
const host = restUrl.replace('https://', '');
const token = process.env.UPSTASH_REDIS_REST_TOKEN;

const redisConnection = new Redis({
    host: host,
    port: 6379,
    password: token,
    tls: {}, // Required for Upstash
    maxRetriesPerRequest: null, // Required for BullMQ
    // keepAlive: 10000,
    // family: 0
});


// Cache-only connection
// const redisConnection = new Redis(process.env.UPSTASH_REDIS_URL);

redisConnection.on('error', (err) => {
    console.error('Redis connection error:', err);
});

export default redisConnection;
