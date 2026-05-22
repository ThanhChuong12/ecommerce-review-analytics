import { Queue } from 'bullmq';
import redisConnection from '../config/redis.mjs';

const analysisQueue = new Queue('Queue', {
    connection: redisConnection,
    defaultJobOptions: {
        removeOnComplete: true,
        removeOnFail: { age: 86400 },
    }
});

export default analysisQueue;
