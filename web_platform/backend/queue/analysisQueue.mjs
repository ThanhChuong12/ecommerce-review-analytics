import { Queue } from 'bullmq';
import redisConnection from '../config/redis.mjs';

const analysisQueue = new Queue('AnalysisQueue', {
    connection: redisConnection
});

export default analysisQueue;
