import express from 'express';
import { finishedWebhook, updateProgressWebhook } from '../controllers/webhookController.mjs';

const router = express.Router();
router.post('/finished', finishedWebhook);
router.post('/update-progress', updateProgressWebhook);

export default router;
