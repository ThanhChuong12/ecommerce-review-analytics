import express from 'express';
import { analyzeUrl } from '../controllers/analyzeController.mjs';

const router = express.Router();
router.post('/analyze', analyzeUrl);

export default router;
