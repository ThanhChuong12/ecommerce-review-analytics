import express from 'express';
import { getHistoryList, getHistoryDetail, deleteHistory } from '../controllers/historyController.mjs';
import { verifyToken } from '../middleware/auth.mjs';

const router = express.Router();

// Lắp "Chốt bảo vệ" cho TẤT CẢ API đi qua route này
router.use(verifyToken);

router.get('/', getHistoryList);
router.get('/:productId', getHistoryDetail);
router.delete('/:productId', deleteHistory);

export default router;
