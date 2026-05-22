import express from 'express';
import { getHistoryList, getHistoryDetail, deleteHistory, exportPDF } from '../controllers/historyController.mjs';
import { verifyToken } from '../middleware/auth.mjs';

const router = express.Router();

// Lắp "Chốt bảo vệ" cho TẤT CẢ API đi qua route này
router.use(verifyToken);

router.get('/', getHistoryList);
router.get('/:productId', getHistoryDetail);
router.get('/:productId/export', exportPDF);
router.delete('/:productId', deleteHistory);

export default router;
