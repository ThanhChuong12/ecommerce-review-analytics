import { DataTypes } from 'sequelize';
import sequelize from '../config/database.mjs';
import Product from './Product.mjs';

const Review = sequelize.define('Review', {
    id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true,
    },
    product_id: {
        type: DataTypes.UUID,
        allowNull: false,
        references: { model: Product, key: 'id' }
    },
    review_text: { type: DataTypes.TEXT },
    rating: { type: DataTypes.INTEGER },
    image_path: { type: DataTypes.TEXT },
    label: { type: DataTypes.ENUM('intact', 'damaged', 'wrong_item', 'irrelevant') },
    sentiment: { type: DataTypes.STRING }
}, { tableName: 'reviews', timestamps: true });

export default Review;
