import { DataTypes } from 'sequelize';
import sequelize from '../config/database.mjs';
import Product from './Product.mjs';

const Report = sequelize.define('Report', {
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
    summary_text: { type: DataTypes.TEXT },
    risk_level: { type: DataTypes.STRING }
}, { tableName: 'reports', timestamps: true });

export default Report;
