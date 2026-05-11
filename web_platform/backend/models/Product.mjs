import { DataTypes } from 'sequelize';
import sequelize from '../config/database.mjs';

const Product = sequelize.define('Product', {
    id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true,
    },
    userId: {
        type: DataTypes.UUID,
        allowNull: true, // Nếu chưa đăng nhập thì field này để trống (ẩn danh)
    },
    name: {
        type: DataTypes.STRING,
        allowNull: true,
    },
    url: {
        type: DataTypes.TEXT,
        allowNull: false,
    },
    thumbnail: {
        type: DataTypes.STRING,
        allowNull: true,
    },
    status: {
        type: DataTypes.ENUM('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'),
        defaultValue: 'PENDING',
    }
}, { tableName: 'products', timestamps: true });

export default Product;
