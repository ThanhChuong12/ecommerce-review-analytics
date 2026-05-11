import { DataTypes } from 'sequelize';
import sequelize from '../config/database.mjs';

const User = sequelize.define('User', {
    id: {
        type: DataTypes.UUID,
        primaryKey: true,
    },
    email: {
        type: DataTypes.STRING,
        unique: true,
        allowNull: false,
    },
    name: {
        type: DataTypes.STRING,
        allowNull: true,
    },
    avatar: {
        type: DataTypes.STRING,
        allowNull: true,
    }
}, { tableName: 'users', timestamps: true });

export default User;
