import sequelize from '../config/database.mjs';
import User from './User.mjs';
import Product from './Product.mjs';
import Review from './Review.mjs';
import Report from './Report.mjs';

// User - Product association
User.hasMany(Product, { foreignKey: 'userId', as: 'products' });
Product.belongsTo(User, { foreignKey: 'userId' });

// Product - Review association
Product.hasMany(Review, { foreignKey: 'product_id', as: 'reviews', onDelete: 'CASCADE' });
Review.belongsTo(Product, { foreignKey: 'product_id' });

// Product - Report association
Product.hasOne(Report, { foreignKey: 'product_id', as: 'report', onDelete: 'CASCADE' });
Report.belongsTo(Product, { foreignKey: 'product_id' });

export { sequelize, User, Product, Review, Report };
