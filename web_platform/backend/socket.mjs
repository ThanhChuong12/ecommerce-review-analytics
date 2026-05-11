import { Server } from 'socket.io';

let io;

export const initSocket = (server) => {
    io = new Server(server, { cors: { origin: "*" } });

    io.on('connection', (socket) => {
        console.log('Client connected:', socket.id);

        // Frontend sẽ gửi event 'join-room' truyền kèm productId
        socket.on('join-room', (productId) => {
            socket.join(`room-${productId}`);
            console.log(`Socket ${socket.id} đã join room: room-${productId}`);
        });

        socket.on('disconnect', () => {
            console.log('Client disconnected:', socket.id);
        });
    });
    return io;
};

export const getIo = () => {
    if (!io) throw new Error('Socket.io chưa khởi tạo!');
    return io;
};
