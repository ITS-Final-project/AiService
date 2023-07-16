import socketio
import asyncio

from flask import Flask, request, jsonify
from WebSocket import WebSocket as ws

app = Flask(__name__)
sio = socketio.AsyncClient()

WS_PORT = 5001
HOST = 'node'

sockets = []
taskts = []

@app.route('/api/join/client', methods=['POST'])
async def joinClient():
    # get token from request (x-www-form-urlencoded)
    if request.method != 'POST':
        return jsonify({'error' : 'Method not allowed', 'status' : 405})

    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error' : 'Content-Type must be application/json', 'status' : 400})

    request_data = request.get_json()
    token = request_data['token']

    if token == None:
        return jsonify({'error' : 'Token is not provided', 'status' : 400})

    socket = ws()

    socket.connect(HOST, WS_PORT)
    socket.joinRoom('ai', token)

    sockets.append(socket)

    return jsonify({'status' : 200})

async def wsClientOnStream(socket):
    await socket.onStream()


async def wsClientConnect(token):
    socket = ws()

    await socket.connect(HOST, WS_PORT)
    await socket.joinRoom('ai', token)

    sockets.append(socket)

if __name__ ==  '__main__':

    app.run(host='0.0.0.0', port=5003, debug=True)
