#!/usr/bin/env python3
# coding:utf-8

from socket import *
from multiprocessing import Process


class MyWebServer(object):

    @staticmethod
    def deal(conn):
        recv_data = conn.recv(1024).decode('utf-8')
        recv_data_head = recv_data.splitlines()[0]
        print('------', recv_data_head)
        request_method, request_path, http_version = recv_data_head.split()

        # 去掉url中的?和之后的参数
        request_path = request_path.split('?')[0]

        if request_path == '/':
            request_path = '/index.html'
        file_name = "../build/html" + request_path

        try:
            f = open(file_name, 'rb')
        except IOError:
            conn.send(b'HTTP/1.1 404 ERROR \r\n\r\n <h1>Page is not exsit .</h1>')
            return

        read_data = f.read()
        send_data = b'HTTP/1.1 200 OK \r\n\r\n' + read_data
        conn.send(send_data)
        f.close()

    def __init__(self):
        self.s = socket(AF_INET, SOCK_STREAM)
        self.s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.s.bind(('',8000))
        self.s.listen(1023)

    def start(self):
        while 1:
            conn, user_info = self.s.accept()
            print(user_info)
            p = Process(target=self.deal, args=(conn,))
            p.start()
            conn.close()  # 进程会复制出一个新的conn,所以这里的conn需要关闭


s = MyWebServer()
s.start()