from wsgiref.simple_server import make_server, demo_app

if __name__ == '__main__':
    with make_server('', 8000, demo_app) as httpd:
        sa = httpd.socket.getsockname()
        print("Serving HTTP:", sa)
        httpd.serve_forever()
