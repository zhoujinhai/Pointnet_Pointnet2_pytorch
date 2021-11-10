import webserver

if __name__ == '__main__':
    server = webserver.MeshWebServer()
    server.run(port=8000)
