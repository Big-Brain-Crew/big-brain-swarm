import socket


def main():
    def repeated_message(cs):
        cs.sendall(b"0.2 0.2")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((socket.gethostname(), 5071))
        s.listen(50)
        clientsocket, address = s.accept()
        with clientsocket:
            print('Connected from:', address)
            while True:
                rcvdmsg = clientsocket.recv(1024)
                if not rcvdmsg:
                    break
                repeated_message(clientsocket)

if __name__ == "__main__":
    main()