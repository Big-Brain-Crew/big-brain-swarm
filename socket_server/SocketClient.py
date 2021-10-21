import socket


def main(msg):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(("35.231.193.25", 5071))
    clientsocket.sendall(bytes(msg,encoding="utf-8"))

    while True:
        rtrnmsg = clientsocket.recv(1024)
        print(rtrnmsg)

if __name__ == "__main__":
    main()