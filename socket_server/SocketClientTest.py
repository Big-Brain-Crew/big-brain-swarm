import json
import SocketClient

test_dict = {"a": "3", "b": "2", "c": "3"}

def main():
    test_json = json.dumps(test_dict)
    SocketClient.main(test_json)

if __name__ == "__main__":
    main()