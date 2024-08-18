import socket

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 12345))  # Bind to all network interfaces
        s.listen()
        print("Server is listening...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                conn.sendall(b"Hello from the server!")  # Send a test message

if __name__ == "__main__":
    start_server()
