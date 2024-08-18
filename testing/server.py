import socket
import pickle
from PIL import Image
import io
import matplotlib.pyplot as plt

def process_image_and_text(image_data, text):
    # Load the image
    image = Image.open(io.BytesIO(image_data))

    # Plot the image with the text as the title
    plt.imshow(image)
    plt.title(text)
    plt.axis('off')  # Hide the axes
    plt.show()

def start_server():
    host = '0.0.0.0'
    port = 12345
    buffer_size = 4096

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)

        print("Server is listening...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")

                # Receive image size first
                image_size_data = conn.recv(buffer_size).decode()
                if image_size_data.startswith('SIZE'):
                    image_size = int(image_size_data.split()[1])
                    conn.sendall(b"GOT SIZE")
                else:
                    print("Did not receive image size information")
                    continue

                # Receive the actual image data
                image_data = b""
                while len(image_data) < image_size:
                    packet = conn.recv(buffer_size)
                    if not packet:
                        break
                    image_data += packet

                if len(image_data) != image_size:
                    print("Incomplete image data received")
                    continue

                # Send acknowledgment
                conn.sendall(b"GOT IMAGE")

                # Receive the text title
                text = conn.recv(buffer_size).decode()

                # Plot the image with the received text as the title
                process_image_and_text(image_data, text)

if __name__ == "__main__":
    start_server()
