import socket
from PIL import Image, ImageDraw
import io
import os

def process_image_and_text(image_data, text):
    # Load the image
    image = Image.open(io.BytesIO(image_data))

    # Convert the image to a GIF with text overlay
    frames = []
    for i in range(5):  # Create 5 frames for the GIF
        frame = image.copy()
        draw = ImageDraw.Draw(frame)
        draw.text((10, 10), text, fill="white")  # Draw text on the image
        frames.append(frame)

    gif_path = "temp.gif"
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Duration for each frame in milliseconds
        loop=0
    )

    return gif_path

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

                # Process the image and text to create a GIF
                gif_path = process_image_and_text(image_data, text)

                # Send the GIF size
                gif_size = os.path.getsize(gif_path)
                conn.sendall(f"GIF_SIZE {gif_size}".encode())
                response = conn.recv(buffer_size).decode()

                if response != "GOT GIF SIZE":
                    print("Failed to get acknowledgment for GIF size")
                    continue

                # Send the GIF data
                with open(gif_path, 'rb') as gif_file:
                    gif_data = gif_file.read()
                    conn.sendall(gif_data)

                # Delete the GIF file after sending
                os.remove(gif_path)
                print("GIF sent and deleted from server.")

if __name__ == "__main__":
    start_server()
