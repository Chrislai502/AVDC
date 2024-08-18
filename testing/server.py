import socket
from PIL import Image, ImageDraw
import io
import os
from flowdiffusion.infer import InferAVDC

class_dir = os.path.dirname(os.path.abspath(__file__))

class AVDCServer:
    def __init__(self, host='0.0.0.0', port=12345, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.model = InferAVDC(checkpoint_num=188000)  # Initialize the model

    def process_image_and_text(self, image_data, text):
        """Process the received image and text to generate a GIF."""
        # Load the image
        image = Image.open(io.BytesIO(image_data))
        image = self.model.preprocess_image(image)
        cache_dir = os.path.join(class_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        gif_path = self.model.generate(image, text, output_gif_path=os.path.join(cache_dir, "temp.gif"))
        # # Convert the image to a GIF with text overlay
        # frames = []
        # for i in range(5):  # Create 5 frames for the GIF
        #     frame = image.copy()
        #     draw = ImageDraw.Draw(frame)
        #     draw.text((10, 10), text, fill="white")  # Draw text on the image
        #     frames.append(frame)

        # gif_path = "temp.gif"
        # frames[0].save(
        #     gif_path,
        #     format="GIF",
        #     save_all=True,
        #     append_images=frames[1:],
        #     duration=200,  # Duration for each frame in milliseconds
        #     loop=0
        # )

        return gif_path

    def handle_client(self, conn, addr):
        """Handle communication with a connected client."""
        print(f"Connected by {addr}")

        try:
            # Receive image size first
            image_size_data = conn.recv(self.buffer_size).decode()
            if image_size_data.startswith('SIZE'):
                image_size = int(image_size_data.split()[1])
                conn.sendall(b"GOT SIZE")
            else:
                print("Did not receive image size information")
                return

            # Receive the actual image data
            image_data = b""
            while len(image_data) < image_size:
                packet = conn.recv(self.buffer_size)
                if not packet:
                    break
                image_data += packet

            if len(image_data) != image_size:
                print("Incomplete image data received")
                return

            # Send acknowledgment
            conn.sendall(b"GOT IMAGE")

            # Receive the text title
            text = conn.recv(self.buffer_size).decode()

            # Process the image and text to create a GIF
            gif_path = self.process_image_and_text(image_data, text)

            # Send the GIF size
            gif_size = os.path.getsize(gif_path)
            conn.sendall(f"GIF_SIZE {gif_size}".encode())
            response = conn.recv(self.buffer_size).decode()

            if response != "GOT GIF SIZE":
                print("Failed to get acknowledgment for GIF size")
                return

            # Send the GIF data
            with open(gif_path, 'rb') as gif_file:
                gif_data = gif_file.read()
                conn.sendall(gif_data)

            # Delete the GIF file after sending
            os.remove(gif_path)
            print("GIF sent and deleted from server.")
        finally:
            conn.close()

    def start(self):
        """Start the server to listen for incoming connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(5)

            print(f"Server is listening on {self.host}:{self.port}...")

            while True:
                conn, addr = s.accept()
                self.handle_client(conn, addr)

if __name__ == "__main__":
    server = AVDCServer()
    server.start()
