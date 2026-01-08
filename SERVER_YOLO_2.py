import socket
import cv2
import numpy as np
import time

# YOLO setup
YOLO_CFG = r"C:\Users\Kostas Dimitropoulos\Desktop\yolo_v4-tiny\FINAL_DATASET\yolov4-tiny.cfg"
YOLO_WEIGHTS = r"C:\Users\Kostas Dimitropoulos\Desktop\yolo_v4-tiny\FINAL_DATASET\yolov4-tiny_last.weights"
OBJ_NAMES = r"C:\Users\Kostas Dimitropoulos\Desktop\yolo_v4-tiny\FINAL_DATASET\obj.names"
SAVE_PATH = r"C:\Users\Kostas Dimitropoulos\Desktop\yolo_v4-tiny\detectedimages"

CONF_THRESHOLD = 0.25
NMS_THRESHOLD  = 0.3

net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open(OBJ_NAMES, "r") as f:
    class_names = f.read().strip().split("\n")

# Server setup
HOST = "192.168.0.117"
PORT = 5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Waiting for connection...")

client_socket, addr = server_socket.accept()
print(f"Connected to {addr}")

try:
    while True:
        size_data = client_socket.recv(4)
        if not size_data:
            break

        size = int.from_bytes(size_data, byteorder='big')

        data = b""
        while len(data) < size:
            packet = client_socket.recv(size - len(data))
            if not packet:
                break
            data += packet

        if not data:
            continue

        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue

        output_img = img.copy()
        height, width = img.shape[:2]

        # YOLO forward
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Collect detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESHOLD:
                    center_x, center_y, w, h = (detection[:4] *
                        np.array([width, height, width, height])).astype(int)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        detections = []

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                detected_object = class_names[class_id]

                detections.append(detected_object)

                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{detected_object} ({confidence:.2f})"
                cv2.putText(output_img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Detected: {detected_object} at ({x},{y}) [{confidence:.2f}]")

        # Save image if detections exist
        if detections:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{SAVE_PATH}/detected_{timestamp}.jpg"
            cv2.imwrite(filename, output_img)

        # Send result back to NAO
        result_message = ", ".join(sorted(set(detections))) if detections else "No objects detected."
        client_socket.sendall(result_message.encode())

except Exception as e:
    print(f"Error: {e}")

finally:
    client_socket.close()
    server_socket.close()
