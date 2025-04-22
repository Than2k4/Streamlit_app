import io
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER

class Inference:
    def __init__(self, model=None):
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = YOLO(model) if model else None

    def web_ui(self):
        st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        st.title("Ultralytics YOLO Streamlit Application 🚀")
        st.subheader("Real-time Object Detection on Webcam or Video Files")

    def sidebar(self):
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg", width=250)
            st.title("User Configuration")
            self.source = st.selectbox("Video Source", ("webcam", "video"))
            self.enable_trk = st.radio("Enable Tracking", ("Yes", "No"))
            self.conf = st.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
            self.iou = st.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01)

        col1, col2 = st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Handles video file uploads and webcam input."""
        self.vid_file_name = None

        if self.source == "video":
            vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file:
                g = io.BytesIO(vid_file.read())
                with open("uploaded_video.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "uploaded_video.mp4"
        elif self.source == "webcam":
            img_file = st.camera_input("Start Webcam")
            if img_file:
                bytes_data = img_file.getvalue()
                self.vid_file_name = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    def configure(self):
        """Configures the model and selects classes for inference."""
        if self.model:
            class_names = list(self.model.names.values())
            selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
            self.selected_ind = [class_names.index(option) for option in selected_classes]

    def inference(self):
        """Performs object detection inference."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if st.sidebar.button("Start"):
            stop_button = st.button("Stop")

            # Check if we're processing video or webcam input
            if isinstance(self.vid_file_name, str):  # Process video file
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    st.error("Could not open video file.")
                    return

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        st.warning("Failed to read frame.")
                        break

                    # Inference on video frames
                    results = self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind) if self.enable_trk == "Yes" else self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    annotated_frame = results[0].plot()

                    if stop_button:
                        cap.release()
                        st.stop()

                    self.org_frame.image(frame, channels="BGR")
                    self.ann_frame.image(annotated_frame, channels="BGR")

                cap.release()
                cv2.destroyAllWindows()

            elif isinstance(self.vid_file_name, np.ndarray):  # Process webcam input
                frame = self.vid_file_name
                results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()

                self.org_frame.image(frame, channels="BGR")
                self.ann_frame.image(annotated_frame, channels="BGR")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model_path).inference()
