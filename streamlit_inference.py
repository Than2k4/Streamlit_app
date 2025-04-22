# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any, List

import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    WebRtcMode,
)

# Cấu hình STUN server để WebRTC hoạt động tốt trên môi trường deploy
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class YOLOTransformer(VideoTransformerBase):
    """Xử lý khung hình từ webcam client"""

    def __init__(self, model: YOLO, conf: float, iou: float, classes: List[int], enable_trk: bool):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.enable_trk = enable_trk

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.enable_trk:
            results = self.model.track(
                img, conf=self.conf, iou=self.iou, classes=self.classes, persist=True
            )
        else:
            results = self.model(img, conf=self.conf, iou=self.iou, classes=self.classes)
        return results[0].plot()


class VideoFileTransformer(VideoTransformerBase):
    """Xử lý video file đã upload"""

    def __init__(self, path: str, model: YOLO, conf: float, iou: float, classes: List[int], enable_trk: bool):
        self.cap = cv2.VideoCapture(path)
        self.model = model
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.enable_trk = enable_trk

    def transform(self, frame):
        success, img = self.cap.read()
        if not success:
            return frame
        if self.enable_trk:
            results = self.model.track(
                img, conf=self.conf, iou=self.iou, classes=self.classes, persist=True
            )
        else:
            results = self.model(img, conf=self.conf, iou=self.iou, classes=self.classes)
        return results[0].plot()


class Inference:
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict["model"]
        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

        # Giá trị mặc định
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.selected_ind: List[int] = []
        self.model: YOLO

    def web_ui(self):
        st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        st.markdown("<style>MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='color:#FF64DA;text-align:center;'>Ultralytics YOLO Streamlit</h1>",
            unsafe_allow_html=True,
        )

    def sidebar(self) -> str:
        with st.sidebar:
            st.image(
                "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg",
                width=200,
            )
            src = st.selectbox("Source", ["webcam", "video"])
            self.enable_trk = st.radio("Enable Tracking", ["Yes", "No"]) == "Yes"
            self.conf = st.slider("Confidence", 0.0, 1.0, self.conf, 0.01)
            self.iou = st.slider("IoU", 0.0, 1.0, self.iou, 0.01)

            # Chọn model
            models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
            if self.model_path:
                models.insert(0, self.model_path.split(".pt")[0])
            chosen = st.selectbox("Model", models)
            with st.spinner("Loading model..."):
                self.model = YOLO(f"{chosen.lower()}.pt")
            st.success("Model loaded!")

            # Chọn classes
            names = list(self.model.names.values())
            picked = st.multiselect("Classes", names, default=names[:3])
            self.selected_ind = [names.index(c) for c in picked]

        return src

    def run(self):
        self.web_ui()
        src = self.sidebar()

        if src == "video":
            vid = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
            if vid:
                path = "uploaded_video.mp4"
                with open(path, "wb") as f:
                    f.write(vid.read())
                webrtc_streamer(
                    key="video",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: VideoFileTransformer(
                        path=path,
                        model=self.model,
                        conf=self.conf,
                        iou=self.iou,
                        classes=self.selected_ind,
                        enable_trk=self.enable_trk,
                    ),
                )
        else:
            webrtc_streamer(
                key="webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: YOLOTransformer(
                    model=self.model,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.selected_ind,
                    enable_trk=self.enable_trk,
                ),
            )


if __name__ == "__main__":
    import sys

    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model_arg).run()
