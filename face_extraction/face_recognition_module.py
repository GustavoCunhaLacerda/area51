from PIL import Image
import mediapipe as mp
import torch
from blazeface import FaceExtractor, BlazeFace, VideoReader

class MyFaceExtractor():
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.facedet = BlazeFace().to(self.device)
        self.facedet.load_weights("./blazeface/blazeface.pth")
        self.facedet.load_anchors("./blazeface/anchors.npy")


    def _getImageFaces(self, source, _type='File'):
        face_extractor = FaceExtractor(facedet=self.facedet)

        if _type == 'File':
            img = Image.open(source)

        return face_extractor.process_image(img=img)['faces'][0]

    def _getVideoFaces(self, source, frames_per_video, _type='File'):
        videoreader = VideoReader(verbose=False)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=self.facedet)

        # if _type == 'File':
        #     video = face_extractor.process_video(source)
            
        return face_extractor.process_video(source)
