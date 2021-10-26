import face_recognition_module as frm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def image_tests():
  source = './data/img/img_03.jpg'

  extractor = frm.MyFaceExtractor()
  face = extractor._getImageFaces(source)

  print(face)

  # cv2.imshow("Face", face)
  # cv2.waitKey(0)
  # plt.imshow()
  fig,ax = plt.subplots(1,2,figsize=(8,4))

  ax[0].imshow(Image.open(source))
  ax[0].set_title('Imagem')

  ax[1].imshow(face)
  ax[1].set_title('Rosto');

  plt.show()

def video_tests():
  source = './data/video/video_04.mp4'
  extractor = frm.MyFaceExtractor()
  faces = extractor._getVideoFaces(source, 60)

  fig = plt.figure(figsize=(8, 8))
  columns = 5
  rows = 5
  for i in range(1, columns*rows +1):
      img = faces[i]['faces'][0]
      fig.add_subplot(rows, columns, i)
      plt.imshow(img)
  plt.show()


if __name__=="__main__":
  image_tests()
  # video_tests()