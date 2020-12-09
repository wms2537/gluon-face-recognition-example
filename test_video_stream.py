from imutils.video import VideoStream
import imutils
import time
import cv2
import os

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import numpy as np
from tinydb import TinyDB, Query
import vptree

db = TinyDB('./d_single_image_ids.json')

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
sample_size = 2
factor = 0.02 * 512 # loosely crop face
output = "output.avi"
writer = None
ctx = [mx.gpu(0)]

# load face detection, face feature extraction and classification model
det_net = mx.gluon.nn.SymbolBlock.imports("./models/face_detection/center_net_resnet18_v1b_face_best-symbol.json", [
                                          'data'], "./models/face_detection/center_net_resnet18_v1b_face_best-0113.params", ctx=ctx)
features_net = mx.gluon.nn.SymbolBlock.imports(
    "./models/face_feature_extraction/mobilefacenet-symbol.json", ['data'], "./models/face_feature_extraction/mobilefacenet-0000.params", ctx=ctx)
mlp_net = mx.gluon.nn.SymbolBlock.imports(
    "./models/mlp-symbol.json", ['data'], "./models/mlp-0029.params", ctx=ctx)

face_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

vs = VideoStream(src=0).start()

enrolledIds = db.all()
synsets = enrolledIds[0]['labels']

while True:
    frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512))
    yr = frame.shape[0] / float(rgb.shape[0])
    xr = frame.shape[1] / float(rgb.shape[1])
    x = mx.nd.image.normalize(mx.nd.image.to_tensor(mx.nd.array(rgb)), mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225)).as_in_context(ctx[0]).expand_dims(0).astype('float16', copy=False)
    _, scores, bboxes = det_net(x)
    names = []

    for (s, (left, top, right, bottom)) in zip(scores.asnumpy()[0], bboxes.asnumpy()[0]):
        if s < 0.4:
            continue
        top = int(max(top-factor, 0) * yr)
        right = int(min(right+factor, 512) * xr)
        bottom = int(min(bottom+factor, 512) * yr)
        left = int(max(left-factor, 0) * xr)
        face_image = frame[top:bottom, left:right]

        color = (0, 0, 255)
        if face_image.shape[0] >= 128 and face_image.shape[1] >= 128:
            rgb = cv2.cvtColor(
                face_image, cv2.COLOR_BGR2RGB)
            face = mx.nd.array(rgb)
            encodings = []
        
            xface = face_transforms(face).expand_dims(0)
            y = features_net(xface.as_in_context(ctx[0]))

            predictions = mlp_net(y)[0]

            p = mx.nd.argmax(predictions)

            y = top - 15 if top - 15 > 15 else top + 15
            
            cv2.putText(frame, synsets[int(p.asscalar())] + '\n' + str(predictions[p].asscalar()), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        cv2.rectangle(frame, (left, top), (right, bottom),
                      color, 2)

    frame = imutils.resize(frame, width=1024)
    if writer is None and output is not "" or None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output, fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
