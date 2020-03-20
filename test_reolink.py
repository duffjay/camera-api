
from PIL import Image
import io
import numpy
import cv2
import time
import base64
import urllib.request


cv2.namedWindow('Reolink', cv2.WINDOW_NORMAL)

# works
url = 'http://192.168.1.122/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=KL94LT46X6&user=admin&password=sT1nkeye'

data = {
    'username' : 'admin',
    'password' : 'sT1nkeye'
}

error_count = 0
success_count = 0
for i in range(2500):
    start_time = time.time()
    print (start_time)

    try:
        reader = urllib.request.urlopen(url)
        print (i, "success", reader.status, time.time() - start_time)
        success_count = success_count + 1
        img_bytes = bytearray(reader.read())
        img_array = Image.open(io.BytesIO(img_bytes))
        img_numpy = numpy.array(img_array)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        print (img_bgr.shape)
        cv2.imshow('Reolink', img_bgr)
    except:
        print (i, "error")
        error_count = error_count + 1

    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



