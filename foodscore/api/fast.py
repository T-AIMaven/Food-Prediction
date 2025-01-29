from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from foodscore.data.datasearch import convert_test_image, get_nutritions
from foodscore.model.modelcreation import predict_label, load_model


import numpy as np
import cv2
import io

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = load_model()

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):

    ### Receiving and decoding the image
    contents = await img.read()
    # converting string format image to nparray
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ##image reshape
    converted_image = convert_test_image(cv2_img)

    ##predicts label
    predicted_label = predict_label(model, converted_image)


    ##get nutrition
    nutritions = get_nutritions(predicted_label)
    return nutritions
