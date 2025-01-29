from foodscore.data.datasearch import get_data, convert_test_image,get_nutritions
from foodscore.model.modelcreation import create_model, fit_model,load_model, predict_label
from foodscore import params
## Get datasets
#train_ds, val_ds, test_ds = get_data(path = params.LOCAL_PATH, validation_split = 0.2, img_height = params.IMG_HEIGHT, img_width = params.IMG_WIDTH, bs = params.BATCH_SIZE)

## Create model
#model = create_model(input_shape = params.INPUT_SHAPE)

## Train model
#fit_history = fit_model(model, train_ds, val_ds)

## Load model
saved_model=load_model(path = params.MODEL_PATH)

## Convert test picture
image = convert_test_image(path = params.IMG_PATH, img_height = params.IMG_HEIGHT, img_width = params.IMG_WIDTH)

## Predict label
predicted_class = predict_label(saved_model,image, most_prob=5)

## Get nutritions
nutritions = get_nutritions(predicted_class)
print(nutritions)
