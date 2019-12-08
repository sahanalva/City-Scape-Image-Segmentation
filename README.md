# City Scape Image Segmentation

## Requirements:
* Please Install the following packages to run the code:
 * tensorflow==1.14.0
 * numpy==1.17.2
 * open>=3.4.2
* requirements.txt(Created using pip freeze) file included in Project run "pip install -r requirements.txt" to install packages using pip
* The code was setup in Python3, but it should still work in Python2 with minor modifications.

## Running Code
* First download CityScapes Coarse segmentation [data](https://www.cityscapes-dataset.com/downloads/)
* Unzip data and edit folder paths in the data_preparation.ipynb Jupyter notebook.
* Run all cells in data_preparation.ipynb to create train, val and test data for our models.
  * We can also modify the frame size over here.
* Now run the cells in ImageSegmentation.ipynb file to train a model.
  * The different model configurations can be found in the models.py file.
  * By default the ResNet50 Encoder - Unet Decoder model is trained, will need to modify ImageSegmentaiton.ipynb to change the model.
 
## Metrics Used:
 * Pixel Accuracy
 * IoU
 
# Results:
Experiments | Acc. | IoU
---------|------|--------
SegNet | 0.8361 |0.6572
UNet | 0.8499 | 0 .6069
SegNet-Skip Connect | 0.8292 | 0.6072
SegNet-Skip Connect Concat| 0.8382 | 0.6363
UNet-Skip Connect | 0.8474 | 0.5303
UNet-Skip Connect Concat| 0.8492 | 0.5714
ResNet50 Encoder-UNet Decoder | 0.8674 | 0.7232

## Example Segmentation:

![Segmentation Using ResNet50 - Encoder - UNet Decoder](https://user-images.githubusercontent.com/7547054/70396877-839aac80-19d2-11ea-8ad6-150350d8df07.png)

## ResNet50 Encoder - UNet Decoder Architecture:

<img width="1180" alt="Screenshot 2019-12-02 at 8 01 18 PM" src="https://user-images.githubusercontent.com/7547054/70397073-92825e80-19d4-11ea-9ce2-8d31c7fc481e.png">

## Project Members:
* Abraham Gerard Sebastian([abgese](https://github.com/abgese))
* Sahan Suresh Alva([sahanalva](https://github.com/sahanalva))
