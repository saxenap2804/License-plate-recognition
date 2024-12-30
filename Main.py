import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import Preprocess
import PossiblePlate
import json

# Module level variables
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
plate = [""]

def main(img):
    # Train KNN model
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
    if not blnKNNTrainingSuccessful:
        print("\nError: KNN training was not successful\n")
        return

    # Load and verify image
    imgOriginalScene = cv2.imread(img)
    if imgOriginalScene is None:
        print("\nError: Image not read from file\n")
        return
    
    imgOriginalScene = resizeImage(imgOriginalScene, (640, 480))
    # Preprocess the image
    imgGrayscale, imgThresh, vehicleColor = Preprocess.preprocess(imgOriginalScene)

    # Detect plates and characters
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    cv2.imshow("imgOriginalScene", imgOriginalScene)

    data={
        "image_path": img,
        "license_plate": None,
        "detection_success": False,
        "vehicle_color": vehicleColor.tolist()
    }
    if len(listOfPossiblePlates) == 0:
        print("\nNo license plates were detected\n")
    else:
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("\nNo characters were detected\n")
            return

        # Draw red rectangle around the plate
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
        plate[0] = licPlate.strChars
        print("\nLicense plate read from image = " + licPlate.strChars + "\n")
        print("----------------------------------------")

        # Write license plate characters on the image
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        data["license_plate"]=licPlate.strChars
        data["detection_success"]=True
        # Display and save the final output
        standard_width, standard_height = 615, 480
        imgOriginalScene = cv2.resize(imgOriginalScene, (standard_width, standard_height))
        cv2.imshow("imgOriginalScene", imgOriginalScene)
        
        # Save as output.jpg
        if cv2.imwrite("output.jpg", imgOriginalScene):
            print("Image successfully saved as output.jpg")
        else:
            print("Failed to save the image.")
    json_file_path = "output_data.json"
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Detection data saved to {json_file_path}")
    cv2.waitKey(0)

def getLicensePlate():
    return plate[0]

def resizeImage(img, size):
    return cv2.resize(img, size)

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    # Get 4 vertices of rotated rect, and convert points to integers
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    p2fRectPoints = np.int32(p2fRectPoints)  # Convert to integer coordinates

    # Draw the 4 red lines around the plate
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)
    (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)
    ptCenterOfTextAreaX = intPlateCenterX

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python Main.py <image_path>")
