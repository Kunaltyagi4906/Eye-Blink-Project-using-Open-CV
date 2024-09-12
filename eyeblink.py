import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize FaceMesh detector
detector = FaceMeshDetector(maxFaces=1)

# Initialize plot
plotY = LivePlot(640, 360, [20, 50], invert=True)

# Define eye landmarks
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

# Initialize variables
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:
    # Read frame
    success, img = cap.read()

    # Check for frame read success
    if not success:
        print("Cannot read frame")
        break

    # Find face mesh
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # Get face landmarks
        face = faces[0]

        # Draw eye landmarks
        for kunal in idList:
            cv2.circle(img, face[kunal], 5, color, cv2.FILLED)

        # Calculate eye aspect ratio
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)

        # Check for blink
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)
        if ratioAvg < 30 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 9:
                counter = 0
                color = (255, 0, 255)

        # Display blink count
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        # Update plot
        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        # Handle no face detection
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    # Display output
    cv2.imshow("Image", imgStack)

    # Exit on key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
