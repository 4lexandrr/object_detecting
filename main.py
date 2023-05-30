import cv2


cap = cv2.VideoCapture('Highway Free footage.mp4')

object_detecror = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)

while True:
    ret, frame = cap.read()
    
    height, width, _ = frame.shape

    # Region of interesting
    roi = frame[400: 720, 400: 900]


    mask = object_detecror.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        object_area = cv2.contourArea(cnt)

        if object_area > 700:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0))
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('CarDetecting', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()