import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import cv2
import pywhatkit as kit
from collections import deque
import json
import requests
import smtplib
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

import pyjokes

webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"))

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

dict1 = {
    "Email to 'Name of person'" : "Email Address of that person"
}


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >= 4 and hour < 12:
        print("Good Morning Sir!!")
        speak("Good morning sir")
    elif hour >= 12 and hour <= 16:
        print("Good Afternoon Sir!!")
        speak("Good afternoon sir")
    elif hour >= 17 and hour <= 19:
        print("Good Evening Sir!!")
        speak("Good evening sir")
    else:
        print("Good Night Sir!!")
        speak("Good night sir")
    speak("I am your personal assistant. Please tell me how may i help you")


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-in")
        print(f"You said:- {query}")
    except Exception as e:
        print("Sorry! Can't recognize you! please say again!!")
        return "None"
    return query


def sendemail(to, content):
    with smtplib.SMTP('smtp.gmail.com', 587) as server:

        server.ehlo()
        server.starttls()
        server.login('Email address', 'password')
        server.sendmail('Email address', to, content)
        server.close


def jokes():
    my_joke = pyjokes.get_joke(language='en', category='neutral')
    print(my_joke)
    speak(my_joke)


def emotion():
    face_classifier = cv2.CascadeClassifier(
        r'C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\haarcascade_frontalface_default.xml')
    classifier = load_model(r'C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear',
                      'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                # print("\nprediction = ", preds)
                label = emotion_labels[preds.argmax()]
                # print("\nprediction max = ", preds.argmax())
                # print("\nlabel = ", label)
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # print("\n\n")
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def filter():

    cap = cv2.VideoCapture(0)

    cascade = cv2.CascadeClassifier(
        r'C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\face.xml')

    specs_ori = cv2.imread(
        r"C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\glass.png", -1)
    cigar_ori = cv2.imread(
        r"C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\cigar.png", -1)
    mus_ori = cv2.imread(
        r"C:\Users\PRATHAM UPADHYAY\Desktop\jrvs\mustache.png", -1)

    def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  # size of foreground image
        rows, cols, _ = src.shape  # size of background image
        y, x = pos[0], pos[1]

        for i in range(h):
            for j in range(w):
                if x + i > rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3]/255)  # read the alpha chanel
                src[x+i][y+j] = alpha * overlay[i][j][:3] + \
                    (1-alpha) * src[x+i][y+j]
        return src

    while cap.isOpened():
        result, frame = cap.read()
        if result:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, 1.3, 5, 0, minSize=(120, 120), maxSize=(350, 350))
            for (x, y, w, h) in faces:
                if h > 0 and w > 0:
                    glass_symin = int(y + 1.5 * h/5)
                    glass_symax = int(y + 2.5 * h / 5)
                    sh_glass = glass_symax - glass_symin

                    cigar_symin = int(y + 4 * h / 6)
                    cigar_symax = int(y + 5.5 * h / 6)
                    sh_cigar = cigar_symax - cigar_symin

                    mus_symin = int(y + 3.5 * h / 6)
                    mus_symax = int(y + 5 * h / 6)
                    sh_mus = mus_symax - mus_symin

                    face_glass_ori = frame[glass_symin:glass_symax, x:x+w]
                    cigar_glass_ori = frame[cigar_symin:cigar_symax, x:x + w]
                    mus_glass_ori = frame[mus_symin:mus_symax, x:x + w]

                    glass = cv2.resize(
                        specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
                    cigar = cv2.resize(
                        cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)
                    mus = cv2.resize(mus_ori, (w, sh_mus),
                                     interpolation=cv2.INTER_CUBIC)

                    transparentOverlay(face_glass_ori, glass)
                    # transparentOverlay(mus_glass_ori , mus)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def Canvas():
    def setValues(x):
        print("")

    # Creating the trackbars needed for
    # adjusting the marker colour These
    # trackbars will be used for setting
    # the upper and lower ranges of the
    # HSV required for particular colour
    cv2.namedWindow("Color detectors")
    cv2.createTrackbar("Upper Hue", "Color detectors",
                       153, 180, setValues)
    cv2.createTrackbar("Upper Saturation", "Color detectors",
                       255, 255, setValues)
    cv2.createTrackbar("Upper Value", "Color detectors",
                       255, 255, setValues)
    cv2.createTrackbar("Lower Hue", "Color detectors",
                       64, 180, setValues)
    cv2.createTrackbar("Lower Saturation", "Color detectors",
                       72, 255, setValues)
    cv2.createTrackbar("Lower Value", "Color detectors",
                       49, 255, setValues)

    # Giving different arrays to handle colour
    # points of different colour These arrays
    # will hold the points of a particular colour
    # in the array which will further be used
    # to draw on canvas
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    # These indexes will be used to mark position
    # of pointers in colour array
    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    # The kernel to be used for dilation purpose
    kernel = np.ones((5, 5), np.uint8)

    # The colours which will be used as ink for
    # the drawing purpose
    colors = [(255, 0, 0), (0, 255, 0),
              (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((471, 636, 3)) + 255

    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # Loading the default webcam of PC.
    cap = cv2.VideoCapture(0)

    # Keep looping
    while True:

        # Reading the frame from the camera
        _, frame = cap.read()

        # Flipping the frame to see same side of yours
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Getting the updated positions of the trackbar
        # and setting the HSV values
        u_hue = cv2.getTrackbarPos("Upper Hue",
                                   "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation",
                                          "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value",
                                     "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue",
                                   "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation",
                                          "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value",
                                     "Color detectors")
        Upper_hsv = np.array([u_hue, u_saturation, u_value])
        Lower_hsv = np.array([l_hue, l_saturation, l_value])

        # Adding the colour buttons to the live frame
        # for colour access
        frame = cv2.rectangle(frame, (40, 1), (140, 65),
                              (122, 122, 122), -1)
        frame = cv2.rectangle(frame, (160, 1), (255, 65),
                              colors[0], -1)
        frame = cv2.rectangle(frame, (275, 1), (370, 65),
                              colors[1], -1)
        frame = cv2.rectangle(frame, (390, 1), (485, 65),
                              colors[2], -1)
        frame = cv2.rectangle(frame, (505, 1), (600, 65),
                              colors[3], -1)

        cv2.putText(frame, "CLEAR ALL", (49, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "BLUE", (185, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "GREEN", (298, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "RED", (420, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "YELLOW", (520, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 2, cv2.LINE_AA)

        # Identifying the pointer by making its
        # mask
        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        Mask = cv2.erode(Mask, kernel, iterations=1)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
        Mask = cv2.dilate(Mask, kernel, iterations=1)

        # Find contours for the pointer after
        # idetifying it
        cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # Ifthe contours are formed
        if len(cnts) > 0:

            # sorting the contours to find biggest
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

            # Get the radius of the enclosing circle
            # around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Calculating the center of the detected contour
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            # Now checking if the user wants to click on
            # any button above the screen
            if center[1] <= 65:

                # Clear Button
                if 40 <= center[0] <= 140:
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

        # Append the next deques when nothing is
        # detected to avois messing up
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw lines of all the colors on the
        # canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):

            for j in range(len(points[i])):

                for k in range(1, len(points[i][j])):

                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue

                    cv2.line(frame, points[i][j][k - 1],
                             points[i][j][k], colors[i], 2)
                    cv2.line(
                        paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Show all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)
        cv2.imshow("mask", Mask)

        # If the 'q' key is pressed then stop the application
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and all resources
    cap.release()
    cv2.destroyAllWindows()


def news():
    url = "https://newsapi.org/v2/top-headlines?country=in&apiKey=APIKEY"
    news = requests.get(url).text
    news_json = json.loads(news)
    arts = news_json["articles"]
    for article in arts:
        
        speak(article['title'])
        speak("Next news ")
        
        


if __name__ == "__main__":
    wishme()
    while True:
        query = takeCommand().lower()

        if 'search' in query:
            query = query.replace("search", "")
            speak(f"searching {query}")
            webbrowser.get('chrome').open(query)
        elif 'wikipedia' in query:
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to wikipedia")
            print(results)
            speak(results)
        elif 'play' in query:
            query = query.replace("play", "")
            speak("Opening youtube")

            yt = "https://www.youtube.com/results?search_query={}"
            search_url = yt.format(query)
            webbrowser.get('chrome').open(search_url)
        elif 'the time' in query:
            strtime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir the current time is {strtime}")

        elif 'open notepad' in query:
            codePath = "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Accessories"
            os.startfile(codePath)

        elif 'open pycharm' in query:
            codePath = "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\JetBrains"
            os.startfile(codePath)
        elif 'open code' in query:
            codePath = "C:\\Users\\PRATHAM UPADHYAY\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Visual Studio Code"
            os.startfile(codePath)
        elif 'movies' in query:
            codePath = "C:\\Users\\PRATHAM UPADHYAY\\Desktop\\Movies"
            os.startfile(codePath)

        elif 'email to' in query:
            # query = query.replace("email to", "")
            if(query in dict1):
                try:
                    speak("what should i say")
                    content = takeCommand()
                    to = dict1[query]
                    print(to)
                    sendemail(to, content)
                    speak("Email has been sent")

                except Exception as e:
                    print(e)
                    speak(
                        "Sorry sir. I can't able to send that email. Please try again.")
            else:
                speak(f"Sorry there is no one name {query} in your contact.")

        elif 'detect emotion' in query:
            emotion()

        elif 'invisible' in query:
            video = cv2.VideoCapture(0)
            background = 0
            for i in range(30):
                ret, background = video.read()
            background = np.flip(background, axis=1)
            while True:
                ret, img = video.read()
                img = np.flip(img, axis=1)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                blur = cv2.GaussianBlur(hsv, (35, 35), 0)

                lower = np.array([0, 120, 70])
                upper = np.array([180, 255, 255])
                mask01 = cv2.inRange(hsv, lower, upper)

                lower_red = np.array([336, 35, 100])
                upper_red = np.array([0, 100, 25])

                mask02 = cv2.inRange(hsv, lower_red, upper_red)

                mask = mask01+mask02

                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

                img[np.where(mask == 255)] = background[np.where(mask == 255)]

                cv2.imshow("Display", img)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    break
            video.release()
            cv2.destroyAllWindows()

        elif 'filter' in query:

            filter()

        elif 'joke' in query:
            jokes()
        elif 'youtube' in query:
            query = query.replace("youtube", "")
            kit.playonyt(query)
        
        elif 'paint' in query:
            Canvas()
        
        elif 'news' in  query:
            speak("News for today is..")
            news()

        elif 'quit' in query:
            speak("Signing off")
            exit()
