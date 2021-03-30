from imageai.Detection import VideoObjectDetection
import os
import numpy
import cv2
import json

def forFrame(frame_number,output_array, output_count):
    print("\nFrame Number :",frame_number)
    print("\nArray for the outputs of the frame \n", output_array)
    print([d['box_points'] for d in output_array])
    file=open("file.txt","w")
    content=str([d['box_points'] for d in output_array])
    file.write(str(frame_number)+content+"\n")
    file.close()
    print("\nArray for output count for unique objects in this frame : ", output_count)
    print("------------END OF A FRAME --------------")
    cap = cv2.VideoCapture("traffic-mini.mp4")
    myFrameNumber = frame_number
    c=myFrameNumber

# get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# check for valid frame number
    if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

    while True:
        ret, img = cap.read()
        if (type(img) == type(None)):
            break
        cv2.imshow("frame",img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == False:
    #         break
    #     cv2.imwrite('frame'+str(i)+'.jpg',frame)
    #     i+=1
        f = open("file.txt")
        contents = f.read().split(']]')
        f.close()

        for content in contents:
            coordinate_string = (content+']]')

            if coordinate_string.find('[[') == -1:
                break

            coordinate_string = coordinate_string[coordinate_string.find('[['):coordinate_string.find(']]')+2]
            coordinate = json.loads(coordinate_string)
            i=0
            for cor in coordinate:
                crop_img = img[cor[1]:cor[3], cor[0]:cor[2]]
                cv2.imwrite(os.path.join("C:/Users/Joel Joseph/Desktop/Object detection/frames",'obj'+str(c)+'.'+str(i)+'.jpg'),crop_img)
                i+=1
        cap.release()

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "traffic-mini.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_mini_detected_1.mp4"),save_detected_video=True,frames_per_second=1,per_frame_function=forFrame, log_progress=True,minimum_percentage_probability=80)
print(video_path)


cv2.destroyAllWindows()
