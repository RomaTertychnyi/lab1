import cv2, os, time

def kaze_match(images_path, good_image_path, pref, f):
    images = []
    for file in os.listdir(images_path):
        if file.endswith(".jpg"):
            images.append(cv2.imread(images_path+file))

    gray_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray);

    good_image = cv2.imread(good_image_path)
    good_gray_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.BRISK_create()

    results = []
    for image in gray_images:
        (kps, desc) = detector.detectAndCompute(image, None)
        results.append((kps,desc,image))

    (kps_good, descs_good) = detector.detectAndCompute(good_gray_image, None)

    i = 0
    for (kps, desc, image) in results:
        f.write(pref+" Image "+str(i)+"\n")
        f.write("keypoints: {}, descriptors: {}".format(len(kps), desc.shape)+"\n")
        print("keypoints: {}, descriptors: {}".format(len(kps), desc.shape))

        start_time = time.time()

        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(desc,descs_good,k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(image, kps, good_gray_image, kps_good, good[:10], None, flags=2)
        cv2.imwrite("../resultsBrisk/"+pref+"res"+str(i)+".jpg", img3)
        f.write("Time: {}\n".format(time.time()-start_time))
        i = i + 1

def main():
    good_image_dir = "../good Roma/"
    bad_image_dir = "../bad Roma/"
    good_image = "../good Roma/IMG_20181106_150647.jpg"

    f = open("resultsBrisk.txt", "w")
    f.write("Results:\n")
    f.close()
    f = open("resultsBrisk.txt", "a")

    kaze_match(good_image_dir,good_image,"good",f)
    #kaze_match(bad_image_dir,good_image,"bad",f)

main()