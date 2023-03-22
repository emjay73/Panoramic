import cv2
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def extract_feature(img, dump_path):
    # reference feature extraction
    # https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des  = sift.detectAndCompute(gray,None)
    img_kp  = cv2.drawKeypoints(gray, kp, gray)
    cv2.imwrite(dump_path, img_kp)    
    return kp, des

def match_features(kp1, kp2, des1, des2, dump_path=None, img1=None, img2=None):
    # feature matching
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # https://deep-learning-study.tistory.com/260
    matches = bf.knnMatch(des1, des2, k=2, compactResult=True)
    
    if (dump_path is not None) and (img1 is not None) and (img2 is not None):
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv.drawMatchesKnn expects list of lists as matches.
        # https://bkshin.tistory.com/entry/OpenCV-28-%ED%8A%B9%EC%A7%95-%EB%A7%A4%EC%B9%ADFeature-Matching
        img_matching = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(dump_path, img_matching)

    return matches