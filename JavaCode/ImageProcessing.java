package com.dji.RDP.JavaCode;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.opencv.imgproc.Imgproc.MORPH_CLOSE;
import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;

public class ImageProcessing {
    private Mat src;
    private Mat dirt;
    private Mat thresh;
    public Mat maskRed, maskBlue;
    public Bitmap bm;
    private boolean isGray;
    private Context context;

    /**
     * Initializes class variables
     *
     * @param context In order to be able to debug using Toast
     * @param path    reads image from path
     */
    public ImageProcessing(Context context, String path) {
        //This has to be the first line before using any OpenCV functions.
        OpenCVLoader.initDebug();
        this.context = context;
        //Trying to read image from path to bitmap, and then from bitmap to Mat
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            Bitmap imageFromPath = BitmapFactory.decodeFile(path, options);
            src = new Mat(imageFromPath.getHeight(), imageFromPath.getWidth(), CvType.CV_8UC3);
            Utils.bitmapToMat(imageFromPath, src);
            //Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2RGB);
        } catch (Exception e) {
            e.printStackTrace();
        }
        dirt = new Mat();
        thresh = new Mat();
        maskRed = new Mat();
        maskBlue = new Mat();
        isGray = false;
        //Check before update, otherwise crash
        if (src.cols() > 0)
            updateBitmap(src);
    }

    /**
     * This is the main function used for ImageProcessing. Here we threshold the colors we need to use from an HSV image,
     * we clean the binary images, and then we try to find in each of the images circles, that may be the Nodes of our graph.
     *
     * @return We return a matrix of points, that represent the ordered nodes of the graph. Each point has x,y and color of Node.
     */
    public MyPoint[][] colorThreshold() {
        Mat raw = new Mat();
        Mat hsv = new Mat();
        Utils.bitmapToMat(bm, raw);
        //In order to threshold, we need to work with an HSV image
        Imgproc.cvtColor(raw, hsv, Imgproc.COLOR_BGR2HSV);
        Mat mask1 = new Mat(), mask2 = new Mat();
        Mat maskGreen = new Mat();
        //=========================RED THRESHOLD===========================================
        double H_MIN_UPPER_RED = 166, S_MIN_UPPER_RED = 95, V_MIN_UPPER_RED = 91, H_MAX_UPPER_RED = 180, S_MAX_UPPER_RED = 215, V_MAX_UPPER_RED = 255;
        double H_MIN_LOWER_RED = 0, S_MIN_LOWER_RED = 70, V_MIN_LOWER_RED = 100, H_MAX_LOWER_RED = 12, S_MAX_LOWER_RED = 230, V_MAX_LOWER_RED = 255;
        Core.inRange(hsv, new Scalar(H_MIN_UPPER_RED, S_MIN_UPPER_RED, V_MIN_UPPER_RED), new Scalar(H_MAX_UPPER_RED, S_MAX_UPPER_RED, V_MAX_UPPER_RED), mask1);
        Core.inRange(hsv, new Scalar(H_MIN_LOWER_RED, S_MIN_LOWER_RED, V_MIN_LOWER_RED), new Scalar(H_MAX_LOWER_RED, S_MAX_LOWER_RED, V_MAX_LOWER_RED), mask2);
        //Combine binary images to one image.
        Core.bitwise_or(mask1, mask2, maskRed);
        //Clean binary image, remove small irrelevant noises.
        Imgproc.GaussianBlur(maskRed, maskRed, new Size(3, 3), 0);
        Imgproc.morphologyEx(maskRed, maskRed, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        //Detect circles from binary image.
        Point[][] redCircles = circles(maskRed);
        //=========================RED THRESHOLD===========================================

        //=========================GREEN THRESHOLD=========================================
        double H_MIN_GREEN = 55, S_MIN_GREEN = 170, V_MIN_GREEN = 70, H_MAX_GREEN = 87, S_MAX_GREEN = 255, V_MAX_GREEN = 255;
        Core.inRange(hsv, new Scalar(H_MIN_GREEN, S_MIN_GREEN, V_MIN_GREEN), new Scalar(H_MAX_GREEN, S_MAX_GREEN, V_MAX_GREEN), maskGreen);
        Imgproc.GaussianBlur(maskGreen, maskGreen, new Size(3, 3), 0);
        Imgproc.morphologyEx(maskGreen, maskGreen, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        Point[][] greenCircles = circles(maskGreen);
        //=========================GREEN THRESHOLD=========================================

        //=========================BLUE THRESHOLD==========================================
        double H_MIN_BLUE = 90, S_MIN_BLUE = 113, V_MIN_BLUE = 111, H_MAX_BLUE = 100, S_MAX_BLUE = 255, V_MAX_BLUE = 255;
        Core.inRange(hsv, new Scalar(H_MIN_BLUE, S_MIN_BLUE, V_MIN_BLUE), new Scalar(H_MAX_BLUE, S_MAX_BLUE, V_MAX_BLUE), maskBlue);
        Imgproc.GaussianBlur(maskBlue, maskBlue, new Size(3, 3), 0);
        Imgproc.morphologyEx(maskBlue, maskBlue, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        Point[][] blueCircles = circles(maskBlue);
        //=========================BLUE THRESHOLD==========================================

        //Combine the binary images to one image.
        Core.bitwise_or(maskRed, maskBlue, thresh);
        Core.bitwise_or(maskGreen, thresh, thresh);

        //Join the red green and blue detected circles to an array, each circle with its location and color.
        ArrayList<MyPoint> points = new ArrayList<>();
        for (Point[] row : redCircles) {
            for (Point redC : row) {
                points.add(new MyPoint(redC.x, redC.y, Color.RED));
            }
        }
        for (Point[] row : greenCircles) {
            for (Point greenC : row) {
                points.add(new MyPoint(greenC.x, greenC.y, Color.GREEN));
            }
        }
        for (Point[] row : blueCircles) {
            for (Point blueC : row) {
                points.add(new MyPoint(blueC.x, blueC.y, Color.BLUE));
            }
        }
        //Convert the list of circles to a 2D ordered matrix, from down up, left to right.
        MyPoint[][] centersOrdered = orderCirclesToMatrix(orderCircles(points.toArray(new MyPoint[points.size()])));

        //===============================FOR DEBUGGING PURPOSES=================================================
        String ss = "";
        for (int i = 0; i < centersOrdered.length; i++) {
            for (int j = 0; j < centersOrdered[i].length; j++) {
                ss += "Circle " + (i * centersOrdered.length + j) + " (" + i + ", " + j + ") " + ": " + centersOrdered[i][j].x + ", " + centersOrdered[i][j].y + " is " + centersOrdered[i][j].color.name() + "\n";
            }
        }
        Toast.makeText(context, ss, Toast.LENGTH_LONG).show();
        //===============================FOR DEBUGGING PURPOSES=================================================
        updateBitmap(thresh);
        return centersOrdered;
    }

    /**
     * We detect the possible circles from a binary image, and return them in a matrix, ordered down up, left to right
     *
     * @param binaryImage the binary image from where we search for circles
     * @return A 2D ordered matrix of the centers of the ordered circles.
     */
    public Point[][] circles(Mat binaryImage) {
        Mat circles = new Mat(), currBinImg = binaryImage != null ? binaryImage : thresh;
        //Imgproc.HoughCircles(thresh, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 30, 100, 70, 1, 80);
        //Imgproc.HoughCircles(thresh, circles, Imgproc.CV_HOUGH_GRADIENT, 1, 95, 100, 10, 50, 50);

        //We used to use different circles before, now we use big paper ones, made by us.
        boolean smallStickers = false;
        //Remove small irrelevant noises.
        Imgproc.morphologyEx(currBinImg, currBinImg, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(13, 13)));
        Imgproc.GaussianBlur(currBinImg, currBinImg, new Size(5, 5), 0);
        Imgproc.morphologyEx(currBinImg, currBinImg, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7)));

        /**
         * minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
         *
         * param1: Gradient value used to handle edge detection in the Yuen et al. method.
         *
         * param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected (including false circles). The larger the threshold is, the more circles will potentially be returned.
         *
         * minRadius: Minimum size of the radius (in pixels).
         *
         * maxRadius: Maximum size of the radius (in pixels).
         */
        if (smallStickers)
            Imgproc.HoughCircles(currBinImg, circles, Imgproc.CV_HOUGH_GRADIENT, 1, 50, 100, 10, 15, 30);
        else//Big red paper circles
            Imgproc.HoughCircles(currBinImg, circles, Imgproc.CV_HOUGH_GRADIENT, 1, 50, 100, 10, 70, 90);

        //Order detected circles to matrix, down up, left to right.
        Point[][] centersOrdered = orderCirclesToMatrix(orderCircles(circles));

        //===================Only if we are working with thresh Mat (Called from MainActivity)===========
        if (binaryImage == null) {
            Mat copySrc = src.clone();
            for (int i = 0; i < circles.cols(); i++) {
                double[] vCircle = circles.get(0, i);
                Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
                int radius = (int) Math.round(vCircle[2]);
                Imgproc.circle(copySrc, pt, radius, new Scalar(50, 255, 0), 5);
            }

            String ss = "";
            for (int i = 0; i < centersOrdered.length; i++) {
                for (int j = 0; j < centersOrdered[i].length; j++) {
                    Imgproc.putText(copySrc, "" + i, centersOrdered[i][j], Core.FONT_HERSHEY_PLAIN, 7.0, new Scalar(0, 255, 255));//circle(copySrc, p, );
                    ss += "Circle " + (i * centersOrdered.length + j) + " (" + i + ", " + j + ") " + ": " + centersOrdered[i][j].x + ", " + centersOrdered[i][j].y + "\n";
                }
            }
            Toast.makeText(context, ss, Toast.LENGTH_LONG).show();
            updateBitmap(copySrc);
        }
        //===================Only if we are working with thresh Mat (Called from MainActivity)===========

        return centersOrdered;
    }


    /**
     * Orders circles by their center and convert them to a 1d array, according to thresh.
     * If x from circle is like x from previous circle +- thresh, then it is in the same column,
     * otherwise it is in the next column.
     *
     * @param circles the Mat we want to convert to an ordered array.
     * @return We return an array, that is a 1D array, it is like we take all columns, and put them in one long column,
     * the column starts from the left down circle, goes up, and right, and finishes with the upper right one.
     */
    public Point[] orderCircles(Mat circles) {
        Point[] centersOfCircles = new Point[circles.cols()];
        for (int i = 0; i < circles.cols(); i++) {
            double[] vCircle = circles.get(0, i);
            centersOfCircles[i] = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
        }
        Arrays.sort(centersOfCircles, new Comparator<Point>() {
            public int compare(Point a, Point b) {
                //=================Important to adapt thresh according to height of picture made by drone=============
                int thresh = 80;
                int xComp = Double.compare(a.x, b.x);
                //if they are on the same column, check who is the higher one.
                if (Math.abs(a.x - b.x) <= thresh) {
                    return -Double.compare(a.y, b.y);
                } else
                    return xComp;
            }
        });
        return centersOfCircles;
    }

    /**
     * Order circles from array, according to compare function in MyPoint.
     *
     * @param circles the array we want to order
     * @return We return an array, that is a 1D array, it is like we take all columns, and put them in one long column,
     * the column starts from the left down circle, goes up, and right, and finishes with the upper right one.
     */
    public MyPoint[] orderCircles(MyPoint[] circles) {
        Arrays.sort(circles, new MyPoint());
        return circles;
    }

    /**
     * After having ordered the circles to one big column, we convert that column, to an array of columns, according to
     * their y, if the y is lower than the y of the last point, it means that the point belongs to a new column
     *
     * @param orderedCircles 1D ordered array we want to convert to 2D ordered matrix
     * @return 2D ordered matrix, from down up, left to right.
     */
    public Point[][] orderCirclesToMatrix(Point[] orderedCircles) {
        ArrayList<ArrayList<Point>> matrixCircles = new ArrayList<>();
        matrixCircles.add(new ArrayList<Point>());
        int currentIndex = 0;
        matrixCircles.get(currentIndex).add(orderedCircles[0]);
        for (int i = 1; i < orderedCircles.length; i++) {
            Point currentP = orderedCircles[i], lastP = orderedCircles[i - 1];
            //Because the coordinate system is different in image( y increases from up to down),
            // here we check if belongs to same column
            if (currentP.y < lastP.y) {
                matrixCircles.get(currentIndex).add(currentP);
            } else {
                matrixCircles.add(new ArrayList<Point>());
                matrixCircles.get(++currentIndex).add(currentP);
            }
        }
        return arrayListsToArrays(matrixCircles);
    }

    /**
     * After having ordered the circles to one big column, we convert that column, to an array of columns, according to
     * their y, if the y is lower than the y of the last point, it means that the point belongs to a new column
     *
     * @param orderedCircles 1D ordered array we want to convert to 2D ordered matrix
     * @return 2D ordered matrix, from down up, left to right.
     */
    public MyPoint[][] orderCirclesToMatrix(MyPoint[] orderedCircles) {
        ArrayList<ArrayList<MyPoint>> matrixCircles = new ArrayList<>();
        matrixCircles.add(new ArrayList<MyPoint>());
        int currentIndex = 0;
        matrixCircles.get(currentIndex).add(orderedCircles[0]);
        for (int i = 1; i < orderedCircles.length; i++) {
            MyPoint currentP = orderedCircles[i], lastP = orderedCircles[i - 1];
            //Because the coordinate system is different in image( y increases from up to down),
            // here we check if belongs to same column
            if (currentP.y < lastP.y) {
                matrixCircles.get(currentIndex).add(currentP);
            } else {
                matrixCircles.add(new ArrayList<MyPoint>());
                matrixCircles.get(++currentIndex).add(currentP);
            }
        }
        return arrayListsToArraysMP(matrixCircles);
    }

    /**
     * Convert 2D ArrayList to 2D array.
     *
     * @param arrayLists, 2D ArrayList to convert
     * @return 2D array of MyPoint
     */
    private MyPoint[][] arrayListsToArraysMP(ArrayList<ArrayList<MyPoint>> arrayLists) {
        MyPoint[][] array = new MyPoint[arrayLists.size()][];
        for (int i = 0; i < arrayLists.size(); i++) {
            ArrayList<MyPoint> row = arrayLists.get(i);
            array[i] = row.toArray(new MyPoint[row.size()]);
        }
        return array;
    }

    /**
     * Convert 2D ArrayList to 2D array.
     *
     * @param arrayLists, 2D ArrayList to convert
     * @return 2D array of Point
     */
    private Point[][] arrayListsToArrays(ArrayList<ArrayList<Point>> arrayLists) {
        Point[][] array = new Point[arrayLists.size()][];
        for (int i = 0; i < arrayLists.size(); i++) {
            ArrayList<Point> row = arrayLists.get(i);
            array[i] = row.toArray(new Point[row.size()]);
        }
        return array;
    }

    /**
     * Update bitmap from Mat image
     *
     * @param image the image we will use to update the bitmap
     */
    private void updateBitmap(Mat image) {
        bm = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, bm);
    }

    //=============================From here down we no longer use it================================

    public static double stdDev(double[] arr) {
        double average = averageOf(arr), variance = 0;
        for (int i = 0; i < arr.length; i++) {
            variance += Math.pow(arr[i] - average, 2);
        }
        variance /= arr.length;
        double stdDeviation = Math.sqrt(variance);
        return stdDeviation;
    }

    public boolean inRange(double num, double lower, double upper) {
        return num <= upper && num >= lower;
    }

    public static double averageOf(double... arr) {
        double average = 0;
        for (int i = 0; i < arr.length; i++) {
            average += arr[i];
        }
        average /= arr.length;
        return average;
    }

    public static boolean inStd(double std, double avr, double n, int mul) {
        return n <= avr + std * mul && n >= avr - std * mul;
    }

    public double[][] filterCirclesOldOne(Mat circles) {
        double average = 0, variance = 0;
        for (int i = 0; i < circles.cols(); i++) {
            double[] vCircle = circles.get(0, i);
            Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
            int radius = (int) Math.round(vCircle[2]);
            average += radius;
        }
        average /= circles.cols();
        for (int i = 0; i < circles.cols(); i++) {
            double[] vCircle = circles.get(0, i);
            Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
            int radius = (int) Math.round(vCircle[2]);
            variance += Math.pow(radius - average, 2);
        }
        variance /= circles.cols();
        double whichStd = 0.7;
        double stdDeviation = Math.sqrt(variance);
        ArrayList<double[]> filteredCir = new ArrayList<>();
        for (int i = 0; i < circles.cols(); i++) {
            double[] vCircle = circles.get(0, i);
            Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
            int radius = (int) Math.round(vCircle[2]);
            if (inRange(radius, average - whichStd * stdDeviation, average + whichStd * stdDeviation))
                filteredCir.add(vCircle);
        }
        double[][] filteredCircles = filteredCir.toArray(new double[filteredCir.size()][]);
        return filteredCircles;
    }

    public double[][] filterCircles(Mat circles) {
        for (int i = 0; i < circles.cols(); i++) {
            double[] circle1 = circles.get(0, i);
            Point pt1 = new Point(Math.round(circle1[0]), Math.round(circle1[1]));
            for (int j = 0; j < circles.cols(); j++) {
                double[] circle2 = circles.get(0, j);
                Point pt2 = new Point(Math.round(circle2[0]), Math.round(circle2[1]));
                //TODO: Calculate distance from all circles. If the distance is bigger than the stdDev of distance, then it means that it is a fake circle.
            }
        }

        return null;
    }

    public void findingOptimalParameters(int numCircles) {
        Mat temp = new Mat(bm.getWidth(), bm.getHeight(), CvType.CV_8UC1), temppp = new Mat();

        Utils.bitmapToMat(bm, temppp);
        Imgproc.cvtColor(temppp, temp, Imgproc.COLOR_RGB2GRAY);

        Mat circles = new Mat();
        Imgproc.HoughCircles(temp, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 100, 100, 100, 30, 400);
        //Make iterative function to find the optimal param2 of function.

        int currentNumCircles = circles.cols();


        while (currentNumCircles < numCircles) {
            Imgproc.HoughCircles(temp, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 100, 100, 100, 30, 400);

        }

    }

    public void closing() {
        Mat temp = new Mat();
        Utils.bitmapToMat(bm, temp);
        Imgproc.morphologyEx(temp, temp, MORPH_CLOSE, Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(7, 7)));
        updateBitmap(temp);
    }

    public void opening() {
        Mat temp = new Mat();
        Utils.bitmapToMat(bm, temp);
        Imgproc.morphologyEx(temp, temp, MORPH_OPEN, Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(3, 3)));
        updateBitmap(temp);
    }


    public void toGrayOrColor() {

        if (isGray) {
            dirt = src.clone();//new Mat(bm.getWidth(), bm.getHeight(), CvType.CV_8UC3);
            isGray = false;
        } else {
            dirt = new Mat(bm.getWidth(), bm.getHeight(), CvType.CV_8UC1);
            Utils.bitmapToMat(bm, dirt);
            Imgproc.cvtColor(dirt, dirt, Imgproc.COLOR_RGB2GRAY);
            //Toast.makeText(context, "To Gray",Toast.LENGTH_SHORT).show();
            isGray = true;
        }
        updateBitmap(dirt);
    }

    public void blur() {
        dirt = new Mat();
        Utils.bitmapToMat(bm, dirt);
        Imgproc.blur(dirt, dirt, new Size(3, 3));
        Imgproc.GaussianBlur(dirt, dirt, new Size(3, 3), 0);
        updateBitmap(dirt);
    }

    private boolean pointInArray(Point p, Point[][] arr) {
        for (Point[] rows : arr) {
            for (Point point : rows) {
                if (p.x == point.x && p.y == point.y) {
                    return true;
                }
            }
        }
        return false;
    }

    public void contours() {

        Mat image = new Mat(), temp = new Mat();
        Utils.bitmapToMat(bm, temp);
        Imgproc.cvtColor(temp, image, Imgproc.COLOR_RGB2GRAY);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat copySrc = src.clone();
        double[] radiuses = new double[contours.size()];
        for (int i = 0; i < contours.size(); i++) {
            Point center = new Point();
            float[] radius = new float[1];
            MatOfPoint c = contours.get(i);
            MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
            Imgproc.minEnclosingCircle(c2f, center, radius);
            radiuses[i] = radius[0];
        }
        double stdDevRadius = stdDev(radiuses), avgRadius = averageOf(radiuses[0]);
        for (int i = 0; i < contours.size(); i++) {
            Point center = new Point();
            float[] radius = new float[1];
            MatOfPoint c = contours.get(i);
            MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
            Imgproc.minEnclosingCircle(c2f, center, radius);
            //Check if the circle is of a "Normal" size.
            if (inStd(stdDevRadius, avgRadius, radius[0], 1)) {
                Imgproc.circle(copySrc, center, (int) radius[0], new Scalar(255, 0, 0), 2);
            }
        }
        updateBitmap(copySrc);
    }

    public void threshGreen() {
        Mat green = new Mat();
        Mat raw = new Mat();
        Mat hsv = new Mat();
        Utils.bitmapToMat(bm, raw);
        Imgproc.cvtColor(raw, hsv, Imgproc.COLOR_RGB2HSV);
        double H_MIN_BLUE = 78, S_MIN_BLUE = 180, V_MIN_BLUE = 70, H_MAX_BLUE = 87, S_MAX_BLUE = 235, V_MAX_BLUE = 227;
        Core.inRange(hsv, new Scalar(H_MIN_BLUE, S_MIN_BLUE, V_MIN_BLUE), new Scalar(H_MAX_BLUE, S_MAX_BLUE, V_MAX_BLUE), green);
        Imgproc.GaussianBlur(green, green, new Size(3, 3), 0);
        Imgproc.morphologyEx(green, green, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        updateBitmap(green);
    }

    public void makeROI() {
        Mat image = new Mat(), temp = new Mat();
        Utils.bitmapToMat(bm, temp);
        Imgproc.cvtColor(temp, image, Imgproc.COLOR_RGB2GRAY);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat copySrc = src.clone();
        double maxArea = 0;
        float[] radius = new float[1];
        Point center = new Point();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint c = contours.get(i);
            if (Imgproc.contourArea(c) > maxArea) {
                MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
                Imgproc.minEnclosingCircle(c2f, center, radius);
            }
        }
        Imgproc.circle(copySrc, center, (int) radius[0], new Scalar(255, 0, 0), 2);


        updateBitmap(copySrc);
    }
}
