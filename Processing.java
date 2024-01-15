package acquisition;
import org.opencv.core.*;  
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import java.util.ArrayList;
import java.util.List;
import org.opencv.highgui.HighGui;


public class Processing {
	
	static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
	
	public Mat loadImage(String filePath) {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat image = Imgcodecs.imread(filePath);

        if (image.empty()) {
            System.err.println("Error: Could not read the image. Check the file path.");
        }

        return image;
    }

    public Mat preprocessImage(Mat image) {
        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        Mat filteredImage = new Mat();
        Imgproc.GaussianBlur(grayImage, filteredImage, new Size(5, 5), 0);

        Mat edges = new Mat();
        Imgproc.Canny(filteredImage, edges, 50, 150);
        
        

        Mat kernel = Mat.ones(new Size(3, 3), CvType.CV_8U);
        Mat dilatedEdges = new Mat();
        Imgproc.dilate(edges, dilatedEdges, kernel);

        Mat erodedEdges = new Mat();
        Imgproc.erode(dilatedEdges, erodedEdges, kernel);
        
        
        String windowName = "Edges";
        HighGui.imshow(windowName, erodedEdges);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        return erodedEdges;
    }
    
    public void detection(Mat image, Mat erodedEdges) {
        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(erodedEdges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            if (40 <= rect.width && rect.width <= 110 && 40 <= rect.height && rect.height <= 110) {
                Imgproc.rectangle(image, rect, new Scalar(0, 0, 255), 2);
                Mat diceRegion = grayImage.submat(rect);
                Mat filteredImage = new Mat();
                Imgproc.GaussianBlur(diceRegion, filteredImage, new Size(5, 5), 0);
                Mat thresholded = new Mat();
                Imgproc.threshold(filteredImage, thresholded, 100, 255, Imgproc.THRESH_BINARY);
                // Erosion
                Mat kernel = Mat.ones(new Size(1, 1), CvType.CV_8U);
                Mat erodedCircles = new Mat();
                Imgproc.erode(thresholded, erodedCircles, kernel);
                
                // Appliquer la fonction bitwise_not pour obtenir le négatif
                Mat dst = new Mat();
                Core.bitwise_not(erodedCircles, dst);
                
                // visualize
                String windowName2 = "Tresh";
                HighGui.imshow(windowName2, dst);
                HighGui.waitKey(0);
                HighGui.destroyAllWindows();
                
                Mat circles = new Mat();
                Imgproc.HoughCircles(dst, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 3, 100, 22, 5, 9);

                if (circles.cols() > 0) {
                    List<Point> validCircles = new ArrayList<>();
                    for (int i = 0; i < circles.cols(); i++) {
                        double[] circleDetails = circles.get(0, i);
                        int x = (int) Math.round(circleDetails[0]);
                        int y = (int) Math.round(circleDetails[1]);
                        int r = (int) Math.round(circleDetails[2]);

                        Point center = new Point(x, y);
                        boolean isValid = true;

                        for (Point validCircle : validCircles) {
                            double distance = Math.sqrt(Math.pow(center.x - validCircle.x, 2) + Math.pow(center.y - validCircle.y, 2));
                            if (distance <= 7) {
                                isValid = false;
                                break;
                            }
                        }

                        if (isValid) {
                            validCircles.add(center);
                            Imgproc.circle(diceRegion, center, r, new Scalar(0, 255, 0), 4);
                        }
                    }
                    

                    int numValidCircles = validCircles.size();
                    System.out.println("Number of circles detected: " + Math.min(numValidCircles, 6));

                 // Display the image with detected circles
                    String windowName = "Detected Circles";
                    HighGui.imshow(windowName, diceRegion);
                    HighGui.waitKey(0);
                    HighGui.destroyAllWindows();
                } else {
                    System.out.println("No circles detected in the image.");
                }
            }
        }
    }
    
    public List<Integer> detection_List(Mat image, Mat erodedEdges) {
        List<Integer> diceCounts = new ArrayList<>();

        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(erodedEdges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            if (40 <= rect.width && rect.width <= 110 && 40 <= rect.height && rect.height <= 110) {
                Mat diceRegion = grayImage.submat(rect);
                Mat filteredImage = new Mat();
                Imgproc.GaussianBlur(diceRegion, filteredImage, new Size(5, 5), 0);
                Mat thresholded = new Mat();
                Imgproc.threshold(filteredImage, thresholded, 100, 255, Imgproc.THRESH_BINARY);
                
                // Erosion
                Mat kernel = Mat.ones(new Size(1, 1), CvType.CV_8U);
                Mat erodedCircles = new Mat();
                Imgproc.erode(thresholded, erodedCircles, kernel);
                
                // Inverse the image
                Mat dst = new Mat();
                Core.bitwise_not(erodedCircles, dst);
                
                Mat circles = new Mat();
                Imgproc.HoughCircles(dst, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 3, 100, 22, 5, 9);

                if (circles.cols() > 0) {
                    List<Point> validCircles = new ArrayList<>();
                    for (int i = 0; i < circles.cols(); i++) {
                        double[] circleDetails = circles.get(0, i);
                        Point center = new Point(Math.round(circleDetails[0]), Math.round(circleDetails[1]));
                        boolean isValid = true;

                        for (Point validCircle : validCircles) {
                            double distance = Math.sqrt(Math.pow(center.x - validCircle.x, 2) + Math.pow(center.y - validCircle.y, 2));
                            if (distance <= 7) {
                                isValid = false;
                                break;
                            }
                        }

                        if (isValid) {
                            validCircles.add(center);
                        }
                    }

                    // Add the number of valid circles (points on the dice) to the list
                    diceCounts.add(validCircles.size());
                }
            }
        }

        return diceCounts;
    }

}
