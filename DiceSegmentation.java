package acquisition;


import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class DiceSegmentation {

static {
System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
}

private Mat image;
private List<Mat> croppedDiceImages;

public DiceSegmentation(String imagePath) {
this.image = Imgcodecs.imread(imagePath);
this.croppedDiceImages = new ArrayList<>();
}

public void segmentDices() {
if (this.image.empty()) {
System.out.println("Image did not load.");
return;
}

Mat gray = new Mat();
Imgproc.cvtColor(this.image, gray, Imgproc.COLOR_BGR2GRAY);

Mat blurred = new Mat();
Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);

Mat thresholded = new Mat();
Imgproc.threshold(blurred, thresholded, 128, 255, Imgproc.THRESH_BINARY_INV);

List<MatOfPoint> contours = new ArrayList<>();
Mat hierarchy = new Mat();
Imgproc.findContours(thresholded, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

for (MatOfPoint contour : contours) {
if (isDiceContour(contour)) {
Rect rect = Imgproc.boundingRect(contour);
Mat croppedDice = new Mat(this.image, rect);
this.croppedDiceImages.add(croppedDice);
}
}

for (int i = 0; i < croppedDiceImages.size(); i++) {
Mat croppedDice = croppedDiceImages.get(i);
processAndShowDiceImage(croppedDice, i);
}
}

private boolean isDiceContour(MatOfPoint contour) {
MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
double contourArea = Imgproc.contourArea(contour);
MatOfPoint2f approxCurve = new MatOfPoint2f();
Imgproc.approxPolyDP(contour2f, approxCurve, 0.1 * Imgproc.arcLength(contour2f, true), true);

return approxCurve.toArray().length >= 4 && contourArea > 500;
}

private void processAndShowDiceImage(Mat diceImage, int index) {
// Convert to grayscale and apply threshold
Mat grayDice = new Mat();
Imgproc.cvtColor(diceImage, grayDice, Imgproc.COLOR_BGR2GRAY);
Mat histogram = calculateHistogram(grayDice);
int optimalThreshold = findOptimalThreshold(histogram);
Mat thresholdedDice = new Mat();
Imgproc.threshold(grayDice, thresholdedDice, optimalThreshold, 255, Imgproc.THRESH_BINARY);
HighGui.imshow("Thresholded Dice " + index, thresholdedDice);
HighGui.waitKey(0); // Wait for any key.
// Fermeture (Dilatation suivie d'une érosion)
Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(12, 12));
Imgproc.morphologyEx(thresholdedDice, thresholdedDice, Imgproc.MORPH_CLOSE, kernel);
HighGui.imshow("Thresholded Dice " + index, thresholdedDice);
HighGui.waitKey(0); // Wait for any key.

// Ouverture (Érosion suivie d'une dilatation)
//Imgproc.morphologyEx(thresholdedDice, thresholdedDice, Imgproc.MORPH_OPEN, kernel);
//HighGui.imshow("Thresholded Dice " + index, thresholdedDice);
//HighGui.waitKey(0); // Wait for any key.

// Detecting circles on the thresholded image
Mat circles = new Mat();
Imgproc.HoughCircles(
thresholdedDice, // Image en niveaux de gris
circles, // Matrice de sortie :(taille colonnes = nbr de cercles)
Imgproc.HOUGH_GRADIENT, // Méthode de détection
1, // Échelle d'image
10, // Distance minimale entre les cercles
100, // Paramètre de détection (si élevé = détection plus rigoureuse )
6, // Seuil pour la détection de cercles
5, // Rayon min du cercle
20 // Rayon max ... (j'ai récup ses valeurs avec datatips, peut ne pas marcher pour autres images)
);

// Filter the circles that are on the top side of the dice
List<Point> topCircles = new ArrayList<>();
for (int i = 0; i < circles.cols(); i++) {
double[] circle = circles.get(0, i);
if (circle[1] < thresholdedDice.rows() / 2) { // Only consider circles in the top half
topCircles.add(new Point(circle[0], circle[1]));
}
}

// Draw the circles and show the dice value
Mat circlesImage = diceImage.clone();
for (Point center : topCircles) {
Imgproc.circle(circlesImage, center, (int) circles.get(0, 0)[2], new Scalar(0, 255, 0), 2);
}
int diceValue = topCircles.size();
System.out.println("Dice " + index + " Value: " + diceValue);
HighGui.imshow("Detected Top Circles Dice " + index, circlesImage);
HighGui.waitKey(0); // Wait for any key.
}

private static Mat calculateHistogram(Mat grayscaleImage) {
List<Mat> images = new ArrayList<>();
images.add(grayscaleImage);
Mat histogram = new Mat();
Imgproc.calcHist(
images,
new MatOfInt(0),
new Mat(),
histogram,
new MatOfInt(256),
new MatOfFloat(0, 256),
false
);

Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX, -1, new Mat());
return histogram;
}

private static int findOptimalThreshold(Mat histogram) {
double maxEntropy = 0;
int optimalThreshold = 0;

for (int threshold = 1; threshold <= 255; threshold++) {
Mat foregroundHist = histogram.rowRange(0, threshold);
Mat backgroundHist = histogram.rowRange(threshold, histogram.rows());

Scalar sumForeground = Core.sumElems(foregroundHist);
Scalar sumBackground = Core.sumElems(backgroundHist);
MatOfFloat foregroundProb = new MatOfFloat();
MatOfFloat backgroundProb = new MatOfFloat();
Core.divide(foregroundHist, sumForeground, foregroundProb);
Core.divide(backgroundHist, sumBackground, backgroundProb);

double entropyForeground = calculateEntropy(foregroundProb);
double entropyBackground = calculateEntropy(backgroundProb);

double totalEntropy = entropyForeground + entropyBackground;

if (totalEntropy > maxEntropy) {
maxEntropy = totalEntropy;
optimalThreshold = threshold;
}
}
return optimalThreshold;
}

private static double calculateEntropy(Mat histogram) {
double entropy = 0;
for (int i = 0; i < histogram.rows(); i++) {
double[] value = histogram.get(i, 0);
if (value[0] > 0) {
entropy -= value[0] * Math.log(value[0]) / Math.log(2);
}
}
return entropy;
}



public static void main(String[] args) {
String imagePath = "C:/Users/JALAL/OneDrive/Bureau/projetInfo/temp/image_captured.jpg";; // Remplacez par le chemin de votre image
DiceSegmentation segmentation = new DiceSegmentation(imagePath);
segmentation.segmentDices();
HighGui.destroyAllWindows();
}
}