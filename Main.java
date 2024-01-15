package acquisition;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;

public class Main {
    public static void main(String[] args) {
    	Camera app = new Camera();
        app.run();
        Processing processing = new Processing();
        String imagePath = "C:/Users/JALAL/OneDrive/Bureau/projetInfo/temp/image_captured.jpg";
        Mat image = processing.loadImage(imagePath);
        if (image.empty()) {
            System.out.println("Could not load the image.");
            return;
        }

        Mat erodedEdges = processing.preprocessImage(image);

        // Performing detection
        processing.detection(image, erodedEdges);
        
        List<Integer> diceCounts = new ArrayList<>();
        diceCounts = processing.detection_List(image, erodedEdges);
        diceCounts.forEach(element -> System.out.println(element));
    }
}


