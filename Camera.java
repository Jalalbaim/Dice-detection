package acquisition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import java.util.concurrent.atomic.AtomicBoolean;

public class Camera {
    private static final int CAMERA_INDEX = 0; // Index de la webcam
    private static final String FILE_PATH = "C:/Users/JALAL/OneDrive/Bureau/projetInfo/temp/image_captured.jpg";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
    /**
    public static void main(String[] args) {
        Camera app = new Camera();
        app.run();
    }
    **/
    public void run() {
        VideoCapture cap = new VideoCapture(CAMERA_INDEX);

        if (!cap.isOpened()) {
            System.out.println("Unable to access the webcam");
            return;
        }

        Mat frame = new Mat();
        AtomicBoolean shouldExit = new AtomicBoolean(false);

        // Crée un thread pour afficher la vidéo en temps réel
        Thread displayThread = new Thread(() -> {
            while (!shouldExit.get()) {
                if (cap.read(frame)) {
                    HighGui.imshow("Webcam", frame);
                    int key = HighGui.waitKey(1);

                    if (key == 32) { // Espace pour capturer une image
                        Imgcodecs.imwrite(FILE_PATH, frame);
                        System.out.println("Image captured successfully");
                        shouldExit.set(true);
                        HighGui.destroyWindow("Webcam");
                    } else if (key == 113) { // 'q' pour quitter sans capturer
                        System.out.println("Exiting without capturing");
                        shouldExit.set(true);
                        HighGui.destroyWindow("Webcam");
                    }
                }
            }
        });

        // Démarrer le thread d'affichage
        displayThread.start();

        try {
            // Attendre que le thread d'affichage se termine
            displayThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Fermeture de la webcam
        cap.release();
        System.out.println("Program terminated");
        HighGui.destroyWindow("Webcam");
    }
}

