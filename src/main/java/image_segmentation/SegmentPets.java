package image_segmentation;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import convolution_autoencoder.NewExp;
import nu.pattern.OpenCV;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.MatFile;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Struct;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.Map;


/**
 *
 * Oxford III pets 2 data set
 * https://www.robots.ox.ac.uk/~vgg/data/pets/
 *
 * I added jmatio through IDE/maven and should be in pom...oops
 *
 *
 */





public class SegmentPets {


    private static JFrame frame;
    private static JLabel label;

    JLabel originalLaabel;
    JLabel resultLabel;
    Java2DNativeImageLoader imageLoader;
    Java2DNativeImageLoader imageLoaderTwo;

    private static final Logger log = LoggerFactory.getLogger(NewExp.class);
    int height = 64;
    int width = 64;

    String sourceLocation = "D:\\Downloads\\";
    String extractionFolder = "petsdataset\\";
    String images = "images\\";
    String annotations = "annotations\\";

    String imagesTar = "images.tar.gz";
    String annotationsTar = "annotations.tar.gz";

    String imageURL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz";
    String annotationsURL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz";


    int channels = 3;




    public static void main(String[] args)
    {
        SegmentPets segmentPets = new SegmentPets();

        //if (!segmentPets.buildDataOnDrive()) return; //make sure we have the data

        //segmentPets.trimapOpen();
        //segmentPets.viewTrimap();
        segmentPets.train();

    }


    public SegmentPets()
    {


        OpenCV.loadLocally();
        String[] file = new File(sourceLocation + extractionFolder + images).list();

        Mat src = Imgcodecs.imread(sourceLocation + extractionFolder + images + file[0]);
        System.out.println("FINAL \n Width: " + src.width() + "\n Height: " + src.height());


        frame = new JFrame("Results");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        imageLoader = new Java2DNativeImageLoader();
        imageLoaderTwo = new Java2DNativeImageLoader();
        originalLaabel = new JLabel();
        originalLaabel.setBounds(0,0, width * 2, height * 2);

        resultLabel = new JLabel();
        resultLabel.setBounds(width * 2,0, width * 2, height * 2);

        frame.add(originalLaabel);
        frame.add(resultLabel);
        frame.setSize(width * 5, height * 4 );
        frame.setEnabled(true);
        frame.setLayout(null);
        frame.setVisible(true);
    }



    public void train()
    {

        DataSetIterator iter = getDataSetIter();

        ComputationGraph net = CAEGraphs.grabShortEncoder(width, 8, width, 128, 1e-4, 100);

        Layer[] names = net.getLayers();

        for (int i = 0; i < 100; i++) {

            DataSet d = null;
            int count = 0;
            while (iter.hasNext()){

                try {
                     d = iter.next();
                }catch (Exception e){
                    System.out.print("IOException: " + count);
                }

                //INDArray ind = net.getGradientsViewArray();
                //Gradient g = net.gradient();

                //net.backpropGradient()


                if ((count % 100) == 0) {

                    BufferedImage bf = imageLoader.asBufferedImage(d.getFeatures());

                    Map<String, INDArray> a = net.feedForward(new INDArray[] {d.getFeatures()}, names.length-1, false);


                    BufferedImage bfO = imageLoaderTwo.asBufferedImage(a.get("out"));

                    originalLaabel.setIcon(new ImageIcon(bf));
                    resultLabel.setIcon(new ImageIcon(bfO));

                    //System.out.println("");

                }




                net.fit( new INDArray[] {d.getFeatures()}, new INDArray[] {d.getFeatures()});


                count++;
            }

        }






    }




    public DataSetIterator getDataSetIter()
    {
        try{

            ImageRecordReader recordReader = new ImageRecordReader(height,width, channels);
            recordReader.initialize(new FileSplit(new File(sourceLocation + extractionFolder + images)));

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 1);

            return dataSetIterator;

        }catch (Exception e){
            e.printStackTrace();
        }

        return null;
    }






    private void trimapOpen()
    {

        OpenCV.loadLocally();

        File file = new File("C:\\Users\\Cagney\\Desktop\\Abyssinian_1.png");


        Mat src = Imgcodecs.imread(file.getAbsolutePath());


        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Java2DNativeImageLoader loader = new Java2DNativeImageLoader();
        JLabel label = new JLabel();
        label.setBounds(0,0,256,256);

        f.add(label);
        f.setSize(256 , 256);
        f.setEnabled(true);
        f.setLayout(null);
        f.setVisible(true);

        BufferedImage buff = null;

        try {

            //BufferedImage ima = ImageIO.read(trimaps[0]);


            NativeImageLoader nil = new NativeImageLoader();
            INDArray image = nil.asMatrix(src);
            image.muli(100);

            buff = loader.asBufferedImage(image);

        } catch (Exception e) {e.printStackTrace();}


        label.setIcon(new ImageIcon(buff));


    }

    private void viewTrimap()
    {


        /*
        OpenCV.loadLocally();
        File file = new File(sourceLocation + extractionFolder + images + "Abyssinian_100.jpg");

        Mat src = Imgcodecs.imread(file.getAbsolutePath());
        System.out.println("FINAL \n Width: " + src.width() + "\n Height: " + src.height());



        File dir = new File(sourceLocation + extractionFolder +  annotations + "trimaps");
        File[] trimaps = dir.listFiles();

        System.out.println(trimaps[0].getName());


        try {

            Mat5File mat = Mat5.readFromFile(trimaps[0]);
            System.out.println(mat.toString());
            System.out.println(mat.getDescription());

            for (MatFile.Entry entry : mat.getEntries()) {
                System.out.println(entry);
            }

            Matrix m = mat.getMatrix(0);



            int i = m.getNumElements();

            for (int j = 0; j < i; j++) {
              double d =  m.getDouble(j);

              System.out.print(d + " ");

              if (j % 100 == 0){
                  System.out.println();
              }

            }

        } catch (Exception e){e.printStackTrace();}


        */



        /*


        OpenCV.loadLocally();

        Mat src = Imgcodecs.imread(trimaps[0].getAbsolutePath());


        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Java2DNativeImageLoader loader = new Java2DNativeImageLoader();
        JLabel label = new JLabel();
        label.setBounds(0,0,256,256);

        f.add(label);
        f.setSize(256 , 256);
        f.setEnabled(true);
        f.setLayout(null);
        f.setVisible(true);

        BufferedImage buff = null;

        try {

            //BufferedImage ima = ImageIO.read(trimaps[0]);


            NativeImageLoader nil = new NativeImageLoader();
            INDArray image = nil.asMatrix(src);

           buff = loader.asBufferedImage(image);

        } catch (Exception e) {e.printStackTrace();}


        //label.setIcon(new ImageIcon(image));


        */



        /*

        try
        {
            FileInputStream fis = new FileInputStream(trimaps[0]);

            byte[] bytes = new byte[354];
            int value = 0;
            int count = 0;
            do
            {
                value = fis.read(bytes);
                System.out.println(toHexFromBytes(bytes));
                count++;

            }while(value != -1);

            System.out.println(count);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }


         */



    }


    // TODO: 12/17/2020 this doesnt work. get it from github
    //download, create dirs, extract
    private boolean buildDataOnDrive()
    {
        File dir = new File(sourceLocation + extractionFolder);
        if (!dir.exists()){
            dir.mkdir();

            //download datset tar files
            try {

                DataUtilities.downloadFile(imageURL, sourceLocation + imagesTar);
                DataUtilities.downloadFile(annotationsURL, sourceLocation + annotationsTar);

            } catch (Exception e)
            {
                e.printStackTrace();
                return false;
            }

            //create extraction folders and extract
            File im = new File(sourceLocation + extractionFolder + images);
            im.mkdir();

            File an = new File(sourceLocation + extractionFolder + annotations);
            an.mkdir();

            try {
                DataUtilities.extractTarGz(sourceLocation + imagesTar, sourceLocation + extractionFolder + images);
                DataUtilities.extractTarGz(sourceLocation + annotationsTar, sourceLocation + extractionFolder + annotations);
            }catch (Exception e){
                e.printStackTrace();
                return false;
            }

        }
        return true;

    }


   

}
