package convolution_autoencoder;




import nu.pattern.OpenCV;
        import org.datavec.api.split.FileSplit;
        import org.datavec.image.loader.Java2DNativeImageLoader;
        import org.datavec.image.recordreader.ImageRecordReader;
        import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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

        import javax.swing.*;
        import java.awt.image.BufferedImage;
        import java.io.File;
        import java.util.List;
        import java.util.Map;

/**
 * convolutional auto encoder
 * you must add a dataset
 * I used  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 *
 */

public class Tester {

    private static JFrame frame;
    private static JLabel label;

    JLabel originalLaabel;
    JLabel resultLabel;
    Java2DNativeImageLoader imageLoader;
    Java2DNativeImageLoader imageLoaderTwo;

    private static final Logger log = LoggerFactory.getLogger(NewExp.class);
    int height = 32;
    int width = 32;

    String incorrect = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";
    String source = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";

    int channels = 3;

    public static void main(String[] args)
    {
        Tester newExp = new Tester();
        newExp.run();
    }


    public Tester() {

        if (incorrect.equals(source)){
            System.err.println("You must set the source to your dataset");

        }

        OpenCV.loadLocally();
        String[] file = new File(source).list();

        Mat src = Imgcodecs.imread(source + file[0]);
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


    public void run()
    {

        DataSetIterator iter = getDataSetIter();

        ComputationGraph net = ShortGraphBuilder.grabEncoder(32, 8, 64, 128, 1e-3);




        for (int i = 0; i < 100; i++) {

            int count = 0;
            while (iter.hasNext()){

                DataSet d = iter.next();

                //Map<String, INDArray> b = net.feedForward(new INDArray[] {d.getFeatures()}, 18, false);


                net.fit(new INDArray[] {d.getFeatures()}, new INDArray[] {d.getFeatures()});

                //INDArray ind = net.getGradientsViewArray();
                //Gradient g = net.gradient();

                //net.backpropGradient()


                if ((count % 100) == 0) {

                    BufferedImage bf = imageLoader.asBufferedImage(d.getFeatures());

                    Map<String, INDArray> a = net.feedForward(new INDArray[] {d.getFeatures()}, 18, false);
                    // List<INDArray> a = net.feedForwardToLayer(10, d.getFeatures());


                    BufferedImage bfO = imageLoaderTwo.asBufferedImage(a.get("out"));

                    originalLaabel.setIcon(new ImageIcon(bf));
                    resultLabel.setIcon(new ImageIcon(bfO));

                    //System.out.println("");

                }
                count++;
            }

        }




    }


    public DataSetIterator getDataSetIter()
    {
        try{
            ImageRecordReader recordReader = new ImageRecordReader(height,width, 3);
            recordReader.initialize(new FileSplit(new File(source)));

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 1);

            return dataSetIterator;

        }catch (Exception e){
            e.printStackTrace();
        }

        return null;
    }









}
