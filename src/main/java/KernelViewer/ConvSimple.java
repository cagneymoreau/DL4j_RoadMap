package KernelViewer;

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
import org.deeplearning4j.nn.gradient.Gradient;
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

/**
 * This is a simple multinet convo encoder from existing project
 *
 */

public class ConvSimple {

    private static JFrame frame;
    private static JLabel label;

      JLabel originalLaabel;
    JLabel resultLabel;
    Java2DNativeImageLoader imageLoader;
    Java2DNativeImageLoader imageLoaderTwo;

    private static final Logger log = LoggerFactory.getLogger(ConvSimple.class);
    int height = 32;
    int width = 32;

    String incorrect = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";
    String source = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";

    int channels = 3;

    public static void main(String[] args)
    {
        ConvSimple newExp = new ConvSimple();
        newExp.run();
    }


    public ConvSimple() {

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

        MultiLayerNetwork net = getNet();

        //KernelView  kernelView = new KernelView(net);
        CovOutViewer covOutViewer = new CovOutViewer(net);


        for (int i = 0; i < 100; i++) {

            int count = 0;
            while (iter.hasNext()){

                DataSet d = iter.next();


                INDArray ind = net.getGradientsViewArray();
                Gradient g = net.gradient();

                //net.backpropGradient();

                if ((count % 100) == 0) {


                    BufferedImage bf = imageLoader.asBufferedImage(d.getFeatures());

                    List<INDArray> a = net.feedForwardToLayer(10, d.getFeatures());

                    Gradient grr = net.gradient();

                    covOutViewer.update();

                    BufferedImage bfO = imageLoaderTwo.asBufferedImage(a.get(a.size() - 1));

                    originalLaabel.setIcon(new ImageIcon(bf));
                    resultLabel.setIcon(new ImageIcon(bfO));

                    //System.out.println("");

                }

                net.fit(d.getFeatures(), d.getFeatures());

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







    private MultiLayerNetwork getNet()
    {
        int rngSeed = 123;
        int dimensions = 256;

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()

                //encode
                .layer(0, new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.LEAKYRELU).nIn(channels).nOut(32).build())
                //(32-3)/1 + 1 = 32x30x30
                //(64-3)/1 + 1 = 32x62x62
                .layer( new BatchNormalization())
                .layer(1, new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                //(30-2)/2 + 1 = 32x15x15
                //(62-3)/2 + 1 = 32x30x30
                .layer(2, new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).activation(Activation.LEAKYRELU).nIn(32).nOut(16).build())
                //(15-2)/1 + 1 = 16x14x14
                //(35-2)/1 + 1 = 16x29x29
                .layer(new BatchNormalization())
                .layer(3, new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                //(14-2)/2 + 1 = 16x7x7
                //(29-2)/2 + 1 = 16x15x15

                //double stuff oreo center
                .layer(4, new DenseLayer.Builder().nIn(784).nOut(dimensions).weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build())
                .layer(5, new DenseLayer.Builder().nIn(dimensions).nOut(784).weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build())

                //decode
                .layer( 6, new Upsampling2D.Builder().size(2) .build())
                //out = 16x14x14
                .layer( new BatchNormalization())
                .layer(7, new Deconvolution2D.Builder().kernelSize(2,2) .stride(1,1).nIn(16).nOut(32).activation(Activation.LEAKYRELU).build())
                //out = 32x15x15
                .layer(8, new Upsampling2D.Builder().size(2).build())
                //out = 32x30x30
                .layer( new BatchNormalization())
                .layer(9, new Deconvolution2D.Builder().kernelSize(3,3).stride(1,1).activation(Activation.LEAKYRELU).nIn(32).nOut(channels).build())
                //out = 3x32x32

                .layer(10, new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.LEAKYRELU).build())
                //out = 3x32x32
                //.layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(3072).nOut(3072).activation(Activation.IDENTITY).build())

                // say hail marys here
                .inputPreProcessor(4, new CnnToFeedForwardPreProcessor(7,7,16))
                .inputPreProcessor(6, new FeedForwardToCnnPreProcessor(7,7,16))
                //.inputPreProcessor(10, new CnnToFeedForwardPreProcessor(32,32,3))
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        System.out.println(net.summary());

        return net;

    }





}
