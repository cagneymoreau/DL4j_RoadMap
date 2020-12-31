package vae;

import convolution_autoencoder.NewExp;
import nu.pattern.OpenCV;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MyVAE {


    private static JFrame frame;
    private static JLabel label;
    private static JPanel panel;

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
        MyVAE myVAE = new MyVAE();
        myVAE.run();
    }

    public MyVAE() {

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
        originalLaabel.setBounds(0,0, width * 10, height * 2);

        resultLabel = new JLabel();
        resultLabel.setBounds(0,height * 3, width * 10, height * 2);

        panel = new JPanel();

        frame.add(originalLaabel);
        frame.add(resultLabel);
        frame.add(panel);
        frame.setSize(width * 20, height * 15 );
        frame.setEnabled(true);
        frame.setLayout(null);
        frame.setVisible(true);


    }


    public void run()
    {

        DataSetIterator iter = getDataSetIter();

        ComputationGraph net = getNet();

        //test/results storage
        ArrayList<INDArray[]> samplesData = new ArrayList<>();
        ArrayList<DataSet> xySamples = new ArrayList<>();

        INDArray images = null;
        for (int i = 0; i < 10; i++) {

            samplesData.add(new INDArray[] {iter.next().getFeatures()});
            if (i == 0){

                images = samplesData.get(0)[0];

            }
            else {
                images = Nd4j.concat(0, images, samplesData.get(i)[0]);
            }


        }

        BufferedImage bf = imageLoader.asBufferedImage(images);
        originalLaabel.setIcon(new ImageIcon(bf));



        for (int i = 0; i < 100; i++) {

            int count = 0;
            while (iter.hasNext()){

                DataSet d = iter.next();

                INDArray[] arr = new INDArray[] {d.getFeatures()};

                net.fit(arr, arr);


                if ((count % 1000) == 0) {

                    INDArray feat = Nd4j.zeros(10);
                    INDArray label = Nd4j.zeros(10);

                    INDArray sample = null;
                    for (int j = 0; j < samplesData.size(); j++) {

                        Map<String, INDArray> a = net.feedForward();
                        feat.putScalar(j, a.get("position").getFloat(0));
                        label.putScalar(j, a.get("position").getFloat(0));

                        if (j == 0){
                            sample = a.get("output");
                        }
                        else{
                            sample = Nd4j.concat(0, sample, a.get("output"));
                        }

                    }


                    //images
                    BufferedImage bfO = imageLoaderTwo.asBufferedImage(sample);
                    resultLabel.setIcon(new ImageIcon(bfO));


                    //save the image locations for review
                    DataSet out = new org.nd4j.linalg.dataset.DataSet();
                    out.setFeatures(feat);
                    out.setLabels(label);
                    xySamples.add(out);

                    panel = plot2DScatterGraph(xySamples, "out");

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




    private ComputationGraph getNet()
    {
        int rngSeed = 123;
        int dimensions = 2;

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        ComputationGraphConfiguration.GraphBuilder confBuilder = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("input");

                //encode
                confBuilder.addLayer("in_1", new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.LEAKYRELU).nIn(channels).nOut(32).build(), "input")
                //(32-3)/1 + 1 = 32x30x30
                //(64-3)/1 + 1 = 32x62x62
                //.layer( new BatchNormalization())
                .layer("in_2", new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(), "in_1")
                //(30-2)/2 + 1 = 32x15x15
                //(62-3)/2 + 1 = 32x30x30
                .layer("in_3", new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).activation(Activation.LEAKYRELU).nIn(32).nOut(16).build(), "in_2")
                //(15-2)/1 + 1 = 16x14x14
                //(35-2)/1 + 1 = 16x29x29
                //.layer(new BatchNormalization())
                .layer("in_4", new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(), "in_3")
                //(14-2)/2 + 1 = 16x7x7
                //(29-2)/2 + 1 = 16x15x15

                //double stuff oreo center
                .layer("in_5", new DenseLayer.Builder().nIn(784).nOut(dimensions).weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build(), "in_4")
                .layer("in_6", new DenseLayer.Builder().nIn(dimensions).nOut(784).weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build(), "in_5")

                //decode
                .layer( "in_7", new Upsampling2D.Builder().size(2) .build(), "in_6")
                //out = 16x14x14
                //.layer( new BatchNormalization())
                .layer("in_8", new Deconvolution2D.Builder().kernelSize(2,2) .stride(1,1).nIn(16).nOut(32).activation(Activation.LEAKYRELU).build(), "in_7")
                //out = 32x15x15
                .layer("in_9", new Upsampling2D.Builder().size(2).build(), "in_8")
                //out = 32x30x30
                //.layer( new BatchNormalization())
                .layer("in_10", new Deconvolution2D.Builder().kernelSize(3,3).stride(1,1).activation(Activation.LEAKYRELU).nIn(32).nOut(channels).build(), "in_9")
                //out = 3x32x32

                .layer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.LEAKYRELU).build(), "in_10")
                //out = 3x32x32
                //.layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(3072).nOut(3072).activation(Activation.IDENTITY).build())

                // say hail marys here
                .inputPreProcessor("in_5", new CnnToFeedForwardPreProcessor(7,7,16))
                .inputPreProcessor("in_6", new FeedForwardToCnnPreProcessor(7,7,16))
                //.inputPreProcessor(10, new CnnToFeedForwardPreProcessor(32,32,3))
                        .setOutputs("output")
                .setInputTypes(InputType.convolutional(32, 32, 3))
                .build();


        ComputationGraph net = new ComputationGraph(confBuilder.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        System.out.println(net.summary());

        return net;

    }





    public static JPanel  plot2DScatterGraph(ArrayList<DataSet> DataSetList, String title) {
        XYSeriesCollection c = new XYSeriesCollection();

        int dscounter = 1; //use to name the dataseries
        for (DataSet ds : DataSetList) {
            INDArray features = ds.getFeatures();
            INDArray outputs = ds.getLabels();

            int nRows = (int) features.length();
            XYSeries series = new XYSeries("S" + dscounter);
            for (int i = 0; i < nRows; i++) {
                series.add(features.getDouble(i), outputs.getDouble(i));
            }

            c.addSeries(series);
        }


        String xAxisLabel = "Date";
        String yAxisLabel = "Price";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);
        return new ChartPanel(chart);

    }




}
