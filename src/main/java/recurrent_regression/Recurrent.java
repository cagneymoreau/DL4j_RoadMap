package recurrent_regression;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.io.File;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple example to est recurrent regression on predictable function
 *
 * Notice I used normalization that I hand wrote. This is discouraged when learning
 * because api has normalization strategies that are tested and durable
 *
 */

public class Recurrent {



    public static void main(String[] args) throws Exception
    {
        ArrayList<DataSet> results = new ArrayList<>();

        DataSet ds = getSineDataSet(20, 100, .1);


        results.add(ds); //ad vanilla dataset

        //copy dataset and normalize
        DataSet out = ds.copy();


        Normies lNorm = new Normies();
        lNorm.linearRange(ds.getLabels(), 2);
        out.setLabels(lNorm.convert(out.getLabels()));



        //submit to network
        LSTMNet r = new LSTMNet();
        DataSet res = r.trainTest(out, .8, 1000);
        res.setLabels(lNorm.revert(res.getLabels()));
        //res.setFeatures(fNorm.revert(res.getFeatures()));
        results.add(res);


        plot2DScatterGraph(results, LocalTime.now().toString());



    }






    private static DataSet getSineDataSet(int period, int sampleSize, double noise)
    {
        period = period/2;

        INDArray x = Nd4j.linspace((period * -1), period, sampleSize).reshape(sampleSize, 1);
        INDArray sinArr = Transforms.sin(x, true);
        if (noise > 0){
            sinArr = sinArr.add((Nd4j.randn(sampleSize,1).mul(noise)));
        }


        DataSet ds = new DataSet();
        ds.setFeatures(x);
        ds.setLabels(sinArr);

        List<String> listy = new ArrayList<>();
        for (int i = 0; i < x.length(); i++) {
            listy.add(String.valueOf(i));
        }
        ds.setLabelNames(listy);

        //System.out.println(ds);

        return ds;

    }



    public static MultiLayerNetwork getLSTMNetwork(int input, int hiddenLayerSize, boolean useExisting, String path) throws Exception
    {
        System.out.println("Engaging Network: " + path);

        if (useExisting){

            try {
                MultiLayerNetwork net = MultiLayerNetwork.load(new File(path), true);
                return net;

            }catch (Exception e){
                //e.printStackTrace();
                System.err.println("CREATING NEW NETWORK - NONE FOUND!");
            }

        }


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(input).nOut(hiddenLayerSize).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(hiddenLayerSize).nOut(1).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(100));

        return network;

    }



    public static void plot2DScatterGraph(ArrayList<DataSet> DataSetList, String title) {
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
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);


    }






}
