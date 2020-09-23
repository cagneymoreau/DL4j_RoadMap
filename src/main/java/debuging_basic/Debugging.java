package debuging_basic;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.io.File;
import java.time.LocalTime;
import java.util.*;

/**
 * See below for multilayer network debugging strategies
 * For computational graph see growablegan/gangrow
 *
 * Another important tool in debugging is to always attempt to overfit
 * single datapoint. If you cant overfit there is a bug
 *
 */

public class Debugging {




    public static void main(String[] args)
    {
        ArrayList<DataSet> results = new ArrayList<>();

        DataSet ds = getSineDataSet(20, 200, .1);

        DataSet res = ttFuncRegression(ds, "");

        results.add(ds);
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






    //region----------------------------regressionMathfunction

    public static DataSet ttFuncRegression(DataSet dataSet, String saveLocation){

        MultiLayerNetwork network = diffNet();

        dataSet.setFeatures(dataSet.getFeatures().reshape(dataSet.getFeatures().length(), 1));
        dataSet.setLabels(dataSet.getLabels().reshape(dataSet.getLabels().length(),1));

        final List<DataSet> list = dataSet.asList();
        Collections.shuffle(list, new Random(123));

        DataSetIterator iter = new ListDataSetIterator<>(list, 50);
        //DataSetIterator iter = new ExistingDataSetIterator(dataSet);

        /**
         * Here is an example of debugging. This can be use to solve issues such as...
         * outG = tesnors so you can review the size and shape of a tensor
         * p = your weights
         * g = your error gradient
         *
         */

        for (int i = 0; i < 500; i++) {

            List<INDArray> outG = network.feedForward (dataSet.getFeatures().getRow(0).reshape(1,1), false );
            INDArray p = network.params();
            network.fit(iter);
            Gradient g = network.gradient();
            System.out.println("Add debugging point here");

        }


        NormalizerMinMaxScaler n = new NormalizerMinMaxScaler();
        n.fit(dataSet);

        int number = 20;
        INDArray x = Nd4j.linspace(Math.round(n.getMin().getFloat(0)), Math.round(n.getMax().getFloat(0)), number).reshape(number, 1);
        INDArray y = network.output(x);

        DataSet result = new DataSet();
        result.setFeatures(x);
        result.setLabels(y);

        if (!saveLocation.isEmpty()){

            try{
                network.save(new File(saveLocation));

            }catch (Exception e)
            {
                e.printStackTrace();
            }
        }


        return result;

    }


    private static MultiLayerNetwork diffNet(){

        int seed = 12345;
        int numInputs = 1;
        int numOutputs = 1;
        int numHidden = 50;



        // Hook up one input to the one output.
        // The resulting model is a straight line.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(.01, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHidden).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nIn(numHidden).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(5));


        return net;

    }


    //endregion





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
