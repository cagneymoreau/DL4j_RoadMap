package INDarrays_dataSets;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

/**
 *  Basics to know are building INDarrays which are the core building blocks of mahine learning
 *  which allow fast linear algeabra calcs
 *
 */


public class ArrayDataExample {

    public static void main(String[] args)
    {
        DataSet d = getSineDataSet(10, 100, .01);

        ArrayList<DataSet> list = new ArrayList<>();
        list.add(d);

        plot2DScatterGraph(list, "testing sine");
    }


    private static DataSet mathFunct(String desc, int period, int sampleSize, double noise){

        switch ( desc){


            case "sine":

                return getSineDataSet(period,sampleSize, noise);

            case "line":

                break;

            case "xor":

                break;

            case "saw":

                break;



        }

        return new DataSet();
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


    private static DataSet getLinearDataSet(int dim) {



        double slope = -3.4;
        double bVal = 4.2;

        INDArray x = Nd4j.rand(dim,1);
        INDArray y = Nd4j.zeros(dim,1);
        y = y.add(slope);
        //System.out.println(x);
        // System.out.println(y);

        x = x.mul(2);
        //System.out.println(x);
        y = y.mul(x);
        //System.out.println(y);
        y = y.add(bVal);
        INDArray g = Nd4j.randn(dim,1);
        //g = g.mul(.5);
        y = y.add(g);
        //System.out.println(y);

        DataSet ds  = new DataSet();
        ds.setFeatures(x);
        ds.setLabels(y);

        return ds;

    }



    private static DataSet gettriangleWaveDataSet(int dim)
    {

        INDArray x = Nd4j.linspace(-10, 10, dim).reshape(dim, 1);

        final double sawtoothPeriod = 4.0;
        //the input data is the intervals at which the wave is being calculated
        final double[] xd2 = x.data().asDouble();
        final double[] yd2 = new double[xd2.length];
        for (int i = 0; i < xd2.length; i++) {  //Using the sawtooth wave function, find the values at the given intervals
            yd2[i] = 2 * (xd2[i] / sawtoothPeriod - Math.floor(xd2[i] / sawtoothPeriod + 0.5));
        }


        INDArray sawArr = Nd4j.create(yd2, xd2.length, 1);  //Column vector
        sawArr = sawArr.add((Nd4j.randn(dim,1).mul(0.5)));

        DataSet ds = new DataSet();
        ds.setFeatures(x);
        ds.setLabels(sawArr);

        //System.out.println(ds);

        return ds;

    }


    private static DataSet getXor(){

        INDArray func = Nd4j.zeros(4,2);
        INDArray res = Nd4j.zeros(4,1);

        //00= 0
        func.putScalar(new int[]{0,0}, 0);
        func.putScalar(new int[]{0,1}, 0);
        res.putScalar(new int[]{0}, 0);
        //01= 1
        func.putScalar(new int[]{1,0}, 0);
        func.putScalar(new int[]{1,1}, 1);
        res.putScalar(new int[]{1}, 1);
        //10= 1
        func.putScalar(new int[]{2,0}, 1);
        func.putScalar(new int[]{2,1}, 0);
        res.putScalar(new int[]{2}, 1);
        //11= 0
        func.putScalar(new int[]{3,0}, 1);
        func.putScalar(new int[]{3,1}, 1);
        res.putScalar(new int[]{3}, 0);

        DataSet data = new DataSet();
        data.setFeatures(func);
        data.setLabels(res);

        return data;

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
