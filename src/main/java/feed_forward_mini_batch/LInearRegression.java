package feed_forward_mini_batch;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.time.LocalTime;
import java.util.ArrayList;

/**
 * Here you can see a simple regression and in the feed foward section I also play with batch sizes
 *
 *
 */


public class LInearRegression {


    public static void main(String[] args)
    {
        testLinear();
    }


    public static void testLinear()
    {

        ArrayList<DataSet> results = new ArrayList<>();

        DataSet ds = getLinearDataSet(200);

        FeedForward f = new FeedForward();
        DataSet res = f.ttLinearRegression(ds);

        results.add(ds);
        results.add(res);


        plot2DScatterGraph(results, LocalTime.now().toString());



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
