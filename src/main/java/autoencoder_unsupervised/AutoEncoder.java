package autoencoder_unsupervised;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * This is a dataset of a large portion of sp500 stocks
 * each tick is the daily change in price as an absolute
 * precentage of the previous days price value
 *
 *
 */


public class AutoEncoder {

    private final static String path = "D:\\Dropbox\\Apps\\RoadMap\\src\\main\\java\\autoencoder_unsupervised\\";

    public static void main(String[] args)throws Exception
    {
        postStocksToLatentSpace();
    }


    static void postStocksToLatentSpace()throws Exception
    {

        DataSetIterator iter = basicIterator(path + "auto.csv", 0, 1);

        DataSet datas = iter.next();

        DataNormalization normalization = new NormalizerMinMaxScaler();
        normalization.fit(iter);
        iter.setPreProcessor(normalization);
        //normalization.preProcess(datas);

        MultiLayerNetwork network = getlittleEncoder(Activation.IDENTITY);


        for (int i = 0; i < 1; i++) {

            while (iter.hasNext())
            {
                INDArray ind = iter.next().getFeatures();

                network.fit(ind, ind);
                List<INDArray> ls = network.feedForwardToLayer(2, ind);
                ls.clear();
            }
            iter.reset();

        }

        iter.reset();

        INDArray t = iter.next().getFeatures();

        List<INDArray> lq  = network.feedForwardToLayer(2,t);


        /*

        INDArray ff = iter.next().getFeatures();

        for (int i = 0; i < 100; i++) {
            network.fit(ff, ff);
        }
        iter.reset();



        for (int i = 0; i < 2; i++) {
            network.pretrain(iter);
        }

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) network.getLayer(0);

         */

        INDArray netP = network.getLayer(0).params();


        ModelSerializer.writeModel(network, path + "model_ident.zip", true);

        TransferLearning.Builder builder = new TransferLearning.Builder(network)
                .removeLayersFromOutput(3);
        //builder.fineTuneConfiguration(new FineTuneConfiguration());

        //builder.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(2).nOut(2).build());
        MultiLayerNetwork encoder = builder.build();
        System.out.println(encoder.summary());

        INDArray encP = encoder.getLayer(0).params();
        INDArray netTwoP = network.getLayer(0).params();


        iter.reset();

        ArrayList<INDArray> output = new ArrayList<>();

        while (iter.hasNext()){

            INDArray f = iter.next().getFeatures();
            INDArray out = encoder.output(f);
            List<INDArray> l  = network.feedForwardToLayer(2,f);
            //INDArray out =  vae.activate(f, false, LayerWorkspaceMgr.noWorkspaces());
            output.add(out);

        }

        plotScatterGraphPairs(output, path + "auto.csv", 0);

    }



    // Iterate feed forward row by row
    public static DataSetIterator basicIterator(String path, int skipRow, int skipColumn)throws Exception
    {
        if (skipColumn > 0){

            CSVBenchMarkRecordReader recordReader = new CSVBenchMarkRecordReader(skipRow, skipColumn);
            recordReader.initialize(new FileSplit(new File(path)));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 1);

            return iterator;

        }else {

            CSVRecordReader recordReader = new CSVRecordReader(skipRow);
            recordReader.initialize(new FileSplit(new File(path)));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 1);

            return iterator;

        }
    }




    //plot 2d indarrays onto a chart
    public static void plotScatterGraphPairs(ArrayList<INDArray> DataSetList, String path, int colunmIndex) throws  Exception {

        XYSeriesCollection dataset = new XYSeriesCollection();

        List<String> names = retreiveColumnFullCSV(path, colunmIndex);

        int counter = 0;
        for (INDArray ind : DataSetList) {

            XYSeries series = new XYSeries(names.get(counter));

            double x = ind.getDouble(0);
            double y = ind.getDouble(1);

            series.add(x,y);


            dataset.addSeries(series);
            counter++;
        }

        String title = "title";
        String xAxisLabel = "xAxisLabel";
        String yAxisLabel = "yAxisLabel";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = false;
        boolean tooltips = true;
        boolean urls = true;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, dataset, orientation, legend, tooltips, urls);
        //JFreeChart chart = ChartFactory.createLineChart(title,xAxisLabel, yAxisLabel, dataset, true, false, false);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);
    }


    public static MultiLayerNetwork getlittleEncoder(Activation activation){

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(.1))
                .list()
                .layer(new DenseLayer.Builder().nIn(512).nOut(64).activation(activation).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(16).activation(activation).build())
                .layer(new DenseLayer.Builder().nIn(16).nOut(2).activation(activation).build())
                .layer(new DenseLayer.Builder().nIn(2).nOut(16).activation(activation).build())
                .layer(new DenseLayer.Builder().nIn(16).nOut(64).activation(activation).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(activation).nIn(64).nOut(512).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        return net;
    }


    public static List<String> retreiveColumnFullCSV(String path, int column) throws Exception {

        List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);

        for (int i = 0; i < lines.size(); i++) {
            lines.set(i, lines.get(i).split(",")[column]);
        }

        return lines;
    }



}
