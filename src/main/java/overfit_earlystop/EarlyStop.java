package overfit_earlystop;

import dual_lstm_csv_manipulation.MergeCSV;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.datavec.local.transforms.misc.StringToWritablesFunction;
import org.datavec.local.transforms.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.RegressionScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
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
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 *
 * Overfitting can be useful because eventually nn overfit should occur.
 * But overfitting is not the goal. This code shows you
 *
 * 1) splitting dataset on the hard drive. You can split datasets within your ram
 * but its not always doable with any non trival dataset
 *
 * 2) How to stop from fitting when you prediction are the best match to unseen data
 *
 *
 */

public class EarlyStop {

    private enum deleteRow {none, first, last}
    public enum combination {simple, combo, tech, full}
    public static String path = "D:\\Dropbox\\Apps\\RoadMap\\src\\main\\java\\overfit_earlystop\\";


    public static void main(String[] args) throws Exception
    {


        /**
         * This uses the api to simplify a dataset for use
         */

        createSimpleDataSet(path + "mmm.csv", path + "simple_mmm.csv", deleteRow.last, false);
        createSeqSingleLabelDataSet(path + "mmm.csv", path + "label_mmm.csv", deleteRow.first);

        EarlyStop ll = new EarlyStop();
        ll.trainandTestSimpleSingle();






    }






    private void trainandTestSimpleSingle()throws Exception
    {
        List<String> list = retreiveFullCSV(path + "simple_mmm.csv");
        List<String> label = retreiveFullCSV(path + "label_mmm.csv");

        int trainLength = (int) Math.round(list.size() * .7);

        rewriteFullCSV(path + "train_feat_mmm.csv", list.subList(0, trainLength));
        rewriteFullCSV(path + "train_label_mmm.csv",  label.subList(0, trainLength));

        rewriteFullCSV(path + "test_feat_mmm.csv", list.subList(trainLength, list.size()));
        rewriteFullCSV(path + "test_label_mmm.csv", label.subList(trainLength, list.size()));


        DataSetIterator trainIter = sequenceIterator("train_feat_mmm.csv",  "train_label_mmm.csv", 1);
        DataSetIterator testIter = sequenceIterator("test_feat_mmm.csv",  "test_label_mmm.csv", 1);


        DataNormalization normalization = new NormalizerMinMaxScaler();
        normalization.fitLabel(true);
        normalization.fit(trainIter);
        trainIter.reset();

        trainIter.setPreProcessor(normalization);
        testIter.setPreProcessor(normalization);


        MultiLayerNetwork network = getStockNetwork(trainIter.inputColumns(), 32, false, "simple");


        //get earlystoppingtrainer
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                //.epochTerminationConditions(new MaxEpochsTerminationCondition(1))
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(10))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(120, TimeUnit.MINUTES))
                .scoreCalculator( new RegressionScoreCalculator(RegressionEvaluation.Metric.MSE, testIter))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(path ))
                .build();


        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, trainIter);

        EarlyStoppingResult result = trainer.fit();

        //result.setBestModel(null);
        //FileManager.serialize(savedirectory  + "\\" + Constants.RESULT_SUFFIX, result);

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());



        network.rnnClearPreviousState();

        INDArray out = Nd4j.zeros(1);
        ArrayList<INDArray> output = new ArrayList<>();

        testIter.reset();
        while (testIter.hasNext()) {
            out = network.output(testIter.next().getFeatures());
            output.add(out);
        }




        normalization.revertLabels(out);

        ArrayList<DataSet> toChart = new ArrayList<>();
        DataSet results = new DataSet();
        results.setFeatures(out);
        toChart.add(results);
        toChart.add(readCSVDataSet(path + "simple_mmm.csv", 1500, -1));



        plotFitCompare(toChart.get(1), toChart.get(0), trainLength , "simplesingle");
    }


    // using datasetiter api
    public static DataSet readCSVDataSet(String path, int maxfileLength, int labelColumn) throws Exception {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(path)));

        if (labelColumn == -1)
        {
            DataSetIterator it = new RecordReaderDataSetIterator(rr, maxfileLength);
            return it.next();
        }else {
            DataSetIterator it = new RecordReaderDataSetIterator(rr, maxfileLength, labelColumn, labelColumn, true);
            return it.next();
        }

    }


    //List of files int a sequence iterator
    public static DataSetIterator sequenceIterator(String feat, String label, int batch) throws Exception
    {
        String featureToPath = "" ;
        String labelToPath = "";


                featureToPath = path + feat;
                labelToPath = path + label;



        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
        featureReader.initialize(new FileSplit(new File(featureToPath)));

        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
        labelReader.initialize(new FileSplit(new File(labelToPath)));


        DataSetIterator it = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, batch, 1, true);



        return it;

    }


    // index 0 will be train and index 1 will be test
    public static ArrayList<DataSet> splitDataSet(DataSet data, double split){

        ArrayList<DataSet> out = new ArrayList<>();

        INDArray features = data.getFeatures();
        INDArray labels = data.getLabels();

        INDArray trainFeat;
        INDArray trainLabel;
        INDArray testFeat;
        INDArray testLabel;

        long[] shape = features.shape();

        if (split >= 1 || split < 0) split = 0;


            //if array is 2d - split last rows
        else if (shape.length < 3){



        }


        //if array is 3d with depth 1
        else if (shape.length == 3 ){

            //if ( shape[0] == 1){

            long testLength = (int) Math.round(shape[2] * (1-split));

            testFeat = features.get( NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(shape[2]-testLength, shape[2]));
            testLabel = labels.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(shape[2]-testLength, shape[2]));

            DataSet test = new DataSet();
            test.setFeatures(testFeat);
            test.setLabels(testLabel);

            trainFeat = features.get( NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, shape[2]-testLength));
            trainLabel = labels.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, shape[2]-testLength));

            DataSet train = new DataSet();
            train.setFeatures(trainFeat);
            train.setLabels(trainLabel);

            out.add(train);
            out.add(test);









        }

        //if array is 3d
        else if (shape.length == 4){




        }






        return out;
    }



    public static MultiLayerNetwork getStockNetwork(int input, int hiddenLayerSize, boolean useExisting, String path) throws Exception
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



    //creates a simple version of stock csv
    //delete last for many to many seq
    private static void createSimpleDataSet(String fromPath, String toPath, deleteRow act, boolean keepDates) throws Exception
    {

        //year, month, day, high, low, close, adjclose, volume

        Schema inputSchema = new Schema.Builder()
                .addColumnInteger("year")
                .addColumnInteger("month")
                .addColumnInteger("day")
                .addColumnDouble("high",0.00,null, false,false)
                .addColumnDouble("low",0.00,null, false,false)
                .addColumnDouble("close",0.00,null, false,false)
                .addColumnDouble("adjclose")
                .addColumnDouble("volume",0.00,null, false,false)
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .removeColumns("year", "month", "day") // TODO: 7/30/2020 adjclose?

                .build();



        List<String> in = retreiveFullCSV(fromPath);

        removeNonNumbers(in);

        switch (act){
            case last:
                in =  in.subList(0, in.size()-1);
                break;
            case first:
                in = in.subList(1, in.size());
                break;
        }


        List<List<Writable>> almost = new ArrayList<>();
        StringToWritablesFunction wt = new StringToWritablesFunction(new CSVRecordReader());

        for (String s :
                in) {
            almost.add(wt.apply(s));
        }

        List<List<Writable>> processedData;
        if (!keepDates) {
            processedData = LocalTransformExecutor.execute(almost, tp);
        }else{
            processedData = almost;
        }


        StringBuilder sb = new StringBuilder();

        for (List ll:
                processedData) {

            sb.append(new WritablesToStringFunction(",").apply(ll));
            sb.append("\n");

        }
        sb.setLength(sb.length()-1);



        rewriteFullCSV(toPath, sb.toString());



    }



    //create a single value label from each close price
    //delete first for many to many seq
    private static void createSeqSingleLabelDataSet(String fromPath, String toPath, deleteRow act) throws Exception
    {
        //year, month, day, high, low, close, adjclose, volume

        Schema inputSchema = new Schema.Builder()
                .addColumnInteger("year")
                .addColumnInteger("month")
                .addColumnInteger("day")
                .addColumnDouble("high",0.00,null, false,false)
                .addColumnDouble("low",0.00,null, false,false)
                .addColumnDouble("close",0.00,null, false,false)
                .addColumnDouble("adjclose")
                .addColumnDouble("volume",0.00,null, false,false)

                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .removeColumns("year", "month", "day", "high", "low", "adjclose", "volume")
                .build();



        List<String> in = retreiveFullCSV(fromPath);

        removeNonNumbers(in);

        switch (act){
            case last:
                in =  in.subList(0, in.size()-1);
                break;
            case first:
                in = in.subList(1, in.size());
                break;
        }


        List<List<Writable>> almost = new ArrayList<>();
        StringToWritablesFunction wt = new StringToWritablesFunction(new CSVRecordReader());

        for (String s :
                in) {
            almost.add(wt.apply(s));
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(almost, tp);


        StringBuilder sb = new StringBuilder();

        for (List ll:
                processedData) {

            sb.append(new WritablesToStringFunction(",").apply(ll));
            sb.append("\n");

        }
        sb.setLength(sb.length()-1);



        rewriteFullCSV(toPath, sb.toString());



    }


    private static void removeNonNumbers(List<String> list){

        for (int a = 0; a < list.size(); a++) {


            String[] v = list.get(a).split(",");
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < v.length; i++) {

                try {
                    Double.valueOf(v[i]);
                }catch (Exception e){
                    v[i] = "0.0";
                }

                sb.append(v[i]).append(",");
            }
            sb.setLength(sb.length()-1);
            list.set(a, sb.toString());
        }

    }


    public static void plotFitCompare(DataSet orig, DataSet res, int offset, String title)
    {
        XYSeriesCollection data = new XYSeriesCollection();

        long[] shape = orig.getFeatures().shape();

        //cycle through columns for each data type
        for (int i = 0; i < shape[1]; i++) {
            XYSeries series = new XYSeries("Ground truth");

            //cycle through rows
            for (int b = 0; b < shape[0]; b++) {

                series.add(b, orig.getFeatures().getFloat(b,i));
                //series.add( new Day(day[b], month[b], year[b]) , d.getFeatures().getFloat(b,i,0));

            }
            data.addSeries(series);

        }

        if (res != null) {

            long[] shapeSec = res.getFeatures().shape();

            if (offset == -1) {
                offset = (int) shape[0] - (int) shapeSec[2];
            }

            //cycle through row length for each data type
            for (int i = 0; i < shapeSec[1]; i++) {
                XYSeries series = new XYSeries("Prediction");

                //cycle through column for each time series
                for (int b = 0; b < shapeSec[2]; b++) {

                    series.add((b + offset), res.getFeatures().getFloat(0, i, b));
                    //series.add( new Day(day[b], month[b], year[b]) , d.getFeatures().getFloat(b,i,0));

                }
                data.addSeries(series);

            }


        }

        String xAxisLabel = "Date";
        String yAxisLabel = "Price";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, data, PlotOrientation.VERTICAL, legend,tooltips,urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);


    }





    public  static void rewriteFullCSV(String path, String vals)throws Exception{


        FileWriter filewriter = new FileWriter(path);

        filewriter.append(vals);

        filewriter.flush();
        filewriter.close();

    }


    public  static List<String> retreiveFullCSV(String path)throws Exception {

        List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);

        return lines;
    }



    public  static void rewriteFullCSV(String path, List<String> newVals)throws Exception{

        FileWriter filewriter = new FileWriter(path);

        for (String st :
                newVals) {

            filewriter.append(st);
            filewriter.append("\n");

        }

        filewriter.flush();
        filewriter.close();

    }




}
