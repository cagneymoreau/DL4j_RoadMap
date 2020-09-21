package recurrent_regression;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;





public class LSTMNet {

    MultiLayerNetwork network;
    DataNormalization normy;
    int hist = 30;


    public LSTMNet(){

        /*
        double[] test = new double[]{1,2,3,4,5,6};

        INDArray y = Nd4j.create(test, 1,1,6);
        System.out.println(y);

        System.out.println(y);
        */

    }


    public DataSet trainTest(DataSet dataSet, double trainSplit, int epochs)
    {

        buildNet(1);

        dataSet.setFeatures(dataSet.getFeatures().reshape(dataSet.getFeatures().length(), 1,1));
        dataSet.setLabels(dataSet.getLabels().reshape(dataSet.getLabels().length(),1,1));

        SplitTestAndTrain thisData = dataSet.splitTestAndTrain(trainSplit);

        INDArray oldTrainfeatures = thisData.getTrain().getFeatures();
        INDArray oldtestFeatures = thisData.getTest().getFeatures();

        // set x axis
        thisData.getTrain().setFeatures(thisData.getTrain().getLabels());

        //set y axis
        thisData.getTrain().getLabels().putScalar(thisData.getTrain().getLabels().length()-1, thisData.getTrain().getFeatures().getFloat(0));
        for (int i = 0; i < thisData.getTrain().getLabels().length()-1; i++) {

            thisData.getTrain().getLabels().putScalar(i, thisData.getTrain().getFeatures().getFloat(i + 1));
        }

        DataSet newDataSet = new DataSet();
        newDataSet.setFeatures(thisData.getTrain().getFeatures());
        newDataSet.setLabels(thisData.getTrain().getLabels());

        final List<DataSet> list = newDataSet.asList();
        //Collections.shuffle(list, new Random(123)); //Dont shuffle as order is important

        DataSetIterator iter = new ListDataSetIterator<>(list, 50);

        //DataSetIterator iter = new ExistingDataSetIterator(thisData.getTrain());

        network.fit(iter, epochs);

        network.rnnClearPreviousState();



        INDArray ind = Nd4j.zeros(1,1,1); //holder for variable

        long start = thisData.getTrain().getFeatures().length()- hist; //where we start priming the lstm
        INDArray out = Nd4j.zeros(thisData.getTest().getLabels().length() + hist,1); //label
        INDArray outFeat = Nd4j.zeros(out.length());    //feature
        //ind.putScalar(0,thisData.getTrain().getFeatures().getDouble(thisData.getTrain().getFeatures().length()-1));


        for (int i = 0; i < out.length(); i++) {

            outFeat.putScalar(i, dataSet.getFeatures().getFloat(start)); //set our x "feature"

            if (i < hist){
                ind.putScalar(0,thisData.getTrain().getFeatures().getDouble(start-1));

            }


            ind = network.rnnTimeStep(ind);
            out.putScalar(i, ind.getDouble(0));
            start++;

        }

        //System.out.println(out);
        //System.out.println(thisData.getTest().getLabels().reshape(thisData.getTest().getFeatures().length(), 1));

        DataSet outDS = new DataSet();
        outDS.setLabels(out);
        outDS.setFeatures(outFeat.reshape(out.length(), 1));

        return outDS;

    }


    private void buildNet(int sampleLength){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)


                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(hist).build())

                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID).nIn(hist).nOut(sampleLength).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(20));
        network = net;

    }





}
